# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This package contains round behaviours of CollectDefiApp."""

from abc import ABC
from pathlib import Path
import random
from tempfile import mkdtemp
from typing import Dict, Generator, List, Optional, Set, Type, cast

from openai import OpenAI

from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.niron.skills.collect_defillama_abci.models import (
    CoingeckoSpecs,
    DefiLlamaSpecs,
    Params,
    SharedState,
)
from packages.niron.skills.collect_defillama_abci.payloads import (
    CollectRandomnessPayload,
    DefiLlamaPullPayload,
    DecisionMakingPayload,
    ExecuteLLMPayload,
    SelectKeeperPayload,
    TxPreparationPayload,
)
from packages.niron.skills.collect_defillama_abci.rounds import (
    CollectRandomnessRound,
    DefiLlamaPullRound,
    DecisionMakingRound,
    Event,
    CollectDefiApp,
    ExecuteLLMRound,
    SelectKeeperRound,
    SynchronizedData,
    TxPreparationRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)
from packages.valory.skills.transaction_settlement_abci.rounds import TX_HASH_LENGTH

from packages.niron.skills.collect_defillama_abci.ml.preprocessing import preprocess_stablecoin_data
from packages.niron.skills.collect_defillama_abci.ml.stats import apply_holtwinters

# Define some constants
ZERO_VALUE = 0
HTTP_OK = 200
GNOSIS_CHAIN_ID = "gnosis"
EMPTY_CALL_DATA = b"0x"
SAFE_GAS = 0
VALUE_KEY = "value"
TO_ADDRESS_KEY = "to_address"
METADATA_FILENAME = "metadata.json"

SYSTEM_MESSAGE = """You are a statistical data scientist.
                You are tasked with providing analysis on the risk and opportunity of the DeFi sector in the near terms.
                Based on the following a trend statistics and Holt Winters function on the global stablecoin market.
                Positive trend in the stablecoin market represents new money inflows into the DeFi sector.
                Analyze the signals and provide a recommendation, always reply within a strictly `JSON` format.
                Do not use markdown or formatting, your output is chained in a data pipeline, use strict data formats.
                - signal: string(Enum): bullish | bearish | neutral,
                - confidence: <float between 0 and 1>
                - forecast: <string: commentary>"""

USER_PROMPT = """
    The recent analysis of the stablecoin market based on Holt-Winters forecasting:
    
    Model Parameters:
    - Trend Direction: {trend_direction}
    - Trend Strength: {trend_strength:.2%}
    - Forecast Mean: ${forecast_mean:,.2f}
    - Last Actual Value: ${last_actual_value:,.2f}
    - Model Confidence (Alpha): {alpha:.3f}
    - Trend Factor (Beta): {beta:.3f}
    - Seasonal Factor (Gamma): {gamma:.3f}
    
    Based on these metrics, what is your market recommendation? 
    Please provide a signal (bullish/bearish/neutral) with confidence level and explanation.
    """


class DefiCollectorBaseBehaviour(BaseBehaviour, ABC):  # pylint: disable=too-many-ancestors
    """Base behaviour for the learning_abci behaviours."""

    @property
    def params(self) -> Params:
        """Return the params. Configs go here"""
        return cast(Params, super().params)

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data. This data is common to all agents"""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def local_state(self) -> SharedState:
        """Return the local state of this particular agent."""
        return cast(SharedState, self.context.state)

    @property
    def defillama_specs(self) -> DefiLlamaSpecs:
        """Get the DefiLlama api specs."""
        return self.context.defillama_specs

    @property
    def metadata_filepath(self) -> str:
        """Get the temporary filepath to the metadata."""
        return str(Path(mkdtemp()) / METADATA_FILENAME)

    def get_sync_timestamp(self) -> float:
        """Get the synchronized time from Tendermint's last block."""
        now = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        return now

class CollectRandomnessBehaviour(DefiCollectorBaseBehaviour):
    """Retrieve randomness."""

    matching_round = CollectRandomnessRound

    def async_act(self) -> Generator:
        """
        Check whether tendermint is running or not.

        Steps:
        - Do a http request to the tendermint health check endpoint
        - Retry until healthcheck passes or timeout is hit.
        - If healthcheck passes set done event.
        """
        if self.context.randomness_api.is_retries_exceeded():
            # now we need to wait and see if the other agents progress the round
            yield from self.wait_until_round_end()
            self.set_done()
            return

        api_specs = self.context.randomness_api.get_spec()
        http_message, http_dialogue = self._build_http_request_message(
            method=api_specs["method"],
            url=api_specs["url"],
        )
        response = yield from self._do_request(http_message, http_dialogue)
        observation = self.context.randomness_api.process_response(response)

        if observation:
            self.context.logger.info(f"Retrieved DRAND values: {observation}.")
            payload = CollectRandomnessPayload(
                self.context.agent_address,
                observation["round"],
                observation["randomness"],
            )
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()
            self.set_done()
        else:
            self.context.logger.error(
                f"Could not get randomness from {self.context.randomness_api.api_id}"
            )
            yield from self.sleep(self.params.sleep_time)
            self.context.randomness_api.increment_retries()

    def clean_up(self) -> None:
        """
        Clean up the resources due to a 'stop' event.

        It can be optionally implemented by the concrete classes.
        """
        self.context.randomness_api.reset_retries()

class SelectKeeperBehaviour(DefiCollectorBaseBehaviour, ABC):
    """Select the keeper agent."""

    matching_round = SelectKeeperRound

    def async_act(self) -> Generator:
        """
        Do the action.

        Steps:
        - Select a keeper randomly.
        - Send the transaction with the keeper and wait for it to be mined.
        - Wait until ABCI application transitions to the next round.
        - Go to the next behaviour (set done event).
        """

        participants = sorted(self.synchronized_data.participants)
        random.seed(self.synchronized_data.most_voted_randomness, 2)  # nosec
        index = random.randint(0, len(participants) - 1)  # nosec

        keeper_address = participants[index]

        self.context.logger.info(f"Selected a new keeper: {keeper_address}.")
        payload = SelectKeeperPayload(self.context.agent_address, keeper_address)

        yield from self.send_a2a_transaction(payload)
        yield from self.wait_until_round_end()

        self.set_done()

class DefiLlamaPullBehaviour(DefiCollectorBaseBehaviour, ABC): # pylint: disable=too-many-ancestors
    """DefiLlamaPullBehaviour"""

    matching_round: Type[AbstractRound] = DefiLlamaPullRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            self.context.logger.info(f"Starting DefiLlamaPullBehaviour from Agent: {sender}")
            stablecoins_data = yield from self.get_stablecoins_history()
            self.context.logger.info(f"Attempting upload to IPFS")

            ipfs_hash = yield from self.send_stablecoins_to_ipfs(stablecoins_data)

            payload = DefiLlamaPullPayload(
                sender=sender,
                stablecoins_history=str(stablecoins_data),
                stablecoins_ipfs_hash=ipfs_hash)
         

        yield from self.send_a2a_transaction(payload)
        yield from self.wait_until_round_end()
     
        self.set_done()

    def get_stablecoins_history(self) -> Generator[None, None, Optional[dict]]:
        """Get Stablecoins Data from DefiLlama"""
        self.context.logger.info("Attempting to fetch Stablecoins Data")
        specs = self.defillama_specs.get_spec()
        raw_response = yield from self.get_http_response(**specs)
        stablecoins_history = eval(raw_response.body)
    
        self.context.logger.info(f"Got Stablecoins Data Example in place 0: {stablecoins_history[0]}")

        return stablecoins_history
   
    def send_stablecoins_to_ipfs(self, stablecoins_data: dict) -> Generator[None, None, str]:
        """Send Stablecoins data to IPFS"""
        data = stablecoins_data
        self.context.logger.info("Uploading Stablecoins data to IPFS")
        stablecoins_ipfs_hash = yield from self.send_to_ipfs(
            filename=self.metadata_filepath, obj=data, filetype=SupportedFiletype.JSON
        )
        self.context.logger.info(f"Uploading object with hash: {stablecoins_ipfs_hash}")

        return stablecoins_ipfs_hash

class ExecuteLLMBehaviour(DefiCollectorBaseBehaviour):
    """PostPreparationBehaviour"""

    matching_round = ExecuteLLMRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        
        raw_data = self.synchronized_data.stablecoins_history
        
        # Preprocess the data
        df = preprocess_stablecoin_data(raw_data)
        
        # Apply Holt-Winters
        model_params = apply_holtwinters(df)

        # Prepare the prompt
        formatted_prompt = USER_PROMPT.format(**model_params)
        
        # Get LLM response
        response = self.get_llm_response(
            system_message=SYSTEM_MESSAGE,
            user_prompt=formatted_prompt
        )

        if response:
            self.context.logger.info(f'########### LLM Response ##########\n{response}')

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload = ExecuteLLMPayload(
                sender=sender, 
                llm_response=response, 
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    # Not generator method
    def get_llm_response(self, system_message: str, user_prompt: str) -> str:
        """Get a response from the LLM."""
        client = OpenAI(api_key=self.params.openai_api_key)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            model='gpt-4-turbo',
            messages=messages,
            temperature=0.7,
            n=1,
            timeout=120,
            stop=None,
        )

        if response:
            return response.choices[0].message.content
        else:
            return False

class DecisionMakingBehaviour(DefiCollectorBaseBehaviour):
    """DecisionMakingBehaviour"""
    
    matching_round: Type[AbstractRound] = DecisionMakingRound
    
    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            
            llm_analysis: dict = eval(self.synchronized_data.llm_response)
            
            if llm_analysis is None:
                self.context.logger.error("LLM Analysis is None. Cannot proceed.")
                event = Event.ERROR.value  # Or handle as appropriate
            else:
                self.context.logger.info(f"LLM Analysis: {llm_analysis}")
                
                if llm_analysis['signal'] == 'bullish' and llm_analysis['confidence'] > 0.7:
                    self.context.logger.info("LLM Response Bullish. Transacting.")
                    event = Event.TRANSACT.value
                else:
                    event = Event.DONE.value
                
            self.context.logger.info(f"Agent {sender} decided to {event}.")
            payload = DecisionMakingPayload(sender=sender, event=event)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

class TxPreparationBehaviour(
    DefiCollectorBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """TxPreparationBehaviour"""

    matching_round: Type[AbstractRound] = TxPreparationRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address

            # Get the transaction hash
            tx_hash = yield from self.get_tx_hash()

            payload = TxPreparationPayload(
                sender=sender, tx_submitter=self.auto_behaviour_id(), tx_hash=tx_hash
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_tx_hash(self) -> Generator[None, None, Optional[str]]:
        """Get the transaction hash"""

        self.context.logger.info("Preparing a native transaction")
        tx_hash = yield from self.get_native_transfer_safe_tx_hash()
        return tx_hash

    def get_native_transfer_safe_tx_hash(self) -> Generator[None, None, Optional[str]]:
        """Prepare a native safe transaction"""

        # Transaction data
        # This method is not a generator, therefore we don't use yield from
        data = self.get_native_transfer_data()

        # Prepare safe transaction
        safe_tx_hash = yield from self._build_safe_tx_hash(**data)
        self.context.logger.info(f"Native transfer hash is {safe_tx_hash}")

        return safe_tx_hash

    def get_native_transfer_data(self) -> Dict:
        """Get the native transaction data"""
        # Send 1 wei to the recipient
        data = {VALUE_KEY: 1, TO_ADDRESS_KEY: self.params.transfer_target_address}
        self.context.logger.info(f"Native transfer data is {data}")
        return data

    def _build_safe_tx_hash(
        self,
        to_address: str,
        value: int = ZERO_VALUE,
        data: bytes = EMPTY_CALL_DATA,
        operation: int = SafeOperation.CALL.value,
    ) -> Generator[None, None, Optional[str]]:
        """Prepares and returns the safe tx hash for a multisend tx."""

        self.context.logger.info(
            f"Preparing Safe transaction [{self.synchronized_data.safe_contract_address}]"
        )

        # Prepare the safe transaction
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=to_address,
            value=value,
            data=data,
            safe_tx_gas=SAFE_GAS,
            chain_id=GNOSIS_CHAIN_ID,
            operation=operation,
        )

        # Check for errors
        if response_msg.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                "Couldn't get safe tx hash. Expected response performative "
                f"{ContractApiMessage.Performative.STATE.value!r}, "  # type: ignore
                f"received {response_msg.performative.value!r}: {response_msg}."
            )
            return None

        # Extract the hash and check it has the correct length
        tx_hash: Optional[str] = response_msg.state.body.get("tx_hash", None)

        if tx_hash is None or len(tx_hash) != TX_HASH_LENGTH:
            self.context.logger.error(
                "Something went wrong while trying to get the safe transaction hash. "
                f"Invalid hash {tx_hash!r} was returned."
            )
            return None

        # Transaction to hex
        tx_hash = tx_hash[2:]  # strip the 0x

        safe_tx_hash = hash_payload_to_hex(
            safe_tx_hash=tx_hash,
            ether_value=value,
            safe_tx_gas=SAFE_GAS,
            to_address=to_address,
            data=data,
            operation=operation,
        )

        self.context.logger.info(f"Safe transaction hash is {safe_tx_hash}")

        return safe_tx_hash


class DefiCollectorRoundBehaviour(AbstractRoundBehaviour):
    """DefiCollectorRoundBehaviour"""

    initial_behaviour_cls = DefiLlamaPullBehaviour
    abci_app_cls = CollectDefiApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [  # type: ignore
        CollectRandomnessBehaviour,
        SelectKeeperBehaviour,
        DefiLlamaPullBehaviour,
        ExecuteLLMBehaviour,
        DecisionMakingBehaviour,
        TxPreparationBehaviour,
    ]
