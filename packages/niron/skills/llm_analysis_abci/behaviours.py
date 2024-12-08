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

"""This package contains round behaviours of LLM Analysis App."""

from abc import ABC
from pathlib import Path
import random
from tempfile import mkdtemp
from typing import Dict, Generator, List, Optional, Set, Type, cast

from openai import OpenAI

from packages.valory.skills.abstract_round_abci.base import AbstractRound

from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.niron.skills.llm_analysis_abci.models import (
    DefiLlamaSpecs,
    Params,
    SharedState,
)
from packages.niron.skills.llm_analysis_abci.payloads import (
    ExecuteLLMPayload
)
from packages.niron.skills.llm_analysis_abci.rounds import (
    ExecuteLLMRound,
    Event,
    LLMAnalysisAbciApp,
    FinalLLMRound,
    SynchronizedData,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)
from packages.valory.skills.transaction_settlement_abci.rounds import TX_HASH_LENGTH

from packages.niron.skills.llm_analysis_abci.ml.preprocessing import preprocess_stablecoin_data
from packages.niron.skills.llm_analysis_abci.ml.stats import apply_holtwinters


# Define some constants
ZERO_VALUE = 0
HTTP_OK = 200
GNOSIS_CHAIN_ID = "gnosis"
EMPTY_CALL_DATA = b"0x"
SAFE_GAS = 0
VALUE_KEY = "value"
TO_ADDRESS_KEY = "to_address"
METADATA_FILENAME = "metadata.json"

SYSTEM_MESSAGE = """You are a statistical scientist.
                You are given a triple-smoothing/holt winters function for state of the global stablecoin market.
                Positive growth in the stablecoin market represents money inflows into the DeFi sector.
                Analyze the signals and provide a recommendation.
                - signal: bullish | bearish | neutral,
                - confidence: <float between 0 and 1>"""

USER_PROMPT = """
    Analysis of the stablecoin market based on Holt-Winters forecasting:
    
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

class LLMAnalysisBaseBehaviour(BaseBehaviour, ABC):  # pylint: disable=too-many-ancestors
    """Base behaviour for the llm_analysis_abci behaviours."""

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


class ExecuteLLMBehaviour(LLMAnalysisBaseBehaviour):
    """PostPreparationBehaviour"""

    matching_round: Type[AbstractRound] = ExecuteLLMRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        
        raw_data = eval(self.synchronized_data.stablecoins_history)

        self.context.logger.info(f'TYPE CHECK: {type(raw_data)}')
        
        # Preprocess the data
        df = preprocess_stablecoin_data(raw_data)
        
        # Apply Holt-Winters
        model_params = apply_holtwinters(df)

        self.context.logger.info(f'########### MODEL PARAMS ##########\n{model_params}\n###############')

        # Prepare the prompt
        formatted_prompt = USER_PROMPT.format(**model_params)
        
        # Get LLM response
        response = self.get_llm_response(
            system_message=SYSTEM_MESSAGE,
            user_prompt=formatted_prompt
        )

        if response:
            self.context.logger.info(f'########### MARKET ANALYSIS ##########\n{response}\n###############')
        else:
            self.context.logger.info(f'########### NO ANALYSIS AVAILABLE ##########\n###############')

        # Rest of your existing code...
        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address
            payload = ExecuteLLMPayload(
                sender=sender, 
                tx_submitter=self.auto_behaviour_id(), 
                # tx_hash=None
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

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

class LLMAnalysisRoundBehaviour(AbstractRoundBehaviour):
    """LLMAnalysisRoundBehaviour"""

    initial_behaviour_cls = ExecuteLLMBehaviour
    abci_app_cls = LLMAnalysisAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [  # type: ignore
        ExecuteLLMBehaviour,
    ]
