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

"""This package contains the rounds of LLMAnalysisAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    OnlyKeeperSendsRound,
    get_name,
)
from packages.niron.skills.llm_analysis_abci.payloads import (
    ExecuteLLMPayload,
)


class Event(Enum):
    """LLMAnalysisApp Events"""

    DONE = "done"
    ERROR = "error"
    TRANSACT = "transact"
    NO_MAJORITY = "no_majority"
    ROUND_TIMEOUT = "round_timeout"
    NONE = "none"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application, so all the agents share the same data.
    """

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)
    
    @property
    def stablecoins_history(self) -> Optional[float]:
        """Get the Stablecoins History."""
        return self.db.get("stablecoins_history", None)
    
    @property
    def stablecoins_ipfs_hash(self) -> Optional[str]:
        """Get the Stablecoins IPFS Hash."""
        return self.db.get("stablecoins_ipfs_hash", None)

    @property
    def participant_to_data_round(self) -> DeserializedCollection:
        """Agent to payload mapping for the DataPullRound."""
        return self._get_deserialized("participant_to_data_round")

    @property
    def most_voted_tx_hash(self) -> Optional[float]:
        """Get the token most_voted_tx_hash."""
        return self.db.get("most_voted_tx_hash", None)

    @property
    def participant_to_tx_round(self) -> DeserializedCollection:
        """Get the participants to the tx round."""
        return self._get_deserialized("participant_to_tx_round")

    @property
    def tx_submitter(self) -> str:
        """Get the round that submitted a tx to transaction_settlement_abci."""
        return str(self.db.get_strict("tx_submitter"))

class ExecuteLLMRound(CollectSameUntilThresholdRound):
    """ExecuteLLMRound"""

    payload_class = ExecuteLLMPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_tx_round)
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.most_voted_tx_hash),
    )

    # Event.ROUND_TIMEOUT  # this needs to be referenced for static checkers

class FinalLLMRound(DegenerateRound):
    """FinalLLMRound"""

class LLMAnalysisAbciApp(AbciApp[Event]):
    """LLMAnalysisAbciApp"""

    initial_round_cls: AppState = ExecuteLLMRound
    initial_states: Set[AppState] = {
        ExecuteLLMRound,
    }
    transition_function: AbciAppTransitionFunction = {
        ExecuteLLMRound: {
            Event.NO_MAJORITY: ExecuteLLMRound,
            Event.ROUND_TIMEOUT: ExecuteLLMRound,
            Event.DONE: FinalLLMRound,
        },
        FinalLLMRound: {},
    }
    final_states: Set[AppState] = {
        FinalLLMRound,
    }
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        ExecuteLLMRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinalLLMRound: set(),
        # FinishedTxPreparationRound: {get_name(SynchronizedData.most_voted_tx_hash)},
    }
