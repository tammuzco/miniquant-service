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

"""This module contains the transaction payloads of the LearningAbciApp."""

from dataclasses import dataclass
from typing import Optional, List, Dict

from packages.valory.skills.abstract_round_abci.base import BaseTxPayload


@dataclass(frozen=True)
class CollectRandomnessPayload(BaseTxPayload):
    """Represent a transaction payload of type 'randomness'."""

    round_id: int
    randomness: str

@dataclass(frozen=True)
class SelectKeeperPayload(BaseTxPayload):
    """Represent a transaction payload of type 'select_keeper'."""

    keeper: str

@dataclass(frozen=True)
class DefiLlamaPullPayload(BaseTxPayload):
    """Represent a transaction payload for the pulling data from DefiLlama."""

    stablecoins_history: Optional[str] = None
    stablecoins_ipfs_hash: Optional[str] = None

@dataclass(frozen=True)
class ExecuteLLMPayload(BaseTxPayload):
    """Represent a response of the llm analysis."""

    llm_response: Optional[str] = None

@dataclass(frozen=True)
class DecisionMakingPayload(BaseTxPayload):
    """Represent a transaction payload for the DecisionMakingRound."""

    event: str

@dataclass(frozen=True)
class TxPreparationPayload(BaseTxPayload):
    """Represent a transaction payload for the TxPreparationRound."""

    tx_submitter: Optional[str] = None
    tx_hash: Optional[str] = None
