# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2022 Valory AG
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

"""Preprocessing operations."""

from typing import Dict, Optional, Tuple, Union

import pandas as pd
from datetime import datetime

def preprocess_stablecoin_data(
    raw_data: dict,
) -> pd.DataFrame:
    """Prepare the timeseries data for all the pairs.

    :param raw_data: stablecoin data response from DefiLlama.
    :return: a DataFrame with DateTime and Stablecoins value in USD.
    """
    # Convert to DataFrame

    df = pd.DataFrame(eval(raw_data))
    
    # Extract peggedUSD value from nested dictionary
    df['value'] = df['totalCirculatingUSD'].apply(lambda x: x['peggedUSD'])
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
    
    # Set date as index
    df = df.set_index('date')
    
    # Keep only the value column
    df = df[['value']]
    
    # Sort index
    df = df.sort_index()
    df = df.asfreq('D')  # Set daily frequency explicitly

    # Remove the last day
    df = df[:-1]
    
    return df