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
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

# HoltWinters_Forecast = Tuple[dict, pd.Series]

def apply_holtwinters(
    df: pd.DataFrame,
    periods_to_forecast=30
) -> Dict:
    """Use Holt Winters to predict the next 30 periods.

    :param df: a stablecoin DataFrame.
    :return: a tuple containing model parameters and forecast values.
    """
    # Fit Holt-Winters model
    model = ExponentialSmoothing(
        df['value'],
        seasonal_periods=7,
        trend='add',
        seasonal='add',
        initialization_method='estimated',
    ).fit()
    
    # Make forecast
    forecast = model.forecast(periods_to_forecast)
    
    # Create model parameters dictionary
    model_params = {
        "alpha": model.params['smoothing_level'],
        "beta": model.params['smoothing_trend'],
        "gamma": model.params['smoothing_seasonal'],
        "trend_direction": "up" if forecast.iloc[-1] > forecast.iloc[0] else "down",
        "trend_strength": abs((forecast.iloc[-1] - forecast.iloc[0]) / forecast.iloc[0]),
        "last_actual_value": df['value'].iloc[-1],
        "forecast_mean": forecast.mean(),
        "forecast_std": forecast.std()
    }
    
    return model_params