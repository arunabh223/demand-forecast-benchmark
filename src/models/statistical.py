"""
Statistical Forecasting Models
================================
Implements classical and intermittent-demand forecasting methods.
 
Models
------
  CrostonModel     : Original Croston (1972) method
  SBAModel         : Syntetos-Boylan Approximation (2001) — bias-corrected Croston
  TSBModel         : Teunter-Syntetos-Babai (2011) — handles obsolescence
  ARIMAModel       : Auto-ARIMA via pmdarima
  ETSModel         : Exponential Smoothing (ETS) via statsmodels
  NaiveModel       : Naive forecasting model
 
All models share a common interface:
    model.fit(train)  →  model
    model.predict(h)  →  np.ndarray of length h
"""

"""
What am I using for this?
1. N-Dimensional array for storing the time series
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional


class BaseForecaster:
    """
    This is the base class and interface for every model
        model.fit(train)  →  model
        model.predict(h)  →  np.ndarray of length h
    Every model will inherit from this class

    We define two core methods

    model.fit(series) - reads the historical demand series and updates the model
    model.predict(h) - uses the updated model to generate a forecast as a numpy array

    All subclasses should have a fit() and predict() method or they will raise a NotImplementedError.
    """
    def fit(self, series: np.ndarray) -> "BaseForecaster":
        raise NotImplementedError
    
    def predict(self, h: int) -> np.ndarray:
        raise NotImplementedError

###################### CROSTON MODEL ###########################

class CrostonModel(BaseForecaster):
    """
    This will model Croston's SBA method for intermittent demand.
    
    Here, we will maintain two series:
    z : the non-zero demand series
    p : the inter-demand interval series
    alpha : the smoothing factor for the demand 
        alpha = 0.1 - 0.3 for slow adaptation, good for stable demand
        alpha = 0.3+ for fast adaptation, tracks recent changes
        (default) alpha = 0.1

    Forecast = z/p (the forecast will be flat)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._z = Optional[float] # stores the smoothed demand series
        self._p = Optional[float] # stores the smoothed interval

    # Defining the fit() method
    def fit(self, series: np.ndarray) -> "CrostonModel":
        """
        Walk through the data and update z and p at each non-zero demand observation using SES.

        The SES formula we will use is 
        new_estimate = alpha * new_obs + (1 - alpha) * old_estimate
        """
        series = np.asarray(series, dtype=float)

        nonzero_index = np.where(series > 0)[0] # find the index positions of all the non-zero demand periods
        z = series[nonzero_index[0]] # initialize on the first non-zero observation
        p = nonzero_index[0] + 1 # interval from period 1 to first demand

        # walk through the series and update z and p
        for k in range(1, len(nonzero_index)):
            i_curr = nonzero_index[k]
            i_prev = nonzero_index[k-1]
            interval = i_curr - i_prev # periods between current and previous demand

            z = self.alpha * series[i_curr] + (1 - self.alpha) * z
            p = self.alpha * interval + (1 - self.alpha) * p

        self._z = z
        self._p = p
        return self
    
    def predict(self, h: int) -> np.ndarray:
        """
        We use the formula 
        forecast = z/p
        """
        forecast = self._z/self._p
        return np.full(h, forecast)
    
################### NAIVE MODEL ##########################

class NaiveModel(BaseForecaster):
    """
    This is model a Naive forecast. Simplest possible forecast where latest demand is carried forward.

    Formula: 
    forecast(t+1) = demand(t)
    """
    def fit(self, series: np.ndarray) -> "NaiveModel":
        self._last = float(series[-1])
        return self
    
    def predict(self, h:int) -> np.ndarray:
        return np.full(h, self._last)


################### MODEL REGISTRY #######################

def get_statistical_models():
    """
    Return fresh unfitted instances of all models
    
    Called once per SKU so that each SKU gets a clean slate.
    """
    return {
        "Croston": CrostonModel(alpha=0.1),
        "Naive": NaiveModel()
    }


