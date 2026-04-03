import pandas as pd
import numpy as np
from typing import Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

"""
Registry for all statistical forecasting models.
"""

class BaseForecaster:
    def fit(self, series:np.ndarray) -> "BaseForecaster":
        raise NotImplementedError
    
    def predict(self, h:int) -> np.ndarray:
        raise NotImplementedError
    
class ETSModel(BaseForecaster):
    def __init__(self, trend:Optional[str]="add", seasonal:Optional[str]="add", seasonal_periods:int=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._model = None

    def fit(self, series:np.ndarray) -> "ETSModel":
        self._model = ExponentialSmoothing(
            endog=series,
            trend=self.trend,
            seasonal= self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self._model = self._model.fit(optimized=True, remove_bias=True)
        return self # Return a fit model as a new instance of the class
    
    def predict(self, h:int) -> np.ndarray:
        fc = self._model.forecast(h)
        return np.asarray(fc)
    
# Model registry

def get_statistical_models() -> dict:
    return {
        "ETSModel"  :   ETSModel()
    }

