"""
Machine Learning Forecasting Models
=====================================
Gradient-boosted tree models (XGBoost, LightGBM) for demand forecasting.
 
The key challenge: tree models expect a table of rows and columns, but
demand data is a time series. We solve this with FEATURE ENGINEERING —
converting the series into lag features, rolling stats, and calendar info
so the model can learn patterns from past demand.
 
Models
------
  XGBoostForecaster  : XGBoost with recursive multi-step forecasting
  LightGBMForecaster : LightGBM with recursive multi-step forecasting
 
All models share a common interface:
    model.fit(train)  →  model
    model.predict(h)  →  np.ndarray of length h
"""

import numpy as np
import pandas as pd
from typing import Optional

