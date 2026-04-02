"""
classify.py

Classifies demand into one of 4 quadrants based on the SBC framework.

The SBC framework used 2 metrics to describe a demand pattern:
1. ADI (Average Demand Interval)
2. CV (Coefficient of Variance)
"""

import numpy as np
import pandas as pd

def compute_adi(series: np.ndarray) -> float:
    """
    Computes the ADI from the demand series.

    Formula for ADI = total periods / periods with non-zero demand

    Arguments:
        series: the demand series

    Returns:   
        ADI: a single number denoting the ADI
    """
    total_periods = len(series)
    periods_with_non_zero_demand = np.sum(series > 0)

    if periods_with_non_zero_demand == 0:
        return np.inf
    adi = total_periods/periods_with_non_zero_demand

    return adi

def compute_cv(series: np.ndarray) -> float:
    """
    Compute CV squared for the demand series. 

    CV = standard deviation / mean

    Arguments:
        series: the demand series

    Result:
        cv(squared)
    """
    non_zero = series[series > 0]

    if len(non_zero) < 2:
        return 0.0
    
    mean = np.mean(non_zero)
    std_dev = np.std(non_zero)
    
    if mean == 0:
        return 0.0
    
    cv_squared = (std_dev/mean) ** 2
    return cv_squared

def classify_demand(adi: float, cv_squared:float) -> str:
    """
    Classified the demand into 1 of 4 categories based on the SBC framework.

    Arguments:
        adi: the ADI returned from the previous function
        cv_squared: the CV**2 returned from the previous function

    Returns:
        demand type
    """
    # Setting thresholds based on empirical data
    ADI_THRESHOLD = 1.32
    CV2_THRESHOLD = 0.49

    if adi == np.inf:
        return "no demand"
    
    high_adi = adi>ADI_THRESHOLD
    high_cv2 = cv_squared>CV2_THRESHOLD

    if high_adi and not high_cv2:
        return "intermittent"
    
    if high_adi and high_cv2:
        return "erratic"
    
    if not high_adi and not high_cv2:
        return "smooth"
    
    else:
        return "lumpy"
