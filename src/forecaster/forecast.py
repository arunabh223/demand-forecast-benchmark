"""
forecast.py

After the classification step, this code will check the demand type of each SKU and forecast for next n periods

smooth -> ETS (Exponential Time Smoothing)
"""

import pandas as pd
import numpy as np
from forecaster.models.statistical import ETSModel


def load_smooth_skus(classifications_filepath:str, demand_agg_filepath:str) -> pd.DataFrame:
    """
    Loads the classification dataset and the demand dataset.
    
    Arguments:
        classification dataset
        demand data
    Returns:
        A new DataFrame with demand for SKUs having only smooth pattern
    """

    # load the datasets
    # load the dataset having the classifications
    classifications = pd.read_csv(classifications_filepath)

    # load the dataset having the aggregated demand
    demand_data = pd.read_csv(demand_agg_filepath)

    # we want the datasets with smooth skus only so we apply a merge
    smooth_skus = classifications[classifications["demand_type"] == "smooth"]

    return smooth_skus.merge(demand_data, on="id", how="inner")

# First we define a function to calculate the forecast of a single SKU as an array
def forecast_sku(series:np.ndarray, model:ETSModel, fc_periods:int = 6) -> dict:
    """
    Calculates the ETS forecast for one SKU.
    Arguments:
        The DataFrame with the SKU id and aggregated demand
        The model used (ETS)
    Returns:
        A numpy array with the forecast for the next n periods (default 6 months)
    """
    # split the series into train and test datasets
    train = series[:-fc_periods]
    test  = series[-fc_periods:]

    # apply the fit() method to the training dataset
    model.fit(train)
    forecast = model.predict(fc_periods)

    return {
        "model_used"    :   model,
        "actual_demand" :   test,
        "model_forecast":   forecast
    }

def forecast_dataframe(portfolio:pd.DataFrame, fc_periods:int=6) -> pd.DataFrame:
    """
    Apply the forecast_sku() method to the whole dataframe having the aggregated demand
    Arguments:
        The portfolio having the sku_id and the aggregated demand
        The forecast periods
    Result:
        A DataFrame having the forecast for the next n periods
    """
    ## extract each row of the dataframe as a numpy array for the forecast

    # extracting the period cols
    period_cols = [c for c in portfolio.columns if c.startswith("period_")]
    # a final dataframe with the sku_id and the period cols on which we will run the forecast
    demand_matrix = portfolio[["id"] + period_cols]

    # applying the logic to each row of the dataframe
    model = ETSModel()
    results =[]

    for _, row in demand_matrix.iterrows():
        series = row[period_cols].to_numpy(dtype=float)
        result = forecast_sku(series, model, fc_periods)
        results.append({
            "id":               row["id"],
            "actual_demand":   result["actual_demand"],
            "model_forecast":   result["model_forecast"]
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # testing out the functions above
    smooth_skus = load_smooth_skus("data/results/classifications.csv", "data/results/aggregated_data.csv")
    # result.to_csv("temp/temp.csv")
    result = forecast_dataframe(smooth_skus)
    result.to_csv("data/results/forecast_results.csv")


