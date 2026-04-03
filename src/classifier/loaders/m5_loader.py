"""
This loader is specific for the "M5" dataset.
The purpose of this loader is to clean the dataset and make it usable for the classify.py file
"""

import pandas as pd
import numpy as np

def load_file_m5(filepath:str) -> pd.DataFrame:
    """
    This function loads the csv file from the filepath and converts it into a DataFrame.
    Only the ID column with the daily demand columns are kept. All other columns are dropped.
    
    Argument:
        filepath: path where the csv file is located
    Returns:
        DataFrame containing the ID and the sales
    """
    df = pd.read_csv(filepath)
    # separate the columns with daily demand
    daily_cols = []
    for c in df.columns:
        if c.startswith("d_"):
            daily_cols.append(c)
    sales = df[["id"] + daily_cols]
    result = _aggregate_to_periods(sales, daily_cols)
    return result

def _aggregate_to_periods(sales:pd.DataFrame, daily_cols:list) -> pd.DataFrame:
    """
    There are a lot of zeros in the demand of every SKU.
    So to smooth out the demand, we will aggregate it into 30 days periods.

    1914 days = 1914 / 30 = ~ 63 periods (with some extra which are dropped)
    
    Arguments:
        sales: the DataFrame with the sales
    Return:
        A DataFrame with the aggregated demand
    """
    # Specify the periodicity
    days_per_period = 30 # Setting a month as a standard period

    daily_values = sales[daily_cols].values # converts the DataFrame into a numpy array
    n_skus, n_days = daily_values.shape
    
    n_periods = n_days // days_per_period
    days_to_use = n_periods * days_per_period
    trimmed = daily_values[:, :days_to_use] # drop the leftover days

    # Reshape the array into 63 rows of 30 numbers each, summed up for every 30 days
    period_total = trimmed.reshape(n_skus, n_periods, days_per_period).sum(axis=2)

    # Put the numbers back into the DataFrame
    period_cols = [f"period_{i+1}" for i in range(n_periods)]
    result = pd.DataFrame(period_total, columns=period_cols)
    result.insert(0,"id", sales["id"].values)

    return result

if __name__ == "__main__":
    path = "data/files/sales_train_validation.csv"
    daily_cols = load_file_m5(path)
    # print(daily_cols.shape)
    print(daily_cols.head())
    daily_cols.to_csv("data/results/aggregated_data.csv")