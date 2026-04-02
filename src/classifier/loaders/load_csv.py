"""
Generic csv loader if we want to load demand as a csv file

There can be two formats of the csv file.
1. Long format
2. Wide format

* The csv should have a consistent structure

  - One column identifying the SKU (e.g. "product_id", "sku", "item")
  - One column for the time period (e.g. "month", "date", "week")
  - One column for demand (e.g. "sales", "units", "quantity")

LONG FORMAT EXAMPLE:
  sku_id   | month   | sales
  SKU-001  | 2023-01 | 120
  SKU-001  | 2023-02 | 95
  SKU-002  | 2023-01 | 0
  SKU-002  | 2023-02 | 45
 
WIDE FORMAT EXAMPLE:
  sku_id   | 2023-01 | 2023-02 | 2023-03
  SKU-001  | 120     | 95      | 110
  SKU-002  | 0       | 45      | 30

This code will return a standard format - one row per SKU, one column per period
"""

import pandas as pd
import numpy as np

def load_long(filepath: str, id_col:str, period_col:str, demand_col:str) -> pd.DataFrame:
    """
    Function to load the long format csv file.

    Arguments:
        filepath: path to the file
        id_col: column name containing the serial no
        demand_col: column name containing the demand
        period_col: columns name contaning the timestamp
    Return:
        Normalized DataFrame containing the time series
    """
    raw = pd.read_csv(filepath)

    # Pivot from long to wide
    wide = raw.pivot_table(
        values=demand_col,
        columns=period_col,
        aggfunc="sum", # if there are duplicate ids sum them
        fill_value=0, # for missing values
        index=id_col
    )
    wide.columns = [f"period_{i+1}" for i in range (len(wide.columns))]
    wide = wide.reset_index().rename(columns={id_col: "id"})

    return wide

def load_wide(filepath:str, id_col:str) -> pd.DataFrame:
    """
    Loads a csv file in the wide format and returns the standard format

    Arguments:
        filepath
        id_col
    Return:
        the DataFrame in the standard format
    """
    raw = pd.read_csv(filepath)

    period_cols = [c for c in raw.columns if c!= id_col]

    # Rename the period cols to the same naming convention -> period_i
    rename_map = {col: f"period_{i+1}" for i,col in enumerate(period_cols)}
    result = raw[id_col + [period_cols]].rename(columns = {id_col:"id", **rename_map})
    result = result.fillna(0)

    return result


