"""
Entry point for the demand classification.
Loads a csv file having demand and classifies the demand based on the SBC framework.
"""

import pandas as pd
import numpy as np
import sys, os
from classify import compute_adi, compute_cv, classify_demand
from loaders.m5_loader import load_file_m5

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCE = "m5"
 
# Path to the input data file.
FILEPATH = "data/files/sales_train_validation.csv"
 
# Output path for the classification results.
OUTPUT = "data/classifications.csv"
 
# Column names — only needed when SOURCE is "csv_long" or "csv_wide".
# For SOURCE = "m5" these are ignored.
ID_COL     = "id"       # column that identifies each SKU
PERIOD_COL = "month"    # column for the time period  (csv_long only)
DEMAND_COL = "sales"    # column for demand quantity  (csv_long only)

def classify_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the SBC classification on every row of the dataframe.
    Applies the functions in classify.py to every row of the portfolio dataframe.
    Arguments:
        df: the dataframe containing the SKUs and their respective demand
    Returns:
        The dataframe containing SKU, demand, ADI, CV and demand type
    """
    num_sku = len(df)
    print(f"Calculating for {num_sku} SKUs....")

    # Find the number of period columns
    period_cols = [c for c in df.columns if c.startswith("period_")]
    
    # Create the demand matrix
    demand_matrix = df[period_cols].values

    # Apply the logic to every row of the dataset
    adi_values = np.apply_along_axis(compute_adi,axis=1,arr=demand_matrix)
    cv_values = np.apply_along_axis(compute_cv, axis=1, arr=demand_matrix)


    # Classify each SKU as per their demand type
    demand_types = [classify_demand(adi,cv2) for adi,cv2 in zip(adi_values, cv_values)]

    result_portfolio = pd.DataFrame({
        "id" : df["id"].values,
        "adi": adi_values.round(2),
        "cv2": cv_values.round(2),
        "demand_type": demand_types
    })

    return result_portfolio

def print_summary(result: pd.DataFrame) -> None:
    """
    Prints a readable summary to the user.
    Arguments:
        result: the matrix with all the details
    Result:
        A readable summary
    """
    total = len(result)

    for demand_type, count in result["demand_type"].value_counts().items():
        pct = count/total * 100
        print(f"  {demand_type:<15}  {count:>7,}  ({pct:.1f}%)")

    print(f"\n  Total SKUs: {total:,}")

    print("\n  Median ADI and CV² by demand type:")
    summary = result.groupby("demand_type")[["adi", "cv2"]].median().round(2)
    print(summary.to_string())
    print("=" * 50)

def run() -> None:
    if SOURCE == "m5":
        df = load_file_m5(FILEPATH)
    else:
        print("unknown source")
        sys.exit(1)

    results = classify_portfolio(df)

    print_summary(results)

if __name__ == "__main__":
    run()



    
