"""
Entry point for the demand classification.
Loads a csv file having demand and classifies the demand based on the SBC framework.
"""

import pandas as pd
import numpy as np
import sys, os
import argparse
from classify import compute_adi, compute_cv, classify_demand
from loaders.m5_loader import load_file_m5
from loaders.load_csv import load_long, load_wide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    # return demand_matrix 

    # Apply the logic to every row of the dataset
    adi_values = np.apply_along_axis(compute_adi,axis=1,arr=demand_matrix)
    cv_values = np.apply_along_axis(compute_cv, axis=1, arr=demand_matrix)

    # Classify each SKU as per their demand type
    demand_types = [classify_demand(adi,cv2) for adi,cv2 in zip(adi_values, cv_values)]

    result_portfolio = pd.DataFrame({
        "id" : df["id"].values,
        "adi": adi_values.round(2),
        "cv2": cv_values.round(2),
        "demand_type": demand_types,
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

# Main entry point for the classification
def run(source:str, filepath:str, id_col:str, period_col:str, demand_col:str, output:str) -> None:
    if source == "m5":
        df = load_file_m5(filepath)
    elif source == "csv_long":
        df = load_long(filepath, id_col, period_col, demand_col)
    elif source == "csv_wide":
        df = load_wide(filepath, id_col)
    else:
        print(f"Unknown source '{source}'. Use: m5, csv_long, csv_wide")
        sys.exit(1)

    results = classify_portfolio(df)
    print(results)

    # print_summary(results)

    # Save results to a csv file
    results.to_csv(output, index=False)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    """
    USAGE
    ------
    # M5 dataset (default)
    python run_classification.py --source m5
    
    # Your own long-format CSV
    python run_classification.py --source csv_long \\
        --filepath data/my_sales.csv \\
        --id_col product_id --period_col month --demand_col units_sold
    
    # Your own wide-format CSV
    python run_classification.py --source csv_wide \\
        --filepath data/monthly_sales.csv --id_col sku_id
    """
    parser = argparse.ArgumentParser(description="Classify demand patterns in a portfolio")
    parser.add_argument("--source", choices=["m5","csv_long","csv_wide"], default="m5")
    parser.add_argument("--filepath", default="data/files/sales_train_validation.csv")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--period_col", default="month")
    parser.add_argument("--demand_col", default= "sales")
    parser.add_argument("--output", default= "data/results/classifications.csv")
    args = parser.parse_args()
    run(args.source, args.filepath, args.id_col,args.period_col, args.demand_col, args.output)

    
