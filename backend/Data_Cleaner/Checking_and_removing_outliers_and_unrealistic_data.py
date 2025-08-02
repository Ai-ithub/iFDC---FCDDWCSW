


"""
detect_and_remove_outliers.py

This script detects and removes outliers from Parquet files in a given directory.
It uses both Z-Score and IQR methods to identify outliers and saves clean and outlier data separately.

Author: mahdis
Date: [1404-03-03]
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

def detect_and_remove_outliers(
    folder_path: str,
    output_clean_path: str,
    output_outliers_path: str,
    columns_to_check: list = None,
    z_threshold: float = 3,
    iqr_multiplier: float = 1.5,
    verbose: bool = True
):
    """
    Detects and removes outliers from Parquet files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing input Parquet files.
        output_clean_path (str): Directory to save cleaned data files.
        output_outliers_path (str): Directory to save outlier data files.
        columns_to_check (list, optional): List of columns to check for outliers.
                                           Defaults to common drilling data features.
        z_threshold (float, optional): Z-Score threshold for outlier detection. Defaults to 3.
        iqr_multiplier (float, optional): Multiplier for IQR method. Defaults to 1.5.
        verbose (bool, optional): Whether to print progress information. Defaults to True.
    """
    if columns_to_check is None:
        columns_to_check = [
            'temperature', 'pressure', 'permeability', 'flow_rate',
            'depth', 'porosity', 'salinity', 'mud_weight'
        ]

    # Create output directories if they don't exist
    os.makedirs(output_clean_path, exist_ok=True)
    os.makedirs(output_outliers_path, exist_ok=True)

    # Find all Parquet files in the folder
    parquet_files = glob.glob(os.path.join(folder_path, '*.parquet'))
    if not parquet_files:
        print(f"⚠️ No Parquet files found in {folder_path}.")
        return

    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        if verbose:
            print(f"\n🔍 Processing file: {file_name}")

        try:
            df = pd.read_parquet(file_path)

            # Check for missing columns
            missing_cols = [col for col in columns_to_check if col not in df.columns]
            if missing_cols:
                if verbose:
                    print(f"⚠️ Missing columns in {file_name}: {missing_cols}")
                continue

            # Z-Score method
            z_scores = np.abs(stats.zscore(df[columns_to_check], nan_policy='omit'))
            outliers_zscore = (z_scores > z_threshold)

            # IQR method
            outliers_iqr = pd.DataFrame(False, index=df.index, columns=columns_to_check)
            for col in columns_to_check:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                outliers_iqr[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

            # Combine outliers from both methods
            outliers_combined = outliers_zscore | outliers_iqr
            any_outlier = outliers_combined.any(axis=1)

            # Separate clean and outlier data
            df_outliers = df[any_outlier]
            df_clean = df[~any_outlier]

            # Save results
            df_clean.to_parquet(os.path.join(output_clean_path, f"clean_{file_name}"))
            df_outliers.to_parquet(os.path.join(output_outliers_path, f"outliers_{file_name}"))

            if verbose:
                print(f"✅ {file_name}: {df_outliers.shape[0]} outliers, {df_clean.shape[0]} clean rows.")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Example usage for local or Google Colab (update paths accordingly)
    # from google.colab import drive
    # drive.mount('/content/drive')

    input_folder = '/content/drive/MyDrive/24may_amin_dataset'  # Replace with your actual input folder
    output_clean_folder = '/path/to/your/output/clean_folder'
    output_outliers_folder = '/path/to/your/output/outliers_folder'

    detect_and_remove_outliers(
        folder_path=input_folder,
        output_clean_path=output_clean_folder,
        output_outliers_path=output_outliers_folder,
        z_threshold=3,
        iqr_multiplier=1.5,
        verbose=True
    )