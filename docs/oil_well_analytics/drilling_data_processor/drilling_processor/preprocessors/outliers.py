from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
from typing import Literal, Union
import pandas as pd
import numpy as np

class OutlierDetector:
    def __init__(self):
        self.report = {}

    def detect(self, 
               df: pd.DataFrame, 
               method: Literal['isolation_forest', 'zscore', 'lof'] = 'isolation_forest',
               contamination: float = 0.05,
               return_df: bool = True
               ) -> Union[pd.DataFrame, np.ndarray]:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a valid pandas DataFrame.")
        
        numeric_df = df.select_dtypes(include=['number']).copy()
        
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(numeric_df)
            outlier_mask = preds == -1

        elif method == 'zscore':
            z_scores = np.abs(zscore(numeric_df))
            outlier_mask = (z_scores > 3).any(axis=1)

        elif method == 'lof':
            clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            preds = clf.fit_predict(numeric_df)
            outlier_mask = preds == -1

        else:
            raise ValueError(f"Unknown method: {method}")

        self.report = {
            "method": method,
            "outliers_count": int(outlier_mask.sum()),
            "percentage": round(outlier_mask.sum() / len(df) * 100, 2)
        }

        if return_df:
            result_df = df.copy()
            result_df['is_outlier'] = outlier_mask
            return result_df
        else:
            return outlier_mask
