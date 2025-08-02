import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Dict

class DataCleaner:
    def __init__(self):
        self.imputation_strategies = {
            'median': SimpleImputer(strategy='median'),
            'mean': SimpleImputer(strategy='mean'),
            'knn': KNNImputer(n_neighbors=5),
            'iterative': IterativeImputer(max_iter=10, random_state=42)
        }
        self.imputation_history = []

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'median',
        custom_strategy: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Advanced handling of missing values with the following capabilities:
        - Numerical imputation with various methods (`median`, `mean`, `knn`, `iterative`)
        - Impute `NaN` values in text columns with the most frequent value (`mode`)
        - Track the history of changes for process analysis
        
        Parameters:
            df: The input DataFrame
            strategy: Default strategy for numeric columns
            custom_strategy: Dictionary specifying strategies for specific columns
            
        Example:
            cleaner.handle_missing_values(df, strategy='mean',
                                custom_strategy={'Salinity_ppm': 'knn'})
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("❌ Error: The input must be a valid DataFrame!")

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # ✅ Custom imputation for specific numeric columns
        if custom_strategy:
            for col, col_strategy in custom_strategy.items():
                if col in numeric_cols:
                    imputer = self.imputation_strategies.get(col_strategy)
                    if imputer:
                        df[[col]] = imputer.fit_transform(df[[col]])
                        self.imputation_history.append(
                            f"Column '{col}' imputed with {col_strategy}"
                        )
                        numeric_cols.remove(col)
                    else:
                        raise ValueError(f"❌ Error: Imputation strategy '{col_strategy}' is invalid!")

        # ✅ General imputation for remaining numeric columns
        if numeric_cols:
            imputer = self.imputation_strategies.get(strategy)
            if imputer:
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputation_history.append(
                    f"Columns {numeric_cols} imputed with {strategy}"
                )
            else:
                raise ValueError(f"❌ Error: Imputation strategy '{strategy}' is invalid!")

        # ✅ Impute `NaN` in text columns with the most frequent value (`mode`)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if df[col].isna().sum() > 0:  # Only if the column has `NaN` values
                df[col].fillna(df[col].mode()[0], inplace=True)
                self.imputation_history.append(f"Categorical column '{col}' imputed with mode")
            
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows while keeping the first occurrence"""
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("❌ Error: The input must be a valid DataFrame!")

        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)
        self.imputation_history.append(
            f"Removed {removed} duplicate rows"
        )
        return df
