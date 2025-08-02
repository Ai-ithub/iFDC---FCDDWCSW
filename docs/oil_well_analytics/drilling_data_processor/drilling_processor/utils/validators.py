import pandas as pd
from typing import Tuple

class DataValidator:
    @staticmethod
    def validate_input_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the structure of the input data"""
        required_columns = {
            'Temperature_C': 'float32',
            'Pressure_psi': 'float32',
            'Formation': 'category'
        }
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
            
        type_errors = []
        for col, dtype in required_columns.items():
            if str(df[col].dtype) != dtype:
                type_errors.append(f"{col} should be {dtype} but found {df[col].dtype}")
        
        if type_errors:
            return False, "Type errors:\n" + "\n".join(type_errors)
            
        return True, "Data validation passed"
