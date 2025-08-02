import pandas as pd
import json
from typing import Dict, Any

class QualityChecker:
    def __init__(self):
        self.report = {}

    def generate_report(self, df) -> Dict[str, Any]:
        """Generate a comprehensive data quality report"""
        self._check_missing_values(df)
        self._check_value_ranges(df)
        self._check_data_distribution(df)
        self._check_logical_inconsistencies(df)
        df = self._check_feature_inconsistencies(df)
        return self.report

    def _check_missing_values(self, df):
        """Check for missing values"""
        self.report['missing_values'] = {
            'total': df.isnull().sum().sum(),
            'by_column': df.isnull().sum().to_dict()
        }

    def _check_value_ranges(self, df):
        """Check logical value ranges"""
        ranges = {
            'Temperature_C': (0, 400),
            'Pressure_psi': (0, 30000),
            'pH': (0, 14)
        }
        violations = {}
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                violations[col] = {
                    'below_min': int((df[col] < min_val).sum()),
                    'above_max': int((df[col] > max_val).sum())
                }
        self.report['value_range_violations'] = violations

    def _check_logical_inconsistencies(self, df: pd.DataFrame):
        """
        Check for logically inconsistent combinations between features
        """
        if {'Fluid_Type', 'Formation_Type', 'Completion_Type'}.issubset(df.columns):
            suspicious_conditions = (
                (df['Fluid_Type'].str.lower() == 'acidic') &
                (df['Formation_Type'].str.lower() == 'shale') &
                (df['Completion_Type'].str.lower() == 'open hole')
            )
            
            suspicious_count = suspicious_conditions.sum()
            total_records = len(df)

            self.report['logical_inconsistencies'] = {
                'suspicious_combinations_count': int(suspicious_count),
                'suspicious_percentage': round((suspicious_count / total_records) * 100, 2),
                'recommendation': 'Review acidic fluid usage in shale formations with open hole completion.'
            }

    def _check_feature_inconsistencies(self, df):
        """
        Check for feature combination inconsistencies
        """
        conditions = [
            (df["Formation_Type"] == "Shale") & (df["Formation_Permeability"] > 50),
            (df["Formation_Type"] == "Limestone") & (df["Reservoir_Temperature"] < 60),
            (df["Formation_Type"] == "Sandstone") & (df["Clay_Content_Percent"] > 60),
            (df["Mud_Type"] == "Water-based") & (df["Viscosity"] > 60),
            (df["Mud_Type"] == "Oil-based") & (df["pH"] < 6),
            (df["Mud_Type"] == "Synthetic") & (df["Mud_Weight_In"] < 7.5),
            (df["Viscosity"] > 70) & (df["In_Rate_Flow_Mud"] > 150),
            (df["Reservoir_Temperature"] > 130) & (df["Formation_Type"] == "Shale"),
            (df["Pressure_Annulus"] > df["Pressure_Standpipe"]),
            (df["Pressure_Reservoir"] < 3000) & (df["Depth_Measured"] > 3000),
            (df["Phase_Operation"] == "Production") & (df["ROP"] > 1),
            (df["Phase_Operation"] == "Completion") & (df["Weight_on_Bit"] > 5000),
            (df["Depth_Bit"] > df["Depth_Measured"]),
            (df["Completion_Type"] == "Cased") & (df["Density_Perforation"] < 5),
            (df["Completion_Type"] == "Open Hole") & (df["Density_Perforation"] > 20),
            (df["Clay_Mineralogy_Type"] == "Montmorillonite") & (df["pH"] > 10),
            (df["Formation_Type"] == "Shale") & (df["Fractures_Presence"] == 1) & (df["Formation_Permeability"] > 100),
            (df["Phase_Operation"] == "Production") & (df["Weight_on_Bit"] > 1000),
            (df["Fluid_Loss_API"] > 3.0) & (df["Mud_Type"] == "Oil-based"),
            (df["Reservoir_Temperature"] > 120) & (df["Clay_Content_Percent"] > 50)
        ]
        anomaly_mask = conditions[0]
        for cond in conditions[1:]:
            anomaly_mask |= cond

        df["Feature_Inconsistency"] = anomaly_mask
        self.report["feature_inconsistencies"] = {
            "total": int(anomaly_mask.sum()),
            "percentage": round((anomaly_mask.sum() / len(df)) * 100, 2)
        }

        return df
    
    def _check_data_distribution(self, df):
        """Check data distribution"""
        pass

    def save_report(self, file_path: str):
        """Save the report to a file"""
        with open(file_path, 'w') as f:
            json.dump(self.report, f, indent=4)
