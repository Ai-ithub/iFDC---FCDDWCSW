import sys
import pytest
import pandas as pd
from drilling_data_processor.drilling_processor.preprocessors.cleaners import DataCleaner

sys.path.append('/Users/parnian/Desktop/oil_well_analytics/drilling_data_processor')

@pytest.fixture
def sample_well_data():
    """Sample test data with NaN values for imputation testing"""
    return pd.DataFrame({
        'Well_ID': ['WELL_001', 'WELL_002', 'WELL_003'],
        'Temperature_C': [80.5, 120.3, None],  # NaN value for numeric imputation test
        'Pressure_psi': [5000, 12000, 8000],
        'Formation': ['Sandstone', 'Carbonate', None],  # NaN value for categorical imputation test
        'Damage_Type': ['Clay & Iron', None, 'Fluid Loss']  # NaN value for categorical imputation test
    })

def test_cleaner_imputation(sample_well_data):
    """Test the imputer functionality on well data"""
    cleaner = DataCleaner()
    
    # ✅ Apply custom imputation for categorical columns
    cleaned = cleaner.handle_missing_values(
        sample_well_data,
        custom_strategy={"Formation": "most_frequent", "Damage_Type": "most_frequent"}
    )
    
    # ✅ Check that no NaN values remain
    assert cleaned.isna().sum().sum() == 0, f"❌ Remaining NaN values: \n{cleaned.isna().sum()}"

    # ✅ Check that the data structure is preserved
    assert set(cleaned.columns) == set(sample_well_data.columns), "❌ Column names have changed!"
