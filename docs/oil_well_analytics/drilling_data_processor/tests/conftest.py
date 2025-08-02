import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_well_data():
    return pd.DataFrame({
        'Well_ID': ['WELL_001', 'WELL_002', 'WELL_003'],
        'Temperature_C': [80.5, 120.3, np.nan],
        'Pressure_psi': [5000, 12000, 8000],
        'Formation': ['Sandstone', 'Carbonate', None],
        'Damage_Type': ['Clay & Iron', None, 'Fluid Loss']
    })
