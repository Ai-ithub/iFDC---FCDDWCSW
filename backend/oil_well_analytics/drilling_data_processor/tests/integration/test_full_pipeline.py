# tests/integration/test_full_pipeline.py
import sys
sys.path.append('/Users/parnian/Desktop/oil_well_analytics/drilling_data_processor')

from drilling_data_processor.drilling_processor.core import DrillingDataProcessor
from pathlib import Path

def test_processing_pipeline(tmp_path, sample_well_data):
    """Full testing of the drilling data processing pipeline"""
    # 1. Save the test data
    test_file = tmp_path / "test_wells.parquet"
    sample_well_data.to_parquet(test_file)
    
    # 2. Run the pipeline
    processor = DrillingDataProcessor(test_file)
    processed = processor.run_pipeline()
    
    # 3. Main assertions
    assert not processed.empty
    assert 'PT_Ratio' in processed.columns  # Check feature engineering
    assert processed.isna().sum().sum() == 0  # Ensure no missing values

