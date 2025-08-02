### **Drilling Data Processing Package Files**

#### **Package Structure**:

```plaintext
drilling_data_processor/
├── setup.py
├── requirements.txt
└── drilling_processor/
    ├── __init__.py
    ├── core.py
    ├── preprocessors/
    │   ├── __init__.py
    │   ├── cleaners.py
    │   ├── outliers.py
    │   ├── feature_engine.py
    │   └── quality.py
    ├── pipelines/
    │   ├── __init__.py
    │   └── ml_pipeline.py
    └── utils/
        ├── __init__.py
        ├── validators.py
        └── loggers.py
```

---

### **File Descriptions**:

#### **1. Root-Level Files**:
| File | Description |
|------|------------|
| `setup.py` | Main package settings including name, version, and dependencies |
| `requirements.txt` | List of required libraries |

#### **2. Main Folder (`drilling_processor`)**:
| File/Folder | Description |
|-------------|------------|
| `__init__.py` | Initialization file to define the module |
| `core.py` | Main class `DrillingDataProcessor` for overall data processing management |

#### **3. Preprocessors Folder**:
| File | Description |
|------|------------|
| `cleaners.py` | `DataCleaner` class for handling missing and invalid data |
| `outliers.py` | `OutlierDetector` class for identifying outlier data |
| `feature_engine.py` | `FeatureEngineer` class for generating new features |
| `quality.py` | `QualityChecker` class for generating data quality reports |

#### **4. Pipelines Folder**:
| File | Description |
|------|------------|
| `ml_pipeline.py` | Contains `build_ml_pipeline()` function for constructing the machine learning pipeline |

#### **5. Utils Folder**:
| File | Description |
|------|------------|
| `validators.py` | Functions for validating input data |
| `loggers.py` | Event and error logging system |

---

### **Test Code Example**:
Test Folder Structure:

```plaintext
tests/
├── unit/
│   ├── test_cleaners.py
│   ├── test_outliers.py
│   └── ...
└── integration/
    ├── test_pipeline.py
    └── ...
```

Example from `cleaners` test:

```python
# tests/unit/test_cleaners.py
def test_missing_value_imputation():
    # Test the functionality of filling missing values
    test_data = pd.DataFrame({
        'Pressure': [1500, np.nan, 3000],
        'Temperature': [80, 120, np.nan]
    })

    cleaner = DataCleaner()
    result = cleaner.handle_missing_values(test_data)
    assert result.isnull().sum().sum() == 0
```

### **How to Run the Tests**:

```bash
# Run all tests
python -m pytest tests/

# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/
```

---

### **Technical Notes**:
1. Each module is independently extendable
2. Tests cover all possible scenarios
3. Documentation for each function is included in the respective files
4. Follows the latest Python coding standards

---

✍️ Developed by **Parnian Mahdian** | [GitHub Profile](https://github.com/Pmahdian)
