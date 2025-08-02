# Issues and Tasks of the iFDC---FCDDWCSW Project

## 🚀 Phase 1: Project Setup & Data Preparation

### 🧠 Issue 1 - Initialize Project Repository & Documentation

**Objective**: Set up version control, folder structure, and documentation.  
**Tasks**:

- Create GitHub/GitLab repository with standardized structure.
- Draft README.md with setup instructions and goals.
- Add LICENSE, CONTRIBUTING.md, and code of conduct.

**Deliverable**: Organized repo with documentation.

## Phase 2: Data Generating and Data Checking

### 🧠 Issue 2 - Create CSV File with Header

**Name**: Create Empty CSV File with Header  
**Objective**: Initialize the structure for synthetic formation data CSV file by creating the necessary header fields as defined in the SRS.  
**Steps**:

1. Define a list of columns as per the SRS, including `depth`, `temperature`, `pressure`, `fluid_type`, `damage_type`, etc.
2. Create a CSV file named `synthetic_formation_damage_data.csv` in the `/data` folder.
3. Write the header as the first row, ensuring it is consistent with the schema described in the SRS.

**Requirements**:

- Use `csv.writer` from Python’s standard library.
- Only the header row should be written at this stage.

---

### 🧠 Issue 3 — Generate Random Pressure, Temperature, Depth Values

**Name**: Generate Base Formation Values  
**Objective**: Populate the CSV with realistic synthetic values for `depth`, `pressure`, and `temperature`, mimicking real-world data distributions.  
**Steps**:

1. Loop through a range of 100,000 (start small and then scale up).
2. For each row, generate:
   - `depth`: Random float between 500m and 5000m.
   - `pressure`: Random value between 2000 and 10000 psi.
   - `temperature`: Random value between 50°C and 180°C.
3. Append each row to the CSV file.

**Requirements**:

- Use `random.uniform` for generating float values.
- Ensure that each value is stored with two decimal precision.

---

### 🧠 Issue 4 — Add Categorical Columns: Fluid Type & Damage Type

**Name**: Add Fluid Type and Damage Type Columns  
**Objective**: Add categorical columns for `fluid_type` and `damage_type`, with randomized values from predefined lists.  
**Steps**:

1. Define fixed lists for fluid types and damage types:
   - `fluid_type`: e.g., `['water-based', 'oil-based', 'acid', 'polymer']`.
   - `damage_type`: e.g., `['Clay & Iron Control', 'Drilling-Induced Damage', 'Fluid Loss', 'Scale / Sludge Incompatibility', ...]`.
2. For each row, randomly assign one value from each list.
3. Append these values to the corresponding columns.

**Requirements**:

- Use `random.choice` for categorical assignment.
- Ensure damage types are chosen from the list in the SRS, with at least 10 unique categories.

---

### 🧠 Issue 5 — Add Optional Noise Injection (Advanced)

**Name**: Add Optional Data Noise for Realism  
**Objective**: Introduce controlled noise to `pressure` and `temperature` fields to simulate real-world data imperfections.  
**Steps**:

1. Create a function `inject_noise(row)` that adds noise to either `pressure` or `temperature`.
2. Introduce a 10% chance of adding +/- 5% noise to each value.
3. Add a CLI argument `--with-noise` to allow toggling of this feature.

**Requirements**:

- Use `argparse` for CLI integration.
- Use `numpy` for generating random noise within a given percentage range.
- Ensure rows are left unchanged if noise injection is disabled.

---

### 🧠 Issue 6 — Generate 1 Million Rows and Save in Batches

**Name**: Generate 1 Million Rows and Save Efficiently  
**Objective**: Efficiently generate 1 million rows of synthetic data and write them to the CSV file in manageable batches.  
**Steps**:

1. Generate data in batches of 100,000 rows to avoid memory overflow.

2. Append each batch to `synthetic_formation_damage_data.csv`.
3. Display progress every 100k rows using `tqdm`.

**Requirements**:

- Use `tqdm` for progress reporting.
- Ensure the total number of rows in the file is exactly 1,000,000.
- Store the file under `/data/synthetic_formation_damage_data.csv`.

---

### 🧠 Issue 7 — Validate Output CSV File

**Name**: CSV Format and Value Validator  
**Objective**: Validate the integrity and structure of the generated CSV file. Ensure that the data adheres to the expected schema.  
**Steps**:

1. Check that the row count matches 1 million.

2. Ensure there are no missing or NaN values.
3. Print unique values for each column and compare them to expected ranges.
4. Save the validation results to a file `validation_report.txt`.

**Requirements**:

- Use `pandas.read_csv` for loading the file.
- Log any discrepancies in `validation_report.txt`.

---

### 🧠 Issue 8 - Synthetic Data Generation

**Objective**: Generate labeled synthetic data for prototyping.  
**Tasks**:

- Finalize data schema (drilling params, fluid properties, damage labels).
- Develop `generate_synthetic_data.py` with realistic distributions.
- Validate data ranges with domain experts.

**Deliverable**: `synthetic_formation_damage_data.csv` (15M+ records).

### 🧠 Issue 9 - Data Pipeline MVP

**Objective**: Build a pipeline to ingest and preprocess data.  
**Tasks**:

- Set up PostgreSQL/MongoDB hybrid storage.
- Write ETL scripts for cleaning/normalization (Python/Pandas).
- Add unit tests for data validation.

**Deliverable**: Pipeline with test cases.

## 🚀 Phase 3: Modeling & Simulation

### 🧠 Issue 10 - Damage Classification Model (XGBoost/LightGBM)

**Objective**: Train and fine-tune the damage classification model using structured data from FCDD.  
**Tasks**:

- Perform exploratory data analysis (`notebooks/eda.ipynb`).
- Train and fine-tune XGBoost or LightGBM model using the preprocessed synthetic data.
- Evaluate model performance using metrics such as precision, recall, and F1-score.
- Save the trained model to `models/xgboost_model.json`.

**Deliverable**: Trained model (`models/xgboost_model.json`).

---

### 🧠 Issue 11 - Anomaly Detection (Autoencoder/Isolation Forest)

**Objective**: Identify and alert anomalies in real-time sensor data using unsupervised learning techniques.  
**Tasks**:

- Implement Autoencoder model using TensorFlow/Keras for unsupervised anomaly detection.
- Integrate Isolation Forest from `sklearn.ensemble` for comparison.
- Integrate Kafka for real-time anomaly alerting.
- Output anomalies to Kafka topic `anomaly_alerts`.

**Deliverable**: Anomaly detection module.

---

### 🧠 Issue 12 - Physics-Based Simulation (OpenFOAM/FEniCS)

**Objective**: Simulate fluid-rock interactions using a Finite Element Model (FEM).  
**Tasks**:

- Develop FEM model for fluid-rock interactions (`simulation/fem_model.py`).
- Validate simulations using lab and field data.
- Store simulation outputs and validation reports in `simulation/results/`.

**Deliverable**: Simulation outputs + validation report.

---

## 🚀 Phase 4: Dashboard & Real-Time Integration

### 🧠 Issue 13 - Dashboard Frontend (React.js/D3.js)

**Objective**: Build an interactive and user-friendly monitoring UI for real-time damage detection and analysis.  
**Tasks**:

- Design and implement filters for location, depth, and time.
- Create time-series charts to display real-time data for pressure, temperature, and fluid loss.
- Implement real-time anomaly alerts using WebSockets or REST API.

**Deliverable**: Deployed React dashboard.

---

### 🧠 Issue 14 - Backend API (FastAPI)

**Objective**: Develop a FastAPI backend to serve model predictions, simulations, and handle data requests.  
**Tasks**:

- Create RESTful API endpoints for damage type prediction and simulation.
- Implement user authentication and logging using JWT tokens.
- Implement rate-limiting for the prediction API.

**Deliverable**: FastAPI backend with Swagger docs.

---

### 🧠 Issue 15 - Real-Time Monitoring (Kafka + Grafana)

**Objective**: Set up real-time monitoring of sensor data and system health using Kafka and Grafana.  
**Tasks**:

- Set up Kafka brokers to handle real-time data ingestion.
- Create Grafana dashboards to visualize fluid loss, pressure, temperature, and anomaly alerts.
- Implement real-time data push using WebSockets.

**Deliverable**: Real-time monitoring system.

---

## 🚀 Phase 5: Testing & Deployment

### 🧠 Issue 16 - End-to-End Testing

**Objective**: Validate full system integration.  
**Tasks**:

- Test the entire data pipeline, from ingestion to prediction, ensuring smooth operation.
- Stress-test APIs and Kafka stream to handle large-scale data inputs.

**Deliverable**: Test report + bug fixes.

---

### 🧠 Issue 17 - Cloud Deployment (AWS/GCP)

**Objective**: Deploy the entire system to the cloud (AWS or GCP) with auto-scaling.  
**Tasks**:

- Containerize all services using Docker.
- Set up a Kubernetes cluster for scalable deployment (EKS/GKE).
- Ensure continuous deployment (CI/CD) pipelines are automated.

**Deliverable**: Cloud deployment with auto-scaling.

---

## 🚀 Phase 6: Roadmap & Maintenance

### 🧠 Issue 18 - Self-Learning Model Pipeline

**Objective**: Enable continuous learning and model updates with new data.  
**Tasks**:

- Implement a feedback loop for continuous model updates using SCADA/PI system data.
- Schedule periodic retraining of models based on performance metrics.

**Deliverable**: CI/CD pipeline for models.

---

### 🧠 Issue 19 - SCADA/PI System Integration

**Objective**: Integrate SCADA/PI system to collect real-time well data for model predictions.  
**Tasks**:

- Develop API adapters for SCADA/PI systems to retrieve real-time data.
- Process and clean incoming data for immediate model use.

**Deliverable**: Integrated production data source.
