# 🛢️ Formation Damage Detection System (FCDD)

---

## 🧩 Overview

**FCDD (Formation Condition & Damage Detection)** is an advanced integrated software system designed to **identify, predict, and analyze formation damage** in oil and gas fields. The system aims to enhance drilling and production efficiency while reducing costs associated with undetected formation damage.

---

## 🎯 System Objectives

- 📌 Automatically detect various types of formation damage during drilling, completion, and production phases  
- 📌 Reduce uncertainty in operational and technical decision-making  
- 📌 Provide multidimensional analysis of geological and operational data  
- 📌 Deliver real-time alerts to prevent serious damage  
- 📌 Recommend preventive and corrective actions using intelligent algorithms

---

## 🔬 Scope of Operation

### Covered Damage Types:

| Damage Type | Description |
|-------------|-------------|
| Clay & Iron Control | Chemical interactions with clay and iron |
| Drilling-Induced Damage | Mechanical and pressure-related damage during drilling |
| Fluid Loss | Loss of drilling or stimulation fluids |
| Scale / Sludge Incompatibility | Formation of inorganic or organic deposits |
| Near-Wellbore Emulsions | Emulsion formation near the wellbore |
| Rock/Fluid Interaction | Incompatibility between formation rock and fluids |
| Completion Damage | Loss of connectivity between formation and completion |
| Stress/Corrosion Cracking | Cracks caused by stress or corrosion |
| Surface Filtration | Surface fluid filtration disruptions |
| Ultra-Clean Fluids Control | Management of high-purity fluids during stimulation |

---

## 🧠 Technologies Used

| Module | Technology / Language | Reason |
|--------|------------------------|--------|
| Data Mining | Python (Pandas, NumPy) | Fast processing of time-series data |
| Machine Learning | XGBoost, LightGBM, TensorFlow | Multiclass classification and prediction |
| Simulation | OpenFOAM (C++), FEniCS (Python) | Physical process simulations |
| UI | React.js, D3.js | Interactive and visual dashboards |
| Backend | FastAPI (Python) | Fast, lightweight RESTful API |
| Storage | PostgreSQL + MongoDB | Hybrid structured and semi-structured data |
| Real-Time Monitoring | Kafka + Grafana | High-speed data transmission and live visualization |

---

## 🔍 Implemented Machine Learning Algorithms

- 🎯 **XGBoost / LightGBM**: For damage type classification using drilling and fluid features  
- 🔁 **LSTM / GRU**: For time-series modeling of fluid loss and critical event prediction  
- 📊 **KMeans / DBSCAN**: For clustering and discovery of hidden damage patterns  
- 🧬 **Autoencoder / Isolation Forest**: For anomaly detection and outlier discovery  
- 🧪 **GAN (Generative Adversarial Networks)**: For synthetic data generation and scenario modeling

---

## 🧪 Synthetic Data Generation

To support initial development, a Python script is used to generate synthetic data, including drilling parameters, fluid properties, pressure and temperature conditions, and damage type labels.

📄 Script: `generate_synthetic_data.py`  
📁 Output File: `synthetic_formation_damage_data.csv`  
📈 Record Count: 15,552,000

---

## 📊 Monitoring & Analysis Dashboard

- Web-based dashboard using React.js and D3.js  
- Filters by location, depth, and well history  
- Time-series charts for losses, pressure, temperature, and damage predictions  
- Real-time alerts for anomalies and critical damage events

---

## 🧰 Project Structure

```
formation-damage-system/
├── data/
│   └── synthetic_formation_damage_data.csv
├── models/
│   └── xgboost_model.json
├── dashboard/
│   ├── frontend/ (React.js)
│   └── backend/ (FastAPI)
├── simulation/
│   └── fem_model.py
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
├── generate_synthetic_data.py
├── predict_damage_type.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-org/formation-damage-system.git
cd formation-damage-system
pip install -r requirements.txt
python generate_synthetic_data.py
python predict_damage_type.py
```

---

## 💻 Recommended Hardware

| Component | Specs |
|-----------|-------|
| CPU | Intel Xeon or AMD EPYC, minimum 16 cores |
| RAM | Minimum 64 GB |
| GPU | NVIDIA RTX A6000 or A100, minimum 16 GB VRAM |
| Storage | 1 TB SSD + HDD for archiving |
| Cloud (Optional) | AWS EC2 (GPU Enabled) or Google Cloud TPU |

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `generate_synthetic_data.py` | Controlled-distribution synthetic data generation |
| `predict_damage_type.py` | Formation damage type prediction using XGBoost |
| `dashboard/frontend/` | Interactive damage monitoring UI |
| `simulation/fem_model.py` | Finite element simulation model for damage processes |

---

## 📌 Future Roadmap

- Real-time prediction module with Kafka integration  
- Self-learning models with continuous updates  
- Direct integration with well databases (SCADA, PI System)  
- Cloud-native deployment (Kubernetes-ready)  
- Automated root cause analysis module

---

## 🤝 Contributions

- Submit Pull Requests to improve models or dashboards  
- Report bugs or suggest enhancements via Issues  
- Contribute to simulation modules, feature extensions, and field evaluations

---

## 🧾 License

> This project is released under the **MIT License**. Free to use, develop, commercialize, and distribute with attribution.

---

## 👨‍🔬 Developed By

- Digital R&D unit in oil and gas industry  
- Reservoir, drilling, and production data analytics team  
- Collaboration with universities and geological research centers
