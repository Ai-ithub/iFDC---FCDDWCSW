## 📁 Datasets

This folder is dedicated to storing the datasets used in this project.
Please place all dataset-related files here to ensure easy access and organized data management.

> 💡 Note: Please avoid uploading large files directly to the repository. If necessary, include a download link to the dataset (e.g., from Google Drive, Kaggle, Zenodo, etc.) or add a script that builds or fetches the dataset..

---
## 📦 Sample output

| Record_ID | API_Well_ID |   LONG    |   LAT    |      DateTime       | Days_Age_Well | Phase_Operation | Formation_Type | Clay_Mineralogy_Type | Fractures_Presence | Reservoir_Temperature | Formation_Permeability | Porosity_Formation | Clay_Content_Percent | Completion_Type | Density_Perforation | Depth_Measured | Depth_Bit | Weight_on_Bit |   RPM   |  ROP  | Torque | Pressure_Standpipe | Pressure_Annulus | Overbalance | Pressure_Reservoir |   Mud_Type   | In_Rate_Flow_Mud | Mud_Weight_In | Mud_Temperature_In | Chloride_Content | Solid_Content | Mud_pH | Out_Rate_Flow_Mud | Volume_Pit | Mud_Temperature_Out | Viscosity | Fluid_Loss_API | Mud_Weight_Out | Active_Damage |       Type_Damage        |
|-----------|-------------|-----------|----------|---------------------|----------------|------------------|----------------|----------------------|---------------------|------------------------|-------------------------|---------------------|------------------------|------------------|----------------------|----------------|-----------|----------------|---------|-------|--------|---------------------|-------------------|-------------|---------------------|--------------|-------------------|----------------|---------------------|-------------------|----------------|--------|--------------------|-------------|----------------------|-----------|------------------|----------------|----------------|---------------------------|
|    0      |  40100050   | -94.86079 | 32.26120 | 2023-01-01 00:00:00 |       0        | Drilling         | Shale          | Illite               |          0          |         70.59          |         153.14          |        18.99        |         25.71         | Cased           |        31.89         |    500.86      |  494.39   |     4902.66     |  130.32 | 13.72 | 533.19 |       3184.97       |       2995.44      |    86.86     |       4877.69       | Water-based  |       92.83       |     8.99       |        42.13        |      506.95       |     13.38      |  7.16  |        89.57        |   567.89    |        37.89         |   14.71    |       0.52       |      9.13      |       No       |      No Damage          |
|    1      |  40100050   | -94.85962 | 32.25940 | 2023-01-01 00:00:01 |       0        | Drilling         | Limestone      | Kaolinite            |          1          |         88.97          |         150.42          |        25.78        |         14.91         | Open Hole       |        12.45         |    507.73      |  498.58   |     4876.45     |  118.77 | 13.21 | 481.28 |       3051.16       |       2880.88      |    93.71     |       5032.48       | Oil-based    |       96.45       |     9.21       |        38.61        |      490.44       |     10.57      |  7.06  |        92.88        |   599.34    |        36.87         |   17.18    |       0.61       |      8.84      |       No       |      No Damage          |
|    2      |  40100050   | -94.85997 | 32.26165 | 2023-01-01 00:00:02 |       0        | Drilling         | Sandstone      | Montmorillonite      |          1          |         86.11          |          82.73          |        29.87        |         41.75         | Liner           |        18.91         |    503.65      |  496.53   |     4894.21     |  126.98 | 11.83 | 519.42 |       3194.31       |       3002.67      |   110.42     |       5096.74       | Synthetic    |       98.34       |     9.34       |        39.07        |      540.16       |     11.69      |  7.28  |        95.11        |   576.42    |        37.14         |   13.62    |       0.58       |      9.01      |      Yes       |   Clay & Iron Control    |

---
## ✅ Model input\output

```python
# Step 1: Load the dataset
df = pd.read_csv("synthetic_formation_damage_data.csv")  # Replace with your actual path

# Step 2: Drop non-numeric and unnecessary columns for LSTM input
non_numerical_cols = [
    "Record_ID", "API_Well_ID", "DateTime", "Type_Damage", "Phase_Operation",
    "Formation_Type", "Clay_Mineralogy_Type", "Completion_Type", "Mud_Type", "Active_Damage"
]

# optional, may be it's better to just drop `Active_Damage` and `Type_Damage`
X = df.drop(columns=non_numerical_cols) 

# Step 3: Encode label (Type_Damage) to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Active_Damage"])  # You can also get class names using label_encoder.classes_

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Recommended Folder Structure:


```
Datasets/
├── raw/                  # Original/raw datasets
├── processed/            # Cleaned or preprocessed datasets
└── README.md             # Documentation and dataset descriptions
```
