import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os

INPUT_DIR = "well_outputs"
OUTPUT_DIR = "well_outputs_flagged"
REPORT_CSV = "anomaly_report_summary.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_combination_anomalies(df):
    conditions = [
        (df["Formation_Type"] == "Shale") & (df["Formation_Permeability"] > 50),
        (df["Formation_Type"] == "Limestone") & (df["Reservoir_Temperature"] < 60),
        (df["Formation_Type"] == "Sandstone") & (df["Clay_Content_Percent"] > 60),

        (df["Mud_Type"] == "Water-based") & (df["Viscosity"] > 60),
        (df["Mud_Type"] == "Oil-based") & (df["Mud_pH"] < 6),
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

        (df["Clay_Mineralogy_Type"] == "Montmorillonite") & (df["Mud_pH"] > 10),
        (df["Formation_Type"] == "Shale") & (df["Fractures_Presence"] == 1) & (df["Formation_Permeability"] > 100),
        (df["Phase_Operation"] == "Production") & (df["Weight_on_Bit"] > 1000),
        (df["Fluid_Loss_API"] > 3.0) & (df["Mud_Type"] == "Oil-based"),
        (df["Reservoir_Temperature"] > 120) & (df["Clay_Content_Percent"] > 50)
    ]

    anomaly_mask = conditions[0]
    for cond in conditions[1:]:
        anomaly_mask |= cond
    
    return anomaly_mask

summary_records = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".parquet"):
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"🔍 Processing {filename} ...")
        
        df = pd.read_parquet(filepath)
        anomaly_mask = check_combination_anomalies(df)
        df["Combination_Anomaly"] = anomaly_mask

        output_path = os.path.join(OUTPUT_DIR, filename)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')

        total = len(df)
        count_anomalies = anomaly_mask.sum()
        percent = round((count_anomalies / total) * 100, 4)

        summary_records.append({
            "Well File": filename,
            "Total Records": total,
            "Anomalies": count_anomalies,
            "Percentage (%)": percent
        })

        print(f"✅ Done: {count_anomalies} anomalies ({percent}%)")

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(REPORT_CSV, index=False)
print(f"📊 Summary report saved to: {REPORT_CSV}")

