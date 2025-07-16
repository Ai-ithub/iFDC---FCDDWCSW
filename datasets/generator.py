import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pyarrow.parquet as pq
import pyarrow as pa


# Set random seed for reproducibility
np.random.seed(42)

# Configuration for data generation
records_per_well = 15_552_000  # Total records to generate per well
chunk_size = 1_000_000         # Number of records to process in each chunk

# List of well IDs with their longitude and latitude
wells_info = [
    (40100050, -94.86, 32.26),
    (40131881, -94.82, 32.26),
    (40134068, -94.78, 32.25),
    (40181715, -94.95, 32.17),
    (36535068, -94.18, 32.32),
    (36500362, -94.13, 32.32),
    (36530944, -94.15, 32.12),
    (18332094, -94.62, 32.37),
    (18331921, -94.6, 32.37),
    (18387931, -94.86, 32.45)
]

start_date = datetime(2023, 1, 1)  # Base start date for DateTime generation

# Domain-specific categories and parameters
formations = ['Shale', 'Limestone', 'Sandstone']
clay_types = ['Kaolinite', 'Illite', 'Montmorillonite']
completion_types = ['Cased', 'Open Hole', 'Liner']
mud_types = ['Water-based', 'Oil-based', 'Synthetic']

fracture_prob = {'Shale': 0.2, 'Limestone': 0.6, 'Sandstone': 0.4}
temp_base = {'Shale': 70, 'Limestone': 90, 'Sandstone': 85}
perm_base = {'Shale': 5, 'Limestone': 150, 'Sandstone': 80}
porosity_base = {'Shale': 8, 'Limestone': 16, 'Sandstone': 12}  # Example porosity base values in percentage
clay_base = {'Kaolinite': 15, 'Illite': 25, 'Montmorillonite': 40}
density_perforation_map = {'Cased': 30, 'Open Hole': 10, 'Liner': 20}
wob_map = {'Drilling': 5000, 'Completion': 2000, 'Production': 0}

# Determine operational phase based on well age in days
def phase_operation(day):
    if day < 100:
        return 'Drilling'
    elif day < 200:
        return 'Completion'
    return 'Production'

# Calculate probability score of damage based on inputs
def calc_damage(prob_base, clay, loss_fluid, fractures, api_loss):
    score = prob_base
    if clay > 30:
        score += 0.2
    if loss_fluid > 1.0:
        score += 0.3
    if fractures == 1:
        score += 0.1
    if api_loss > 0.6:
        score += 0.2
    return min(score, 0.95)

# Determine damage type based on domain-driven conditions
def determine_damage_type(row):
    if row['Clay_Content_Percent'] > 35 and row['Clay_Mineralogy_Type'] == "Montmorillonite":
        return "Clay & Iron Control"
    if row['Formation_Type'] == "Shale" and row['Fluid_Loss_API'] > 0.8:  # previous threshhold was 1.2
        return "Drilling-Induced Damage"
    if row['Fluid_Loss_API'] > 1.0 and row['Mud_Type'] == "Water-based":  # previous threshhold was 1.5
        return "Fluid Loss"
    if row['Chloride_Content'] > 500 and row['Solid_Content'] > 10:  # previous threshhols were 600 and 12
        return "Scale / Sludge Incompatibility"
    if row['Completion_Type'] == "Open Hole" and row['Mud_pH'] < 7.0:  # previous threshhold was 6.5
        return "Near-Wellbore Emulsions"
    if row['Formation_Permeability'] < 30 and row['Reservoir_Temperature'] > 85:  # previous threshholds were 90 and 20
        return "Rock/Fluid Interaction"
    if row['Completion_Type'] == "Cased" and row['Overbalance'] > 100:  # previous threshhold was 120
        return "Completion Damage"
    if row['Reservoir_Temperature'] > 95 and row['Mud_Weight_In'] > 9.5:  # previous threshholds were 100 and 10
        return "Stress/Corrosion Cracking"
    if row['Viscosity'] > 18 and row['Mud_Type'] == "Oil-based":  # previous threshhold was 20
        return "Surface Filtration"
    if row['Viscosity'] < 12 and row['Mud_Type'] == "Synthetic":  # previous threshhold was 20
        return "Ultra-Clean Fluids Control"
    return "Generic Damage"

# Generate a synthetic dataset for one well
def generate_data_for_well(well_id, long_val, lat_val, records, start_record_id=0):
    ID_Well_API = np.array([well_id]*records)
    LONG = long_val + np.random.randn(records)*0.001
    LAT = lat_val + np.random.randn(records)*0.001
    
    ID_Record = np.arange(start_record_id, start_record_id + records)
    DateTime = np.array([start_date + timedelta(seconds=i) for i in range(records)])
    Days_Age_Well = ((DateTime - start_date).astype('timedelta64[s]').astype(int) // 86400)

    # Categorical and numerical synthetic features
    Phase_Operation = np.array([phase_operation(d) for d in Days_Age_Well])
    Type_Formation = np.random.choice(formations, size=records, p=[0.4, 0.3, 0.3])
    Type_Mineralogy_Clay = np.random.choice(clay_types, size=records)
    Fractures_of_Presence = np.array([np.random.binomial(1, fracture_prob[t]) for t in Type_Formation])
    
    Temperature_Reservoir = np.array([temp_base[t] + np.random.randn()*2 for t in Type_Formation])
    Permeability_Formation = np.array([perm_base[t] + np.random.randn()*5 for t in Type_Formation])
    Porosity_Formation = np.array([porosity_base[t] + np.random.randn()*2 for t in Type_Formation])
    Percent_Content_Clay = np.array([clay_base[t] + np.random.randn()*3 for t in Type_Mineralogy_Clay])
    
    Type_Completion = np.random.choice(completion_types, size=records)
    Density_Perforation = np.array([density_perforation_map[t] + np.random.randn()*2 for t in Type_Completion])

    # Drilling dynamics
    Depth_Measured = Days_Age_Well * 5 + np.random.randn(records)*10 + 500
    Depth_Bit = Depth_Measured - (np.random.rand(records)*10)
    WOB = np.array([wob_map[phase] + np.random.randn()*300 for phase in Phase_Operation])
    RPM = np.array([120 + np.random.randn()*10 if phase == 'Drilling' else 50 + np.random.randn()*5 for phase in Phase_Operation])
    ROP = np.array([10 + 5*np.random.rand() if phase == 'Drilling' else 0 for phase in Phase_Operation])
    Torque = WOB / 10 + np.random.randn(records)*50
    Pressure_Standpipe = 3000 + ROP*20 + np.random.randn(records)*100
    Pressure_Annulus = Pressure_Standpipe - 200 + np.random.randn(records)*50
    Overbalance = 100 + np.random.randn(records)*20
    Pressure_Reservoir = 5000 + np.random.randn(records)*300

    # Mud parameters
    Type_Mud = np.random.choice(mud_types, size=records, p=[0.6, 0.3, 0.1])
    In_Rate_Flow_Mud = 100 + 10*np.random.randn(records)
    In_Weight_Mud = 9 + 0.5*np.random.randn(records)
    In_Temperature_Mud = 40 + 5*np.random.randn(records)
    Content_Chloride = 500 + 50*np.random.randn(records)
    Content_Solid = 10 + 5*np.random.randn(records)
    pH_Mud = 7 + np.random.randn(records)*0.5
    Out_Rate_Flow_Mud = In_Rate_Flow_Mud * (0.95 + 0.05*np.random.rand(records))
    Volume_Pit = 500 + 100*np.random.randn(records)
    Out_Temperature_Mud = In_Temperature_Mud - (1 + 0.5*np.random.rand(records))
    Viscosity = 15 + 5*np.random.randn(records)
    API_Loss_Fluid = np.clip(0.5 + 0.1*np.random.randn(records), 0, None)
    Out_Weight_Mud = In_Weight_Mud * (0.95 + 0.05*np.random.rand(records))
    num_injected = int(records * 0.002)
    
    # Inject controlled damage cases (if needed)
    if num_injected > 0:
        Temperature_Reservoir[:num_injected] = np.random.uniform(86, 100, size=num_injected)
        Permeability_Formation[:num_injected] = np.random.uniform(5, 29, size=num_injected)

    # Determine damage presence and type
    Active_Damage = []
    Type_Damage = []

    for i in range(records):
        damage_prob = calc_damage(0.1, Percent_Content_Clay[i], API_Loss_Fluid[i], Fractures_of_Presence[i], API_Loss_Fluid[i])
        is_damaged = np.random.rand() < damage_prob
        Active_Damage.append('Yes' if is_damaged else 'No')
        if is_damaged:
            row = {
                'Clay_Content_Percent': Percent_Content_Clay[i],
                'Clay_Mineralogy_Type': Type_Mineralogy_Clay[i],
                'Formation_Type': Type_Formation[i],
                'Fluid_Loss_API': API_Loss_Fluid[i],
                'Mud_Type': Type_Mud[i],
                'Chloride_Content': Content_Chloride[i],
                'Solid_Content': Content_Solid[i],
                'Completion_Type': Type_Completion[i],
                'Mud_pH': pH_Mud[i],
                'Formation_Permeability': Permeability_Formation[i],
                'Reservoir_Temperature': Temperature_Reservoir[i],
                'Overbalance': Overbalance[i],
                'Viscosity': Viscosity[i],
                'Mud_Weight_In': In_Weight_Mud[i],
            }
            Type_Damage.append(determine_damage_type(row))
        else:
            Type_Damage.append("No Damage")

    # Create final DataFrame
    df = pd.DataFrame({
        'Record_ID': ID_Record,
        'API_Well_ID': ID_Well_API,
        'LONG': LONG,
        'LAT': LAT,
        'DateTime': DateTime,
        'Days_Age_Well': Days_Age_Well,
        'Phase_Operation': Phase_Operation,
        'Formation_Type': Type_Formation,
        'Clay_Mineralogy_Type': Type_Mineralogy_Clay,
        'Fractures_Presence': Fractures_of_Presence,
        'Reservoir_Temperature': Temperature_Reservoir,
        'Formation_Permeability': Permeability_Formation,
        'Porosity_Formation': Porosity_Formation,
        'Clay_Content_Percent': Percent_Content_Clay,
        'Completion_Type': Type_Completion,
        'Density_Perforation': Density_Perforation,
        'Depth_Measured': Depth_Measured,
        'Depth_Bit': Depth_Bit,
        'Weight_on_Bit': WOB,
        'RPM': RPM,
        'ROP': ROP,
        'Torque': Torque,
        'Pressure_Standpipe': Pressure_Standpipe,
        'Pressure_Annulus': Pressure_Annulus,
        'Overbalance': Overbalance,
        'Pressure_Reservoir': Pressure_Reservoir,
        'Mud_Type': Type_Mud,
        'In_Rate_Flow_Mud': In_Rate_Flow_Mud,
        'Mud_Weight_In': In_Weight_Mud,
        'Mud_Temperature_In': In_Temperature_Mud,
        'Chloride_Content': Content_Chloride,
        'Solid_Content': Content_Solid,
        'Mud_pH': pH_Mud,
        'Out_Rate_Flow_Mud': Out_Rate_Flow_Mud,
        'Volume_Pit': Volume_Pit,
        'Mud_Temperature_Out': Out_Temperature_Mud,
        'Viscosity': Viscosity,
        'Fluid_Loss_API': API_Loss_Fluid,
        'Mud_Weight_Out': Out_Weight_Mud,
        'Active_Damage': Active_Damage,
        'Type_Damage': Type_Damage
    })
    return df

# Main saving loop for all wells
output_dir = 'well_outputs'
os.makedirs(output_dir, exist_ok=True)

# generate and save each chunk in Parquet format
for well_id, long_val, lat_val in wells_info:
    print(f"Generating data for well {well_id} ...")
    num_chunks = records_per_well // chunk_size
    remainder = records_per_well % chunk_size

    file_path = os.path.join(output_dir, f'well_{well_id}.parquet')
    if os.path.exists(file_path):
        os.remove(file_path)
    writer = None
    record_start = 0

    for chunk_i in range(num_chunks + (1 if remainder > 0 else 0)):
        current_chunk_size = chunk_size if chunk_i < num_chunks else remainder
        if current_chunk_size == 0:
            break
        print(f"  chunk {chunk_i + 1} / {num_chunks + (1 if remainder > 0 else 0)} size: {current_chunk_size}")
        df_chunk = generate_data_for_well(well_id, long_val, lat_val, current_chunk_size, record_start)
        record_start += current_chunk_size
        table = pa.Table.from_pandas(df_chunk)
        if writer is None:
            writer = pq.ParquetWriter(file_path, table.schema, compression='snappy')
        writer.write_table(table)

    if writer is not None:
        writer.close()

print("Data generation and saving done.")
