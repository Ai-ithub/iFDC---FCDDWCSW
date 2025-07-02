import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from confluent_kafka import Producer

# ---------- Kafka Configuration ----------
producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'infinite-well-generator'
})
topic = 'fdms.well.stream'

# ---------- Simulation Configuration ----------
well_id = 40100050
long_val = -94.86
lat_val = 32.26

formations = ['Shale', 'Limestone', 'Sandstone']
clay_types = ['Kaolinite', 'Illite', 'Montmorillonite']
completion_types = ['Cased', 'Open Hole', 'Liner']
mud_types = ['Water-based', 'Oil-based', 'Synthetic']

fracture_prob = {'Shale': 0.2, 'Limestone': 0.6, 'Sandstone': 0.4}
temp_base = {'Shale': 70, 'Limestone': 90, 'Sandstone': 85}
perm_base = {'Shale': 5, 'Limestone': 150, 'Sandstone': 80}
clay_base = {'Kaolinite': 15, 'Illite': 25, 'Montmorillonite': 40}
density_perforation_map = {'Cased': 30, 'Open Hole': 10, 'Liner': 20}
wob_map = {'Drilling': 5000, 'Completion': 2000, 'Production': 0}

start_date = datetime(2023, 1, 1)
record_id = 0

def phase_operation(day):
    if day < 100:
        return 'Drilling'
    elif day < 200:
        return 'Completion'
    return 'Production'

def generate_one_record(record_id, seconds_since_start):
    current_time = start_date + timedelta(seconds=seconds_since_start)
    day = seconds_since_start // 86400

    formation = np.random.choice(formations, p=[0.4, 0.3, 0.3])
    clay = np.random.choice(clay_types)
    completion = np.random.choice(completion_types)
    mud_type = np.random.choice(mud_types, p=[0.6, 0.3, 0.1])
    fractures = np.random.binomial(1, fracture_prob[formation])

    record = {
        'Record_ID': record_id,
        'API_Well_ID': well_id,
        'LONG': long_val + np.random.randn() * 0.001,
        'LAT': lat_val + np.random.randn() * 0.001,
        'DateTime': current_time.isoformat(),
        'Days_Age_Well': day,
        'Phase_Operation': phase_operation(day),
        'Formation_Type': formation,
        'Clay_Mineralogy_Type': clay,
        'Fractures_Presence': fractures,
        'Reservoir_Temperature': temp_base[formation] + np.random.randn() * 2,
        'Formation_Permeability': perm_base[formation] + np.random.randn() * 5,
        'Clay_Content_Percent': clay_base[clay] + np.random.randn() * 3,
        'Completion_Type': completion,
        'Density_Perforation': density_perforation_map[completion] + np.random.randn() * 2,
        'Depth_Measured': day * 5 + 500 + np.random.randn() * 10,
        'Depth_Bit': day * 5 + 495 + np.random.randn() * 10,
        'Weight_on_Bit': wob_map[phase_operation(day)] + np.random.randn() * 300,
        'RPM': 120 + np.random.randn()*10 if day < 100 else 50 + np.random.randn()*5,
        'ROP': 10 + 5*np.random.rand() if day < 100 else 0,
        'Torque': 500 + np.random.randn() * 50,
        'Pressure_Standpipe': 3000 + np.random.randn()*100,
        'Pressure_Annulus': 2800 + np.random.randn()*50,
        'Overbalance': 100 + np.random.randn()*20,
        'Pressure_Reservoir': 5000 + np.random.randn()*300,
        'Mud_Type': mud_type,
        'In_Rate_Flow_Mud': 100 + np.random.randn() * 10,
        'Mud_Weight_In': 9 + 0.5*np.random.randn(),
        'Mud_Temperature_In': 40 + 5*np.random.randn(),
        'Chloride_Content': 500 + 50*np.random.randn(),
        'Solid_Content': 10 + 5*np.random.randn(),
        'Mud_pH': 7 + np.random.randn() * 0.5,
        'Out_Rate_Flow_Mud': 95 + np.random.randn() * 5,
        'Volume_Pit': 500 + 100*np.random.randn(),
        'Mud_Temperature_Out': 38 + np.random.randn() * 2,
        'Viscosity': 15 + 5*np.random.randn(),
        'Fluid_Loss_API': np.clip(0.5 + 0.1*np.random.randn(), 0, None),
        'Mud_Weight_Out': 9 + 0.5*np.random.randn()
    }

    return record

# ---------- Infinite Loop for Kafka Publishing ----------
print(f"💡 Starting Kafka data stream for well {well_id} to topic '{topic}' ...")

seconds_since_start = 0

try:
    while True:
        record = generate_one_record(record_id, seconds_since_start)
        producer.produce(topic, key=str(record_id), value=json.dumps(record))
        producer.poll(0)
        print(f"[{record_id}] Sent record at {record['DateTime']}")
        time.sleep(1)

        record_id += 1
        seconds_since_start += 1

except KeyboardInterrupt:
    print("\n⛔️ Stopped by user.")
    producer.flush()
