import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm.auto import tqdm

# Configuration
np.random.seed(42)
# records_per_well = 864000  # 10 days of 1-second data
# chunk_size = 10000  # Records per chunk

records_per_well = 6 * 30 * 24 * 60 * 60  # 6 month in second
chunk_size = 1_000_000         # Number of records to process in each chunk
output_dir = 'well_data'
os.makedirs(output_dir, exist_ok=True)

# Well configuration
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
start_date = datetime(2023, 1, 1)

# Enhanced formation definitions
formations = {
    'Topsoil': {
        'depth_range': (0, 2), 'density': 1.5, 'base_rop': 65, 'wob_factor': 0.7,
        'perm_range': (1000, 5000), 'porosity': 40, 'clay_content_range': (5, 10),
        'fracture_grad': 0.6, 'temp_grad': 0.01, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.01, 'color': 'brown'
    },
    'Sandstone': {
        'density': 2.3, 'base_rop': 20, 'wob_factor': 1.0,
        'perm_range': (10, 1000), 'porosity': 22, 'clay_content_range': (10, 15),
        'fracture_grad': 0.7, 'temp_grad': 0.026, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.07, 'pressure_mult': 1.0, 'reservoir': True
    },
    'Shale': {
        'density': 2.4, 'base_rop': 12, 'wob_factor': 1.3,
        'perm_range': (0.001, 0.1), 'porosity': 10, 'clay_content_range': (25, 35),
        'fracture_grad': 0.75, 'temp_grad': 0.025, 'clay_type': 'Illite',
        'fracture_prob': 0.15, 'pressure_mult': 0.85, 'seal': True
    },
    'Weathered_Rock': {
        'depth_range': (2, 10), 'density': 1.8, 'base_rop': 45, 'wob_factor': 0.8,
        'perm_range': (500, 2000), 'porosity': 30, 'clay_content_range': (10, 15),
        'fracture_grad': 0.65, 'temp_grad': 0.015, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.03, 'color': 'tan'
    },
    'Limestone': {
        'density': 2.7, 'base_rop': 15, 'wob_factor': 1.2,
        'perm_range': (1, 100), 'porosity': 15, 'clay_content_range': (5, 10),
        'fracture_grad': 0.8, 'temp_grad': 0.028, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.1, 'pressure_mult': 1.1,'reservoir': True,
        'seal': False, 'color': 'light gray'
    },
}

bit_types = {
    'PDC': {'wob_eff': 0.9, 'rop_eff': 1.3},
    'Tricone': {'wob_eff': 1.1, 'rop_eff': 0.8}
}

mud_programs = [
    # Surface mud (0-50m)
    {
        'conditions': {'max_depth': 50},
        'mud': {
            'type': 'Water-based',
            'base_weight': 8.5,  # ppg at surface
            'weight_grad': 0.015,  # ppg/m increase with depth
            'base_temp': 20,  # °C at surface
            'temp_grad': 0.035,  # °C/m increase with depth
            'viscosity': 15, 
            'ph': 8.0,
            'chloride_base': 300, 
            'chloride_grad': 0.3,
            'temp_diff': (2, 5)  # Min/max temp difference between in/out
        }
    },
    # Intermediate mud (50-300m)
    {
        'conditions': {'min_depth': 50, 'max_depth': 300},
        'mud': {
            'type': 'Water-based',
            'base_weight': 9.5,
            'weight_grad': 0.02,
            'base_temp': 22,
            'temp_grad': 0.040,
            'viscosity': 25,
            'ph': 8.5,
            'chloride_base': 400,
            'chloride_grad': 0.5,
            'temp_diff': (3, 6)
        }
    },
    # Shale-specific mud (300-600m shale)
    {
        'conditions': {'min_depth': 300, 'max_depth': 600, 'formations': ['Shale']},
        'mud': {
            'type': 'Water-based',
            'base_weight': 10.5,
            'weight_grad': 0.0025,
            'base_temp': 25,
            'temp_grad': 0.022,
            'viscosity': 30,
            'ph': 9.0,
            'chloride_base': 600,
            'chloride_grad': 0.7,
            'temp_diff': (4, 7)
        }
    },
    # Reservoir drilling mud (600m+)
    {
        'conditions': {'min_depth': 600},
        'mud': {
            'type': 'Oil-based',
            'base_weight': 12.0,
            'weight_grad': 0.003,
            'base_temp': 30,
            'temp_grad': 0.025,
            'viscosity': 40,
            'ph': 7.5,
            'chloride_base': 50,
            'chloride_grad': 0.1,
            'temp_diff': (5, 10)  # Larger difference for oil-based mud
        }
    }
]

def select_formation(depth):
    """Select formation based on depth with realistic probabilities"""
    formations = [
        'Topsoil',
        'Weathered_Rock', 
        'Sandstone',
        'Shale',
        'Limestone'
    ]
    
    if depth < 50:
        return str(np.random.choice(formations[:3], p=[0.1, 0.7, 0.2]))
    elif depth < 300:
        return str(np.random.choice(formations[1:4], p=[0.6, 0.3, 0.1]))
    elif depth < 600:
        return str(np.random.choice(formations[2:], p=[0.5, 0.3, 0.2]))
    else:
        return str(np.random.choice(formations[2:], p=[0.6, 0.3, 0.1]))

def get_formation_thickness(formation, current_depth):
    """Get realistic thickness ranges for each formation type"""
    base_thickness_ranges = {
        'Topsoil': (1, 3),
        'Weathered_Rock': (5, 15),
        'Sandstone': (10, 100),
        'Shale': (50, 300),
        'Limestone': (20, 150)
    }
    
    # Default to Sandstone thickness if formation not specified
    base_range = base_thickness_ranges.get(formation, (10, 100))
    
    # Depth scaling factor (formations get thinner with depth)
    depth_factor = max(0.3, 1 - (current_depth / 3000))
    
    min_thick = base_range[0] * depth_factor
    max_thick = base_range[1] * depth_factor
    
    return np.random.uniform(min_thick, max_thick)

def generate_formation_layers(total_depth=3000):
    layers = []
    current_depth = 0
    
    # Surface layers
    layers.extend([
        create_layer(0, 2, 'Topsoil'),
        create_layer(2, 10, 'Weathered_Rock')
    ])
    current_depth = 10
    
    # Subsurface layers
    while current_depth < total_depth:
        formation = select_formation(current_depth)
        thickness = get_formation_thickness(formation, current_depth)
        thickness = min(thickness, total_depth - current_depth)
        
        layer = create_layer(
            current_depth,
            current_depth + thickness,
            formation
        )
        layers.append(layer)
        current_depth += thickness
    
    return layers

def create_layer(top, bottom, formation):
    # Convert any numpy string/bytes type to Python string
    if isinstance(formation, (np.str_, np.bytes_, np.object_)):
        formation = str(formation)
    
    if formation not in formations:
        raise ValueError(f"Formation '{formation}' not defined. Available formations: {list(formations.keys())}")
    
    props = formations[formation]
    depth = (top + bottom) / 2
    
    wob = calculate_wob(depth, props)
    rop = calculate_rop(wob, depth, props)
    
    return {
        'top': float(top),
        'bottom': float(bottom),
        'formation': str(formation),
        'density': float(props['density']),
        'rop': float(rop),
        'wob': float(wob),
        'porosity': float(props['porosity']),
        'perm': float(np.random.uniform(*props['perm_range'])),
        'clay_content': float(np.random.uniform(*props['clay_content_range'])),
        'clay_type': str(props['clay_type']),
        'fracture_prob': float(props['fracture_prob']),
        'temp_grad': float(props['temp_grad']),
        'reservoir': bool(props.get('reservoir', False)),
        'seal': bool(props.get('seal', False))
    }

def calculate_wob(depth, formation, bit_type='PDC'):
    bit = bit_types[bit_type]
    base_wob = 5000 + (depth * 5)
    formation_factor = formation['wob_factor']
    wob = base_wob * formation_factor * bit['wob_eff']
    max_wob = 10000 + (depth * 8)
    return min(wob, max_wob)

def calculate_rop(wob, depth, formation, bit_type='PDC'):
    bit = bit_types[bit_type]
    effective_wob = min(wob, 25000)
    formation_factor = (2.5 / formation['density']) * formation['base_rop']
    depth_factor = max(0.3, 1 - (depth / 4000))
    wob_factor = (effective_wob / 10000) ** 0.7
    rop = formation_factor * depth_factor * wob_factor * bit['rop_eff']
    return max(5, min(rop, 100))

def select_completion_type(layers):
    reservoir_depths = [layer['top'] for layer in layers if layer.get('reservoir', False)]
    has_shale = any(layer['formation'] == 'Shale' for layer in layers)
    
    if not reservoir_depths:
        return 'Open Hole', 10  # No reservoir, simple completion
    
    reservoir_depth = min(reservoir_depths)
    
    if has_shale and reservoir_depth > 500:
        return 'Liner', 20
    elif has_shale:
        return 'Cased', 30
    elif reservoir_depth > 800:
        return 'Liner', 20
    else:
        return 'Open Hole', 10

def get_mud_program(depth, formation):
    """Get mud program with proper weight initialization"""
    for program in reversed(mud_programs):
        conditions = program['conditions']
        depth_ok = True
        formation_ok = True
        
        if 'min_depth' in conditions and depth < conditions['min_depth']:
            depth_ok = False
        if 'max_depth' in conditions and depth >= conditions['max_depth']:
            depth_ok = False
        if 'formations' in conditions and formation not in conditions['formations']:
            formation_ok = False
            
        if depth_ok and formation_ok:
            mud = program['mud'].copy()
            # Initialize weight if not present
            if 'weight' not in mud:
                mud['weight'] = mud['base_weight'] + (depth * mud['weight_grad'])
            return mud
    # Fallback with default values
    return {
        'type': 'Water-based',
        'weight': 10.0,  # Default weight
        'viscosity': 30,
        'ph': 8.0,
        'temp_diff': (3, 5)
    }

def calculate_pressures(depth, mud_weight, formation_props):
    # Standpipe pressure = hydrostatic + friction + safety margin
    hydrostatic = mud_weight * depth * 0.052 * 3.281  # ppg to psi/m
    standpipe = hydrostatic * 1.05 + np.random.uniform(100, 300)
    
    # Annulus pressure
    annulus = hydrostatic * 0.95
    
    # Reservoir pressure (only in reservoir formations)
    if formation_props.get('reservoir', False):
        # Normal pressure gradient (0.465 psi/ft)
        reservoir = depth * 0.465 * 3.281
        # Adjust for formation type
        if formation_props['formation'] == 'Sandstone':
            reservoir *= np.random.uniform(0.95, 1.05)
        elif formation_props['formation'] == 'Limestone':
            reservoir *= np.random.uniform(0.9, 1.1)
    else:
        reservoir = 0
    
    return {
        'standpipe': standpipe,
        'annulus': annulus,
        'reservoir': float(reservoir),  # Ensure float type
        'overbalance': float(standpipe - reservoir if reservoir > 0 else 0)
    }

def calculate_chloride_content(depth, mud_program, formation):
    """Calculate realistic chloride content with safe defaults"""
    # Get base value with fallback
    base = mud_program.get('chloride_base', 300)  # Default 300 mg/l if not specified
    
    # Get gradient with fallback
    gradient = mud_program.get('chloride_grad', 0.3)  # Default 0.3 mg/l/m
    
    # Formation adjustment
    if formation == 'Shale':
        base *= 1.5  # Increase chloride content in shale formations
    
    return base + (depth * gradient) * np.random.uniform(0.8, 1.2)

def calculate_mud_temperatures(mud_program, formation_temp):
    """Calculate realistic mud temperatures based on mud program and formation temp"""
    # Mud temperature in is slightly cooler than formation
    mud_temp_in = formation_temp - np.random.uniform(1, 3)
    
    # Mud temperature out is heated by friction
    temp_diff = np.random.uniform(*mud_program['temp_diff'])
    mud_temp_out = mud_temp_in + temp_diff
    
    # Ensure physical consistency
    mud_temp_out = max(mud_temp_out, mud_temp_in + 0.5)  # Always some heating
    mud_temp_out = min(mud_temp_out, formation_temp + 15)  # Don't exceed reasonable limits
    
    return mud_temp_in, mud_temp_out


def generate_damage_indicator(row):
    base_prob = {
        'Shale': 0.15,
        'Sandstone': 0.07,
        'Limestone': 0.05,
        'Topsoil': 0.02
    }.get(row['Formation_Type'], 0.05)
    
    risk_factors = {
        'high_clay': (row['Clay_Content_Percent_%'] > 30, 0.25),
        'overbalance': (row['Overbalance_psi'] > 500, 0.15),
        'mud_weight': (row['Mud_Weight_In_ppg'] > 11, 0.1),
        'low_perm': (row['Formation_Permeability_mD'] < 10, 0.2)
    }
    
    damage_prob = base_prob
    for factor, (condition, weight) in risk_factors.items():
        if condition:
            damage_prob = min(0.95, damage_prob + weight)
    
    damage_types = [
        ("Fluid Invasion", 0.4),
        ("Clay Swelling", 0.3 if row['Clay_Content_Percent_%'] > 20 else 0.1),
        ("Fines Migration", 0.2),
        ("Scale Deposition", 0.1),
        ("Emulsion Blockage", 0.05)
    ]
    
    if np.random.random() < damage_prob:
        types, probs = zip(*damage_types)
        return 'Yes', np.random.choice(types, p=np.array(probs)/sum(probs))
    return 'No', 'None'

# PyArrow schema definition
schema = pa.schema([
    ('Record_ID', pa.int64()),
    ('API_Well_ID', pa.int64()),
    ('LONG_deg', pa.float64()),
    ('LAT_deg', pa.float64()),
    ('DateTime', pa.timestamp('ns')),
    ('Days_Age_Well_days', pa.int64()),
    ('Depth_Measured_m', pa.float64()),
    ('Depth_Bit_m', pa.float64()),
    ('Layer_ID', pa.int64()),
    ('Formation_Type', pa.string()),
    ('Clay_Mineralogy_Type', pa.string()),
    ('Clay_Content_Percent_%', pa.float64()),
    ('Formation_Permeability_mD', pa.float64()),
    ('Porosity_Formation_%', pa.float64()),
    ('Fractures_Presence', pa.int64()),
    ('Phase_Operation', pa.string()),
    ('Weight_on_Bit_kg', pa.float64()),
    ('ROP_m_hr', pa.float64()),
    ('RPM_rev_min', pa.float64()),
    ('Torque_Nm', pa.float64()),
    ('Mud_Type', pa.string()),
    ('Mud_Weight_In_ppg', pa.float64()),
    ('Mud_Weight_Out_ppg', pa.float64()),
    ('Mud_Temperature_In_C', pa.float64()),
    ('Mud_Temperature_Out_C', pa.float64()),
    ('In_Rate_Flow_Mud_l_min', pa.float64()),
    ('Out_Rate_Flow_Mud_l_min', pa.float64()),
    ('Viscosity_cP', pa.float64()),
    ('Mud_pH', pa.float64()),
    ('Fluid_Loss_API_ml_30min', pa.float64()),
    ('Solid_Content_%', pa.float64()),
    ('Chloride_Content_mg_l', pa.float64()),
    ('Volume_Pit_bbl', pa.float64()),
    ('Pressure_Standpipe_psi', pa.float64()),
    ('Pressure_Annulus_psi', pa.float64()),
    ('Pressure_Reservoir_psi', pa.float64()),
    ('Overbalance_psi', pa.float64()),
    ('Reservoir_Temperature_C', pa.float64()),
    ('Completion_Type', pa.string()),
    ('Density_Perforation_shots_m', pa.int64()),
    ('Active_Damage', pa.string()),
    ('Type_Damage', pa.string())
])

def generate_well_data_chunk(well_id, long_val, lat_val, num_records, start_time, start_depth=0.0):
    layers = generate_formation_layers()
    completion_type, perf_density = select_completion_type(layers)
    
    current_layer_idx = 0
    current_depth = start_depth
    current_time = start_time
    bit_wear = 0.0
    mud_contamination = 0.0
    
    data = defaultdict(list)
    
    for i in range(num_records):
        # Update current layer
        while current_depth >= layers[current_layer_idx]['bottom'] and current_layer_idx < len(layers)-1:
            current_layer_idx += 1
            bit_wear *= 0.7  # Bit refresh when entering new formation
        
        layer = layers[current_layer_idx]
        formation = str(layer['formation'])
        days_age = (current_time - start_date).days
        
        # Calculate dynamic parameters
        effective_rop = layer['rop'] * (1 - bit_wear)
        depth_inc = effective_rop / 3600
        current_depth += depth_inc
        current_time += timedelta(seconds=1)
        
        # Get mud properties with safe access
        mud = get_mud_program(current_depth, formation)
        if 'base_weight' in mud and 'weight_grad' in mud:
            mud['weight'] = mud.get('weight', mud['base_weight'] + (current_depth * mud['weight_grad']))
        else:
            mud['weight'] = mud.get('weight', 10.0)  # Default fallback
        
        # Apply contamination effect
        mud['weight'] *= (1 + mud_contamination * 0.01)
        mud['viscosity'] *= (1 + mud_contamination * 0.02)
        
        # Calculate mud temperatures
        formation_temp = 20 + current_depth * layer['temp_grad']
        mud_temp_in, mud_temp_out = calculate_mud_temperatures(mud, formation_temp)
        
        # Calculate pressures
        pressures = calculate_pressures(current_depth, mud['weight'], layer)
        
        # Calculate chloride content
        chloride = calculate_chloride_content(current_depth, mud, formation)
        
        # Generate damage indicator
        damage_row = {
            'Clay_Content_Percent_%': layer['clay_content'],
            'Overbalance_psi': pressures['overbalance'],
            'Formation_Type': formation,
            'Mud_Weight_In_ppg': mud['weight'],
            'Formation_Permeability_mD': layer['perm']
        }
        damage_active, damage_type = generate_damage_indicator(damage_row)
        
        # Generate flow rates (must be calculated before being referenced)
        in_flow_rate = np.random.normal(100 if mud['type'] == 'Water-based' else 80, 5)
        out_flow_rate = in_flow_rate * np.random.uniform(0.95, 1.0)
        
        # Create measurements dictionary
        measurements = {
            'Record_ID': i,
            'API_Well_ID': well_id,
            'LONG_deg': float(long_val + np.random.normal(0, 0.0001)),
            'LAT_deg': float(lat_val + np.random.normal(0, 0.0001)),
            'DateTime': current_time,
            'Days_Age_Well_days': int(days_age),
            'Depth_Measured_m': float(current_depth),
            'Depth_Bit_m': float(max(0, current_depth - np.random.uniform(0, 5))),
            'Layer_ID': int(current_layer_idx),
            'Formation_Type': formation,
            'Clay_Mineralogy_Type': layer['clay_type'],
            'Clay_Content_Percent_%': float(layer['clay_content']),
            'Formation_Permeability_mD': float(layer['perm']),
            'Porosity_Formation_%': float(layer['porosity']),
            'Fractures_Presence': int(np.random.binomial(1, layer['fracture_prob'])),
            'Phase_Operation': 'Drilling' if current_depth < 1000 else 'Completion',
            'Weight_on_Bit_kg': float(layer['wob'] if current_depth < 1000 else 0),
            'ROP_m_hr': float(effective_rop if current_depth < 1000 else 0),
            'RPM_rev_min': float(np.random.normal(120, 10) if current_depth < 1000 else np.random.normal(50, 5)),
            'Torque_Nm': float(layer['wob']/10 * np.random.uniform(0.8, 1.2)),
            'Mud_Type': mud['type'],
            'Mud_Weight_In_ppg': float(mud['weight']),
            'Mud_Weight_Out_ppg': float(mud['weight'] * np.random.uniform(0.995, 1.005)),
            'Mud_Temperature_In_C': float(mud_temp_in),
            'Mud_Temperature_Out_C': float(mud_temp_out),
            'In_Rate_Flow_Mud_l_min': float(in_flow_rate),
            'Out_Rate_Flow_Mud_l_min': float(out_flow_rate),
            'Viscosity_cP': float(mud['viscosity'] * np.random.uniform(0.9, 1.1)),
            'Mud_pH': float(mud['ph']),
            'Fluid_Loss_API_ml_30min': float(np.random.uniform(0.5, 2.0) if mud['type'] == 'Water-based' else np.random.uniform(0.1, 0.5)),
            'Solid_Content_%': float(np.random.uniform(5, 15)),
            'Chloride_Content_mg_l': float(chloride),
            'Volume_Pit_bbl': float(np.random.uniform(400, 600)),
            'Pressure_Standpipe_psi': float(pressures['standpipe']),
            'Pressure_Annulus_psi': float(pressures['annulus']),
            'Pressure_Reservoir_psi': float(pressures['reservoir']),
            'Overbalance_psi': float(pressures['overbalance']),
            'Reservoir_Temperature_C': float(formation_temp),
            'Completion_Type': completion_type,
            'Density_Perforation_shots_m': int(perf_density),
            'Active_Damage': damage_active,
            'Type_Damage': damage_type,
            'Bit_Wear_%': float(bit_wear * 100),
            'Mud_Contamination_%': float(mud_contamination * 100)
        }
        
        # Update equipment state
        bit_wear = min(1.0, bit_wear + np.random.uniform(0.0001, 0.0005))
        mud_contamination = min(1.0, mud_contamination + np.random.uniform(0.0005, 0.001))
        
        # Store data
        for key, value in measurements.items():
            data[key].append(value)
    
    return pd.DataFrame(data), current_time, current_depth

# Validate all mud programs have required fields
for i, program in enumerate(mud_programs):
    mud = program['mud']
    required_fields = ['type', 'base_weight', 'weight_grad', 'viscosity', 'ph']
    
    for field in required_fields:
        if field not in mud:
            raise ValueError(f"Missing required field '{field}' in mud_programs[{i}]")
    
    # Set default chloride parameters if missing
    if 'chloride_base' not in mud:
        mud['chloride_base'] = 300 if mud['type'] == 'Water-based' else 50
    if 'chloride_grad' not in mud:
        mud['chloride_grad'] = 0.3 if mud['type'] == 'Water-based' else 0.1

# Main execution
for well_id, long_val, lat_val in wells_info:
    file_path = os.path.join(output_dir, f'well_{well_id}.parquet')
    writer = None
    remaining_records = records_per_well
    chunk_start_time = start_date
    chunk_start_depth = 0.0
    
    with tqdm(total=records_per_well, 
              desc=f"Generating well {well_id}", 
              leave=False) as pbar:
        
        while remaining_records > 0:
            current_chunk = min(chunk_size, remaining_records)
            df_chunk, chunk_start_time, chunk_start_depth = generate_well_data_chunk(
                well_id, long_val, lat_val, current_chunk, chunk_start_time, chunk_start_depth
            )
            
            table = pa.Table.from_pandas(df_chunk, schema=schema)
            
            if writer is None:
                writer = pq.ParquetWriter(file_path, schema, compression='snappy')
            writer.write_table(table)
            
            remaining_records -= current_chunk
            pbar.update(current_chunk)  # Update progress bar
    
    if writer:
        writer.close()

print("Data generation completed successfully.")
