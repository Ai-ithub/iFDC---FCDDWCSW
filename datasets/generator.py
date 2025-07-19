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

records_per_well = 1 * 30 * 24 * 60 * 60  # 1 month in second
chunk_size = 1_000       # Number of records to process in each chunk

output_dir = 'well_data'
os.makedirs(output_dir, exist_ok=True)

# Well configuration
wells_info = [
    (40100050, -94.86, 32.26),
    (40131881, -94.82, 32.26),
#     (40134068, -94.78, 32.25),
#     (40181715, -94.95, 32.17),
#     (36535068, -94.18, 32.32),
#     (36500362, -94.13, 32.32),
#     (36530944, -94.15, 32.12),
#     (18332094, -94.62, 32.37),
#     (18331921, -94.6, 32.37),
#     (18387931, -94.86, 32.45)
]
start_date = datetime(2023, 1, 1)

# Enhanced formation definitions
formations = {
    'Topsoil': {
        'depth_range': (0, 2), 'density': 1.5, 'base_rop': 65, 'wob_factor': 0.5,
        'perm_range': (1000, 5000), 'porosity': 40, 'clay_content_range': (5, 10),
        'fracture_grad': 0.6, 'temp_grad': 0.01, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.01, 'color': 'brown', 'min_mud_weight': 8.5,
        'max_mud_weight': 10.0, 'pressure_grad': 0.433
    },
    'Sandstone': {
        'density': 2.3, 'base_rop': 20, 'wob_factor': 1.0,
        'perm_range': (10, 1000), 'porosity': 22, 'clay_content_range': (10, 15),
        'fracture_grad': 0.7, 'temp_grad': 0.026, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.07, 'pressure_mult': 1.0, 'reservoir': True,
        'pressure_grad': 0.433, 'swelling_potential': 'low', 'reactivity': 'low',
        'min_mud_weight': 8.5, 'max_mud_weight': 12.5, 'preferred_mud_type': 'Water-based',
        'additives': ['Lubricants'], 'min_mud_weight': 8.5, 'max_mud_weight': 12.5,
        'reservoir': False, 'seal': False
    },
    'Shale': {
        'density': 2.4, 'base_rop': 12, 'wob_factor': 1.2,
        'perm_range': (0.001, 0.1), 'porosity': 10, 'clay_content_range': (25, 35),
        'fracture_grad': 0.75, 'temp_grad': 0.025, 'clay_type': 'Illite',
        'fracture_prob': 0.15, 'pressure_mult': 0.85, 'pressure_grad': 0.465,  # psi/ft
        'swelling_potential': 'high', 'reactivity': 'high', 'min_mud_weight': 9.5,
        'max_mud_weight': 14.0, 'preferred_mud_type': 'Inhibitive Water-based',
        'additives': ['KCl', 'PHPA'], 'min_mud_weight': 9.5, 'max_mud_weight': 14.0
    },
    'Weathered_Rock': {
        'depth_range': (2, 10), 'density': 1.8, 'base_rop': 45, 'wob_factor': 0.85,
        'perm_range': (500, 2000), 'porosity': 30, 'clay_content_range': (10, 15),
        'fracture_grad': 0.65, 'temp_grad': 0.015, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.03, 'color': 'tan', 'min_mud_weight': 8.5,
        'max_mud_weight': 11.0, 'pressure_grad': 0.433
    },
    'Limestone': {
        'density': 2.7, 'base_rop': 15, 'wob_factor': 1.5,
        'perm_range': (1, 100), 'porosity': 15, 'clay_content_range': (5, 10),
        'fracture_grad': 0.8, 'temp_grad': 0.028, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.1, 'pressure_mult': 1.1,'reservoir': True,
        'seal': False, 'color': 'light gray', 'min_mud_weight': 9.0,
        'max_mud_weight': 13.0, 'pressure_grad': 0.445,
        'reservoir': False, 'seal': False
    },
    'Shale_Seal': {  # Dedicated shale cap rock variant
        'density': 2.4, 'base_rop': 8, 'wob_factor': 1.5, 'perm_range': (0.001, 0.01),
        'porosity': 5, 'clay_content_range': (30, 40), 'fracture_grad': 0.78,
        'temp_grad': 0.025, 'clay_type': 'Illite/Smectite', 'fracture_prob': 0.02,
        'min_mud_weight': 10.0, 'max_mud_weight': 14.0, 'pressure_grad': 0.470,
        'reservoir': False,'seal': True,
    },
    'Salt': { # Cap rock
        'density': 2.2, 'base_rop': 8, 'wob_factor': 1.5,
        'perm_range': (0.0001, 0.001), 'porosity': 3, 'clay_content_range': (0, 2),
        'fracture_grad': 0.85, 'temp_grad': 0.02, 'clay_type': 'None', 'fracture_prob': 0.01,
        'seal': True, 'reservoir': False
    },
    'Tight_Carbonate': { # Cap rock
        'density': 2.6, 'base_rop': 10, 'wob_factor': 1.3,
        'perm_range': (0.01, 0.1), 'porosity': 5, 'clay_content_range': (5, 10),
        'fracture_grad': 0.8, 'temp_grad': 0.028, 'clay_type': 'Kaolinite',
        'fracture_prob': 0.1, 'pressure_mult': 1.1,'reservoir': True,
        'seal': False, 'color': 'light gray', 'min_mud_weight': 9.0,
        'max_mud_weight': 13.0, 'pressure_grad': 0.445,
        'seal': True, 'reservoir': False
    }
}

bit_types = {
    'PDC': {'wob_eff': 1.0, 'rop_eff': 1.3},  # PDC bits handle WOB well
    'Tricone': {'wob_eff': 0.8, 'rop_eff': 0.9}  # Tricone bits need lower WOB
}

mud_programs = [
    # Surface mud (0-50m)
    {
        'conditions': {'max_depth': 50},
        'mud': {
            'type': 'Water-based',
            'base_weight': 8.5,
            'weight_grad': 0.015,
            'base_temp': 20,
            'temp_grad': 0.035,
            'viscosity': 15,
            'ph': 8.0,
            'chloride_base': 300,
            'chloride_grad': 0.3,
            'temp_diff': (2, 5),  # Explicit temp difference range
            'additives': ['Bentonite']
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
            'temp_diff': (3, 6),
            'additives': ['Barite']
        }
    },
    # Shale-specific mud
    {
        'conditions': {'formations': ['Shale'], 'min_pressure': 1500},
        'mud': {
            'type': 'Inhibitive Water-based',
            'base_weight': 10.5,
            'weight_grad': 0.0025,
            'viscosity': 35,
            'ph': 9.0,
            'shale_inhibitor': 'KCl',
            'minimum_weight': 10.0,
            'maximum_weight': 14.0,
            'temp_diff': (4, 7),
            'additives': ['KCl', 'PHPA']
        }
    },
    # High-pressure reservoir mud
    {
        'conditions': {'min_depth': 600, 'min_pressure': 3000},
        'mud': {
            'type': 'Synthetic Oil-based',
            'base_weight': 12.0,
            'weight_grad': 0.003,
            'viscosity': 45,
            'ph': 7.0,
            'minimum_weight': 11.0,
            'maximum_weight': 18.0,
            'temp_diff': (5, 10),
            'additives': ['Lubricants']
        }
    },
    # Default fallback mud program
    {
        'conditions': {},  # Will match if no others do
        'mud': {
            'type': 'Water-based',
            'base_weight': 9.5,
            'weight_grad': 0.02,
            'viscosity': 30,
            'ph': 8.5,
            'temp_diff': (3, 6),  # Default fallback range
            'additives': []
        }
    }
]

def select_formation(depth, exclude_caprocks=False, exclude_reservoirs=False):
    """Select formation based on depth, optionally excluding cap rocks"""
    options = []

    for formation in formations:
        # Skip if excluded
        if exclude_caprocks and formations[formation].get('seal', False):
            continue
        if exclude_reservoirs and formations[formation].get('reservoir', False):
            continue
        options.append(formation)

    # Create a copy of the formation properties without reservoir flag
    props = formations[formation].copy()
    props['reservoir'] = False  # Force disable for all but main reservoir
    options.append((formation, props))

    # Surface layers (0-2m)
    if depth < 2:
        return 'Topsoil'  # Always topsoil at the very surface

    # Near-surface layers (2-50m)
    elif depth < 50:
        # Weathered rock dominates, occasional sandstone lenses
        return str(np.random.choice(
            ['Weathered_Rock', 'Sandstone', 'Topsoil'],
            p=[0.85, 0.12, 0.03]  # Rare to find topsoil below 2m
        ))

    # Shallow subsurface (50-300m)
    elif depth < 300:
        # Mixed sedimentary layers, more sandstone in this range
        return str(np.random.choice(
            ['Sandstone', 'Shale', 'Weathered_Rock', 'Limestone'],
            p=[0.55, 0.30, 0.10, 0.05]  # Sandstone dominates
        ))

    # Intermediate depth (300-1500m)
    elif depth < 1500:
        # More shale as we go deeper, but still significant sandstone
        # Limestone becomes more common
        depth_factor = min(1.0, (depth - 300) / 1200)  # 0 at 300m, 1 at 1500m
        return str(np.random.choice(
            ['Shale', 'Sandstone', 'Limestone'],
            p=[
                0.4 + 0.3 * depth_factor,    # Shale increases with depth
                0.5 - 0.4 * depth_factor,    # Sandstone decreases
                0.1 + 0.1 * depth_factor     # Limestone increases slightly
            ]
        ))

    # Deep formations (1500-3000m)
    elif depth < 3000:
        # Shale dominates, with some limestone reservoirs
        # Sandstone becomes rare at these depths
        return str(np.random.choice(
            ['Shale', 'Limestone', 'Sandstone'],
            p=[0.75, 0.20, 0.05]
        ))

    # Very deep (3000m+)
    else:
        # Mostly shale with occasional limestone formations
        # Sandstone is very rare at these depths
        return str(np.random.choice(
            ['Shale', 'Limestone'],
            p=[0.85, 0.15]
        ))

def select_caprock():
    """Select a caprock formation with weighted probabilities"""
    caprocks = {
        'Shale_Seal': 0.6,  # 60% chance
        'Salt': 0.3,        # 30% chance
        'Tight_Carbonate': 0.1  # 10% chance
    }
    return np.random.choice(list(caprocks.keys()), p=list(caprocks.values()))

def get_formation_thickness(formation, current_depth):
    """Get realistic thickness ranges for each formation type"""
    # Special handling for reservoirs
    if formation in ['Sandstone', 'Limestone'] and current_depth > MAX_DEPTH - 500:  # Last 500m
        return np.random.uniform(50, 200)  # 50-200m thick reservoirs

    base_thickness_ranges = {
        'Topsoil': (1, 3),
        'Weathered_Rock': (5, 15),
        'Sandstone': (10, 100),
        'Shale': (50, 300),
        'Limestone': (20, 150),
        'Shale_Seal': (30, 100),   # Caprocks
        'Salt': (50, 200),         # Salt domes can be thick
        'Tight_Carbonate': (20, 80)
    }


    # Depth scaling factor (formations get thinner with depth)
    depth_factor = max(0.5, 1 - (current_depth / 3000))

    min_thick, max_thick = base_thickness_ranges.get(formation, (10, 50))
    return np.random.uniform(min_thick*depth_factor, max_thick*depth_factor)

def generate_formation_layers(total_depth, has_horizontal=False, vertical_ratio=0.75, tangent_ratio=0.15):

    layers = []
    current_depth = 0

    # Always create surface layers
    layers.extend([
        create_layer(0, 2, 'Topsoil', is_horizontal=False),
        create_layer(2, 10, 'Weathered_Rock', is_horizontal=False)
    ])
    current_depth = 10

    if has_horizontal:
        # Vertical section only goes down to the start of tangent section
        vertical_depth = total_depth * vertical_ratio

        # Drill through regular formations until we reach the tangent section start
        while current_depth < vertical_depth:
            formation = select_formation(current_depth, exclude_reservoirs=True)
            thickness = get_formation_thickness(formation, current_depth)
            if current_depth + thickness > vertical_depth:
                thickness = vertical_depth - current_depth
            if thickness > 0:
                layer = create_layer(current_depth, current_depth + thickness, formation, False)
                layers.append(layer)
                current_depth += thickness

        # Calculate tangent section parameters
        tangent_section_length = total_depth * tangent_ratio
        tangent_end = current_depth + tangent_section_length

        # Make caprock take the entire tangent section
        caprock_layer = create_layer(
            current_depth,
            tangent_end,
            select_caprock(),
            is_horizontal=False
        )
        caprock_layer.update({
            'seal': True,
            'reservoir': False,
            'layer_type': 'CapRock'
        })
        layers.append(caprock_layer)
        current_depth = tangent_end

        # Immediately transition to reservoir after caprock
        reservoir_thickness = total_depth - current_depth
        reservoir_formation = np.random.choice(['Sandstone', 'Limestone'], p=[0.7, 0.3])

        reservoir_layer = create_layer(
            current_depth,
            current_depth + reservoir_thickness,
            reservoir_formation,
            is_horizontal=True
        )
        reservoir_layer.update({
            'reservoir': True,
            'is_main_reservoir': True,
            'layer_type': 'Reservoir',
            'seal': False
        })
        layers.append(reservoir_layer)

    else:
        # Vertical-only well configuration
        reservoir_start_depth = total_depth * 0.8

        while current_depth < reservoir_start_depth:
            formation = select_formation(current_depth)
            thickness = get_formation_thickness(formation, current_depth)
            if current_depth + thickness > reservoir_start_depth:
                thickness = reservoir_start_depth - current_depth
            if thickness > 0:
                layer = create_layer(current_depth, current_depth + thickness, formation, False)
                layers.append(layer)
                current_depth += thickness

        # Add caprock (50-150m thick)
        caprock_thickness = np.random.uniform(50, 150)
        caprock_layer = create_layer(
            current_depth,
            current_depth + caprock_thickness,
            select_caprock(),
            is_horizontal=False
        )
        caprock_layer.update({
            'seal': True,
            'reservoir': False,
            'layer_type': 'CapRock'
        })
        layers.append(caprock_layer)
        current_depth += caprock_thickness

        # Add reservoir (remaining depth)
        reservoir_formation = np.random.choice(['Sandstone', 'Limestone'], p=[0.7, 0.3])
        reservoir_thickness = total_depth - current_depth

        reservoir_layer = create_layer(
            current_depth,
            current_depth + reservoir_thickness,
            reservoir_formation,
            is_horizontal=False
        )
        reservoir_layer.update({
            'reservoir': True,
            'is_main_reservoir': True,
            'layer_type': 'Reservoir',
            'seal': False
        })
        layers.append(reservoir_layer)

    return layers

def create_layer(top, bottom, formation, is_horizontal=False):
    """Enhanced layer creation with built-in damage potential"""
    if isinstance(formation, (np.str_, np.bytes_, np.object_)):
        formation = str(formation)

    if formation not in formations:
        raise ValueError(f"Formation '{formation}' not defined")

    props = formations[formation].copy()
    depth = (top + bottom) / 2

    # Calculate WOB and ROP first (needs original properties)
    wob = calculate_wob(depth, props, is_horizontal=is_horizontal)
    rop = calculate_rop(wob, depth, props, is_horizontal=is_horizontal)

    # Base damage probability (0-20%) based on formation
    base_damage_prob = {
        'Shale': 0.25,
        'Shale_Seal': 0.27,
        'Sandstone': 0.20,
        'Limestone': 0.17,
        'Topsoil': 0.05,
        'Weathered_Rock': 0.08,
        'Salt': 0.15,
        'Tight_Carbonate': 0.16
    }.get(formation, 0.10)

    # Damage types this formation is prone to
    formation_damages = {
        'Shale': ['Clay Swelling', 'Borehole Collapse', 'Shale Instability'],
        'Shale_Seal': ['Caprock Integrity Loss', 'Microannulus Formation'],
        'Sandstone': ['Fines Migration', 'Sand Production'],
        'Limestone': ['Acid Solubility Damage', 'Fracture Plugging'],
        'Topsoil': ['Surface Contamination'],
        'Weathered_Rock': ['Unconsolidated Formation'],
        'Salt': ['Salt Creep', 'Brine Influx'],
        'Tight_Carbonate': ['Fracture Face Damage']
    }.get(formation, ['Formation Damage'])

    # Clay types
    clay_types = {
        'Shale': np.random.choice(['Illite', 'Smectite', 'Mixed-Layer'], p=[0.5, 0.3, 0.2]),
        'Shale_Seal': 'Illite/Smectite',
        'Sandstone': 'Kaolinite',
        'Limestone': 'Kaolinite/Chlorite',
        'Topsoil': 'Montmorillonite',
        'Weathered_Rock': 'Kaolinite/Montmorillonite'
    }

    layer = {
        'top': float(top),
        'bottom': float(bottom),
        'formation': str(formation),
        'density': float(props['density']),
        'rop': float(rop),
        'wob': float(wob),
        'porosity': float(props['porosity']),
        'perm': float(np.random.uniform(*props['perm_range'])),
        'clay_content': float(np.random.uniform(*props['clay_content_range'])),
        'clay_type': clay_types.get(formation, 'Kaolinite'),
        'fracture_prob': float(props['fracture_prob']),
        'temp_grad': float(props['temp_grad']),
        'is_horizontal': is_horizontal,
        'reservoir': False,
        'seal': False,
        'layer_type': 'None',
        # New damage properties
        'base_damage_prob': base_damage_prob,
        'potential_damages': formation_damages,
        'max_damage_severity': np.random.uniform(0.1, 1.0),
        'damage_triggers': {
            'wob': np.random.normal(props.get('wob_factor', 1.0), 0.2),
            'rop': np.random.normal(props.get('base_rop', 15), 5),
            'temp': 20 + (depth * props.get('temp_grad', 0.025))
        }
    }

    return layer

def calculate_wob(depth, formation_props, bit_type='PDC', is_horizontal=False):
    bit = bit_types[bit_type]

    # Get the formation name from the properties if available (some formations have it)
    formation_name = formation_props.get('formation', None)

    # Caprocks often require higher WOB
    if formation_props.get('seal', False):
        wob_range = {
            'Shale_Seal': (8000, 20000),
            'Salt': (10000, 22000),
            'Tight_Carbonate': (12000, 25000)
        }.get(formation_name, (10000, 20000))
    # Reservoir rocks often drill best with moderate WOB
    elif formation_props.get('reservoir', False):
        wob_range = (6000, 15000)
    else:
        wob_range = {
            'Topsoil': (500, 2000),
            'Weathered_Rock': (1000, 4000),
            'Sandstone': (2000, 8000),
            'Shale': (3000, 12000),
            'Limestone': (5000, 18000)
        }.get(formation_name, (2000, 10000))

    # Horizontal section adjustment (reduced effective WOB)
    if is_horizontal:
        wob_range = (wob_range[0]*0.7, wob_range[1]*0.8)

    # Depth adjustment
    depth_factor = min(1.0, depth / 3000)
    wob_base = wob_range[0] + (wob_range[1] - wob_range[0]) * depth_factor

    # Apply formation-specific wob_factor
    wob = wob_base * formation_props['wob_factor'] * bit['wob_eff']

    # Add random variation
    wob *= np.random.uniform(0.9, 1.1)

    return min(wob, 25000)  # Absolute limit


def calculate_rop(wob, depth, formation, bit_type='PDC', is_horizontal=False):
    bit = bit_types[bit_type]
    effective_wob = min(wob, 25000)

    # Base formation factors
    formation_factor = (2.5 / formation['density']) * formation['base_rop']

    # Depth factor - ROP decreases with depth
    depth_factor = max(0.3, 1 - (depth / 6000))

    # WOB factor - diminishing returns at higher WOB
    wob_factor = (effective_wob / 10000) ** 0.7

    # Horizontal section penalty (30-50% reduction)
    if is_horizontal:
        horizontal_factor = 0.6
        # Additional reduction in caprocks
        if formation.get('seal', False):
            horizontal_factor *= 0.8
    else:
        horizontal_factor = 1.0

    # Reservoir rock typically drills faster
    if formation.get('reservoir', False):
        reservoir_factor = 1.2
    else:
        reservoir_factor = 1.0

    rop = (formation_factor * depth_factor * wob_factor *
           bit['rop_eff'] * horizontal_factor * reservoir_factor)

    return max(2, min(rop, 100))  # Keep within realistic bounds

def select_completion_type(layers, drilling_phase=True):
    if drilling_phase:
        return 'Open Hole', 10  # Always Open Hole during drilling
    else:
        # Your original completion logic for casing phase
        reservoir_depths = [layer['top'] for layer in layers if layer.get('reservoir', False)]
        has_shale = any(layer['formation'] == 'Shale' for layer in layers)

        if not reservoir_depths:
            return 'Open Hole', 10

        reservoir_depth = min(reservoir_depths)
        if has_shale and reservoir_depth > 500:
            return 'Liner', 20
        elif has_shale:
            return 'Cased', 30
        elif reservoir_depth > 800:
            return 'Liner', 20
        else:
            return 'Open Hole', 10

def get_mud_program(depth, formation_props, is_horizontal=False):
    """Get mud program considering formation properties, pressures, and drilling direction"""
    formation = formation_props['formation']

    # Special handling for caprocks (higher weights to prevent fluid influx)
    if formation_props.get('seal', False):  # Caprock
        return {
            'type': 'Inhibitive Water-based' if 'Shale' in formation else 'Synthetic Oil-based',
            'weight': max(14.0, formation_props.get('min_mud_weight', 14.0)),
            'viscosity': 45,
            'ph': 9.0 if 'Shale' in formation else 7.5,
            'additives': ['Bridging agents', 'Lost circulation materials']
        }

    # Special reservoir drilling fluid
    if formation_props.get('reservoir', False) and depth > 1500:
        return {
            'type': 'Synthetic Oil-based',
            'weight': 11.5 + (depth/1000)*0.5,  # Gradual weight increase
            'viscosity': 40,
            'ph': 7.0,
            'additives': ['Lubricants', 'Filter cake reducers']
        }

    # Horizontal section adjustments
    if is_horizontal:
        base_program = get_mud_program(depth, formation_props)  # Get normal program first
        # Increase viscosity for better cuttings transport
        base_program['viscosity'] *= 1.3
        # Add lubricants for reduced friction
        if 'additives' not in base_program:
            base_program['additives'] = []
        base_program['additives'].append('Lubricants')
        return base_program

    # Calculate formation pressure with default gradient if not specified
    pressure_grad = formation_props.get('pressure_grad',
                  0.465 if depth < 3000 else np.random.uniform(0.45, 0.8))
    formation_pressure = depth * pressure_grad * 3.281  # Convert to psi

    # Rest of the function remains the same...
    # Calculate required mud weight to balance formation pressure
    required_weight = (formation_pressure / (depth * 0.052 * 3.281)) if depth > 0 else 8.5
    required_weight = max(formation_props.get('min_mud_weight', 8.5),
                         min(required_weight, formation_props.get('max_mud_weight', 18.0)))

    # Select mud program based on multiple factors
    for program in reversed(mud_programs):
        conditions = program['conditions']
        match = True

        # Depth conditions
        if 'min_depth' in conditions and depth < conditions['min_depth']:
            match = False
        if 'max_depth' in conditions and depth >= conditions['max_depth']:
            match = False

        # Formation conditions
        if 'formations' in conditions and formation not in conditions['formations']:
            match = False

        # Pressure conditions
        if 'min_pressure' in conditions and formation_pressure < conditions['min_pressure']:
            match = False
        if 'max_pressure' in conditions and formation_pressure >= conditions['max_pressure']:
            match = False

        if match:
            mud = program['mud'].copy()
            # Adjust weight based on formation requirements
            mud['weight'] = max(mud['base_weight'] + (depth * mud['weight_grad']),
                              required_weight)
            # Add formation-specific additives
            mud['additives'] = formation_props.get('additives', [])
            return mud

    viscosity = np.random.uniform(25, 50) if depth > 1000 else np.random.uniform(15, 30)
    # Fallback with safety margins
    return {
        'type': formation_props.get('preferred_mud_type', 'Water-based'),
        'weight': required_weight * 1.05,  # 5% overbalance
        'viscosity': viscosity,
        'ph': 9.0 if formation == 'Shale' else 8.5,
        'temp_diff': (3, 5),
        'additives': formation_props.get('additives', [])
    }

def validate_mud_weight(mud_weight, formation_props):
    """Ensure mud weight stays within safe operating limits"""
    min_weight = formation_props.get('min_mud_weight', 8.5)
    max_weight = formation_props.get('max_mud_weight', 18.0)
    fracture_grad = formation_props.get('fracture_grad', 0.8)

    # Convert fracture gradient to equivalent mud weight
    fracture_mw = fracture_grad / 0.052

    # Apply safety margins (90% of fracture gradient)
    safe_max = min(max_weight, fracture_mw * 0.9)

    # Ensure we stay within formation-specific limits
    return max(min_weight, min(mud_weight, safe_max))

def calculate_equipment_efficiency(bit_wear, mud_contamination, hours_operating):
    """Calculate overall equipment efficiency factor"""
    bit_efficiency = 1 - (bit_wear * 0.7)  # Bit contributes 70% to efficiency loss
    mud_efficiency = 1 - (mud_contamination * 0.3)  # Mud contributes 30%
    time_efficiency = max(0.6, 1 - (hours_operating / 500))  # Time-based wear

    return bit_efficiency * mud_efficiency * time_efficiency

def calculate_pressures(depth, mud_weight, formation_props):
    # Hydrostatic pressure
    hydrostatic = mud_weight * depth * 0.052 * 3.281  # psi

    # Formation pressure - use default gradient of 0.465 psi/ft if not specified
    pressure_grad = formation_props.get('pressure_grad', 0.465)
    formation_pressure = depth * pressure_grad * 3.281

    # Adjust for abnormal pressure in some formations
    if formation_props.get('abnormal_pressure', False):
        formation_pressure *= np.random.uniform(1.1, 1.3)

    # Standpipe pressure (hydrostatic + friction + safety)
    standpipe = hydrostatic * 1.05 + np.random.uniform(100, 500)

    # Annulus pressure (hydrostatic - friction)
    annulus = hydrostatic * 0.97

    # Overbalance calculation
    overbalance = hydrostatic - formation_pressure

    return {
        'standpipe': standpipe,
        'annulus': annulus,
        'reservoir': formation_pressure,
        'overbalance': overbalance,
        'fracture': depth * formation_props.get('fracture_grad', 0.8) * 3.281
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
    """Calculate realistic mud temperatures with safe defaults"""
    # Get temp_diff with fallback to default range
    temp_diff_range = mud_program.get('temp_diff', (3, 6))  # Default range if not specified

    # Mud temperature in is slightly cooler than formation
    mud_temp_in = formation_temp - np.random.uniform(1, 3)

    # Calculate temperature difference
    try:
        temp_diff = np.random.uniform(*temp_diff_range)
    except:
        temp_diff = np.random.uniform(3, 6)  # Hardcoded fallback

    mud_temp_out = mud_temp_in + temp_diff

    # Ensure physical consistency
    mud_temp_out = max(mud_temp_out, mud_temp_in + 0.5)  # Always some heating
    mud_temp_out = min(mud_temp_out, formation_temp + 15)  # Don't exceed reasonable limits

    return mud_temp_in, mud_temp_out

def generate_damage_indicator(row, layer):

    # Calculate base probability using layer properties
    current_wob = row.get('Weight_on_Bit_kg', 0)
    current_rop = row.get('ROP_m_hr', 0)
    current_temp = row.get('Reservoir_Temperature_C', 0)
    
    wob_factor = min(1.5, current_wob / layer['damage_triggers']['wob']) if layer['damage_triggers']['wob'] > 0 else 1.0
    rop_factor = min(2.0, current_rop / layer['damage_triggers']['rop']) if layer['damage_triggers']['rop'] > 0 else 1.0
    temp_factor = min(1.8, current_temp / layer['damage_triggers']['temp']) if layer['damage_triggers']['temp'] > 0 else 1.0
    
    # Increase base damage probability to 30% (from 20%)
    damage_prob = min(0.3, layer['base_damage_prob'] * wob_factor * rop_factor * temp_factor)
    
    if np.random.random() > damage_prob:
        return "No", "None", None, "No"  # Added "No" for controllable
    
    # Damage types with adjusted base severities for 50-50 distribution
    formation = row.get('Formation_Type', 'Unknown')
    clay_content = row.get('Clay_Content_Percent_%', 0)
    clay_type = row.get('Clay_Mineralogy_Type', '')
    
    damage_types = {
        'Shale': [
            ("Clay Swelling", 0.50, clay_content > 35 and 'Montmorillonite' in clay_type, "Yes"),
            ("Shale Instability", 0.45, clay_content > 30, "Yes"),
            ("Borehole Collapse", 0.40, row.get('Overbalance_psi', 0) > 500, "No")
        ],
        'Sandstone': [
            ("Fines Migration", 0.55, row.get('Formation_Permeability_mD', 0) < 50, "Yes"),
            ("Sand Production", 0.45, row.get('ROP_m_hr', 0) > 25, "No")
        ],
        'Limestone': [
            ("Acid Solubility", 0.60, row.get('Mud_pH', 0) < 7.0, "Yes"),
            ("Fracture Plugging", 0.50, row.get('Fluid_Loss_API_ml_30min', 0) > 1.0, "No")
        ],
        'Salt': [
            ("Salt Creep", 0.70, row.get('Reservoir_Temperature_C', 0) > 90, "No"),
            ("Brine Influx", 0.60, row.get('Chloride_Content_mg_l', 0) > 1000, "No")
        ]
    }
    
    # Get applicable damage types for this formation
    candidates = []
    for damage, base_severity, condition, controllable in damage_types.get(formation, []):
        if condition:
            # Adjust severity to be more evenly distributed around 0.5
            severity = min(1.0, base_severity * np.random.uniform(0.7, 1.3))
            candidates.append((damage, severity, controllable))
    
    # If no specific damage matches, use generic formation damage with 50% chance for each type
    if not candidates:
        severity = np.random.uniform(0.4, 0.9)  # Wider range centered around 0.65
        controllable = "Yes" if np.random.random() > 0.5 else "No"
        return "Yes", "Formation Damage", round(severity, 2), controllable
    
    # Select a damage type weighted by severity
    damages, severities, controllables = zip(*candidates)
    chosen_idx = np.random.choice(len(candidates), p=np.array(severities)/sum(severities))
    damage, severity, controllable = candidates[chosen_idx]
    
    return "Yes", damage, round(severity, 2), controllable

# PyArrow schema definition
schema = pa.schema([
    ('Record_ID', pa.int64()),
    ('API_Well_ID', pa.int64()),
    ('LONG_deg', pa.float64()),
    ('LAT_deg', pa.float64()),
    ('DateTime', pa.timestamp('ns')),
    ('Days_Age_Well_days', pa.int64()),
    ('Drilling_Direction', pa.string()),
    ('Depth_Measured_m', pa.float64()),
    ('Depth_Bit_m', pa.float64()),
    ('Layer_ID', pa.int64()),
    ('Formation_Type', pa.string()),
    ('Layer_Type', pa.string()),  # Can be 'CapRock', 'Reservoir', or None
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
    ('Type_Damage', pa.string()),
    ('Damage_Severity', pa.float64()),
    ('Controllable', pa.string()), 
])

def generate_well_data_chunk(well_id, long_val, lat_val, num_records, start_time,
                           start_depth=0.0, max_depth=3000, vertical_ratio=0.75,
                           tangent_ratio=0.15, has_horizontal=False):

    if has_horizontal:
        vertical_depth = max_depth * vertical_ratio
        tangent_depth = vertical_depth + (max_depth * tangent_ratio)
    else:
        # For vertical wells, everything is vertical
        vertical_depth = max_depth
        tangent_depth = max_depth  # Never reached

    horizontal_depth = max_depth

    # Horizontal section tracking
    in_horizontal = False
    reservoir_formation = None
    reservoir_layer = None
    horizontal_start_depth = None

    # Return empty dataframe if we've already reached max depth
    if start_depth >= max_depth:
        return pd.DataFrame(), start_time, start_depth

    layers = generate_formation_layers(max_depth, has_horizontal=has_horizontal,
                                       vertical_ratio=vertical_ratio, tangent_ratio=tangent_ratio)
    completion_type, perf_density = select_completion_type(layers)

    current_layer_idx = 0
    current_depth = start_depth
    current_time = start_time
    bit_wear = 0.0
    mud_contamination = 0.0
    equipment_efficiency = np.random.uniform(0.95, 1.0)  # 95-100% for new

    data = defaultdict(list)

    for i in range(num_records):
        # Check if we would exceed max depth with next increment
        if current_depth >= max_depth:
            break

        # Determine drilling direction based on current depth and well configuration
        if not any(layer.get('is_horizontal', False) for layer in layers):
            # Pure vertical well
            drilling_direction = "Vertical"
        else:
            # Horizontal well configuration
            if current_depth < vertical_depth:
                drilling_direction = "Vertical"
            elif current_depth < tangent_depth:
                drilling_direction = "Tangent"
            else:
                drilling_direction = "Horizontal"
                if not in_horizontal:
                    in_horizontal = True
                    horizontal_start_depth = current_depth
                    # Find the reservoir formation
                    for layer in layers:
                        if layer.get('reservoir', False):
                            reservoir_formation = layer['formation']
                            reservoir_layer = layer
                            break
                    # Fallback to current layer if no reservoir found
                    if reservoir_formation is None:
                        reservoir_formation = layers[current_layer_idx]['formation']
                        reservoir_layer = layers[current_layer_idx]

        # Use reservoir properties in horizontal section
        if in_horizontal:
            formation = reservoir_formation
            layer = reservoir_layer
        else:
            # Update current layer for vertical/tangent sections
            while current_depth >= layers[current_layer_idx]['bottom'] and current_layer_idx < len(layers)-1:
                current_layer_idx += 1
                # Bit refresh when entering new formation
                formation_change_factor = {
                    'Shale': 0.6, 'Sandstone': 0.7, 'Limestone': 0.5,
                    'Topsoil': 0.9, 'Weathered_Rock': 0.8
                }.get(layers[current_layer_idx]['formation'], 0.7)
                bit_wear *= formation_change_factor

            layer = layers[current_layer_idx]
            formation = str(layer['formation'])

        days_age = (current_time - start_date).days

        # Adjust ROP based on drilling direction
        direction_rop_factor = {
            "Vertical": 1.0,
            "Tangent": 0.8,
            "Horizontal": 0.6
        }.get(drilling_direction, 1.0)

        # Get mud properties
        is_horizontal = (drilling_direction == "Horizontal")
        mud = get_mud_program(current_depth, layer, is_horizontal)
        mud['weight'] = validate_mud_weight(mud['weight'], layer)

        # ROP calculation with direction factor
        base_rop = layer['rop'] * direction_rop_factor
        wob_factor = (layer['wob'] / 10000) ** 0.7
        min_mud_weight = layer.get('min_mud_weight', 8.5)
        pressure_factor = min(1.5, 1 + (mud['weight'] - min_mud_weight) / 2)
        effective_rop = (base_rop * wob_factor * pressure_factor *
                        equipment_efficiency * (1 - bit_wear))

        depth_inc = effective_rop / 3600  # ROP in m/hr to m/sec

        # Adjust depth increment if it would exceed max depth
        if current_depth + depth_inc > max_depth:
            depth_inc = max_depth - current_depth

        current_depth += depth_inc
        current_time += timedelta(seconds=1)

        # If we've reached max depth, set final values
        if current_depth >= max_depth:
            current_depth = max_depth
            effective_rop = 0
            layer['wob'] = 5000

        # Temperature calculations - use vertical depth in horizontal section
        if in_horizontal:
            formation_temp = 20 + vertical_depth * layer['temp_grad']
        else:
            formation_temp = 20 + current_depth * layer['temp_grad']

        mud_temp_in, mud_temp_out = calculate_mud_temperatures(mud, formation_temp)

        # Adjust temperatures for formation thermal properties
        if formation == 'Shale':
            mud_temp_out += np.random.uniform(2, 4)
        elif formation == 'Limestone':
            mud_temp_out += np.random.uniform(1, 3)

        # Pressure calculations - stabilize in horizontal section
        if in_horizontal:
            # Get pressure gradient from layer properties with a default fallback
            pressure_grad = layer.get('pressure_grad', 0.465)  # Default 0.465 psi/ft if not specified
            pressures = {
                'standpipe': np.random.normal(2500, 100),
                'annulus': np.random.normal(2300, 100),
                'reservoir': vertical_depth * pressure_grad * 3.281,
                'overbalance': np.random.normal(200, 50),
                'fracture': vertical_depth * layer.get('fracture_grad', 0.8) * 3.281
            }
        else:
            pressures = calculate_pressures(current_depth, mud['weight'], layer)

        # Flow rates - slightly lower in horizontal section
        base_flow = 100 if mud['type'] == 'Water-based' else 80
        if in_horizontal:
            base_flow *= 0.9  # Reduce flow slightly in horizontal

        formation_flow_factor = {
            'Shale': 1.1, 'Sandstone': 1.0, 'Limestone': 0.9,
            'Topsoil': 1.0, 'Weathered_Rock': 1.05
        }.get(formation, 1.0)

        in_flow_rate = np.random.normal(base_flow * formation_flow_factor, 5)
        out_flow_rate = in_flow_rate * np.random.uniform(0.93, 0.98)

        # Damage indicator
        damage_active, damage_type, damage_severity, controllable = generate_damage_indicator({
            'Clay_Content_Percent_%': layer['clay_content'],
            'Overbalance_psi': pressures['overbalance'],
            'Formation_Type': formation,
            'Mud_Weight_In_ppg': mud['weight'],
            'Formation_Permeability_mD': layer['perm'],
            'Mud_Type': mud['type'],
            'Mud_pH': mud['ph'],
            'Weight_on_Bit_kg': layer['wob'],
            'ROP_m_hr': effective_rop,
            'Reservoir_Temperature_C': formation_temp,
            'Depth_Measured_m': current_depth,
            'Clay_Mineralogy_Type': layer['clay_type'],
            'Fluid_Loss_API_ml_30min': np.random.uniform(0.5, 2.0) if mud['type'] == 'Water-based' else np.random.uniform(0.1, 0.5),
            'Chloride_Content_mg_l': calculate_chloride_content(current_depth, mud, formation),
            'Solid_Content_%': np.random.uniform(5, 15),
            'Completion_Type': completion_type,
            'Porosity_Formation_%': layer['porosity']
        }, layer)

        # Torque calculation
        base_torque = layer['wob'] / 10
        formation_torque_factor = {
            'Shale': 0.9, 'Sandstone': 1.1, 'Limestone': 1.3,
            'Topsoil': 0.7, 'Weathered_Rock': 0.8
        }.get(formation, 1.0)
        torque = base_torque * formation_torque_factor * equipment_efficiency

        # Create measurements dictionary
        measurements = {
            'Record_ID': i,
            'API_Well_ID': well_id,
            'LONG_deg': float(long_val + np.random.normal(0, 0.0001)),
            'LAT_deg': float(lat_val + np.random.normal(0, 0.0001)),
            'DateTime': current_time,
            'Days_Age_Well_days': int(days_age),
            'Drilling_Direction': drilling_direction,
            'Depth_Measured_m': float(current_depth),
            'Depth_Bit_m': float(max(0, current_depth - np.random.uniform(0, 5))),
            'Depth_Vertical_m': float(vertical_depth if in_horizontal else current_depth),
            'Layer_ID': int(current_layer_idx),
            'Formation_Type': formation,
            'Layer_Type': layer.get('layer_type', 'None'),
            'Clay_Mineralogy_Type': layer['clay_type'],
            'Clay_Content_Percent_%': float(layer['clay_content']),
            'Formation_Permeability_mD': float(layer['perm']),
            'Porosity_Formation_%': float(layer['porosity']),
            'Fractures_Presence': int(np.random.binomial(1, layer['fracture_prob'])),
            'Phase_Operation': 'Drilling' if current_depth < max_depth else 'Completed',
            'Weight_on_Bit_kg': float(layer['wob']),
            'ROP_m_hr': float(effective_rop),
            'RPM_rev_min': float(np.random.normal(120, 10) * equipment_efficiency),
            'Torque_Nm': float(torque),
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
            'Chloride_Content_mg_l': float(calculate_chloride_content(current_depth, mud, formation)),
            'Volume_Pit_bbl': float(np.random.uniform(400, 600)),
            'Pressure_Standpipe_psi': float(pressures['standpipe']),
            'Pressure_Annulus_psi': float(pressures['annulus']),
            'Pressure_Reservoir_psi': float(pressures['reservoir']),
            'Overbalance_psi': float(pressures['overbalance']),
            'Fracture_Pressure_psi': float(pressures['fracture']),
            'Reservoir_Temperature_C': float(formation_temp),
            'Completion_Type': completion_type,
            'Density_Perforation_shots_m': int(perf_density),
            'Active_Damage': damage_active,
            'Type_Damage': damage_type,
            'Damage_Severity': damage_severity if damage_severity is not None else 0.0,
            'Controllable': controllable,
            'Bit_Wear_%': float(bit_wear * 100),
            'Mud_Contamination_%': float(mud_contamination * 100),
            'Equipment_Efficiency_%': float(equipment_efficiency * 100),
            'Mud_Additives': ', '.join(mud.get('additives', [])),
            'Inclination_deg': (
                0.0 if drilling_direction == "Vertical" else
                np.random.uniform(30, 60) if drilling_direction == "Tangent" else
                90.0  # Horizontal
            )
        }

        # Update equipment state
        bit_wear_increment = {
            'Shale': 0.0003, 'Sandstone': 0.0005, 'Limestone': 0.0008,
            'Topsoil': 0.0001, 'Weathered_Rock': 0.0002
        }.get(formation, 0.0004)

        bit_wear = min(1.0, bit_wear + np.random.uniform(
            bit_wear_increment * 0.8,
            bit_wear_increment * 1.2
        ))

        # Mud contamination
        contamination_increment = {
            'Shale': 0.0015 if 'KCl' not in mud.get('additives', []) else 0.0005,
            'Sandstone': 0.0008, 'Limestone': 0.0006,
            'Topsoil': 0.0020, 'Weathered_Rock': 0.0012
        }.get(formation, 0.0010)

        mud_contamination = min(1.0, mud_contamination + contamination_increment)

        # Equipment efficiency degradation
        equipment_efficiency = max(0.5, equipment_efficiency - np.random.uniform(0.00001, 0.00003))

        # Store data
        for key, value in measurements.items():
            data[key].append(value)

        # Break if we've reached max depth
        if current_depth >= max_depth:
            break

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


MIN_TOTAL_DEPTH = 3000  # Minimum well depth in meters
MAX_TOTAL_DEPTH = 5000  # Maximum well depth in meters

MAX_DEPTH = np.random.randint(3000, 5000)
VERTICAL_SECTION_RATIO = 0.75  # Percentage of well that's vertical
TANGENT_SECTION_RATIO = 0.15  # Percentage of well that's tangent
# Horizontal section will be the remainder (1 - VERTICAL - TANGENT)

# Main execution
for well_idx, (well_id, long_val, lat_val) in enumerate(wells_info, start=1):
    # First well is always vertical, others have 50% chance
    # horizontal_prob = 1.0 if well_idx == 1 else 0.5
    horizontal_prob = np.random.random()
    has_horizontal = horizontal_prob > 0.5
    vertical_ratio = VERTICAL_SECTION_RATIO if has_horizontal else 1.0
    tangent_ratio = TANGENT_SECTION_RATIO if has_horizontal else 0.0

    # Reset MAX_DEPTH for each well
    MAX_DEPTH = 500 # np.random.randint(MIN_TOTAL_DEPTH, MAX_TOTAL_DEPTH)

    # Generate layers with the current parameters
    layers = generate_formation_layers(
        total_depth = MAX_DEPTH,
        has_horizontal=has_horizontal,
        vertical_ratio = vertical_ratio,
        tangent_ratio = tangent_ratio
    )

    # Debugging
    print(f"\nWell {well_id} configuration:")
    print(f"  Total depth: {MAX_DEPTH}m")
    print(f"  Horizontal probability: {horizontal_prob:.2f} -- Horizontal: {has_horizontal}")
    print(f"  Vertical section: {vertical_ratio*100}%")
    print(f"  Tangent section: {tangent_ratio*100}%")
    print("Formation layers:")
    for layer in layers:
        print(f"  {layer['top']:.1f}-{layer['bottom']:.1f}m: {layer['formation']} "
            f"\t\t - porosity: {layer['porosity']:.2f}%, permeability: {layer['perm']:.2f} mD "
            f"{'(Horizontal)' if layer.get('is_horizontal', False) else ''} "
            f"{'(Reservoir)' if layer.get('reservoir', False) else ''} "
            f"{'(Caprock)' if layer.get('seal', False) else ''}")

    actual_max_depth = layers[-1]['bottom']

    file_path = os.path.join(output_dir, f'well_{well_id}.parquet')
    writer = None
    remaining_records = records_per_well
    chunk_start_time = start_date
    chunk_start_depth = 0.0
    total_wells = len(wells_info)

    # Calculate total time duration (assuming 1 record = 1 second)
    total_hours = records_per_well / (1 * 60 * 60)

    with tqdm(total=total_hours,
              desc=f"Well {well_idx}/{total_wells} (ID: {well_id})",
              unit='day',
              leave=True) as pbar:

        last_day_shown = 0
        while remaining_records > 0 and chunk_start_depth < MAX_DEPTH:

            current_chunk = min(chunk_size, remaining_records)  # Determine how many records to process

            df_chunk, chunk_start_time, chunk_start_depth = generate_well_data_chunk(
                well_id, long_val, lat_val, current_chunk, chunk_start_time,
                start_depth=chunk_start_depth,
                max_depth=MAX_DEPTH,
                vertical_ratio=vertical_ratio,
                tangent_ratio=tangent_ratio,
                has_horizontal=has_horizontal
            )

            remaining_records -= current_chunk  # Decrement remaining records
            # If we got an empty dataframe, we've reached max depth

            if df_chunk.empty:
                break

            table = pa.Table.from_pandas(df_chunk, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(file_path, schema, compression='snappy')
            writer.write_table(table)

            remaining_records -= current_chunk

            # Update progress based on days elapsed
            current_hours = (chunk_start_time - start_date).total_seconds() / (1 * 60 * 60)
            days_to_update = int(current_hours) - last_day_shown
            if days_to_update > 0:
                pbar.update(days_to_update)
                last_day_shown = int(current_hours)

    if writer:
        writer.close()

print(f"Data generated successfully... saved at {output_dir} directory")