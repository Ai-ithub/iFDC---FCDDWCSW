import pytest
from generator import calc_damage, determine_damage_type, phase_operation

# -----------------------------
# Test for calc_damage
# -----------------------------
def test_calc_damage_base_only():
    score = calc_damage(prob_base=0.1, clay=10, loss_fluid=0.5, fractures=0, api_loss=0.4)
    assert score == 0.1  

def test_calc_damage_with_all_risks():
    score = calc_damage(prob_base=0.1, clay=35, loss_fluid=1.5, fractures=1, api_loss=0.8)
    expected = 0.1 + 0.2 + 0.3 + 0.1 + 0.2
    assert round(score, 3) == round(expected, 3)

def test_calc_damage_capped_at_0_95():
    score = calc_damage(prob_base=0.8, clay=50, loss_fluid=2.0, fractures=1, api_loss=1.0)
    assert score == 0.95  

# -----------------------------
# Test for determine_damage_type
# -----------------------------
def test_determine_damage_clay_iron_control():
    row = {
        'Clay_Content_Percent': 40,
        'Clay_Mineralogy_Type': "Montmorillonite",
        'Formation_Type': "Shale",
        'Fluid_Loss_API': 0.5,
        'Mud_Type': "Oil-based",
        'Chloride_Content': 100,
        'Solid_Content': 5,
        'Completion_Type': "Cased",
        'Mud_pH': 7.0,
        'Formation_Permeability': 100,
        'Reservoir_Temperature': 75,
        'Overbalance': 80,
        'Viscosity': 10,
        'Mud_Weight_In': 9.0
    }
    assert determine_damage_type(row) == "Clay & Iron Control"

def test_determine_damage_near_wellbore_emulsions():
    row = {
        'Clay_Content_Percent': 20,
        'Clay_Mineralogy_Type': "Illite",
        'Formation_Type': "Sandstone",
        'Fluid_Loss_API': 0.4,
        'Mud_Type': "Water-based",
        'Chloride_Content': 200,
        'Solid_Content': 5,
        'Completion_Type': "Open Hole",
        'Mud_pH': 6.8,
        'Formation_Permeability': 100,
        'Reservoir_Temperature': 75,
        'Overbalance': 80,
        'Viscosity': 10,
        'Mud_Weight_In': 9.0
    }
    assert determine_damage_type(row) == "Near-Wellbore Emulsions"

def test_determine_damage_generic():
    row = {
        'Clay_Content_Percent': 10,
        'Clay_Mineralogy_Type': "Kaolinite",
        'Formation_Type': "Sandstone",
        'Fluid_Loss_API': 0.4,
        'Mud_Type': "Oil-based",
        'Chloride_Content': 100,
        'Solid_Content': 5,
        'Completion_Type': "Cased",
        'Mud_pH': 7.0,
        'Formation_Permeability': 100,
        'Reservoir_Temperature': 75,
        'Overbalance': 50,
        'Viscosity': 10,
        'Mud_Weight_In': 9.0
    }
    assert determine_damage_type(row) == "Generic Damage"

# -----------------------------
# Test for phase_operation
# -----------------------------
def test_phase_operation_drilling():
    assert phase_operation(50) == "Drilling"

def test_phase_operation_completion():
    assert phase_operation(150) == "Completion"

def test_phase_operation_production():
    assert phase_operation(250) == "Production"
