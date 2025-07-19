# Road map of the project

__Table 0f content__
<ul>
  <li><a href="#generating-data">Generating data</a></li>
  <li><a href="#add-new-label">Add new label</a></li>
  <li><a href="#ml-model">ML model</a></li>
  <li><a href="#deep-learning-models">Deep learning models</a></li>  
</ul>

---
## Generating data

We want to create a dataset to be almost like a real senario. which has three phase:
- Drilling
- Completion
- Production

Each phase has its attibutes(X: model input) and label(y: model output for supervise learning)

---
## Add new label 
[Controllable vs. Non-Controllable](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/37)
### Team Assignment
- Merged into generated data task

We’ll use the following columns from dataset to decide controllability:
- `Type_Damage` (Primary classification)
- `Formation_Type` (Shale, Limestone, Sandstone)
- `Formation_Permeability` (Low permeability → Harder to treat)
- `Reservoir_Temperature` (High temp → More severe damage)
- `Active_Damage` (Yes/No)
- `Mud_Type` (Water/Oil/Synthetic-based mud affects treatability)
- `Completion_Type` (Cased/Open Hole/Liner impacts intervention ease)


✅ Controllable Damage (Can be mitigated with interventions)
Damage Type | Criteria
-- | --
Clay & Iron Control | High clay content (Clay_Content_Percent > 35) + Montmorillonite clay type. Treatable with clay stabilizers.
Fluid Loss | Fluid_Loss_API > 1.0 + Mud_Type == "Water-based". Fixable with fluid loss additives.
Scale/Sludge Incompatibility | Chloride_Content > 500 + Solid_Content > 10. Reversible with acidizing or scale inhibitors.
Near-Wellbore Emulsions | Completion_Type == "Open Hole" + Mud_pH < 7.0. Treatable with demulsifiers.
Completion Damage | Completion_Type == "Cased" + Overbalance > 100. Fixable with reperforation.
Surface Filtration | Viscosity > 18 + Mud_Type == "Oil-based". Adjustable via filtration.
Ultra-Clean Fluids Control | Viscosity < 12 + Mud_Type == "Synthetic". Managed via fluid engineering.

✅ Non-Controllable Damage (Difficult or irreversible)
Damage  Type | Criteria
-- | --
Drilling-Induced Damage | Formation_Type == "Shale" + Fluid_Loss_API > 0.8. Permanent formation damage.
Rock/Fluid Interaction | Formation_Permeability < 30 + Reservoir_Temperature > 85. Irreversible permeability loss.
Stress/Corrosion Cracking | Reservoir_Temperature > 95 + Mud_Weight_In > 9.5. Requires major workover.
Generic Damage | Default category for severe, unspecified damage. Hard to treat.

---

## ML model
### Team Assignment
- Aida,
- Anahita


using `generator.py` located in `/dataset/` directory generated a dataset with `np.random.seed(42)`

- usage:
```python
python generator.py
```
- It has mulitple columns containing features and two different label: `Active_Damage` and `Type_Damage`
- There are 4/5 colummns that are reduntant and has no impact on output

### Task
Check the [issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/58)
Create a model (pytorch preferred) with `torch.manual_seed()` that accept generated data. As you know the main consideration is
preprocessing the data and apply a window (number of time steps that need to predict/generated the next item)

---

## Deep learning models

### Team Assignment
- Amin,
- Juan,
- Mehmoot

using `generator.py` located in `/dataset/` directory generated a dataset with `np.random.seed(42)`

- usage:
```python
python generator.py
```

- It has mulitple columns containing features and two different label: `Active_Damage` and `Type_Damage`
- There are 4/5 colummns that are reduntant and has no impact on output

### Task
Check the [issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/58)
Create a model (pytorch preferred) with `torch.manual_seed()` that accept generated data. As you know the main consideration is
preprocessing the data and apply a window (number of time steps that need to predict/generated the next item)

- LSTM
- GRU
- Transformer --> Amin
