# Road map of the project

__Table 0f content__
<ul>
  <li><a href="#us-patent-search">US patent search --> In progress 🚧 Milad Pour yazdani</a></li>
  <li><a href="#generating-data">Generating data --> Drilling data completed 🚧 MohammadDaraee</a></li>
  <li><a href="#handling-missing-data"> Handling Missing Data --> In progress 🚧 Mohammad Amin, Arezoo jafarpour</a></li>
  <li><a href="#handling-outlier"> Handling Outlier --> waiting😮‍💨 </a></li>
  <li><a href="#add-new-label">Add new label --> Done ✅ Mohammad Daraee</a></li>
  <li><a href="#ml-model">ML model --> In progress 🚧 MOhammad Amin, Aida</a></li>
  <li><a href="#deep-learning-models">Deep learning models --> In progress 🚧️ MohammadDaraee, Amin Ashoori, juan Mehmmot</a></li>  
  <li><a href="#hyperparameter-optimization">Hyperparameter Optimization --> In progress 🚧️ eveybody implement the model</a></li>
  <li><a href="#rto-using-reinforcement"> Real-Time Optimization (RTO) using Reinforcement --> In progress 🚧️ juan Mehmmot </a></li>
</ul>

---
## US patent search
### Team Assignment
- <code style="color : cyan">Milad Pour yazdani</code> 

[issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/61)

---
## Generating data

There are two issues connected to this problem:
1. [issue1](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/48)
2. [issue for impromvement](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/8)

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
## Handling Missing Data
### Team Assignment
- Mohammad Amin
[issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/45)

In real data or in the feature datasets there might be some datasets that need correction, this
task goal is to handle these kinds of problems

---
## Handling Outlier
### Team Assignment
- Arezoo jafarpour

[issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/46)

create a new file called, `handling_missing_value.py` and put it in `.\dataset\` folder

the code should show if the dataset has missing value or not, if so, show the percentage of missing value

---

## ML model
### Team Assignment
- Aida,
- Mohammad Amin


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

---
## Hyperparameter Optimization

### Team Assignment
- Eveyone in Deep learning models, who implemented the model

There are two issues connected with this problem:
1. [issue1](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/32)
2. [issue2](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/47)

---
## RTO Using Reinforcement

### Team Assignment
- juan
- mehmoot

[issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/53)

Consider two type of RL: offline learning and real time learning
