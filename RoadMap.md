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
- Aida,

---

## ML model
### Team Assignment
- Aida,
- Anahita


using `generator.py` located in `/dataset/` directory generated a dataset with `np.random.seed(42)`

- usage:
```python
python genereator.py
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
python genereator.py
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
