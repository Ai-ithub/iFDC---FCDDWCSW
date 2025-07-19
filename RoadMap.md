## Generating data

using `generator.py` located in `/dataset/` directory generated you dataset

it should

---

## ML model
---

## Deep learning models --> Amin, Juan, Mehmoot

using `generator.py` located in `/dataset/` directory generated a dataset with `np.random.seed(42)`

- It has mulitple columns containing features and two different label: `Active_Damage` and `Type_Damage`
- There are 4/5 colummns that are reduntant and has no impact on output

### Task
Check the[issue](https://github.com/Ai-ithub/iFDC---FCDDWCSW/issues/58)
Create a model (pytorch preferred) with `torch.manual_seed()` that accept generated data. As you know the main consideration is
preprocessing the data and apply a window (number of time steps that need to predict/generated the next item)

- LSTM
- GRU
- Transformer --> Amin
