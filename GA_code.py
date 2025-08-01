import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pygad
import joblib

# بارگذاری دیتاست

# خواندن فایل

df = pd.read_csv(r"D:\ppt\projects\iFDC---FCDDWCSW--------------\tasks\task47\clean_processed.csv")
# نمایش نام همه ستون‌ها برای بررسی

print("ستون‌های دیتافریم شما:")
print(df.columns)
print(df.head())
print(df.shape)
print(df.dtypes)

#شتاسایی هایپرپارامترهای مدل 


model = RandomForestClassifier()
print('Hyperparameter of dmodel:')
print(model.get_params())

#هایپرهای مهم و موثر مدل رندوم کلسیفایر 
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


# داده‌ها

X = df.drop(['Record_ID', 'API_Well_ID', 'DateTime', 'Active_Damage'], axis=1).head(10000)
y = df['Active_Damage'].head(10000)


# تبدیل ژن به دیکشنری از هایپرپارامترها
def decode_genes(solution):
    return {
        'n_estimators': int(solution[0]),
        'max_depth': int(solution[1]),
        'min_samples_split': int(solution[2]),
        'min_samples_leaf': int(solution[3]),
    }
print('it is ok 1')

# تابع تناسب با استفاده از cross-validation

def fitness_func(ga_instance, solution, solution_idx):
    print(f"🧬 نسل: {ga_instance.generations_completed} - فرد {solution_idx} - ژن‌ها: {solution}")
    params = decode_genes(solution)
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=2, scoring='accuracy', n_jobs=1)
    return scores.mean()

print('it is ok 2')

# فضای جستجو: محدوده‌ی هر ژن
gene_space = [
    {'low': 50, 'high': 300, 'step': 10},   # n_estimators
    {'low': 5, 'high': 50, 'step': 5},      # max_depth
    {'low': 2, 'high': 10, 'step': 1},      # min_samples_split
    {'low': 1, 'high': 5, 'step': 1},       # min_samples_leaf
]
print('it is ok 3')


# تنظیمات الگوریتم ژنتیک
ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=len(gene_space),
    gene_space=gene_space,
    mutation_percent_genes=2,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random"
)
print('it is ok 4')


import time
print("شروع اجرا...")
start = time.time()

# اجرای الگوریتم ژنتیک
ga_instance.run()
end = time.time()
print("✅ اجرا تموم شد در", end - start, "ثانیه")
print('it is ok 5')

# نمایش بهترین ترکیب هایپرپارامترها
solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_params_ga = decode_genes(solution)
print("Best Hyperparameters:", best_params_ga)
print("Best Fitness (CV accuracy):", solution_fitness)

# آموزش مدل نهایی و ارزیابی
def evaluate_model(X, y, params):
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=2, scoring='accuracy')
    print(f"✅ Cross-Val Accuracy (cv=2): {scores.mean():.4f}")
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1-score': f1_score(y, y_pred, zero_division=0),
        
    }
    print("✅ evaluate is done")
    return model, metrics
model_ga, metrics_ga = evaluate_model(X, y, best_params_ga)
print("GA metrics:", metrics_ga)

import os
print("Current working directory:", os.getcwd())

# ذخیره مدل‌ها

joblib.dump(model_ga, 'model_genetic.pkl')
print('model is saved')

# ذخیره پارامترها

pd.DataFrame([best_params_ga]).to_csv("genetic_best_params.csv", index=False)
print(' best_param is saved')


#اگر بخوای این رو برای مدل‌های دیگه 

#مدل رو عوض کنی

#param_grid مخصوص اون مدل رو بنویسی



  