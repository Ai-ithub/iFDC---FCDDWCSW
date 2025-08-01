import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pygad
import joblib
import random

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
print('Hyperparameter of model:')
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

print('it is ok1')

#Simulated Annealing 

# تابع ارزیابی برای dual_annealing (Minimize → پس منفی دقت می‌گیریم)
def objective_function(params, X, y):
    # تبدیل پارامترهای پیوسته به صحیح یا دسته‌ای
    n_estimators = int(round(params[0]))
    max_depth = int(round(params[1]))
    min_samples_split = int(round(params[2]))
    min_samples_leaf = int(round(params[3]))
    
    # ایجاد مدل
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # دقت با cross-validation (منفی چون dual_annealing مینیمم می‌کنه)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return -score  # چون می‌خوایم maximize کنیم
print('it is ok2')

# محدوده‌ها برای هر پارامتر: (min, max)
bounds = [
    (50, 300),   # n_estimators
    (5, 50),     # max_depth
    (2, 10),     # min_samples_split
    (1, 5)       # min_samples_leaf
]
print('it is ok3')

# اجرای dual_annealing
result = dual_annealing(objective_function, bounds=bounds, args=(X, y), maxiter=50)
print('it is ok4')
# نمایش بهترین پارامترها
best_params = result.x
best_params_dict = {
    'n_estimators': int(round(best_params[0])),
    'max_depth': int(round(best_params[1])),
    'min_samples_split': int(round(best_params[2])),
    'min_samples_leaf': int(round(best_params[3]))
}
print("🔍 Best Parameters (dual_annealing):", best_params_dict)

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
model_SA, metrics_SA = evaluate_model(X, y, best_params_dict)
print("SA metrics:", metrics_SA)

import os
print("Current working directory:", os.getcwd())

# ذخیره مدل‌ها

joblib.dump(model_SA, 'model_simulated_annealing.pkl')
print('model is saved')

# ذخیره پارامترها

pd.DataFrame([best_params_dict]).to_csv("simulated_annealing_best_params.csv", index=False)
print(' best_param is saved')


#اگر بخوای این رو برای مدل‌های دیگه 

#مدل رو عوض کنی

#param_grid مخصوص اون مدل رو بنویسی



  