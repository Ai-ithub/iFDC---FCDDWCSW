import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler

# بارگذاری داده‌ها
df = pd.read_parquet(r"C:\Users\arezoo\Desktop\پترو پالاتوس\projects\iFDC---FCDDWCSW\کدهای من\well_40181715.parquet")
print(df.head())

# انتخاب ستون‌های عددی
numeric_cols = df.select_dtypes(include=[np.number]).columns

# ----------------------------
# پیدا کردن داده‌های پرت با روش Z-Score
z_scores = np.abs(zscore(df[numeric_cols], nan_policy='omit'))
z_outliers = (z_scores > 3).any(axis=1)  # True/False برای هر ردیف

# ----------------------------
# پیدا کردن داده‌های پرت با روش IQR
iqr_outliers = pd.Series(False, index=df.index)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    iqr_outliers = iqr_outliers | is_outlier  # ترکیب با روش Z

# ----------------------------
# ترکیب دو روش: Z-Score و IQR
combined_outliers = z_outliers | iqr_outliers

# استخراج داده‌های پرت
outlier_data = df[combined_outliers]

# استخراج داده‌های تمیز
clean_data = df[~combined_outliers]

# ----------------------------
# ذخیره هر دو در فایل جدا
outlier_data.to_parquet(r"C:\Users\arezoo\Desktop\پترو پالاتوس\projects\iFDC---FCDDWCSW\کدهای من\outliers_only.parquet")
clean_data.to_parquet(r"C:\Users\arezoo\Desktop\پترو پالاتوس\projects\iFDC---FCDDWCSW\کدهای من\clean_data.parquet")

print("✅ Number of outliers:", len(outlier_data))
print("✅ Number of clean records:", len(clean_data))
print("📁 Files saved successfully.")

# بارگذاری داده‌ی تمیز
df = pd.read_parquet(r"C:\Users\arezoo\Desktop\پترو پالاتوس\projects\iFDC---FCDDWCSW\کدهای من\clean_data.parquet")
# ----------------------------
# Label Encoding روی ستون‌های دسته‌ای
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
le_dict = {}  # برای نگه‌داشتن encoder هر ستون
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
# ----------------------------
# Standard Scaler روی ستون‌های عددی
numeric_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ----------------------------
# ذخیره داده‌ی آماده‌شده
df.to_parquet(r"C:\Users\arezoo\Desktop\پترو پالاتوس\projects\iFDC---FCDDWCSW\کدهای من\clean_processed.parquet")

print("✅ Preprocessing done. Output saved to: clean_processed.parquet")