import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path



# تابع افزودن داده گمشده و نویز گوسی
# این تابع جنریک است و فقط روی ستون‌های عددی اعمال می‌شود

def add_missing_and_noise(df, missing_percent=5.0, noise_mean=0.0, noise_std=0.1):
    df_modified = df.copy()
    numeric_cols = df_modified.select_dtypes(include=[np.number]).columns

    total_values = df_modified[numeric_cols].size
    num_missing = int((missing_percent / 100.0) * total_values)

    for _ in range(num_missing):
        col = np.random.choice(numeric_cols)
        row = np.random.randint(0, len(df_modified))
        df_modified.at[row, col] = np.nan

    for col in numeric_cols:
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(df_modified))
        mask = df_modified[col].notnull()
        df_modified.loc[mask, col] += noise[mask]

    return df_modified


# پوشه حاوی فایل‌های ورودی پارکت
input_dir = Path(os.getenv('INPUT_DIR', 'well_outputs')) 
output_dir = Path(os.getenv('OUTPUT_DIR', 'modified_outputs_chunked'))
output_dir.mkdir(parents=True, exist_ok=True)



# پارامترهای پردازش
missing_percent = 5.0
noise_mean = 0.0
noise_std = 0.1

# پیدا کردن همه فایل‌های پارکت در مسیر ورودی
files = sorted([f for f in input_dir.glob("*.parquet")])
print(f"Found {len(files)} parquet files in '{input_dir}'.")

# پردازش هر فایل پارکت به صورت چانک‌به‌چانک
for file_i, file_path in enumerate(files):
    print(f"Processing file {file_i + 1}/{len(files)}: {file_path.name}")
    pf = pq.ParquetFile(file_path)
    num_row_groups = pf.num_row_groups

    for rg in range(num_row_groups):
        print(f"  Reading row group {rg + 1}/{num_row_groups}")
        table = pf.read_row_group(rg)
        df_chunk = table.to_pandas()

        # اعمال تغییرات روی هر چانک
        df_modified = add_missing_and_noise(
            df_chunk,
            missing_percent=missing_percent,
            noise_mean=noise_mean,
            noise_std=noise_std
        )

        # ذخیره خروجی به صورت فشرده
        output_file = output_dir / f"modified_{file_path.stem}_rg{rg + 1}.parquet"
        df_modified.to_parquet(output_file, compression='snappy', index=False)

        print(f"  Saved chunk {rg + 1} to {output_file.name}")
        print("  Sample data after modification:")
        print(df_modified.head(3))
        print("-" * 30)

print("All files processed chunk-by-chunk.")