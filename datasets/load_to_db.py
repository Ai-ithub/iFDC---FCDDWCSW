import os
import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection settings
DB_USER = 'shayantalebian'
DB_PASS = 'Dbfnhju89$'
DB_NAME = 'iFDC---FCDDWCSWdb'
DB_HOST = 'localhost'
DB_PORT = '5432'

# Parquet folder
parquet_folder = '/home/shayantalebian/Documents/projects/PetroPala/iFDC---FCDDWCSW/datasets/well_data'

# Connect to DB
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Rename only the columns with special characters
column_renames = {
    'Clay_Content_Percent_%': 'Clay_Content_Percent_P',
    'Porosity_Formation_%': 'Porosity_Formation_P',
    'Solid_Content_%': 'Solid_Content_P',
}

# Load and insert first 500 rows from each Parquet file
for filename in os.listdir(parquet_folder):
    if not filename.endswith('.parquet'):
        continue

    file_path = os.path.join(parquet_folder, filename)
    print(f"\n📦 Loading {filename}")

    try:
        # Read only the first 500 rows
        df = pd.read_parquet(file_path, engine='pyarrow')
        df = df.head(500)  # Take only first 500 rows
        print(f"✅ Read {len(df)} rows")

        # Rename problematic columns
        df.rename(columns=column_renames, inplace=True)

        # Strip whitespace & convert columns to lowercase to match DB
        df.columns = [col.strip().lower() for col in df.columns]

        # Drop record_id if it's auto-incremented
        if 'record_id' in df.columns:
            df.drop(columns=['record_id'], inplace=True)

        # Convert booleans if those columns exist
        if 'fractures_presence' in df.columns:
            df['fractures_presence'] = df['fractures_presence'].astype(bool)

        if 'controllable' in df.columns:
            df['controllable'] = df['controllable'].map({'Yes': True, 'No': False})

        if 'active_damage' in df.columns:
            df['active_damage'] = df['active_damage'].map({'Yes': True, 'No': False})

        print("🧾 Columns in DataFrame:", df.columns.tolist())

        # Insert into database
        print("📤 Inserting into DB...")
        df.to_sql('well_records', engine, if_exists='append', index=False, chunksize=100, method='multi')
        print(f"✅ Inserted {len(df)} rows from {filename}")

    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")

print("\n🎉 All files processed.")
