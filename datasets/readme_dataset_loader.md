# 🛠️ iFDC Dataset Loader

This module handles the ingestion of well record `.parquet` files into a PostgreSQL database for the **iFDC project**.

## 📁 Directory Structure

```
datasets/
├── load_to_db.py                  # Main script to load data
├── well_data                      # Raw well data folder
└── readme_dataset_loader.md       # This file
```

---

## ✨ How to Use

### 1. Set Up PostgreSQL

Make sure PostgreSQL is installed and running. Create a database:

```bash
createdb iFDC---FCDDWCSWdb
```

You can verify with:

```bash
psql -U <your-username> -d iFDC---FCDDWCSWdb
```

> 🔐 **NOTE:** Replace `<your-username>` with your PostgreSQL username.

### 2. Create the Table

In `psql`, run:

```sql
\i datasets/create_table.sql
```

Or manually create the table using:

```sql
CREATE TABLE IF NOT EXISTS well_records (
    record_id SERIAL PRIMARY KEY,
    api_well_id TEXT,
    long_deg DOUBLE PRECISION,
    lat_deg DOUBLE PRECISION,
    datetime TIMESTAMP,
    days_age_well_days INTEGER,
    drilling_direction TEXT,
    depth_measured_m DOUBLE PRECISION,
    depth_bit_m DOUBLE PRECISION,
    layer_id TEXT,
    formation_type TEXT,
    layer_type TEXT,
    clay_mineralogy_type TEXT,
    clay_content_percent_p DOUBLE PRECISION,
    formation_permeability_md DOUBLE PRECISION,
    porosity_formation_p DOUBLE PRECISION,
    fractures_presence BOOLEAN,
    phase_operation TEXT,
    weight_on_bit_kg DOUBLE PRECISION,
    rop_m_hr DOUBLE PRECISION,
    rpm_rev_min DOUBLE PRECISION,
    torque_nm DOUBLE PRECISION,
    mud_type TEXT,
    mud_weight_in_ppg DOUBLE PRECISION,
    mud_weight_out_ppg DOUBLE PRECISION,
    mud_temperature_in_c DOUBLE PRECISION,
    mud_temperature_out_c DOUBLE PRECISION,
    in_rate_flow_mud_l_min DOUBLE PRECISION,
    out_rate_flow_mud_l_min DOUBLE PRECISION,
    viscosity_cp DOUBLE PRECISION,
    mud_ph DOUBLE PRECISION,
    fluid_loss_api_ml_30min DOUBLE PRECISION,
    solid_content_p DOUBLE PRECISION,
    chloride_content_mg_l DOUBLE PRECISION,
    volume_pit_bbl DOUBLE PRECISION,
    pressure_standpipe_psi DOUBLE PRECISION,
    pressure_annulus_psi DOUBLE PRECISION,
    pressure_reservoir_psi DOUBLE PRECISION,
    overbalance_psi DOUBLE PRECISION,
    reservoir_temperature_c DOUBLE PRECISION,
    completion_type TEXT,
    density_perforation_shots_m DOUBLE PRECISION,
    active_damage BOOLEAN,
    type_damage TEXT,
    damage_severity TEXT,
    controllable BOOLEAN
);
```

---

## 📦 Load Data into the DB

To load the `.parquet` files into the database (only 500 rows per file for dev/testing):

```bash
python datasets/load_to_db.py
```

This will:

- Load all `.parquet` files in the folder
- Read 500 rows per file
- Insert the data into the `well_records` table

📅 Output logs will confirm each insert.

---

## 🔍 Querying the Database

Enter the database:

```bash
psql -U <your-username> -d iFDC---FCDDWCSWdb
```

Example queries:

```sql
-- Count total records
SELECT COUNT(*) FROM well_records;

-- Preview data
SELECT * FROM well_records LIMIT 10;
```

---

## 🧪 Developer Notes

- You can change the number of rows loaded per file by modifying `load_to_db.py`:
  ```python
  df = pd.read_parquet(file).head(500)
  ```
- To load full datasets, just remove `.head(500)`.

---

## 🤝 Team Collaboration

This setup currently uses a **local PostgreSQL server**.
Created and Published by **Shayan Talebian**.
