# 📊 Outlier Detection and Clean Data Generation

This project involves identifying and removing outliers from the input dataset. The input data consists of **10 parquet files**. After processing, two outputs are generated for each input file:  
1️⃣ Outliers file  
2️⃣ Clean data file  

---

## 🗂️ Input and Output Structure
- **Input:** A folder containing 10 parquet files  
- **Outputs:**  
  - **10 Outliers files**  
  - **10 Clean files**  

A total of **20 output files** will be generated.

---

## 📝 Features
- Outlier detection based on the following columns:
  - `temperature`
  - `pressure`
  - `permeability`
  - `flow_rate`
- Statistical methods (z-score and IQR) used for outlier detection.
- Separate storage of outliers and clean data.

---

## 🚀 General Execution:
1️⃣ Input: 10 parquet files in a folder  
2️⃣ Processing: Outlier detection and clean data generation  
3️⃣ Output: 20 files (10 outliers + 10 clean)
