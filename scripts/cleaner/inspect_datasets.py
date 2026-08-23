import os
import pandas as pd
import numpy as np

res_path = "data/raw/CV/Resume/Resume.csv" if os.path.exists("data/raw/CV/Resume/Resume.csv") else "data/CV/Resume/Resume.csv"
job_path = "data/raw/Jobdesc/postings.csv" if os.path.exists("data/raw/Jobdesc/postings.csv") else "data/Jobdesc/postings.csv"

print("=== PROFILING RESUME DATASET ===")
res_df = pd.read_csv(res_path)
print(f"Total Rows: {len(res_df):,}")
print(f"Total Columns: {len(res_df.columns)}")
print("Columns & Dtypes & Missing:")
for col in res_df.columns:
    null_count = res_df[col].isnull().sum()
    null_pct = (null_count / len(res_df)) * 100
    print(f" - {col} ({res_df[col].dtype}): {null_count} nulls ({null_pct:.2f}%)")
print(f"Total Exact Duplicate Rows: {res_df.duplicated().sum()}")
print(f"Duplicate Resume_str: {res_df.duplicated(subset=['Resume_str']).sum()}")
print(f"Unique Categories: {res_df['Category'].nunique()}")
print("Categories distribution:")
print(res_df['Category'].value_counts())

print("\n=== PROFILING LINKEDIN JOB POSTINGS DATASET ===")
job_df = pd.read_csv(job_path)
print(f"Total Rows: {len(job_df):,}")
print(f"Total Columns: {len(job_df.columns)}")
print("Columns & Dtypes & Missing:")
for col in job_df.columns:
    null_count = job_df[col].isnull().sum()
    null_pct = (null_count / len(job_df)) * 100
    print(f" - {col} ({job_df[col].dtype}): {null_count} nulls ({null_pct:.2f}%)")
print(f"Total Exact Duplicate Rows: {job_df.duplicated().sum()}")
print(f"Duplicate description: {job_df.duplicated(subset=['description']).sum()}")
print(f"Duplicate (title, description): {job_df.duplicated(subset=['title', 'description']).sum()}")
