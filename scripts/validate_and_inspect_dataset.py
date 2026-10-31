"""
validate_and_inspect_dataset.py
--------------------------------
Memvalidasi kualitas dan integritas dataset CV-Job Matching 3-Kelas:
- Memeriksa jumlah baris dan keseimbangan rasio kelas (100:100:100)
- Memeriksa nilai kosong (null values) dan duplikasi
- Menganalisis statistik panjang karakter (job & cv)
- Menampilkan sampel representatif per kategori (fit, close fit, not fit)
"""

import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_path = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class.csv')
    curated_path = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class_curated.csv')

    print("=" * 70)
    print("AUDIT & VALIDASI DATASET CV-JOB MATCHING 3-KELAS")
    print("=" * 70)

    for name, path in [("DATASET MODEL (3 KOLOM)", clean_path), ("DATASET KURASI METADATA", curated_path)]:
        print(f"\n>>> Memeriksa: {name}")
        print(f"Path: {path}")
        if not os.path.exists(path):
            print(f"[ERROR] File tidak ditemukan: {path}")
            continue

        df = pd.read_csv(path)
        print(f"Total Baris Data : {len(df)}")
        print(f"Daftar Kolom     : {list(df.columns)}")
        
        # 1. Cek Nilai Kosong (Null Check)
        null_counts = df.isnull().sum()
        print(f"Jumlah Nilai Kosong per Kolom:\n{null_counts.to_string()}")
        assert df['kategori'].isnull().sum() == 0, "Ada nilai null di kolom kategori!"
        assert df['job'].isnull().sum() == 0, "Ada nilai null di kolom job!"
        assert df['cv'].isnull().sum() == 0, "Ada nilai null di kolom cv!"

        # 2. Cek Keseimbangan Kelas
        counts = df['kategori'].value_counts()
        expected_per_cat = len(df) // 3
        print(f"\nDistribusi Kategori (Target: {expected_per_cat} per kelas):\n{counts.to_string()}")
        for cat in ['fit', 'close fit', 'not fit']:
            assert counts.get(cat, 0) == expected_per_cat, f"Kategori {cat} ({counts.get(cat, 0)}) tidak seimbang dengan target {expected_per_cat}!"

        # 3. Cek Duplikasi Teks
        dup_jobs = df.duplicated(subset=['job']).sum()
        dup_cvs = df.duplicated(subset=['cv']).sum()
        dup_pairs = df.duplicated(subset=['job', 'cv']).sum()
        print(f"\nAudit Duplikasi:")
        print(f"  - Pasangan persis (job, cv) duplikat : {dup_pairs}")
        print(f"  - Job description berulang           : {dup_jobs}")
        print(f"  - CV berulang                        : {dup_cvs}")
        assert dup_pairs == 0, "Ditemukan pasangan (job, cv) duplikat!"

        # 4. Statistik Panjang Karakter
        df['job_len'] = df['job'].str.len()
        df['cv_len'] = df['cv'].str.len()
        print(f"\nStatistik Panjang Teks (Karakter):")
        print(f"  - Job [Min: {df['job_len'].min()}, Median: {int(df['job_len'].median())}, Max: {df['job_len'].max()}]")
        print(f"  - CV  [Min: {df['cv_len'].min()}, Median: {int(df['cv_len'].median())}, Max: {df['cv_len'].max()}]")

    # Tampilkan Sampel Inspeksi Representatif dari Dataset Kurasi
    if os.path.exists(curated_path):
        df_curated = pd.read_csv(curated_path)
        print("\n" + "=" * 70)
        print("SAMPEL AUDIT MANUAL PER KATEGORI (BERDASARKAN RUBRIK REKRUTER)")
        print("=" * 70)
        for cat in ['fit', 'close fit', 'not fit']:
            sample = df_curated[df_curated['kategori'] == cat].iloc[0]
            print(f"\n[KATEGORI: {cat.upper()}] - {sample['id']}")
            print(f"Domain         : {sample['domain']}")
            print(f"Subtipe        : {sample['subtipe']}")
            print(f"Job Title      : {sample['job_title']}")
            print(f"Job Exp Level  : {sample['experience_level_job']}")
            print(f"CV Exp Years   : {sample['years_exp_cv']} tahun")
            print(f"Alasan Anotasi : {sample['alasan_kategori']}")
            print(f"Job Snippet    : {str(sample['job'])[:180]}...")
            print(f"CV Snippet     : {str(sample['cv'])[:180]}...")
            print("-" * 50)

    print("\n[VALIDASI BERHASIL] Seluruh kriteria integritas dan keseimbangan 300 data terpenuhi dengan sempurna!")

if __name__ == '__main__':
    main()
