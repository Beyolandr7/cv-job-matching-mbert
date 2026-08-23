import os
import sys
import re
import html
import time
import numpy as np
import pandas as pd

# Reconfigure stdout to UTF-8 for Windows console support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def clean_text_advanced(text: str) -> str:
    """
    NLP Text Cleaning Pipeline specifically designed for mBERT and NER preservation:
    1. Unescapes HTML entities (&amp; -> &, &quot; -> ", etc.)
    2. Strips HTML tags (<p>, <br>, <div>, etc.)
    3. Removes URLs (http/https/www)
    4. Cleans non-printable visual escape characters and unicode artifacts (\r, \n, \t, \xa0, \u200b, bullet points)
    5. Normalizes whitespace while strictly preserving original CASE (NO case-folding/lowercasing).
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Unescape HTML entities
    cleaned = html.unescape(text)
    
    # 2. Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    
    # 3. Strip URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    
    # 4. Normalize escape characters and invisible unicode characters
    cleaned = re.sub(r"[\r\n\t\xa0\u200b\u200c\u200d\u200e\u200f\ufeff\u3000]", " ", cleaned)
    
    # 5. Normalize visual bullets and special decorative symbols to spaces
    cleaned = re.sub(r"[\u2022\u2023\u25e6\u2043\u2219\u25cf\u25aa\u25ab\u25c6\u25c7\u2713\u2714\u2013\u2014]", " ", cleaned)
    
    # 6. Normalize multiple consecutive spaces into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned

def compute_nlp_statistics(series: pd.Series, name: str) -> dict:
    """
    Calculates word counts and estimates WordPiece token lengths for mBERT.
    Standard WordPiece subword expansion factor is typically ~1.25x - 1.35x for English/technical corpora.
    """
    word_counts = series.apply(lambda x: len(x.split()) if isinstance(x, str) and len(x) > 0 else 0)
    # Estimated WordPiece tokens (factor 1.30)
    token_estimates = (word_counts * 1.30).round().astype(int)
    
    stats = {
        "dataset": name,
        "count": len(word_counts),
        "word_min": int(word_counts.min()),
        "word_mean": float(word_counts.mean()),
        "word_std": float(word_counts.std()),
        "word_median": float(word_counts.median()),
        "word_p25": float(np.percentile(word_counts, 25)),
        "word_p75": float(np.percentile(word_counts, 75)),
        "word_p95": float(np.percentile(word_counts, 95)),
        "word_p99": float(np.percentile(word_counts, 99)),
        "word_max": int(word_counts.max()),
        "token_min": int(token_estimates.min()),
        "token_mean": float(token_estimates.mean()),
        "token_median": float(token_estimates.median()),
        "token_p95": float(np.percentile(token_estimates, 95)),
        "token_p99": float(np.percentile(token_estimates, 99)),
        "token_max": int(token_estimates.max()),
        "pct_exceeding_512_tokens": float((token_estimates > 512).mean() * 100)
    }
    return stats, word_counts, token_estimates

def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print(">>> MEMULAI AUDIT DATASET, KURASI FITUR & PEMBERSIHAN NLP")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. AUDIT & PROFILING AWAL: DATASET RESUME
    # -------------------------------------------------------------
    resume_path = "data/raw/CV/Resume/Resume.csv" if os.path.exists("data/raw/CV/Resume/Resume.csv") else "data/CV/Resume/Resume.csv"
    print(f"\n[FASE 1] Memuat Dataset CV / Resume: {resume_path} ...")
    raw_res_df = pd.read_csv(resume_path)
    res_initial_rows, res_initial_cols = raw_res_df.shape
    print(f"-> Ukuran Awal Resume: {res_initial_rows:,} baris x {res_initial_cols} kolom")
    print("-> Kolom awal:", raw_res_df.columns.tolist())
    print("-> Profil Missing Value Resume Awal:")
    for c in raw_res_df.columns:
        n_null = raw_res_df[c].isnull().sum()
        pct_null = (n_null / res_initial_rows) * 100
        print(f"   * {c:15s} | Null: {n_null:5d} ({pct_null:5.2f}%) | Dtype: {raw_res_df[c].dtype}")
    
    res_exact_dups = raw_res_df.duplicated().sum()
    res_text_dups = raw_res_df.duplicated(subset=['Resume_str']).sum()
    print(f"-> Duplikat Baris Penuh: {res_exact_dups} | Duplikat Konten Resume_str: {res_text_dups}")
    
    # -------------------------------------------------------------
    # 1. AUDIT & PROFILING AWAL: DATASET LINKEDIN JOB POSTINGS
    # -------------------------------------------------------------
    job_path = "data/raw/Jobdesc/postings.csv" if os.path.exists("data/raw/Jobdesc/postings.csv") else "data/Jobdesc/postings.csv"
    print(f"\n[FASE 1] Memuat Dataset Lowongan LinkedIn: {job_path} ...")
    raw_job_df = pd.read_csv(job_path)
    job_initial_rows, job_initial_cols = raw_job_df.shape
    print(f"-> Ukuran Awal Job Postings: {job_initial_rows:,} baris x {job_initial_cols} kolom")
    print("-> Kolom awal:", raw_job_df.columns.tolist())
    print("-> Profil Missing Value Job Postings Awal:")
    for c in raw_job_df.columns:
        n_null = raw_job_df[c].isnull().sum()
        pct_null = (n_null / job_initial_rows) * 100
        print(f"   * {c:28s} | Null: {n_null:6d} ({pct_null:5.2f}%) | Dtype: {raw_job_df[c].dtype}")
    
    job_exact_dups = raw_job_df.duplicated().sum()
    job_text_dups = raw_job_df.duplicated(subset=['description']).sum()
    job_pair_dups = raw_job_df.duplicated(subset=['title', 'description']).sum()
    print(f"-> Duplikat Baris Penuh: {job_exact_dups}")
    print(f"-> Duplikat Konten description: {job_text_dups:,}")
    print(f"-> Duplikat Pasangan (title, description): {job_pair_dups:,}")

    # -------------------------------------------------------------
    # 2. KURASI FITUR (KEEP VS DROP)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[FASE 2] KURASI & FILTER FITUR (KEEP VS DROP)")
    print("=" * 70)
    
    # Resume: Keep ['Resume_str', 'Category'], Drop ['ID', 'Resume_html']
    resume_keep_cols = ['Resume_str', 'Category']
    resume_dropped_cols = [c for c in raw_res_df.columns if c not in resume_keep_cols]
    print(f"\n* Kurasi Resume Dataset:")
    print(f"  - Kolom Dipertahankan ({len(resume_keep_cols)}): {resume_keep_cols}")
    print(f"  - Kolom Dieliminasi ({len(resume_dropped_cols)}): {resume_dropped_cols}")
    curated_res_df = raw_res_df[resume_keep_cols].copy()
    
    # LinkedIn Job Postings: Keep ['title', 'description', 'skills_desc', 'formatted_experience_level']
    job_keep_cols = ['title', 'description', 'skills_desc', 'formatted_experience_level']
    job_dropped_cols = [c for c in raw_job_df.columns if c not in job_keep_cols]
    print(f"\n* Kurasi LinkedIn Job Postings Dataset:")
    print(f"  - Kolom Dipertahankan ({len(job_keep_cols)}): {job_keep_cols}")
    print(f"  - Kolom Dieliminasi ({len(job_dropped_cols)}): {job_dropped_cols}")
    curated_job_df = raw_job_df[job_keep_cols].copy()

    # -------------------------------------------------------------
    # 3. PEMBERSIHAN DATA TEKS (TEXT CLEANING)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[FASE 3] PEMBERSIHAN TEKS (TEXT CLEANING BERBASIS NLP & REGEX)")
    print("=" * 70)
    
    # --- Resume Cleaning ---
    print("\n-> Membersihkan Resume Dataset...")
    # 3.1 Drop nulls on primary text column
    res_before_null_drop = len(curated_res_df)
    curated_res_df = curated_res_df.dropna(subset=['Resume_str'])
    curated_res_df = curated_res_df[curated_res_df['Resume_str'].str.strip() != '']
    print(f"   * Hapus Null/Kosong pada 'Resume_str': {res_before_null_drop - len(curated_res_df)} baris terhapus.")
    
    # 3.2 Apply text cleaning (preserves CASE!)
    curated_res_df['Resume_str_clean'] = curated_res_df['Resume_str'].apply(clean_text_advanced)
    
    # Filter out any that became empty after cleaning
    curated_res_df = curated_res_df[curated_res_df['Resume_str_clean'] != '']
    
    # 3.3 Deduplication based on cleaned text content
    res_before_dedup = len(curated_res_df)
    curated_res_df = curated_res_df.drop_duplicates(subset=['Resume_str_clean'], keep='first')
    res_dedup_removed = res_before_dedup - len(curated_res_df)
    print(f"   * Hapus Duplikat Konten Teks Resume: {res_dedup_removed} baris terhapus.")
    
    # Standardize final column names for Resume
    clean_res_final = pd.DataFrame({
        'category': curated_res_df['Category'],
        'resume_text': curated_res_df['Resume_str_clean']
    })

    # --- Job Postings Cleaning ---
    print("\n-> Membersihkan LinkedIn Job Postings Dataset...")
    # 3.1 Drop nulls on primary text columns ('description' and 'title')
    job_before_null_drop = len(curated_job_df)
    curated_job_df = curated_job_df.dropna(subset=['description', 'title'])
    curated_job_df = curated_job_df[
        (curated_job_df['description'].str.strip() != '') & 
        (curated_job_df['title'].str.strip() != '')
    ]
    print(f"   * Hapus Null/Kosong pada 'description' / 'title': {job_before_null_drop - len(curated_job_df):,} baris terhapus.")
    
    # 3.2 Apply text cleaning to title, description, skills_desc
    print("   * Menjalankan pembersihan regex pada title, description, skills_desc (mempertahankan huruf kapital)...")
    curated_job_df['title_clean'] = curated_job_df['title'].apply(clean_text_advanced)
    curated_job_df['description_clean'] = curated_job_df['description'].apply(clean_text_advanced)
    curated_job_df['skills_desc_clean'] = curated_job_df['skills_desc'].apply(
        lambda x: clean_text_advanced(x) if pd.notnull(x) else ""
    )
    curated_job_df['experience_level'] = curated_job_df['formatted_experience_level'].fillna("Not Specified")
    
    # Remove empty after cleaning
    curated_job_df = curated_job_df[
        (curated_job_df['description_clean'] != '') & 
        (curated_job_df['title_clean'] != '')
    ]
    
    # 3.3 Deduplication based on cleaned text content (title + description)
    job_before_dedup = len(curated_job_df)
    curated_job_df = curated_job_df.drop_duplicates(subset=['title_clean', 'description_clean'], keep='first')
    job_dedup_removed = job_before_dedup - len(curated_job_df)
    print(f"   * Hapus Duplikat Konten (title + description): {job_dedup_removed:,} baris terhapus.")
    
    clean_job_final = pd.DataFrame({
        'job_title': curated_job_df['title_clean'],
        'job_description': curated_job_df['description_clean'],
        'skills_description': curated_job_df['skills_desc_clean'],
        'experience_level': curated_job_df['experience_level']
    })

    # -------------------------------------------------------------
    # 4. LAPORAN AUDIT, STATISTIK TOKEN & EKSPOR
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[FASE 4] STATISTIK DISTRIBUSI KATA & TOKENISASI mBERT")
    print("=" * 70)
    
    res_stats, res_words, res_tokens = compute_nlp_statistics(clean_res_final['resume_text'], "Clean Resumes")
    job_desc_stats, job_words, job_tokens = compute_nlp_statistics(clean_job_final['job_description'], "Clean Job Descriptions")
    job_title_stats, _, _ = compute_nlp_statistics(clean_job_final['job_title'], "Clean Job Titles")
    
    print(f"\n[+] Statistik Resume (Word & WordPiece Tokens):")
    print(f"   * Jumlah Dokumen: {res_stats['count']:,}")
    print(f"   * Word Count -> Min: {res_stats['word_min']}, Mean: {res_stats['word_mean']:.1f}, Median: {res_stats['word_median']:.1f}, Max: {res_stats['word_max']}, p95: {res_stats['word_p95']:.1f}, p99: {res_stats['word_p99']:.1f}")
    print(f"   * WordPiece Token Est. -> Mean: {res_stats['token_mean']:.1f}, Median: {res_stats['token_median']:.1f}, Max: {res_stats['token_max']}, p95: {res_stats['token_p95']:.1f}")
    print(f"   * % Dokumen melebihi batas konteks 512 mBERT tokens: {res_stats['pct_exceeding_512_tokens']:.2f}%")
    
    print(f"\n[+] Statistik Job Postings Description (Word & WordPiece Tokens):")
    print(f"   * Jumlah Dokumen: {job_desc_stats['count']:,}")
    print(f"   * Word Count -> Min: {job_desc_stats['word_min']}, Mean: {job_desc_stats['word_mean']:.1f}, Median: {job_desc_stats['word_median']:.1f}, Max: {job_desc_stats['word_max']}, p95: {job_desc_stats['word_p95']:.1f}, p99: {job_desc_stats['word_p99']:.1f}")
    print(f"   * WordPiece Token Est. -> Mean: {job_desc_stats['token_mean']:.1f}, Median: {job_desc_stats['token_median']:.1f}, Max: {job_desc_stats['token_max']}, p95: {job_desc_stats['token_p95']:.1f}")
    print(f"   * % Dokumen melebihi batas konteks 512 mBERT tokens: {job_desc_stats['pct_exceeding_512_tokens']:.2f}%")

    print(f"\n[+] Statistik Job Titles:")
    print(f"   * Word Count -> Min: {job_title_stats['word_min']}, Mean: {job_title_stats['word_mean']:.1f}, Median: {job_title_stats['word_median']:.1f}, Max: {job_title_stats['word_max']}")

    # Ekspor File
    print("\n" + "=" * 70)
    print("[+] MENGEKSPOR DATASET HASIL KURASI & PEMBERSIHAN")
    print("=" * 70)
    
    output_res_path_data = "data/clean_resumes.csv"
    output_job_path_data = "data/clean_job_postings.csv"
    
    clean_res_final.to_csv(output_res_path_data, index=False, encoding="utf-8")
    print(f"[OK] Tersimpan: {output_res_path_data} ({len(clean_res_final):,} baris)")
    
    clean_job_final.to_csv(output_job_path_data, index=False, encoding="utf-8")
    print(f"[OK] Tersimpan: {output_job_path_data} ({len(clean_job_final):,} baris)")
    
    # Save a detailed summary json/csv for verification
    summary_data = {
        "Dataset": ["Resume Dataset", "LinkedIn Job Postings"],
        "Initial_Rows": [res_initial_rows, job_initial_rows],
        "Initial_Columns": [res_initial_cols, job_initial_cols],
        "Curated_Columns_Kept": [len(clean_res_final.columns), len(clean_job_final.columns)],
        "Columns_Eliminated": [len(resume_dropped_cols), len(job_dropped_cols)],
        "Final_Clean_Rows": [len(clean_res_final), len(clean_job_final)],
        "Rows_Dropped_Total": [res_initial_rows - len(clean_res_final), job_initial_rows - len(clean_job_final)],
        "Remaining_Missing_Values": [int(clean_res_final.isnull().sum().sum()), int(clean_job_final.isnull().sum().sum())],
        "Word_Count_Mean": [round(res_stats['word_mean'], 1), round(job_desc_stats['word_mean'], 1)],
        "WordPiece_Token_Mean": [round(res_stats['token_mean'], 1), round(job_desc_stats['token_mean'], 1)],
        "Pct_Exceeding_512_Tokens": [f"{res_stats['pct_exceeding_512_tokens']:.2f}%", f"{job_desc_stats['pct_exceeding_512_tokens']:.2f}%"]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("data/dataset_audit_summary.csv", index=False)
    print("[OK] Tersimpan: data/dataset_audit_summary.csv")
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Pipeline selesai dieksekusi dalam {elapsed:.2f} detik.")

if __name__ == "__main__":
    run_pipeline()
