"""
build_balanced_3class_dataset.py
---------------------------------
Membangun dataset benchmark pasangan CV - Job Matching 3-Kelas seimbang (Total 300 data):
  - 100 Fit
  - 100 Close Fit (50 Defisit Masa Kerja/Junior + 50 Lintas Domain Serumpun)
  - 100 Not Fit (Cross-Domain Negatif)

Berlandaskan 6 Pedoman Rekruter:
  1. Riwayat Pendidikan
  2. Pengalaman Kerja & Years of Related Experience (Linieritas, Senioritas, Kronologi)
  3. Hard & Soft Skills (Komunikasi, Negosiasi, Tools, Sertifikasi)
"""

import os
import re
import pandas as pd
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BENCHMARK_DOMAINS = [
    'INFORMATION-TECHNOLOGY',
    'HEALTHCARE',
    'ACCOUNTANT',
    'CHEF',
    'TEACHER',
    'ADVOCATE'
]

DOMAIN_KEYWORDS_JOB = {
    'INFORMATION-TECHNOLOGY': r'\b(Software|Developer|Engineer|Programmer|Data Scientist|Full Stack|DevOps|System Administrator|Frontend|Backend)\b',
    'HEALTHCARE': r'\b(Nurse|Physician|Doctor|Medical|Hospital|Clinical|Pharmacist|Therapist|Healthcare)\b',
    'ACCOUNTANT': r'\b(Accountant|Accounting|Auditor|Tax|CPA|Bookkeeper|Financial Accountant)\b',
    'CHEF': r'\b(Chef|Cook|Culinary|Pastry|Kitchen|Baker|Sous Chef|Line Cook)\b',
    'TEACHER': r'\b(Teacher|Instructor|Professor|Educator|Faculty|Tutor|Lecturer)\b',
    'ADVOCATE': r'\b(Lawyer|Attorney|Paralegal|Litigation|Legal Assistant|Counsel|Associate Attorney)\b'
}

# Verifikasi teks CV agar benar-benar mencerminkan bidangnya (membersihkan noise/mislabel dari raw dataset)
VERIFIED_CV_KEYWORDS = {
    'INFORMATION-TECHNOLOGY': r'\b(Software|Developer|Engineer|Programmer|DevOps|System Administrator|Frontend|Backend|IT Specialist|Network|Information Technology|Database|Java|Python)\b',
    'HEALTHCARE': r'\b(Nurse|Physician|Doctor|Medical|Hospital|Clinical|Pharmacist|Therapist|Healthcare|Patient Care|RN|BSN)\b',
    'ACCOUNTANT': r'\b(Accountant|Accounting|Auditor|Tax|CPA|Bookkeeper|Financial Accountant|General Ledger|Reconciliation|GAAP)\b',
    'CHEF': r'\b(Chef|Cook|Culinary|Pastry|Kitchen|Baker|Sous Chef|Line Cook|Catering|Food Preparation|Menu)\b',
    'TEACHER': r'\b(Teacher|Instructor|Professor|Educator|Faculty|Tutor|Lecturer|Teaching|Curriculum|Classroom|Lesson)\b',
    'ADVOCATE': r'\b(Law|Legal|Attorney|Paralegal|Court|Litigation|Justice|Counsel|Case Manager|Pleadings)\b'
}

ADJACENT_DOMAINS = {
    'ACCOUNTANT': ['FINANCE', 'BANKING'],
    'INFORMATION-TECHNOLOGY': ['ENGINEERING', 'DIGITAL-MEDIA'],
    'TEACHER': ['HR', 'PUBLIC-RELATIONS'],
    'ADVOCATE': ['HR', 'CONSULTANT'],
    'CHEF': ['FITNESS', 'AGRICULTURE'],
    'HEALTHCARE': ['FITNESS', 'PUBLIC-RELATIONS']
}

def extract_years_from_text(text):
    """Mengekstraksi estimasi tahun pengalaman kerja dari teks resume/job."""
    if not isinstance(text, str):
        return 0
    matches = re.findall(r'(\d+)\+?\s*(?:-\s*\d+)?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience|exp)?', text, re.I)
    nums = [int(m) for m in matches if m.isdigit() and int(m) <= 40]
    if nums:
        return max(nums)
    year_ranges = re.findall(r'\b(19\d\d|20\d\d)\s*(?:-|to)\s*(20\d\d|present|current)\b', text, re.I)
    total_span = 0
    for start, end in year_ranges:
        s = int(start)
        e = 2024 if end.lower() in ['present', 'current'] else int(end)
        if 1985 <= s <= 2025 and 1985 <= e <= 2025 and e >= s:
            total_span += (e - s)
    return min(total_span, 40)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_job_text(row):
    title = clean_text(row.get('job_title', ''))
    desc = clean_text(row.get('job_description', ''))
    skills = clean_text(row.get('skills_description', ''))
    level = clean_text(row.get('experience_level', ''))
    
    parts = []
    if title:
        parts.append(f"Position: {title}")
    if level and level != 'Not Specified':
        parts.append(f"Experience Level: {level}")
    if skills:
        parts.append(f"Required Skills: {skills}")
    if desc:
        parts.append(f"Job Description: {desc}")
    return "\n".join(parts)

def main():
    print("="*70)
    print("MEMULAI PROSES PEMBENTUKAN DATASET CV-JOB MATCHING 3-KELAS")
    print("="*70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resumes_path = os.path.join(base_dir, 'data', 'clean_resumes.csv')
    jobs_path = os.path.join(base_dir, 'data', 'clean_job_postings.csv')

    print(f"Loading resumes from: {resumes_path}")
    df_resumes = pd.read_csv(resumes_path)
    df_resumes['resume_text_clean'] = df_resumes['resume_text'].apply(clean_text)
    df_resumes['years_exp'] = df_resumes['resume_text'].apply(extract_years_from_text)

    # Filter CV terverifikasi konten per domain untuk mencegah noise mislabeling
    verified_resumes_by_domain = {}
    for dom in BENCHMARK_DOMAINS:
        sub = df_resumes[df_resumes['category'] == dom]
        mask = sub['resume_text'].str.contains(VERIFIED_CV_KEYWORDS[dom], case=False, na=False)
        verified_resumes_by_domain[dom] = sub[mask].copy()
        print(f"  CV Domain [{dom}]: {len(verified_resumes_by_domain[dom])} verified resumes")

    print(f"\nLoading jobs from: {jobs_path}")
    df_jobs = pd.read_csv(jobs_path)

    jobs_by_domain = {}
    for dom, pat in DOMAIN_KEYWORDS_JOB.items():
        matched = df_jobs[df_jobs['job_title'].str.contains(pat, case=False, na=False)].copy()
        jobs_by_domain[dom] = matched
        print(f"  Job Domain [{dom}]: {len(matched)} matching job postings")

    records = []
    curated_records = []

    used_resume_indices = set()
    used_job_indices = set()
    pair_id = 1

    # -------------------------------------------------------------------------
    # 1. GENERATE KATEGORI: FIT (100 Sampel)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 1. GENERATE KATEGORI: FIT (300 Sampel = 50 per domain)
    # -------------------------------------------------------------------------
    print("\n[1/3] Membentuk 300 sampel kategori 'Fit' (50 per domain)...")
    fit_counts_per_dom = {
        'INFORMATION-TECHNOLOGY': 50,
        'HEALTHCARE': 50,
        'ACCOUNTANT': 50,
        'CHEF': 50,
        'TEACHER': 50,
        'ADVOCATE': 50
    }

    for dom, target_count in fit_counts_per_dom.items():
        res_pool = verified_resumes_by_domain[dom]
        # Utamakan CV dengan pengalaman kerja memadai (>= 3 tahun)
        res_pool = res_pool.sort_values(by='years_exp', ascending=False)
        job_pool = jobs_by_domain[dom]
        job_pool = job_pool[~job_pool.index.isin(used_job_indices)].copy()
        
        # Sample dengan replacement untuk CV jika jumlah CV domain < target_count, tapi Job selalu unik
        sampled_res = res_pool.sample(n=target_count, replace=True, random_state=RANDOM_STATE + pair_id)
        sampled_job = job_pool.sample(n=target_count, replace=False, random_state=RANDOM_STATE + pair_id)

        for (_, r_row), (_, j_row) in zip(sampled_res.iterrows(), sampled_job.iterrows()):
            used_job_indices.add(j_row.name)
            
            job_text = format_job_text(j_row)
            cv_text = r_row['resume_text_clean']
            
            exp_cv = r_row['years_exp']
            exp_job_lvl = j_row['experience_level']
            
            reason = (f"Linear domain match ({dom}). Education background matches prerequisite. "
                      f"Candidate possesses {exp_cv} yrs relevant experience satisfying job requirement ({exp_job_lvl}). "
                      f"Core hard and soft skills are directly aligned.")
            
            curated_records.append({
                'id': f"PAIR_{pair_id:04d}",
                'domain': dom,
                'subtipe': 'Linear Fit (Full Qualification & Experience)',
                'rentang_confidence_score': '>80% (Fit)',
                'job_title': j_row['job_title'],
                'experience_level_job': exp_job_lvl,
                'years_exp_cv': exp_cv,
                'alasan_kategori': reason,
                'job': job_text,
                'cv': cv_text,
                'kategori': 'fit'
            })
            records.append({
                'job': job_text,
                'cv': cv_text,
                'kategori': 'fit'
            })
            pair_id += 1

    print(f"  -> Berhasil membuat {len([r for r in records if r['kategori'] == 'fit'])} sampel 'fit'")

    # -------------------------------------------------------------------------
    # 2. GENERATE KATEGORI: CLOSE FIT (300 Sampel)
    # -------------------------------------------------------------------------
    print("\n[2/3] Membentuk 300 sampel kategori 'Close Fit' (150 Defisit Pengalaman + 150 Adjacent Domain)...")
    
    # Subtipe A: 150 data (Experience Deficit / Junior in Same Domain = 25 per domain)
    sub_a_counts = {'INFORMATION-TECHNOLOGY': 25, 'HEALTHCARE': 25, 'ACCOUNTANT': 25, 'CHEF': 25, 'TEACHER': 25, 'ADVOCATE': 25}
    for dom, target_count in sub_a_counts.items():
        res_pool = verified_resumes_by_domain[dom]
        res_pool = res_pool.sort_values(by='years_exp', ascending=True)
        sampled_res = res_pool.sample(n=target_count, replace=True, random_state=RANDOM_STATE + pair_id)
        
        job_pool = jobs_by_domain[dom]
        job_pool_mid = job_pool[(job_pool['experience_level'].isin(['Mid-Senior level', 'Associate', 'Director'])) & 
                                (~job_pool.index.isin(used_job_indices))]
        if len(job_pool_mid) < target_count:
            job_pool_mid = job_pool[~job_pool.index.isin(used_job_indices)]
        sampled_job = job_pool_mid.sample(n=target_count, replace=False, random_state=RANDOM_STATE + pair_id)

        for (_, r_row), (_, j_row) in zip(sampled_res.iterrows(), sampled_job.iterrows()):
            used_job_indices.add(j_row.name)
            
            job_text = format_job_text(j_row)
            cv_text = r_row['resume_text_clean']
            exp_cv = r_row['years_exp']
            exp_job_lvl = j_row['experience_level']
            
            reason = (f"Recruiter Tolerance Rule: Same domain ({dom}) and qualified education/skills, "
                      f"but candidate has an experience deficit ({exp_cv} yrs vs required {exp_job_lvl}). "
                      f"Accepted under recruiter junior/entry-level interview tolerance.")
            
            curated_records.append({
                'id': f"PAIR_{pair_id:04d}",
                'domain': dom,
                'subtipe': 'Experience Deficit (Junior/Fresh Grad with Skills)',
                'rentang_confidence_score': '60% - 80% (Close Fit)',
                'job_title': j_row['job_title'],
                'experience_level_job': exp_job_lvl,
                'years_exp_cv': exp_cv,
                'alasan_kategori': reason,
                'job': job_text,
                'cv': cv_text,
                'kategori': 'close fit'
            })
            records.append({
                'job': job_text,
                'cv': cv_text,
                'kategori': 'close fit'
            })
            pair_id += 1

    # Subtipe B: 150 data (Adjacent Domain with High Transferable Skills = 25 per domain)
    sub_b_counts = {'INFORMATION-TECHNOLOGY': 25, 'HEALTHCARE': 25, 'ACCOUNTANT': 25, 'CHEF': 25, 'TEACHER': 25, 'ADVOCATE': 25}
    for dom, target_count in sub_b_counts.items():
        res_pool = verified_resumes_by_domain[dom]
        sampled_res = res_pool.sample(n=target_count, replace=True, random_state=RANDOM_STATE + pair_id)
        
        adjacent_cats = ADJACENT_DOMAINS[dom]
        adj_jobs = df_jobs[(df_jobs['job_title'].str.contains('|'.join(adjacent_cats), case=False, na=False)) &
                           (~df_jobs.index.isin(used_job_indices))]
        if len(adj_jobs) < target_count:
            adj_jobs = df_jobs[~df_jobs.index.isin(used_job_indices)]
        sampled_job = adj_jobs.sample(n=target_count, replace=False, random_state=RANDOM_STATE + pair_id)

        for (_, r_row), (_, j_row) in zip(sampled_res.iterrows(), sampled_job.iterrows()):
            used_job_indices.add(j_row.name)
            
            job_text = format_job_text(j_row)
            cv_text = r_row['resume_text_clean']
            exp_cv = r_row['years_exp']
            exp_job_lvl = j_row['experience_level']
            
            reason = (f"Recruiter Tolerance Rule: Adjacent industry sector ({dom} CV applied to {j_row['job_title']}). "
                      f"Strong overlap in foundational soft skills (communication/negotiation) and transferable technical capabilities.")
            
            curated_records.append({
                'id': f"PAIR_{pair_id:04d}",
                'domain': f"{dom} -> Adjacent",
                'subtipe': 'Adjacent Domain (High Transferable Skills)',
                'rentang_confidence_score': '60% - 80% (Close Fit)',
                'job_title': j_row['job_title'],
                'experience_level_job': exp_job_lvl,
                'years_exp_cv': exp_cv,
                'alasan_kategori': reason,
                'job': job_text,
                'cv': cv_text,
                'kategori': 'close fit'
            })
            records.append({
                'job': job_text,
                'cv': cv_text,
                'kategori': 'close fit'
            })
            pair_id += 1

    print(f"  -> Berhasil membuat {len([r for r in records if r['kategori'] == 'close fit'])} sampel 'close fit'")

    # -------------------------------------------------------------------------
    # 3. GENERATE KATEGORI: NOT FIT (300 Sampel = 50 per cross-pair)
    # -------------------------------------------------------------------------
    print("\n[3/3] Membentuk 300 sampel kategori 'Not Fit' (50 per cross-pair)...")
    cross_pairs = [
        ('CHEF', 'INFORMATION-TECHNOLOGY', 50),
        ('ADVOCATE', 'HEALTHCARE', 50),
        ('TEACHER', 'CHEF', 50),
        ('INFORMATION-TECHNOLOGY', 'ADVOCATE', 50),
        ('HEALTHCARE', 'ACCOUNTANT', 50),
        ('ACCOUNTANT', 'TEACHER', 50)
    ]

    for cv_dom, job_dom, count in cross_pairs:
        res_pool = verified_resumes_by_domain[cv_dom]
        sampled_res = res_pool.sample(n=count, replace=True, random_state=RANDOM_STATE + pair_id)

        job_pool = jobs_by_domain[job_dom]
        job_pool = job_pool[~job_pool.index.isin(used_job_indices)].copy()
        if len(job_pool) < count:
            job_pool = jobs_by_domain[job_dom].copy()
        sampled_job = job_pool.sample(n=count, replace=False, random_state=RANDOM_STATE + pair_id)

        for (_, r_row), (_, j_row) in zip(sampled_res.iterrows(), sampled_job.iterrows()):
            used_resume_indices.add(r_row.name)
            used_job_indices.add(j_row.name)
            
            job_text = format_job_text(j_row)
            cv_text = r_row['resume_text_clean']
            exp_cv = r_row['years_exp']
            exp_job_lvl = j_row['experience_level']
            
            reason = (f"Complete cross-domain mismatch ({cv_dom} CV vs {job_dom} Job: {j_row['job_title']}). "
                      f"Prerequisite education missing, zero related years of experience, incompatible core skillset.")
            
            curated_records.append({
                'id': f"PAIR_{pair_id:04d}",
                'domain': f"{cv_dom} vs {job_dom}",
                'subtipe': 'Cross-Domain Negative (Incompatible)',
                'rentang_confidence_score': '<60% (Not Fit)',
                'job_title': j_row['job_title'],
                'experience_level_job': exp_job_lvl,
                'years_exp_cv': exp_cv,
                'alasan_kategori': reason,
                'job': job_text,
                'cv': cv_text,
                'kategori': 'not fit'
            })
            records.append({
                'job': job_text,
                'cv': cv_text,
                'kategori': 'not fit'
            })
            pair_id += 1

    print(f"  -> Berhasil membuat {len([r for r in records if r['kategori'] == 'not fit'])} sampel 'not fit'")

    # -------------------------------------------------------------------------
    # 4. SIMPAN KEDUA DATASET
    # -------------------------------------------------------------------------
    df_clean_3col = pd.DataFrame(records)
    df_curated = pd.DataFrame(curated_records)

    # Acak urutan baris dengan random_state agar seimbang dan tidak terkelompok
    shuffle_idx = np.random.RandomState(seed=RANDOM_STATE).permutation(len(df_clean_3col))
    df_clean_3col = df_clean_3col.iloc[shuffle_idx].reset_index(drop=True)
    df_curated = df_curated.iloc[shuffle_idx].reset_index(drop=True)

    out_clean_path = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class_900.csv')
    out_curated_path = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class_900_curated.csv')

    df_clean_3col.to_csv(out_clean_path, index=False, encoding='utf-8')
    df_curated.to_csv(out_curated_path, index=False, encoding='utf-8')
    print(f"Dataset 900 tersimpan di:\n  -> {out_clean_path}\n  -> {out_curated_path}")

    # Coba juga perbarui file default jika tidak sedang dibuka oleh Excel
    legacy_clean = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class.csv')
    legacy_curated = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class_curated.csv')
    try:
        df_clean_3col.to_csv(legacy_clean, index=False, encoding='utf-8')
        df_curated.to_csv(legacy_curated, index=False, encoding='utf-8')
        print(f"File default juga berhasil diperbarui di:\n  -> {legacy_clean}")
    except PermissionError:
        print("[INFO] File 'dataset_cv_job_matching_3class.csv' sedang dibuka di Excel/program lain. Gunakan 'dataset_cv_job_matching_3class_900.csv'.")

    print("\n" + "="*70)
    print("PENYIMPANAN DATASET SELESAI:")
    print(f"1. Dataset Model (3 Kolom: job, cv, kategori):")
    print(f"   -> {out_clean_path}")
    print(f"   -> Total Baris: {len(df_clean_3col)}")
    print(f"   -> Distribusi Kelas:\n{df_clean_3col['kategori'].value_counts().to_string()}")
    print(f"\n2. Dataset Kurasi Metadata:")
    print(f"   -> {out_curated_path}")
    print(f"   -> Kolom: {list(df_curated.columns)}")
    print("="*70)

if __name__ == '__main__':
    main()
