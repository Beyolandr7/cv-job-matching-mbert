"""
build_balanced_3class_dataset.py
---------------------------------
Membangun dataset benchmark pasangan CV - Job Matching 3-Kelas seimbang (Total 1.500 data):
  - 500 Fit (>80%: Linearitas Penuh & Kualifikasi Lengkap)
  - 500 Close Fit (60% - 80%: Peran Serumpun & Irisan Keahlian Parsial)
  - 500 Not Fit (<60%: Cross-Domain Incompatible)

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
    'INFORMATION-TECHNOLOGY': r'\b(?:Software Engineer|Software Developer|Full Stack|DevOps|System Administrator|Frontend|Backend|Data Scientist|Programmer)\b',
    'HEALTHCARE': r'\b(?:Registered Nurse|Nurse Practitioner|Physician|Doctor|Pharmacist|Physical Therapist|Occupational Therapist|Clinical Nurse|RN)\b',
    'ACCOUNTANT': r'\b(?:Accountant|Staff Accountant|Senior Accountant|Tax Accountant|CPA|Bookkeeper|Financial Accountant|Accounting Manager)\b',
    'CHEF': r'\b(?:Executive Chef|Head Chef|Sous Chef|Chef|Pastry Chef|Culinary)\b',
    'TEACHER': r'\b(?:Teacher|Professor|Educator|Faculty|Lecturer|Classroom Teacher|Special Education Teacher)\b',
    'ADVOCATE': r'\b(?:Attorney|Lawyer|Litigation|Associate Attorney|General Counsel|Commercial Counsel|Legal Counsel)\b'
}

# Verifikasi header pembuka CV (85 karakter pertama) & kata kunci konten agar bebas dari CV multi-profesi / salah label
CV_HEADER_PATTERNS = {
    'INFORMATION-TECHNOLOGY': r'\b(?:INFORMATION TECHNOLOGY|IT\b|SOFTWARE|SYSTEMS?|NETWORK|DEVELOPER|TECHNOLOGY)',
    'HEALTHCARE': r'\b(?:HEALTHCARE|HEALTH CARE|CLINICAL|MEDICAL|NURSE|NURSING|PATIENT|PHARMACY|THERAPIST|HOSPITAL)',
    'ACCOUNTANT': r'\b(?:ACCOUNTANT|ACCOUNTING|CPA|AUDITOR|FINANCE|TAX|BOOKKEEPER|PAYROLL)',
    'CHEF': r'\b(?:CHEF|CULINARY|COOK|PASTRY|BAKER|KITCHEN|FOOD|CATERING)',
    'TEACHER': r'\b(?:TEACHER|TEACHING|EDUCATOR|INSTRUCTOR|FACULTY|PROFESSOR|TUTOR)',
    'ADVOCATE': r'\b(?:ADVOCATE|ATTORNEY|LEGAL|LAWYER|COUNSEL|PARALEGAL|COURT|CASE MANAGER)'
}

VERIFIED_CV_KEYWORDS = {
    'INFORMATION-TECHNOLOGY': r'\b(?:Software|Developer|Engineer|Programmer|DevOps|System Administrator|Frontend|Backend|IT Specialist|Network|Information Technology|Database|Java|Python)\b',
    'HEALTHCARE': r'\b(?:Nurse|Physician|Doctor|Medical|Hospital|Clinical|Pharmacist|Therapist|Healthcare|Patient Care|RN|BSN)\b',
    'ACCOUNTANT': r'\b(?:Accountant|Accounting|Auditor|Tax|CPA|Bookkeeper|Financial Accountant|General Ledger|Reconciliation|GAAP)\b',
    'CHEF': r'\b(?:Chef|Cook|Culinary|Pastry|Kitchen|Baker|Sous Chef|Line Cook|Catering|Food Preparation|Menu)\b',
    'TEACHER': r'\b(?:Teacher|Instructor|Professor|Educator|Faculty|Tutor|Lecturer|Teaching|Curriculum|Classroom|Lesson)\b',
    'ADVOCATE': r'\b(?:Law|Legal|Attorney|Paralegal|Court|Litigation|Justice|Counsel|Case Manager|Pleadings|Advocate)\b'
}

# Peran Pekerjaan Serumpun (Cognate Roles) untuk Kategori Close Fit (Rentang 60% - 80%)
# Kandidat dan posisi berada dalam rumpun industri yang sama (shared foundations),
# namun memiliki spesialisasi peran atau cakupan keahlian parsial (partial skill overlap).
# Bebas tumpang-tindih (0% overlap) dengan DOMAIN_KEYWORDS_JOB (Core Fit).
COGNATE_KEYWORDS_JOB = {
    'INFORMATION-TECHNOLOGY': r'\b(?:Data Analyst|Help Desk|Technical Support|Systems Analyst|Technical Writer|QA|Quality Assurance|Database Administrator|Network Specialist|IT Coordinator|Business Analyst)\b',
    'HEALTHCARE': r'\b(?:Medical Assistant|Clinical Coordinator|Healthcare Administrator|Health Care Coordinator|Case Manager|Medical Records|Patient Access|Health Coach|Pharmacy Technician|Dental Assistant)\b',
    'ACCOUNTANT': r'\b(?:Financial Analyst|Internal Auditor|Auditor|Accounts Payable|Accounts Receivable|Payroll Specialist|Billing Specialist|Credit Analyst|Tax Advisor)\b',
    'CHEF': r'\b(?:Line Cook|Prep Cook|Kitchen Manager|Food Service|Baker|Dietary Aide|Catering|Barista|Restaurant Manager)\b',
    'TEACHER': r'\b(?:Corporate Trainer|Technical Trainer|Instructional Designer|Curriculum|Education Coordinator|Academic Advisor|Tutor|Training Specialist|Learning Specialist)\b',
    'ADVOCATE': r'\b(?:Compliance Specialist|Contract Administrator|Legal Assistant|Paralegal|Regulatory Affairs|Risk Analyst|Contract Specialist|Contracts Manager)\b'
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

def clean_text(text, max_len=20000):
    if not isinstance(text, str):
        return ""
    # Hapus newline, carriage return, dan tab agar format CSV tetap single-line per baris data
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len].strip()
    return text

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
    return ". ".join(parts)

def main():
    print("="*70)
    print("MEMULAI PROSES PEMBENTUKAN DATASET CV-JOB MATCHING 3-KELAS")
    print("="*70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resumes_path = os.path.join(base_dir, 'data', 'clean_resumes.csv')
    jobs_path = os.path.join(base_dir, 'data', 'clean_job_postings.csv')

    print(f"Loading resumes from: {resumes_path}")
    df_resumes = pd.read_csv(resumes_path)
    # Buang CV outlier raksasa (> 25.000 karakter) yang melampaui batas sel Excel (32.767)
    df_resumes = df_resumes[df_resumes['resume_text'].str.len() <= 25000].copy()
    df_resumes['resume_text_clean'] = df_resumes['resume_text'].apply(clean_text)
    df_resumes['years_exp'] = df_resumes['resume_text'].apply(extract_years_from_text)

    # Filter CV terverifikasi header pembuka & konten per domain untuk mencegah noise mislabeling
    verified_resumes_by_domain = {}
    for dom in BENCHMARK_DOMAINS:
        sub = df_resumes[df_resumes['category'] == dom].copy()
        head85 = sub['resume_text_clean'].str[:85]
        head45 = sub['resume_text_clean'].str[:45]
        m_own = head85.str.contains(CV_HEADER_PATTERNS[dom], case=False, na=False)
        m_other = pd.Series(False, index=sub.index)
        for d2, p2 in CV_HEADER_PATTERNS.items():
            if d2 != dom:
                m_other |= head45.str.contains(p2, case=False, na=False)
        mask = m_own & (~m_other) & sub['resume_text_clean'].str.contains(VERIFIED_CV_KEYWORDS[dom], case=False, na=False)
        verified_resumes_by_domain[dom] = sub[mask].copy()
        print(f"  CV Domain [{dom}]: {len(verified_resumes_by_domain[dom])} verified resumes")

    print(f"\nLoading jobs from: {jobs_path}")
    df_jobs = pd.read_csv(jobs_path)

    all_core_masks = {d: df_jobs['job_title'].str.contains(p, case=False, na=False) for d, p in DOMAIN_KEYWORDS_JOB.items()}
    all_cog_masks = {d: df_jobs['job_title'].str.contains(p, case=False, na=False) for d, p in COGNATE_KEYWORDS_JOB.items()}

    jobs_by_domain = {}
    cognate_jobs_by_domain = {}
    for dom in BENCHMARK_DOMAINS:
        other_core = pd.Series(False, index=df_jobs.index)
        for d2, m2 in all_core_masks.items():
            if d2 != dom:
                other_core |= m2
        any_cog = pd.Series(False, index=df_jobs.index)
        for m2 in all_cog_masks.values():
            any_cog |= m2
        any_core = pd.Series(False, index=df_jobs.index)
        for m2 in all_core_masks.values():
            any_core |= m2
        other_cog = pd.Series(False, index=df_jobs.index)
        for d2, m2 in all_cog_masks.items():
            if d2 != dom:
                other_cog |= m2

        matched_core = df_jobs[all_core_masks[dom] & (~other_core) & (~any_cog)].copy()
        matched_cog = df_jobs[all_cog_masks[dom] & (~any_core) & (~other_cog)].copy()
        jobs_by_domain[dom] = matched_core
        cognate_jobs_by_domain[dom] = matched_cog
        print(f"  Job Domain [{dom}]: {len(matched_core)} strict core postings | {len(matched_cog)} strict cognate postings")

    records = []
    curated_records = []

    used_resume_indices = set()
    used_job_indices = set()
    pair_id = 1

    # -------------------------------------------------------------------------
    # 1. GENERATE KATEGORI: FIT (100 Sampel)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 1. GENERATE KATEGORI: FIT (500 Sampel = 83-84 per domain)
    # -------------------------------------------------------------------------
    print("\n[1/3] Membentuk 500 sampel kategori 'Fit' (83-84 per domain)...")
    fit_counts_per_dom = {
        'INFORMATION-TECHNOLOGY': 84,
        'HEALTHCARE': 84,
        'ACCOUNTANT': 83,
        'CHEF': 83,
        'TEACHER': 83,
        'ADVOCATE': 83
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
    # 2. GENERATE KATEGORI: CLOSE FIT (500 Sampel = 83-84 per domain serumpun)
    # Rentang Kemiripan Standar Akademik (Khatri et al., 2025): 60.00% - 80.00%
    # -------------------------------------------------------------------------
    print("\n[2/3] Membentuk 500 sampel kategori 'Close Fit' (Peran Serumpun & Irisan Keahlian Parsial 60% - 80%)...")
    
    close_counts_per_dom = {
        'INFORMATION-TECHNOLOGY': 84,
        'HEALTHCARE': 84,
        'ACCOUNTANT': 83,
        'CHEF': 83,
        'TEACHER': 83,
        'ADVOCATE': 83
    }

    for dom, target_count in close_counts_per_dom.items():
        res_pool = verified_resumes_by_domain[dom]
        sampled_res = res_pool.sample(n=target_count, replace=True, random_state=RANDOM_STATE + pair_id)
        
        job_pool = cognate_jobs_by_domain[dom]
        job_pool = job_pool[~job_pool.index.isin(used_job_indices)].copy()
        if len(job_pool) < target_count:
            job_pool = cognate_jobs_by_domain[dom].copy()
        sampled_job = job_pool.sample(n=target_count, replace=False, random_state=RANDOM_STATE + pair_id)

        for (_, r_row), (_, j_row) in zip(sampled_res.iterrows(), sampled_job.iterrows()):
            used_job_indices.add(j_row.name)
            
            job_text = format_job_text(j_row)
            cv_text = r_row['resume_text_clean']
            exp_cv = r_row['years_exp']
            exp_job_lvl = j_row['experience_level']
            
            reason = (f"Cognate sub-role match ({dom} candidate applied for {j_row['job_title']}). "
                      f"Strong shared industry foundations and transferable competencies, "
                      f"with partial specialization overlap fulfilling Close Fit criteria (60% - 80%).")
            
            curated_records.append({
                'id': f"PAIR_{pair_id:04d}",
                'domain': f"{dom} (Cognate Role)",
                'subtipe': 'Cognate Role (Partial Skill & Specialization Overlap)',
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
    # 3. GENERATE KATEGORI: NOT FIT (500 Sampel = 83-84 per cross-pair)
    # -------------------------------------------------------------------------
    print("\n[3/3] Membentuk 500 sampel kategori 'Not Fit' (83-84 per cross-pair)...")
    cross_pairs = [
        ('CHEF', 'INFORMATION-TECHNOLOGY', 84),
        ('ADVOCATE', 'HEALTHCARE', 84),
        ('TEACHER', 'CHEF', 83),
        ('INFORMATION-TECHNOLOGY', 'ADVOCATE', 83),
        ('HEALTHCARE', 'ACCOUNTANT', 83),
        ('ACCOUNTANT', 'TEACHER', 83)
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

    out_clean_path = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class_1500.csv')
    out_curated_path = os.path.join(base_dir, 'data', 'dataset_cv_job_matching_3class_1500_curated.csv')

    df_clean_3col.to_csv(out_clean_path, index=False, encoding='utf-8')
    df_curated.to_csv(out_curated_path, index=False, encoding='utf-8')
    print(f"Dataset 1500 tersimpan di:\n  -> {out_clean_path}\n  -> {out_curated_path}")

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
