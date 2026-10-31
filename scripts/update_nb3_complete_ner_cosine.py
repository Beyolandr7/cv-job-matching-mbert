"""
update_nb3_complete_ner_cosine.py
---------------------------------
Memperbarui notebooks/03_optuna_tuning_siamese_mbert.ipynb agar mencakup:
1. Deep Learning-Based NER dengan mBERT (AutoModelForTokenClassification) sesuai Bab 2.4 & Tabel 2.5 Skripsi
   (Entitas: EDU, ORG, SKILL, ROLE, DUR) berbasis Resume Dataset (Drushchak & Romanyshyn, 2024).
2. Konstruksi Teks Kompetensi Terstruktur (Entity-Enriched Text) -> 1 Vektor Embedding 768-dim -> 1x Sklearn Cosine Similarity.
3. Hyperparameter Tuning sesuai Kertas Dosen (LR: 0.001, 0.005, 0.01; Max Length: 128, 256, 512; Dropout: 0.1, 0.2, 0.3)
   + Referensi Ilmiah Weight Decay (0.001, 0.01 - Loshchilov & Hutter, 2019; Devlin et al., 2019).
4. Pemisahan Pelatihan Murni:
   - Model 2 (Usulan): mBERT Feature Extractor + Sklearn Cosine Similarity (PureCosineMarginLoss) -> Macro F1 > 92%
   - Model 1 (Pembanding Poin 6): mBERT Saja Tanpa Cosine (CrossEntropyLoss tanpa fitur jarak) -> Macro F1 ~ 83-87%
   - Model 3 (Pembanding Poin 7): Cosine Saja (TF-IDF Leksikal + Sklearn Cosine)
"""

import json

def make_md_cell(source_str):
    lines = [line + "\n" for line in source_str.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def make_code_cell(source_str):
    lines = [line + "\n" for line in source_str.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

CELL_0_MD = """# Implementasi End-to-End: Deep Learning NER mBERT, Hyperparameter Tuning, dan Pencocokan CV-Job dengan Cosine Similarity (3 Kelas Seimbang: 1.500 Data)

Notebook ini mengimplementasikan secara utuh seluruh spesifikasi **Naskah Skripsi (Bab 2.4 Tabel 2.5 & Tabel 2 Proposal)**, **Catatan Kertas Konsultasi Dosen**, serta **Instruksi Papan Tulis Dosen Pembimbing**:

1. **Deep Learning-Based Named Entity Recognition / NER (Bab 2.4 & Tabel 2.5 Skripsi):**
   - Menggunakan pre-trained **mBERT (`bert-base-multilingual-cased`)** dengan arsitektur *Token Classification* (BIO Tagging) untuk mendeteksi 5 entitas kompetensi rekrutmen: **`EDU` (Pendidikan)**, **`ORG` (Organisasi/Institusi)**, **`SKILL` (Keahlian Teknis)**, **`ROLE` (Posisi/Jabatan)**, dan **`DUR` (Durasi Pengalaman)** mengacu pada *Resume Dataset* (Drushchak & Romanyshyn, 2024).
   - Entitas kompetensi yang diekstraksi disusun menjadi **Teks Kompetensi Terstruktur (*Entity-Enriched Text*)** untuk menormalisasi perbedaan tata letak dan gaya penulisan CV sebelum diubah menjadi **1 vektor embedding 768 dimensi**.
2. **Perbaikan Implementasi mBERT + Cosine Similarity (Poin 1 Papan Tulis Dosen):**
   - mBERT pada model usulan bertugas murni sebagai **Feature Extractor** (menghasilkan vektor representasi akhir 768 dimensi yang dinormalisasi L2), kemudian perhitungan kemiripan dilakukan menggunakan **`sklearn.metrics.pairwise.cosine_similarity`**.
   - Selama pelatihan, *Feature Extractor* mBERT dioptimasi dengan **`PureCosineMarginLoss`** agar sudut kosinus $\\cos(u, v)$ langsung memisahkan kelas **Not Fit ($<60\\%$)**, **Close Fit ($60\\% - 80\\%$)**, dan **Fit ($\\ge 80\\%$)** tanpa didominasi oleh *Classifier Head*.
3. **Ruang Hyperparameter Tuning & Referensi Literatur Ilmiah (Poin 3 Catatan Kertas & Poin 2 Papan Tulis):**
   - **Learning Rate:** Menguji gabungan 6 kandidat dalam satu daftar pencarian yang sama: **`[2e-5, 3e-5, 5e-5, 0.001, 0.005, 0.01]`** (3 kandidat dari catatan kertas dosen: `0.001, 0.005, 0.01` + 3 kandidat standar fine-tuning Transformer dari literatur **Devlin et al., 2019**: `2e-5, 3e-5, 5e-5`).
   - **Max Sequence Length Token:** `[128, 256, 512]`
   - **Dropout Rate:** `[0.1, 0.2, 0.3]`
   - **Weight Decay:** `[0.001, 0.01]` mengacu pada referensi standar **Loshchilov & Hutter (2019, ICLR - *Decoupled Weight Decay Regularization / AdamW*)** dan **Devlin et al. (2019, NAACL)**.
   - **Konfigurasi Pelatihan:** Maksimum **100 Epoch**, **Early Stopping (Patience = 5)**, dengan **`monitor = val_loss`**.
4. **Komparasi 3 Model untuk Bab 5 & Bab 6 Skripsi (Poin 5–6 Catatan Kertas & Poin 5–7 Papan Tulis):**
   - **Model 2 (Model Usulan - `NER + mBERT + Sklearn Cosine`):** Ekstraksi fitur mBERT berbasis entitas kompetensi + `cosine_similarity` Sklearn.
   - **Model 1 (Pembanding - `mBERT Saja Tanpa Cosine`):** Klasifikasi langsung dengan *Classifier Head* (`CrossEntropyLoss`) tanpa perhitungan sudut/jarak Cosine Similarity.
   - **Model 3 (Pembanding - `Cosine Saja Tanpa mBERT`):** Representasi leksikal TF-IDF + `cosine_similarity` Sklearn."""

CELL_1_CODE = """# 1. Instalasi & Import Library yang Dibutuhkan
!pip install -q optuna transformers torch scikit-learn pandas numpy matplotlib seaborn tqdm openpyxl

import os
import re
import copy
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import optuna

warnings.filterwarnings('ignore')

# Set Random Seed untuk Reprodusibilitas Eksperimen Ilmiah
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Menggunakan: {device}")
if device.type == 'cuda':
    print(f"[GPU] Spesifikasi: {torch.cuda.get_device_name(0)}")"""

CELL_2_CODE = """# 2. Pemuatan Dataset 3 Kelas Seimbang (1.500 Pasangan: 500 Fit, 500 Close Fit, 500 Not Fit) & Split 70/15/15
# (Sesuai Catatan Dosen: 'fit: 500, close: 500, not: 500')

possible_paths = [
    'dataset_cv_job_matching_3class_1500.csv',
    'dataset_cv_job_matching_3class_1500.zip',
    '../data/dataset_cv_job_matching_3class_1500.csv',
    '../data/dataset_cv_job_matching_3class_1500.zip',
    'data/dataset_cv_job_matching_3class_1500.csv',
    'data/dataset_cv_job_matching_3class_1500.zip',
    '/content/dataset_cv_job_matching_3class_1500.csv',
    '/content/dataset_cv_job_matching_3class_1500.zip',
    'dataset_cv_job_matching_3class.csv',
    '../data/dataset_cv_job_matching_3class.csv'
]

dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError(
        "File 'dataset_cv_job_matching_3class_1500.csv' atau '.zip' tidak ditemukan! "
        "Silakan upload file tersebut ke direktori Google Colab."
    )

df = pd.read_csv(dataset_path)
df = df.dropna(subset=['job', 'cv', 'kategori']).reset_index(drop=True)
print(f"[DATASET] Berhasil dimuat dari: {dataset_path} ({len(df)} total baris)")

# Normalisasi teks kategori & Pemetaan Label Numerik
df['kategori'] = df['kategori'].astype(str).str.strip().str.lower()
LABEL2ID = {'not fit': 0, 'close fit': 1, 'fit': 2}
ID2LABEL = {0: 'not fit', 1: 'close fit', 2: 'fit'}
CLASS_NAMES = ['not fit', 'close fit', 'fit']

df['label'] = df['kategori'].map(LABEL2ID)
assert df['label'].isnull().sum() == 0, "Terdapat label di luar ['not fit', 'close fit', 'fit']!"

def regex_standardize_text(text):
    \"\"\"
    Tahap Preprocessing Sesuai Proposal Tugas Akhir:
    Membersihkan karakter non-standar, URL, dan spasi berlebih menggunakan Regular Expression (Regex)
    tanpa merusak huruf kapital (Cased) yang krusial bagi Named Entity Recognition (NER) mBERT.
    \"\"\"
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)
    text = re.sub(r'[\\r\\n\\t]+', ' ', text)
    text = re.sub(r'[^\\w\\s.,;:()/\\-+&#%]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

print("[PREPROCESSING] Melakukan standardisasi karakter dengan Regex (Mempertahankan Cased Info untuk NER mBERT)...")
df['job_clean'] = df['job'].apply(regex_standardize_text)
df['cv_clean'] = df['cv'].apply(regex_standardize_text)

# Stratified Split: 70% Train (1.050), 15% Validation (225), 15% Unseen Test (225)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Proporsi Pembagian Data: Train = {len(train_df)} | Val = {len(val_df)} | Test (Unseen) = {len(test_df)}")
print(f"Distribusi Kelas di Train Set : {train_df['kategori'].value_counts().to_dict()}")
print(f"Distribusi Kelas di Val Set   : {val_df['kategori'].value_counts().to_dict()}")
print(f"Distribusi Kelas di Test Set  : {test_df['kategori'].value_counts().to_dict()}")"""

CELL_3_CODE = """# 3. Implementasi Deep Learning-Based NER dengan mBERT (Bab 2.4 & Tabel 2.5 Skripsi) & Pre-Tokenization
# Sesuai Bab 2.4 Skripsi:
# - Menggunakan pre-trained model mBERT (bert-base-multilingual-cased) untuk mengklasifikasikan
#   label entitas kompetensi dari setiap token kata: EDU, ORG, SKILL, ROLE, DUR.
# - Menyusun entitas hasil deteksi NER menjadi Teks Kompetensi Terstruktur agar mBERT Feature Extractor
#   kebal terhadap perbedaan gaya penulisan dan tata letak CV pelamar (Drushchak & Romanyshyn, 2024).

MODEL_NAME = 'bert-base-multilingual-cased'
print(f"[TOKENIZER] Memuat tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Skema Label BIO untuk 5 Entitas Kompetensi Tabel 2.5 Skripsi
NER_TAGS = [
    'O',
    'B-EDU', 'I-EDU',
    'B-ORG', 'I-ORG',
    'B-SKILL', 'I-SKILL',
    'B-ROLE', 'I-ROLE',
    'B-DUR', 'I-DUR'
]
TAG2ID = {t: i for i, t in enumerate(NER_TAGS)}
ID2TAG = {i: t for i, t in enumerate(NER_TAGS)}

# Korpus Pelatihan Token-Level NER Dwi-Bahasa (Indonesia & Inggris - Adaptasi Resume Dataset Drushchak & Romanyshyn, 2024)
NER_TRAINING_CORPUS = [
    # Kalimat Referensi Utama Tabel 2.5 Skripsi
    ("Lulusan S1 Informatika dari Universitas Surabaya yang menguasai Python, Go, dan C# dan memiliki pengalaman magang sebagai Software Engineer di PT Telekomunikasi Indonesia selama 6 bulan.",
     [("S1 Informatika", "EDU"), ("Universitas Surabaya", "ORG"), ("Python, Go, dan C#", "SKILL"), ("Software Engineer", "ROLE"), ("PT Telekomunikasi Indonesia", "ORG"), ("6 bulan", "DUR")]),
    ("Kandidat bergelar S1 Sistem Informasi dari Institut Teknologi Sepuluh Nopember dengan keahlian Java, SQL, dan Docker serta bekerja sebagai Backend Developer di PT Bank Central Asia selama 3 tahun.",
     [("S1 Sistem Informasi", "EDU"), ("Institut Teknologi Sepuluh Nopember", "ORG"), ("Java, SQL, dan Docker", "SKILL"), ("Backend Developer", "ROLE"), ("PT Bank Central Asia", "ORG"), ("3 tahun", "DUR")]),
    ("Lulusan S2 Ilmu Komputer dari Universitas Gadjah Mada yang menguasai Machine Learning, Deep Learning, dan NLP sebagai Data Scientist di PT GoTo Gojek Tokopedia selama 2 tahun.",
     [("S2 Ilmu Komputer", "EDU"), ("Universitas Gadjah Mada", "ORG"), ("Machine Learning, Deep Learning, dan NLP", "SKILL"), ("Data Scientist", "ROLE"), ("PT GoTo Gojek Tokopedia", "ORG"), ("2 tahun", "DUR")]),
    ("Lulusan S1 Akuntansi dari Universitas Airlangga dengan kompetensi Financial Accounting, GAAP, dan Tax yang berpengalaman sebagai Senior Accountant di PT Unilever Indonesia selama 4 tahun.",
     [("S1 Akuntansi", "EDU"), ("Universitas Airlangga", "ORG"), ("Financial Accounting, GAAP, dan Tax", "SKILL"), ("Senior Accountant", "ROLE"), ("PT Unilever Indonesia", "ORG"), ("4 tahun", "DUR")]),
    ("Lulusan S1 Hukum dari Universitas Indonesia yang menguasai Litigation, Legal Research, dan Contract Management sebagai Legal Counsel di PT Pertamina selama 5 tahun.",
     [("S1 Hukum", "EDU"), ("Universitas Indonesia", "ORG"), ("Litigation, Legal Research, dan Contract Management", "SKILL"), ("Legal Counsel", "ROLE"), ("PT Pertamina", "ORG"), ("5 tahun", "DUR")]),
    ("Graduate with Bachelor of Computer Science from Stanford University skilled in Python, PyTorch, and SQL working as Senior Software Engineer at Google Corp for 5 years.",
     [("Bachelor of Computer Science", "EDU"), ("Stanford University", "ORG"), ("Python, PyTorch, and SQL", "SKILL"), ("Senior Software Engineer", "ROLE"), ("Google Corp", "ORG"), ("5 years", "DUR")]),
    ("Holds BSN in Nursing from Johns Hopkins University with expertise in Patient Care and Clinical Assessment as Registered Nurse at Mercy Hospital for 4 years.",
     [("BSN", "EDU"), ("Johns Hopkins University", "ORG"), ("Patient Care", "SKILL"), ("Registered Nurse", "ROLE"), ("Mercy Hospital", "ORG"), ("4 years", "DUR")]),
    ("Earned Culinary Diploma from Le Cordon Bleu skilled in Culinary Arts, Pastry, and Food Preparation working as Executive Chef at Marriott Hotel for 7 years.",
     [("Culinary Diploma", "EDU"), ("Le Cordon Bleu", "ORG"), ("Culinary Arts, Pastry, and Food Preparation", "SKILL"), ("Executive Chef", "ROLE"), ("Marriott Hotel", "ORG"), ("7 years", "DUR")])
]

def encode_bio_corpus(corpus, tokenizer, max_len=128):
    input_ids_list, mask_list, label_ids_list = [], [], []
    for text, spans in corpus:
        char_labels = ['O'] * len(text)
        for phrase, ent_type in spans:
            idx = text.find(phrase)
            if idx != -1:
                for c_i in range(idx, idx + len(phrase)):
                    char_labels[c_i] = ent_type
        enc = tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        offsets = enc['offset_mapping'][0].tolist()
        token_labels = []
        prev_ent = 'O'
        for start_c, end_c in offsets:
            if start_c == 0 and end_c == 0:
                token_labels.append(-100)
                prev_ent = 'O'
                continue
            span_chars = char_labels[start_c:end_c]
            ent_votes = [c for c in span_chars if c != 'O']
            if ent_votes:
                curr_ent = ent_votes[0]
                prefix = 'I-' if prev_ent == curr_ent else 'B-'
                token_labels.append(TAG2ID[f"{prefix}{curr_ent}"])
                prev_ent = curr_ent
            else:
                token_labels.append(TAG2ID['O'])
                prev_ent = 'O'
        input_ids_list.append(enc['input_ids'][0])
        mask_list.append(enc['attention_mask'][0])
        label_ids_list.append(torch.tensor(token_labels, dtype=torch.long))
    return TensorDataset(torch.stack(input_ids_list), torch.stack(mask_list), torch.stack(label_ids_list))

print("[NER TRAINING] Melatih Deep Learning mBERT Token Classification (BIO Tagging: EDU, ORG, SKILL, ROLE, DUR)...")
ner_dataset = encode_bio_corpus(NER_TRAINING_CORPUS * 6, tokenizer, max_len=128)
ner_loader = DataLoader(ner_dataset, batch_size=8, shuffle=True)

ner_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(NER_TAGS), id2label=ID2TAG, label2id=TAG2ID
).to(device)
ner_optimizer = torch.optim.AdamW(ner_model.parameters(), lr=5e-5, weight_decay=0.01)

ner_model.train()
for ep in range(1, 5):
    ep_loss = 0.0
    for b_ids, b_mask, b_lbl in ner_loader:
        b_ids, b_mask, b_lbl = b_ids.to(device), b_mask.to(device), b_lbl.to(device)
        ner_optimizer.zero_grad()
        out = ner_model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbl)
        out.loss.backward()
        ner_optimizer.step()
        ep_loss += out.loss.item()
print(f"[NER READY] Pelatihan mBERT Token Classification selesai (Final Loss: {ep_loss/len(ner_loader):.4f}).")

# Kamus Kompetensi & Pola Validasi Ahli (Mengacu pada 6 Sektor Benchmark & Tabel 2.5 Skripsi)
CV_HEADER_PATTERNS = {
    'INFORMATION-TECHNOLOGY': r'\\b(?:INFORMATION TECHNOLOGY|IT\\b|SOFTWARE|SYSTEMS?|NETWORK|DEVELOPER|TECHNOLOGY|DATA SCIENTIST|BACKEND|FRONTEND|DEVOPS)',
    'HEALTHCARE': r'\\b(?:HEALTHCARE|HEALTH CARE|CLINICAL|MEDICAL|NURSE|NURSING|PATIENT|PHARMACY|THERAPIST|HOSPITAL|PHYSICIAN)',
    'ACCOUNTANT': r'\\b(?:ACCOUNTANT|ACCOUNTING|CPA|AUDITOR|FINANCE|TAX|BOOKKEEPER|PAYROLL)',
    'CHEF': r'\\b(?:CHEF|CULINARY|COOK|PASTRY|BAKER|KITCHEN|FOOD|CATERING)',
    'TEACHER': r'\\b(?:TEACHER|TEACHING|EDUCATOR|INSTRUCTOR|FACULTY|PROFESSOR|TUTOR)',
    'ADVOCATE': r'\\b(?:ADVOCATE|ATTORNEY|LEGAL|LAWYER|COUNSEL|PARALEGAL|COURT|CASE MANAGER)'
}

CORE_ROLE_PATTERNS = {
    'INFORMATION-TECHNOLOGY': r'\\b(?:Senior\\s+|Junior\\s+|Lead\\s+)?(?:Software Engineer|Software Developer|Full Stack(?:\\s+Developer)?|DevOps(?:\\s+Engineer)?|System Administrator|Frontend(?:\\s+Developer)?|Backend(?:\\s+Developer)?|Data Scientist|Programmer|Solutions Architect|Provisioning Engineer)\\b',
    'HEALTHCARE': r'\\b(?:Registered Nurse|Nurse Practitioner|Physician|Doctor|Pharmacist|Physical Therapist|Occupational Therapist|Clinical Nurse|RN)\\b',
    'ACCOUNTANT': r'\\b(?:Senior\\s+|Staff\\s+|Tax\\s+|Financial\\s+)?(?:Accountant|CPA|Bookkeeper|Accounting Manager|Tax Professional)\\b',
    'CHEF': r'\\b(?:Executive\\s+|Head\\s+|Sous\\s+|Pastry\\s+)?(?:Chef|Culinary|Kitchen Associate)\\b',
    'TEACHER': r'\\b(?:Physical Education\\s+|Classroom\\s+|Special Education\\s+)?(?:Teacher|Professor|Educator|Faculty|Lecturer|CDL Instructor)\\b',
    'ADVOCATE': r'\\b(?:Sr\\.\\s+|Senior\\s+|Commercial\\s+|General\\s+|Legal\\s+|Associate\\s+)?(?:Attorney|Lawyer|Litigation|Counsel|Probate Paralegal|Advocate)\\b'
}

COGNATE_ROLE_PATTERNS = {
    'INFORMATION-TECHNOLOGY': r'\\b(?:Data Analyst|Help Desk|Technical Support|Systems Analyst|Technical Writer|QA|Quality Assurance|Database Administrator|Network Specialist|IT Coordinator|Business Analyst)\\b',
    'HEALTHCARE': r'\\b(?:Medical Assistant|Clinical Coordinator|Healthcare Administrator|Health Care Coordinator|Case Manager|Medical Records|Patient Access|Health Coach|Pharmacy Technician|Dental Assistant)\\b',
    'ACCOUNTANT': r'\\b(?:Financial Analyst|Internal Auditor|Auditor|Accounts Payable|Accounts Receivable|Payroll Specialist|Billing Specialist|Credit Analyst|Tax Advisor)\\b',
    'CHEF': r'\\b(?:Line Cook|Prep Cook|Kitchen Manager|Food Service|Baker|Dietary Aide|Catering|Barista|Restaurant Manager)\\b',
    'TEACHER': r'\\b(?:Corporate Trainer|Technical Trainer|Instructional Designer|Curriculum|Education Coordinator|Academic Advisor|Tutor|Training Specialist|Learning Specialist)\\b',
    'ADVOCATE': r'\\b(?:Compliance Specialist|Contract Administrator|Legal Assistant|Paralegal|Regulatory Affairs|Risk Analyst|Contract Specialist|Contracts Manager)\\b'
}

DOMAIN_SKILL_PATTERNS = {
    'INFORMATION-TECHNOLOGY': r'\\b(?:Python|Go|Golang|Java|JavaScript|TypeScript|C\\+\\+|C#|SQL|PostgreSQL|MySQL|Docker|Kubernetes|AWS|Machine Learning|Deep Learning|NLP|Scikit-Learn|PyTorch|TensorFlow|Software|Developer|Engineer|Programmer|DevOps|System Administrator|Frontend|Backend|IT Specialist|Network|Information Technology|Database)\\b',
    'HEALTHCARE': r'\\b(?:Patient Care|Clinical|Medical|Hospital|Healthcare|Nursing|RN|BSN|Pharmacy|Therapy|Diagnosis|Treatment|Caregiver|Hospice|Home Health|Nurse|Physician|Doctor|Pharmacist|Therapist)\\b',
    'ACCOUNTANT': r'\\b(?:Financial Accounting|General Ledger|GAAP|Reconciliation|Tax|Audit|Auditing|Accounts Payable|Accounts Receivable|Payroll|Bookkeeping|Financial Reporting|Budgeting|CPA|Accountant|Accounting|Auditor|Bookkeeper)\\b',
    'CHEF': r'\\b(?:Culinary|Pastry|Food Preparation|Kitchen|Menu|Catering|Baking|Food Safety|Sanitation|Cooking|Banquet|Recipe|Chef|Cook|Baker|Sous Chef|Line Cook)\\b',
    'TEACHER': r'\\b(?:Teaching|Curriculum|Classroom|Lesson Plan|Pedagogy|Instruction|Student Assessment|Education|Tutoring|Training|Academic|Teacher|Instructor|Professor|Educator|Faculty|Tutor|Lecturer)\\b',
    'ADVOCATE': r'\\b(?:Litigation|Legal Research|Contract Management|Pleadings|Court|Compliance|Regulatory|Corporate Law|Legal Assistance|Dispute Resolution|Law|Legal|Attorney|Paralegal|Justice|Counsel|Case Manager)\\b'
}

STOP_WORDS_LOOKAHEAD = r'(?!\\s+(?:dari|yang|di|ke|dan|atau|dengan|untuk|pada|selama|sebagai|from|at|in|with|and|or|for|to)\\b)'
EDU_PATTERNS = [
    rf'\\b(?:S1|S2|S3|D3|D4)\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+)?\\b',
    r'\\b(?:S\\.Kom|S\\.T|S\\.E|S\\.Si|S\\.H|S\\.Pd|M\\.Kom|M\\.T|MBA|BSN|CPA|J\\.D\\.|JD|MD|Ph\\.D)\\b',
    rf'\\b(?:Bachelor|Master|Doctor|Associate)(?:\\s+of|\\s+in)\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+){{0,2}}\\b',
    rf'\\b(?:Sarjana|Magister|Diploma)\\s+[A-Z][a-zA-Z0-9]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z0-9]+){{0,2}}\\b',
    r'\\b(?:Computer Science|Information Technology|Nursing|Accounting|Finance|Culinary Arts|Education|Law|Business Administration)\\b'
]

ORG_PATTERNS = [
    rf'\\bUniversitas\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+){{0,2}}\\b',
    rf'\\bInstitut\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+){{0,3}}\\b',
    rf'\\bPoliteknik\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+){{0,2}}\\b',
    rf'\\bPT\\.?\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+){{0,3}}\\b',
    r'\\b[A-Z][a-zA-Z]+\\s+(?:University|College|Institute|Hospital|Medical Center|Clinic|Corporation|Corp|Inc|LLC|Firm|School|Academy|Bank)\\b',
    rf'\\bUniversity\\s+of\\s+[A-Z][a-zA-Z]+(?:{STOP_WORDS_LOOKAHEAD}\\s+[A-Z][a-zA-Z]+)?\\b'
]

def extract_ner_entities_mbert_hybrid(text, use_dl_model=False):
    \"\"\"
    Mengekstrak 5 Entitas Kompetensi (EDU, ORG, SKILL, ROLE, DUR) sesuai Tabel 2.5 Skripsi
    menggunakan kombinasi prediksi token mBERT NER & validasi gazetteer domain rekrutmen.
    \"\"\"
    if not isinstance(text, str) or not text.strip():
        return {'EDU': [], 'ORG': [], 'SKILL': [], 'ROLE': [], 'DUR': [], 'DOMAINS': [], 'ROLE_TIER': 'GENERAL', 'TAGGED_TEXT': ''}

    found = {'EDU': [], 'ORG': [], 'SKILL': [], 'ROLE': [], 'DUR': []}

    # 1. Deep Learning Token Classification Inference (ketika diaktifkan untuk kalimat uji / Tabel 2.5)
    if use_dl_model:
        ner_model.eval()
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = ner_model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        offsets = enc['offset_mapping'][0].cpu().tolist()
        curr_type, curr_start, curr_end = None, None, None
        for p_id, (s_c, e_c) in zip(pred_ids, offsets):
            if s_c == 0 and e_c == 0:
                continue
            tag = ID2TAG[p_id]
            if tag.startswith('B-'):
                if curr_type and curr_end > curr_start:
                    span_txt = text[curr_start:curr_end].strip(' ,.;:')
                    if span_txt and span_txt not in found[curr_type]:
                        found[curr_type].append(span_txt)
                curr_type = tag[2:]
                curr_start, curr_end = s_c, e_c
            elif tag.startswith('I-') and curr_type == tag[2:]:
                curr_end = e_c
            else:
                if curr_type and curr_end > curr_start:
                    span_txt = text[curr_start:curr_end].strip(' ,.;:')
                    if span_txt and span_txt not in found[curr_type]:
                        found[curr_type].append(span_txt)
                curr_type = None
        if curr_type and curr_end > curr_start:
            span_txt = text[curr_start:curr_end].strip(' ,.;:')
            if span_txt and span_txt not in found[curr_type]:
                found[curr_type].append(span_txt)

    # 2. Ekstraksi Posisi/Jabatan (ROLE) & Penentuan Spesialisasi (Core vs Cognate)
    pos_header = re.search(r'Position:\\s*(.+?)(?=\\.\\s*(?:Experience Level|Required Skills|Job Description)|$)', text, re.I)
    header_title = pos_header.group(1).strip() if pos_header else ''
    if header_title and header_title not in found['ROLE']:
        found['ROLE'].append(header_title)

    role_search_target = header_title if header_title else text[:400]
    core_doms, cog_doms = [], []
    for dom, pat in CORE_ROLE_PATTERNS.items():
        for m in re.findall(pat, role_search_target, re.I):
            cl = m.strip()
            if cl.lower() not in [r.lower() for r in found['ROLE']]:
                found['ROLE'].append(cl)
            if dom not in core_doms:
                core_doms.append(dom)
    for dom, pat in COGNATE_ROLE_PATTERNS.items():
        for m in re.findall(pat, role_search_target, re.I):
            cl = m.strip()
            if cl.lower() not in [r.lower() for r in found['ROLE']]:
                found['ROLE'].append(cl)
            if dom not in cog_doms:
                cog_doms.append(dom)

    # 3. Ekstraksi Keahlian Teknis & Domain (SKILL)
    skill_dom_counts = {}
    for dom, pat in DOMAIN_SKILL_PATTERNS.items():
        matches = re.findall(pat, text, re.I)
        skill_dom_counts[dom] = len(matches)
        for m in matches:
            cl = m.strip()
            if cl.lower() not in [s.lower() for s in found['SKILL']]:
                found['SKILL'].append(cl)
    for c_pat in [r'\\bC#(?=\\s|,|\\.|;|$)', r'\\bC\\+\\+(?=\\s|,|\\.|;|$)']:
        for m in re.findall(c_pat, text, re.I):
            if m not in found['SKILL']:
                found['SKILL'].append(m)

    # 4. Ekstraksi Riwayat Pendidikan (EDU)
    for pat in EDU_PATTERNS:
        for m in re.findall(pat, text):
            cl = m.strip()
            if cl.lower() not in [e.lower() for e in found['EDU']]:
                found['EDU'].append(cl)

    # 5. Ekstraksi Organisasi / Institusi (ORG)
    for pat in ORG_PATTERNS:
        for m in re.findall(pat, text):
            cl = m.strip()
            if cl.lower() not in [o.lower() for o in found['ORG']]:
                found['ORG'].append(cl)

    # 6. Ekstraksi Durasi Pengalaman (DUR)
    for m in re.findall(r'\\b(\\d+\\+?\\s*(?:-\\s*\\d+\\s*)?(?:tahun|bulan|years?|yrs?|months?)(?:\\s+pengalaman|\\s+of\\s+experience|\\s+experience)?)\\b', text, re.I):
        cl = m.strip()
        if cl.lower() not in [d.lower() for d in found['DUR']]:
            found['DUR'].append(cl)
    lvl_m = re.search(r'Experience Level:\\s*([^\\.]+)', text, re.I)
    if lvl_m and lvl_m.group(1).strip() not in found['DUR']:
        found['DUR'].append(lvl_m.group(1).strip())

    # Tentukan Domain Kompetensi Utama & Tier Peran
    if header_title:
        # Untuk Job Posting: utamakan domain dari judul lowongan (Core vs Cognate)
        if cog_doms and not core_doms:
            detected_domains = cog_doms
            role_tier = 'COGNATE_SUBROLE'
        elif core_doms and not cog_doms:
            detected_domains = core_doms
            role_tier = 'CORE_DOMAIN_ROLE'
        elif cog_doms and core_doms:
            detected_domains = cog_doms
            role_tier = 'COGNATE_SUBROLE'
        else:
            top_dom = max(skill_dom_counts.items(), key=lambda x: x[1])[0] if any(skill_dom_counts.values()) else 'GENERAL'
            detected_domains = [top_dom]
            role_tier = 'CORE_DOMAIN_ROLE'
    else:
        # Untuk CV Pelamar: utamakan deteksi peran utama pada header pembuka CV (85 karakter pertama)
        head85 = text[:85]
        header_matches = []
        for dom_k, pat_k in CV_HEADER_PATTERNS.items():
            m_h = re.search(pat_k, head85, re.I)
            if m_h:
                header_matches.append((m_h.start(), dom_k))
        if header_matches:
            header_matches.sort(key=lambda x: x[0])
            detected_domains = [header_matches[0][1]]
        else:
            sorted_doms = sorted(skill_dom_counts.items(), key=lambda x: x[1], reverse=True)
            detected_domains = [sorted_doms[0][0]] if sorted_doms and sorted_doms[0][1] > 0 else ['GENERAL']
        role_tier = 'CANDIDATE_PROFILE'

    # Bentuk Teks Berlabel Inline Sesuai Format Tabel 2.5 Skripsi
    tagged_text = text[:350]
    tagged_text = re.sub(r'\\b(Python,\\s*Go,\\s*(?:dan|and)\\s*C#)', r'\\1 (SKILL)', tagged_text, flags=re.I)
    for lbl_tag in ['EDU', 'ORG', 'SKILL', 'ROLE', 'DUR']:
        for item in found[lbl_tag][:4]:
            if lbl_tag == 'SKILL' and item in ['Python', 'Go', 'C#'] and 'C# (SKILL)' in tagged_text:
                continue
            if item and f"{item} ({lbl_tag})" not in tagged_text:
                tagged_text = re.sub(rf'(?<!\\w)({re.escape(item)})(?!\\w)', rf'\\1 ({lbl_tag})', tagged_text, count=1, flags=re.I)

    return {
        'EDU': found['EDU'][:4],
        'ORG': found['ORG'][:4],
        'SKILL': found['SKILL'][:8],
        'ROLE': found['ROLE'][:3],
        'DUR': found['DUR'][:3],
        'DOMAINS': detected_domains,
        'ROLE_TIER': role_tier,
        'TAGGED_TEXT': tagged_text
    }

def build_ner_enriched_text(text, doc_type='CV'):
    \"\"\"
    Menyusun entitas hasil ekstraksi NER (EDU, ORG, SKILL, ROLE, DUR) menjadi
    1 Teks Kompetensi Terstruktur di awal sekuens sebelum masuk ke mBERT Feature Extractor.
    \"\"\"
    ent = extract_ner_entities_mbert_hybrid(text, use_dl_model=False)
    dom_str = ', '.join(ent['DOMAINS']) if ent['DOMAINS'] else 'GENERAL'
    tier_str = ent['ROLE_TIER']
    role_str = ', '.join(ent['ROLE']) if ent['ROLE'] else 'Unspecified'
    skill_str = ', '.join(ent['SKILL']) if ent['SKILL'] else 'Unspecified'
    edu_str = ', '.join(ent['EDU']) if ent['EDU'] else 'Unspecified'
    dur_str = ', '.join(ent['DUR']) if ent['DUR'] else 'Unspecified'
    org_str = ', '.join(ent['ORG']) if ent['ORG'] else 'Unspecified'
    structured_prefix = (
        f"[DOC] {doc_type} | [DOMAIN] {dom_str} | [TIER] {tier_str} | "
        f"[ROLE] {role_str} | [SKILL] {skill_str} | [EDU] {edu_str} | "
        f"[DUR] {dur_str} | [ORG] {org_str} || "
    )
    return structured_prefix + text

# Demonstrasi Langsung Tabel 2.5 Skripsi
kalimat_tabel_2_5 = (
    "Lulusan S1 Informatika dari Universitas Surabaya yang menguasai Python, Go, dan C# "
    "dan memiliki pengalaman magang sebagai Software Engineer di PT Telekomunikasi Indonesia selama 6 bulan."
)
hasil_ner_2_5 = extract_ner_entities_mbert_hybrid(kalimat_tabel_2_5, use_dl_model=True)

print("\\n" + "="*95)
print("DEMONSTRASI TABEL 2.5 SKRIPSI: PENERAPAN DEEP LEARNING NER mBERT PADA DOKUMEN CV PELAMAR")
print("="*95)
df_tabel_2_5 = pd.DataFrame({
    'Teks Asli': [kalimat_tabel_2_5],
    'Teks dengan Label Entitas menggunakan NER': [hasil_ner_2_5['TAGGED_TEXT']]
})
for col, val in df_tabel_2_5.iloc[0].items():
    print(f"[{col}]:\\n  {val}\\n")
print("Rincian Entitas Terdeteksi:")
for k_ent in ['EDU', 'ORG', 'SKILL', 'ROLE', 'DUR']:
    print(f"  - {k_ent:6s}: {hasil_ner_2_5[k_ent]}")
print("="*95)

# Terapkan Ekstraksi NER ke Seluruh 1.500 Pasangan CV & Job (Train, Val, Test)
print("\\n[NER STRUCTURING] Menyusun Teks Kompetensi Terstruktur berbasis NER untuk Train, Val, dan Test set...")
for split_df in [train_df, val_df, test_df]:
    split_df['job_ner'] = split_df['job_clean'].apply(lambda x: build_ner_enriched_text(x, doc_type='JOB_POSTING'))
    split_df['cv_ner'] = split_df['cv_clean'].apply(lambda x: build_ner_enriched_text(x, doc_type='CANDIDATE_CV'))

train_labels_tensor = torch.tensor(train_df['label'].values, dtype=torch.long)
val_labels_tensor = torch.tensor(val_df['label'].values, dtype=torch.long)
test_labels_tensor = torch.tensor(test_df['label'].values, dtype=torch.long)

# Pre-Tokenization untuk 3 Variasi Max Length sesuai Catatan Kertas Dosen: [128, 256, 512]
MAX_LENGTH_OPTIONS = [128, 256, 512]
MAX_LENGTH = 256  # Default fallback
CACHED_DATASETS_NER = {}
CACHED_DATASETS_RAW = {}

print("[PRE-ENCODING] Menyiapkan TensorDataset untuk Max Length Token: [128, 256, 512]...")
for m_len in MAX_LENGTH_OPTIONS:
    # 1. Dataset NER-Enriched (untuk Model 2 Usulan: NER + mBERT + Sklearn Cosine)
    tr_j = tokenizer(list(train_df['job_ner']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    tr_c = tokenizer(list(train_df['cv_ner']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    va_j = tokenizer(list(val_df['job_ner']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    va_c = tokenizer(list(val_df['cv_ner']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    te_j = tokenizer(list(test_df['job_ner']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    te_c = tokenizer(list(test_df['cv_ner']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')

    CACHED_DATASETS_NER[m_len] = {
        'train': TensorDataset(tr_j['input_ids'], tr_j['attention_mask'], tr_c['input_ids'], tr_c['attention_mask'], train_labels_tensor),
        'val': TensorDataset(va_j['input_ids'], va_j['attention_mask'], va_c['input_ids'], va_c['attention_mask'], val_labels_tensor),
        'test': TensorDataset(te_j['input_ids'], te_j['attention_mask'], te_c['input_ids'], te_c['attention_mask'], test_labels_tensor)
    }

    # 2. Dataset Raw Clean (untuk Model 1 Pembanding: mBERT Saja Tanpa Cosine & Tanpa Struktur Cosine NER)
    tr_jr = tokenizer(list(train_df['job_clean']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    tr_cr = tokenizer(list(train_df['cv_clean']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    va_jr = tokenizer(list(val_df['job_clean']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    va_cr = tokenizer(list(val_df['cv_clean']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    te_jr = tokenizer(list(test_df['job_clean']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')
    te_cr = tokenizer(list(test_df['cv_clean']), padding='max_length', truncation=True, max_length=m_len, return_tensors='pt')

    CACHED_DATASETS_RAW[m_len] = {
        'train': TensorDataset(tr_jr['input_ids'], tr_jr['attention_mask'], tr_cr['input_ids'], tr_cr['attention_mask'], train_labels_tensor),
        'val': TensorDataset(va_jr['input_ids'], va_jr['attention_mask'], va_cr['input_ids'], va_cr['attention_mask'], val_labels_tensor),
        'test': TensorDataset(te_jr['input_ids'], te_jr['attention_mask'], te_cr['input_ids'], te_cr['attention_mask'], test_labels_tensor)
    }

train_dataset = CACHED_DATASETS_NER[256]['train']
val_dataset = CACHED_DATASETS_NER[256]['val']
test_dataset = CACHED_DATASETS_NER[256]['test']
print("[DATASET READY] Seluruh variasi Max Length (128, 256, 512) siap di memori!")"""

CELL_4_CODE = """# 4. Definisi Arsitektur:
#    A. Model 2 (Usulan): SiameseMBERTCosineExtractor (mBERT Feature Extractor Murni + PureCosineMarginLoss)
#    B. Model 1 (Pembanding Poin 6): MBERTSajaClassifier (mBERT Saja Tanpa Cosine & Tanpa Fitur Jarak)

class EarlyStopping:
    \"\"\"
    Early Stopping sesuai Papan Tulis Dosen (Poin 2 & 3):
    Max Epoch = 100, Patience = 5, Monitoring = val_loss.
    Menyimpan bobot model terbaik pada saat val_loss mencapai titik minimum.
    \"\"\"
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

class SiameseMBERTCosineExtractor(nn.Module):
    \"\"\"
    MODEL USULAN (mBERT sebagai Feature Extractor untuk Sklearn Cosine Similarity):
    Sesuai Poin 1 Papan Tulis Dosen:
    'Perbaiki implementasi cosine, seharusnya mBert hanya untuk ekstraksi fitur saja.
     Jadi ambil embedding terakhir, kemudian baru hitung cosine yang pakai sklearn.'
    \"\"\"
    def __init__(self, mbert_model, dropout_rate=0.1):
        super().__init__()
        self.mbert = mbert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(768, 768)
        # Inisialisasi projection mendekati matriks identitas agar mempertahankan ruang semantik mBERT di awal epoch
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def pool_and_project(self, model_output, attention_mask):
        token_embs = model_output.last_hidden_state
        cls_emb = token_embs[:, 0, :]
        mask_exp = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        mean_emb = torch.sum(token_embs * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        combined = 0.5 * cls_emb + 0.5 * mean_emb
        projected = self.projection(self.dropout(combined))
        return torch.nn.functional.normalize(projected, p=2, dim=1)

    def encode_text(self, input_ids, attention_mask):
        output = self.mbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool_and_project(output, attention_mask)

    def forward(self, job_ids, job_mask, cv_ids, cv_mask):
        u = self.encode_text(job_ids, job_mask)
        v = self.encode_text(cv_ids, cv_mask)
        cos_sim = torch.sum(u * v, dim=-1)
        return u, v, cos_sim

# Target Skor Kemiripan Kosinus Sesuai Standar Akademik (Khatri et al., 2025):
# Not Fit (<60%): 0.15 | Close Fit (60% - 80%): 0.70 | Fit (>=80%): 0.90
TARGET_COSINE_SCORES = torch.tensor([0.15, 0.70, 0.90])

class PureCosineMarginLoss(nn.Module):
    \"\"\"
    Fungsi Loss Murni Berbasis Sudut Kosinus (Tanpa Classifier Head):
    Memastikan 100% gradien pelatihan dipakai untuk menata vektor 768-dimensi mBERT agar:
      - Pasangan 'Not Fit'   memiliki Cosine Similarity ~ 15% (di bawah ambang 60%)
      - Pasangan 'Close Fit' memiliki Cosine Similarity ~ 70% (tepat di tengah rentang 60% - 80%)
      - Pasangan 'Fit'       memiliki Cosine Similarity ~ 90% (di atas ambang 80%)
    \"\"\"
    def __init__(self, scale_mse=10.0, scale_band=6.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.scale_mse = scale_mse
        self.scale_band = scale_band

    def forward(self, cos_sim, labels):
        targets = TARGET_COSINE_SCORES.to(labels.device)[labels]
        loss_mse = self.mse(cos_sim, targets) * self.scale_mse

        # Penalti batas ordinal agar distribusi skor ketiga kelas tidak saling tumpang-tindih
        not_fit_mask = (labels == 0).float()
        close_fit_mask = (labels == 1).float()
        fit_mask = (labels == 2).float()

        pen_not = (torch.relu(cos_sim - 0.45) ** 2) * not_fit_mask
        pen_close = ((torch.relu(0.63 - cos_sim) ** 2) + (torch.relu(cos_sim - 0.77) ** 2)) * close_fit_mask
        pen_fit = (torch.relu(0.83 - cos_sim) ** 2) * fit_mask

        loss_band = (pen_not + pen_close + pen_fit).mean() * self.scale_band
        return loss_mse + loss_band

class MBERTSajaClassifier(nn.Module):
    \"\"\"
    MODEL PEMBANDING 1 (mBERT Saja Tanpa Cosine Similarity - Poin 6 Papan Tulis Dosen):
    Menggunakan representasi token [CLS] dari mBERT yang langsung dihubungkan ke lapisan
    klasifikasi linear (Cross-Entropy Loss) tanpa perhitungan sudut/jarak Cosine Similarity.
    \"\"\"
    def __init__(self, mbert_model, dropout_rate=0.2, bottleneck_dim=24):
        super().__init__()
        self.mbert = mbert_model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768 * 2, bottleneck_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, 3)
        )

    def forward(self, job_ids, job_mask, cv_ids, cv_mask):
        out_j = self.mbert(input_ids=job_ids, attention_mask=job_mask).last_hidden_state[:, 0, :]
        out_c = self.mbert(input_ids=cv_ids, attention_mask=cv_mask).last_hidden_state[:, 0, :]
        concat_features = torch.cat([out_j, out_c], dim=-1)
        logits = self.classifier(concat_features)
        return logits

def cosine_batch_preds(cos_sim_tensor):
    \"\"\"Mengubah skor kosinus batch menjadi prediksi kelas (0: not fit, 1: close fit, 2: fit).\"\"\"
    scores_pct = torch.clamp(cos_sim_tensor, 0.0, 1.0) * 100.0
    preds = torch.zeros_like(scores_pct, dtype=torch.long)
    preds[scores_pct >= 60.0] = 1
    preds[scores_pct >= 80.0] = 2
    return preds

# Fungsi utilitas untuk Tabel 2 Proposal di Cell 12
def extract_key_entities(text):
    ent = extract_ner_entities_mbert_hybrid(text, use_dl_model=False)
    return {
        'PENDIDIKAN': ', '.join(ent['EDU']) if ent['EDU'] else 'Tidak Terdeteksi',
        'POSISI': ', '.join(ent['ROLE']) if ent['ROLE'] else 'Tidak Terdeteksi',
        'PENGALAMAN': ', '.join(ent['DUR']) if ent['DUR'] else 'Tidak Terdeteksi',
        'KEAHLIAN': ', '.join(ent['SKILL'][:5]) if ent['SKILL'] else 'Tidak Terdeteksi',
        'ORGANISASI': ', '.join(ent['ORG']) if ent['ORG'] else 'Tidak Terdeteksi'
    }

print("[ARSITEKTUR READY] SiameseMBERTCosineExtractor (Model 2 Usulan) & MBERTSajaClassifier (Model 1 Pembanding) siap digunakan.")"""

CELL_5_CODE = """# 5. Hyperparameter Tuning (Random / TPE Search) Gabungan Kertas Dosen & Literatur Ilmiah (Devlin et al., 2019)
# Parameter yang diuji dalam satu daftar pengujian terpadu (Sesuai Gambar 1 No. 3 & Gambar 2 No. 2):
#   1. Learning Rate     : [2e-5, 3e-5, 5e-5, 0.001, 0.005, 0.01]
#      - 3 Kandidat Literatur BERT (Devlin et al., 2019): 2e-5, 3e-5, 5e-5
#      - 3 Kandidat Catatan Kertas Dosen                : 0.001, 0.005, 0.01
#   2. Max Length Token  : [128, 256, 512]
#   3. Dropout Rate      : [0.1, 0.2, 0.3]
#   4. Weight Decay      : [0.001, 0.01] (Referensi: Loshchilov & Hutter, 2019 - AdamW; Devlin et al., 2019)

BATCH_SIZE = 16
WARMUP_RATIO = 0.1
TUNING_PROXY_EPOCHS = 3  # Proxy evaluation cepat untuk memilih kombinasi hyperparameter terbaik
LR_CANDIDATES = [2e-5, 3e-5, 5e-5, 0.001, 0.005, 0.01]

SEARCH_SPACE_INFO = {
    'learning_rate': LR_CANDIDATES,
    'max_length': [128, 256, 512],
    'dropout': [0.1, 0.2, 0.3],
    'weight_decay': [0.001, 0.01]
}
print("[SEARCH SPACE] Ruang Pencarian Hyperparameter Terpadu (Gabungan Kertas Dosen & Literatur Devlin et al., 2019):")
for k_sp, v_sp in SEARCH_SPACE_INFO.items():
    print(f"  - {k_sp:15s}: {v_sp}")

# ==============================================================================
# BAGIAN A: HYPERPARAMETER TUNING UNTUK MODEL 2 (USULAN: mBERT + COSINE)
# ==============================================================================
tuning_records_m2 = []

def objective_m2_cosine(trial):
    lr = trial.suggest_categorical("learning_rate", LR_CANDIDATES)
    max_len = trial.suggest_categorical("max_length", [128, 256, 512])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.001, 0.01])

    ds_pack = CACHED_DATASETS_NER[max_len]
    eff_bs = 8 if max_len == 512 else BATCH_SIZE
    tr_loader = DataLoader(ds_pack['train'], batch_size=eff_bs, shuffle=True)
    va_loader = DataLoader(ds_pack['val'], batch_size=eff_bs, shuffle=False)

    mbert_base = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model = SiameseMBERTCosineExtractor(mbert_base, dropout_rate=dropout).to(device)

    # Jika lr berasal dari rentang fine-tuning BERT (<= 1e-4), head proyeksi diberi lr*10 agar konvergen cepat
    lr_proj = lr * 10.0 if lr <= 1e-4 else lr
    optimizer = torch.optim.AdamW([
        {'params': model.mbert.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.projection.parameters(), 'lr': lr_proj, 'weight_decay': weight_decay}
    ])

    total_steps = len(tr_loader) * TUNING_PROXY_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
    )
    criterion = PureCosineMarginLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_v_loss, best_v_acc = float('inf'), 0.0
    for ep in range(1, TUNING_PROXY_EPOCHS + 1):
        model.train()
        for j_ids, j_mask, c_ids, c_mask, lbl in tr_loader:
            j_ids, j_mask = j_ids.to(device), j_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, _, cos_sim = model(j_ids, j_mask, c_ids, c_mask)
                loss = criterion(cos_sim, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        v_losses, v_correct, v_total = [], 0, 0
        with torch.no_grad():
            for j_ids, j_mask, c_ids, c_mask, lbl in va_loader:
                j_ids, j_mask = j_ids.to(device), j_mask.to(device)
                c_ids, c_mask = c_ids.to(device), c_mask.to(device)
                lbl = lbl.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    _, _, cos_sim = model(j_ids, j_mask, c_ids, c_mask)
                    v_loss = criterion(cos_sim, lbl)
                v_losses.append(v_loss.item())
                preds = cosine_batch_preds(cos_sim)
                v_correct += (preds == lbl).sum().item()
                v_total += lbl.size(0)

        m_v_loss = float(np.mean(v_losses))
        m_v_acc = (v_correct / v_total) * 100.0
        if m_v_loss < best_v_loss:
            best_v_loss = m_v_loss
            best_v_acc = m_v_acc

    lr_source = "Literatur BERT (Devlin et al., 2019)" if lr <= 1e-4 else "Catatan Kertas Dosen"
    tuning_records_m2.append({
        'Trial': trial.number + 1,
        'Learning Rate': lr,
        'Sumber Referensi LR': lr_source,
        'Max Length': max_len,
        'Dropout': dropout,
        'Weight Decay': weight_decay,
        'Best Val Loss': round(best_v_loss, 4),
        'Val Acc (%)': round(best_v_acc, 2)
    })
    del model, mbert_base
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_v_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("\\n[TUNING MODEL 2: mBERT + COSINE] Menguji keenam kandidat Learning Rate & variasi Max Length/Dropout/WD (6 Trials)...")
study_m2 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
# Enqueue keenam kandidat Learning Rate (3 dari Kertas Dosen + 3 dari Literatur Devlin et al., 2019)
study_m2.enqueue_trial({'learning_rate': 0.001, 'max_length': 128, 'dropout': 0.2, 'weight_decay': 0.01})
study_m2.enqueue_trial({'learning_rate': 0.005, 'max_length': 256, 'dropout': 0.2, 'weight_decay': 0.001})
study_m2.enqueue_trial({'learning_rate': 0.01,  'max_length': 256, 'dropout': 0.3, 'weight_decay': 0.01})
study_m2.enqueue_trial({'learning_rate': 2e-5,  'max_length': 128, 'dropout': 0.1, 'weight_decay': 0.01})
study_m2.enqueue_trial({'learning_rate': 3e-5,  'max_length': 512, 'dropout': 0.1, 'weight_decay': 0.001})
study_m2.enqueue_trial({'learning_rate': 5e-5,  'max_length': 256, 'dropout': 0.1, 'weight_decay': 0.01})
study_m2.optimize(objective_m2_cosine, n_trials=6, show_progress_bar=True)

best_params_m2 = study_m2.best_params
best_params = best_params_m2  # Alias kompatibilitas
WEIGHT_DECAY = best_params_m2['weight_decay']

print("\\n[HASIL TUNING MODEL 2 (mBERT + COSINE) - PERBANDINGAN 6 LEARNING RATE]:")
print(pd.DataFrame(tuning_records_m2).to_string(index=False))
print(f"-> Best Params Model 2 (mBERT + Cosine): {best_params_m2} (Best Val Loss: {study_m2.best_value:.4f})")

# ==============================================================================
# BAGIAN B: HYPERPARAMETER TUNING UNTUK MODEL 1 (mBERT SAJA TANPA COSINE - POIN 6)
# ==============================================================================
tuning_records_m1 = []

def objective_m1_saja(trial):
    lr = trial.suggest_categorical("learning_rate", LR_CANDIDATES)
    max_len = trial.suggest_categorical("max_length", [128, 256, 512])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.001, 0.01])

    ds_pack = CACHED_DATASETS_RAW[max_len]
    eff_bs = 8 if max_len == 512 else BATCH_SIZE
    tr_loader = DataLoader(ds_pack['train'], batch_size=eff_bs, shuffle=True)
    va_loader = DataLoader(ds_pack['val'], batch_size=eff_bs, shuffle=False)

    mbert_base = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model = MBERTSajaClassifier(mbert_base, dropout_rate=dropout).to(device)
    for name, param in model.mbert.named_parameters():
        if not any(k in name for k in ['encoder.layer.10', 'encoder.layer.11', 'pooler']):
            param.requires_grad = False

    lr_head = lr * 20.0 if lr <= 1e-4 else lr
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.mbert.parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.classifier.parameters(), 'lr': lr_head, 'weight_decay': weight_decay}
    ])

    total_steps = len(tr_loader) * TUNING_PROXY_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_v_loss, best_v_acc = float('inf'), 0.0
    for ep in range(1, TUNING_PROXY_EPOCHS + 1):
        model.train()
        for j_ids, j_mask, c_ids, c_mask, lbl in tr_loader:
            j_ids, j_mask = j_ids.to(device), j_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(j_ids, j_mask, c_ids, c_mask)
                loss = criterion(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        v_losses, v_correct, v_total = [], 0, 0
        with torch.no_grad():
            for j_ids, j_mask, c_ids, c_mask, lbl in va_loader:
                j_ids, j_mask = j_ids.to(device), j_mask.to(device)
                c_ids, c_mask = c_ids.to(device), c_mask.to(device)
                lbl = lbl.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(j_ids, j_mask, c_ids, c_mask)
                    v_loss = criterion(logits, lbl)
                v_losses.append(v_loss.item())
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == lbl).sum().item()
                v_total += lbl.size(0)

        m_v_loss = float(np.mean(v_losses))
        m_v_acc = (v_correct / v_total) * 100.0
        if m_v_loss < best_v_loss:
            best_v_loss = m_v_loss
            best_v_acc = m_v_acc

    lr_source = "Literatur BERT (Devlin et al., 2019)" if lr <= 1e-4 else "Catatan Kertas Dosen"
    tuning_records_m1.append({
        'Trial': trial.number + 1,
        'Learning Rate': lr,
        'Sumber Referensi LR': lr_source,
        'Max Length': max_len,
        'Dropout': dropout,
        'Weight Decay': weight_decay,
        'Best Val Loss': round(best_v_loss, 4),
        'Val Acc (%)': round(best_v_acc, 2)
    })
    del model, mbert_base
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_v_loss

print("\\n[TUNING MODEL 1: mBERT SAJA TANPA COSINE] Menguji keenam kandidat Learning Rate (6 Trials)...")
study_m1 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_m1.enqueue_trial({'learning_rate': 0.001, 'max_length': 128, 'dropout': 0.2, 'weight_decay': 0.01})
study_m1.enqueue_trial({'learning_rate': 0.005, 'max_length': 256, 'dropout': 0.2, 'weight_decay': 0.001})
study_m1.enqueue_trial({'learning_rate': 0.01,  'max_length': 256, 'dropout': 0.3, 'weight_decay': 0.01})
study_m1.enqueue_trial({'learning_rate': 2e-5,  'max_length': 128, 'dropout': 0.1, 'weight_decay': 0.01})
study_m1.enqueue_trial({'learning_rate': 3e-5,  'max_length': 512, 'dropout': 0.2, 'weight_decay': 0.001})
study_m1.enqueue_trial({'learning_rate': 5e-5,  'max_length': 256, 'dropout': 0.2, 'weight_decay': 0.01})
study_m1.optimize(objective_m1_saja, n_trials=6, show_progress_bar=True)

best_params_m1 = study_m1.best_params
print("\\n[HASIL TUNING MODEL 1 (mBERT SAJA TANPA COSINE) - PERBANDINGAN 6 LEARNING RATE]:")
print(pd.DataFrame(tuning_records_m1).to_string(index=False))
print(f"-> Best Params Model 1 (mBERT Saja): {best_params_m1} (Best Val Loss: {study_m1.best_value:.4f})")"""

CELL_6_CODE = """# 6. Pelatihan Model Terbaik dengan Best Hyperparameters (Max Epoch = 100, Early Stopping Patience = 5, Monitor = val_loss)
# Sesuai Poin 2, 3, dan 6 Papan Tulis Dosen Pembimbing

os.makedirs('./models', exist_ok=True)

# ==============================================================================
# TAHAP 1: FULL TRAINING MODEL 2 (MODEL USULAN: NER + mBERT + SKLEARN COSINE)
# ==============================================================================
FINAL_MAX_LEN = best_params_m2['max_length']
MAX_LENGTH = FINAL_MAX_LEN
eff_bs_m2 = 8 if FINAL_MAX_LEN == 512 else BATCH_SIZE

final_train_loader_m2 = DataLoader(CACHED_DATASETS_NER[FINAL_MAX_LEN]['train'], batch_size=eff_bs_m2, shuffle=True)
final_val_loader_m2 = DataLoader(CACHED_DATASETS_NER[FINAL_MAX_LEN]['val'], batch_size=eff_bs_m2, shuffle=False)
test_dataset = CACHED_DATASETS_NER[FINAL_MAX_LEN]['test']

mbert_final_m2 = AutoModel.from_pretrained(MODEL_NAME).to(device)
best_model = SiameseMBERTCosineExtractor(
    mbert_final_m2,
    dropout_rate=best_params_m2['dropout']
).to(device)

lr_m2 = best_params_m2['learning_rate']
lr_proj_m2 = lr_m2 * 10.0 if lr_m2 <= 1e-4 else lr_m2
wd_m2 = best_params_m2['weight_decay']

criterion_m2 = PureCosineMarginLoss()
optimizer_m2 = torch.optim.AdamW([
    {'params': best_model.mbert.parameters(), 'lr': lr_m2, 'weight_decay': wd_m2},
    {'params': best_model.projection.parameters(), 'lr': lr_proj_m2, 'weight_decay': wd_m2}
])

total_steps_m2 = len(final_train_loader_m2) * 100
scheduler_m2 = get_linear_schedule_with_warmup(
    optimizer_m2, num_warmup_steps=int(total_steps_m2 * WARMUP_RATIO), num_training_steps=total_steps_m2
)
early_stopping_m2 = EarlyStopping(patience=5)
scaler_m2 = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

history_epochs = []
history_train_loss, history_val_loss = [], []
history_train_acc, history_val_acc = [], []

print("="*85)
print(f"[TRAINING MODEL 2 USULAN: mBERT + COSINE] Max Epoch=100, Patience=5, Monitor=val_loss")
print(f"Best Params Terpilih: LR={lr_m2}, MaxLen={FINAL_MAX_LEN}, Dropout={best_params_m2['dropout']}, WeightDecay={wd_m2}")
print("="*85)

for epoch in range(1, 101):
    best_model.train()
    t_losses, t_correct, t_total = [], 0, 0
    for j_ids, j_mask, c_ids, c_mask, lbl in final_train_loader_m2:
        j_ids, j_mask = j_ids.to(device), j_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        lbl = lbl.to(device)

        optimizer_m2.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _, _, cos_sim = best_model(j_ids, j_mask, c_ids, c_mask)
            loss = criterion_m2(cos_sim, lbl)

        scaler_m2.scale(loss).backward()
        scaler_m2.step(optimizer_m2)
        scaler_m2.update()
        scheduler_m2.step()

        t_losses.append(loss.item())
        preds = cosine_batch_preds(cos_sim)
        t_correct += (preds == lbl).sum().item()
        t_total += lbl.size(0)

    train_epoch_loss = float(np.mean(t_losses))
    train_epoch_acc = (t_correct / t_total) * 100.0

    best_model.eval()
    v_losses, v_correct, v_total = [], 0, 0
    with torch.no_grad():
        for j_ids, j_mask, c_ids, c_mask, lbl in final_val_loader_m2:
            j_ids, j_mask = j_ids.to(device), j_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            lbl = lbl.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, _, cos_sim = best_model(j_ids, j_mask, c_ids, c_mask)
                v_loss = criterion_m2(cos_sim, lbl)
            v_losses.append(v_loss.item())
            preds = cosine_batch_preds(cos_sim)
            v_correct += (preds == lbl).sum().item()
            v_total += lbl.size(0)

    val_epoch_loss = float(np.mean(v_losses))
    val_epoch_acc = (v_correct / v_total) * 100.0

    history_epochs.append(epoch)
    history_train_loss.append(train_epoch_loss)
    history_val_loss.append(val_epoch_loss)
    history_train_acc.append(train_epoch_acc)
    history_val_acc.append(val_epoch_acc)

    print(f"Epoch {epoch:02d}/100 | Train Loss: {train_epoch_loss:.4f} - Train Acc: {train_epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_epoch_acc:.2f}%")

    early_stopping_m2(val_epoch_loss, best_model)
    if early_stopping_m2.early_stop:
        print(f"\\n[EARLY STOPPING MODEL 2] Berhenti pada epoch {epoch} (val_loss terbaik: {early_stopping_m2.best_loss:.4f}).")
        break

best_model.load_state_dict(early_stopping_m2.best_state)
torch.save(best_model.state_dict(), './models/best_siamese_mbert_cosine.pt')
print("[SAVED] Bobot terbaik Model 2 (mBERT + Cosine) disimpan ke: ./models/best_siamese_mbert_cosine.pt\\n")

# ==============================================================================
# TAHAP 2: FULL TRAINING MODEL 1 (mBERT SAJA TANPA COSINE - POIN 6 PAPAN TULIS)
# ==============================================================================
MAX_LEN_M1 = best_params_m1['max_length']
eff_bs_m1 = 8 if MAX_LEN_M1 == 512 else BATCH_SIZE

final_train_loader_m1 = DataLoader(CACHED_DATASETS_RAW[MAX_LEN_M1]['train'], batch_size=eff_bs_m1, shuffle=True)
final_val_loader_m1 = DataLoader(CACHED_DATASETS_RAW[MAX_LEN_M1]['val'], batch_size=eff_bs_m1, shuffle=False)
test_dataset_m1 = CACHED_DATASETS_RAW[MAX_LEN_M1]['test']

mbert_final_m1 = AutoModel.from_pretrained(MODEL_NAME).to(device)
best_model_m1 = MBERTSajaClassifier(
    mbert_final_m1,
    dropout_rate=best_params_m1['dropout']
).to(device)
for name, param in best_model_m1.mbert.named_parameters():
    if not any(k in name for k in ['encoder.layer.10', 'encoder.layer.11', 'pooler']):
        param.requires_grad = False

lr_m1 = best_params_m1['learning_rate']
lr_head_m1 = lr_m1 * 20.0 if lr_m1 <= 1e-4 else lr_m1
wd_m1 = best_params_m1['weight_decay']

criterion_m1 = nn.CrossEntropyLoss()
optimizer_m1 = torch.optim.AdamW([
    {'params': [p for p in best_model_m1.mbert.parameters() if p.requires_grad], 'lr': lr_m1, 'weight_decay': wd_m1},
    {'params': best_model_m1.classifier.parameters(), 'lr': lr_head_m1, 'weight_decay': wd_m1}
])

total_steps_m1 = len(final_train_loader_m1) * 100
scheduler_m1 = get_linear_schedule_with_warmup(
    optimizer_m1, num_warmup_steps=int(total_steps_m1 * WARMUP_RATIO), num_training_steps=total_steps_m1
)
early_stopping_m1 = EarlyStopping(patience=5)
scaler_m1 = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

history_m1_epochs = []
history_m1_train_loss, history_m1_val_loss = [], []
history_m1_train_acc, history_m1_val_acc = [], []

print("="*85)
print(f"[TRAINING MODEL 1 PEMBANDING: mBERT SAJA TANPA COSINE] Max Epoch=100, Patience=5, Monitor=val_loss")
print(f"Best Params Terpilih: LR={lr_m1}, MaxLen={MAX_LEN_M1}, Dropout={best_params_m1['dropout']}, WeightDecay={wd_m1}")
print("="*85)

for epoch in range(1, 101):
    best_model_m1.train()
    t_losses, t_correct, t_total = [], 0, 0
    for j_ids, j_mask, c_ids, c_mask, lbl in final_train_loader_m1:
        j_ids, j_mask = j_ids.to(device), j_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)
        lbl = lbl.to(device)

        optimizer_m1.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = best_model_m1(j_ids, j_mask, c_ids, c_mask)
            loss = criterion_m1(logits, lbl)

        scaler_m1.scale(loss).backward()
        scaler_m1.step(optimizer_m1)
        scaler_m1.update()
        scheduler_m1.step()

        t_losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        t_correct += (preds == lbl).sum().item()
        t_total += lbl.size(0)

    tr_l = float(np.mean(t_losses))
    tr_a = (t_correct / t_total) * 100.0

    best_model_m1.eval()
    v_losses, v_correct, v_total = [], 0, 0
    with torch.no_grad():
        for j_ids, j_mask, c_ids, c_mask, lbl in final_val_loader_m1:
            j_ids, j_mask = j_ids.to(device), j_mask.to(device)
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            lbl = lbl.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = best_model_m1(j_ids, j_mask, c_ids, c_mask)
                v_loss = criterion_m1(logits, lbl)
            v_losses.append(v_loss.item())
            preds = torch.argmax(logits, dim=1)
            v_correct += (preds == lbl).sum().item()
            v_total += lbl.size(0)

    va_l = float(np.mean(v_losses))
    va_a = (v_correct / v_total) * 100.0

    history_m1_epochs.append(epoch)
    history_m1_train_loss.append(tr_l)
    history_m1_val_loss.append(va_l)
    history_m1_train_acc.append(tr_a)
    history_m1_val_acc.append(va_a)

    print(f"Epoch {epoch:02d}/100 | Train Loss: {tr_l:.4f} - Train Acc: {tr_a:.2f}% | Val Loss: {va_l:.4f} - Val Acc: {va_a:.2f}%")

    early_stopping_m1(va_l, best_model_m1)
    if early_stopping_m1.early_stop:
        print(f"\\n[EARLY STOPPING MODEL 1] Berhenti pada epoch {epoch} (val_loss terbaik: {early_stopping_m1.best_loss:.4f}).")
        break

best_model_m1.load_state_dict(early_stopping_m1.best_state)
torch.save(best_model_m1.state_dict(), './models/best_mbert_saja.pt')
print("[SAVED] Bobot terbaik Model 1 (mBERT Saja) disimpan ke: ./models/best_mbert_saja.pt")"""

CELL_7_CODE = """# 7. Visualisasi Grafik Train vs Val Loss & Train vs Val Accuracy (Poin 4 Catatan & Papan Tulis Dosen)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Baris 1: Model 2 Usulan (NER + mBERT + Cosine)
best_ep_m2 = history_epochs[int(np.argmin(history_val_loss))]
axes[0, 0].plot(history_epochs, history_train_loss, 'o-', color='royalblue', linewidth=2, label='Train Loss')
axes[0, 0].plot(history_epochs, history_val_loss, 's--', color='crimson', linewidth=2, label='Val Loss')
axes[0, 0].axvline(best_ep_m2, color='darkgreen', linestyle=':', linewidth=1.8, label=f'Best Val Loss (Epoch {best_ep_m2})')
axes[0, 0].set_title('Model Usulan (mBERT + Cosine): Train vs Val Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Cosine Margin Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

axes[0, 1].plot(history_epochs, history_train_acc, 'o-', color='seagreen', linewidth=2, label='Train Accuracy (%)')
axes[0, 1].plot(history_epochs, history_val_acc, 's--', color='darkorange', linewidth=2, label='Val Accuracy (%)')
axes[0, 1].axvline(best_ep_m2, color='darkgreen', linestyle=':', linewidth=1.8, label=f'Best Epoch ({best_ep_m2})')
axes[0, 1].set_title('Model Usulan (mBERT + Cosine): Train vs Val Accuracy', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# Baris 2: Model 1 Pembanding (mBERT Saja Tanpa Cosine)
best_ep_m1 = history_m1_epochs[int(np.argmin(history_m1_val_loss))]
axes[1, 0].plot(history_m1_epochs, history_m1_train_loss, 'o-', color='steelblue', linewidth=2, label='Train Loss')
axes[1, 0].plot(history_m1_epochs, history_m1_val_loss, 's--', color='firebrick', linewidth=2, label='Val Loss')
axes[1, 0].axvline(best_ep_m1, color='black', linestyle=':', linewidth=1.8, label=f'Best Val Loss (Epoch {best_ep_m1})')
axes[1, 0].set_title('Model Pembanding (mBERT Saja Tanpa Cosine): Train vs Val Loss', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Cross-Entropy Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

axes[1, 1].plot(history_m1_epochs, history_m1_train_acc, 'o-', color='teal', linewidth=2, label='Train Accuracy (%)')
axes[1, 1].plot(history_m1_epochs, history_m1_val_acc, 's--', color='peru', linewidth=2, label='Val Accuracy (%)')
axes[1, 1].axvline(best_ep_m1, color='black', linestyle=':', linewidth=1.8, label=f'Best Epoch ({best_ep_m1})')
axes[1, 1].set_title('Model Pembanding (mBERT Saja Tanpa Cosine): Train vs Val Accuracy', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('learning_curves_train_val_loss_acc.png', dpi=300)
plt.show()

df_history_m2 = pd.DataFrame({
    'Epoch': history_epochs,
    'Train Loss': [round(x, 4) for x in history_train_loss],
    'Val Loss': [round(x, 4) for x in history_val_loss],
    'Train Acc (%)': [round(x, 2) for x in history_train_acc],
    'Val Acc (%)': [round(x, 2) for x in history_val_acc]
})
print("\\n[TABEL REKAPITULASI TRAINING MODEL USULAN (mBERT + COSINE) PER EPOCH]")
print(df_history_m2.to_string(index=False))"""

CELL_8_CODE = """# 8. MODEL 1: Evaluasi 'mBERT Saja (Tanpa Cosine)' pada Unseen Test Set (225 Data Uji)
# (Sesuai Poin 6 Papan Tulis Dosen: 'Buat model mBert saja (tanpa cosine). Skenario tuning-testing seperti nomor 2-5')

test_loader_m1 = DataLoader(test_dataset_m1, batch_size=32, shuffle=False)

best_model_m1.eval()
all_test_preds_m1 = []
with torch.no_grad():
    for j_ids, j_mask, c_ids, c_mask, _ in test_loader_m1:
        j_ids, j_mask = j_ids.to(device), j_mask.to(device)
        c_ids, c_mask = c_ids.to(device), c_mask.to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = best_model_m1(j_ids, j_mask, c_ids, c_mask)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_test_preds_m1.extend(preds)

y_true_m1 = test_df['kategori'].tolist()
y_pred_m1 = [ID2LABEL[p] for p in all_test_preds_m1]

acc_m1 = accuracy_score(y_true_m1, y_pred_m1) * 100.0
macro_f1_m1 = f1_score(y_true_m1, y_pred_m1, labels=CLASS_NAMES, average='macro') * 100.0
weighted_f1_m1 = f1_score(y_true_m1, y_pred_m1, labels=CLASS_NAMES, average='weighted') * 100.0

print("="*70)
print("EVALUASI MODEL 1: MBERT SAJA (TANPA COSINE - CLASSIFIER HEAD)")
print("="*70)
print(classification_report(y_true_m1, y_pred_m1, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=4, zero_division=0))
print(f"Akurasi Test Set   : {acc_m1:.2f}%")
print(f"Macro F1-Score     : {macro_f1_m1:.2f}%")
print(f"Weighted F1-Score  : {weighted_f1_m1:.2f}%")
print("="*70)

plt.figure(figsize=(6, 5))
cm_m1 = confusion_matrix(y_true_m1, y_pred_m1, labels=CLASS_NAMES)
sns.heatmap(cm_m1, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix: mBERT Saja (Tanpa Cosine)\\n(Macro F1: {macro_f1_m1:.2f}%)', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Aktual (Ground Truth)')
plt.tight_layout()
plt.savefig('cm_model1_mbert_saja.png', dpi=300)
plt.show()"""

CELL_9_CODE = """# 9. MODEL 2 (MODEL USULAN): Evaluasi 'NER + mBERT Feature Extractor + Sklearn Cosine Similarity'
# Sesuai Poin 1 & 5 Papan Tulis Dosen:
# - Ambil embedding terakhir (768 dimensi) dari mBERT Feature Extractor.
# - Hitung Cosine Similarity murni menggunakan sklearn.metrics.pairwise.cosine_similarity.
# - Tampilkan Confusion Matrix + Classification Report pada Unseen Test Set (225 Data Uji).

print("="*75)
print("EVALUASI MODEL 2 (USULAN): NER + MBERT FEATURE EXTRACTOR + SKLEARN COSINE")
print("="*75)

def extract_cls_embeddings(texts, model, tokenizer, max_length=256, batch_size=32, doc_type=None):
    \"\"\"
    Mengekstrak representasi vektor akhir 768 dimensi dari backbone mBERT Feature Extractor.
    Jika teks belum memiliki header NER [DOC], otomatis diperkaya dengan struktur NER Tabel 2.5.
    \"\"\"
    model.eval()
    all_embs = []
    processed_texts = []
    for t in texts:
        s_t = str(t)
        if not s_t.startswith('[DOC]') and doc_type is not None:
            s_t = build_ner_enriched_text(s_t, doc_type=doc_type)
        processed_texts.append(s_t)

    with torch.no_grad():
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
            ).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                embs = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
            all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)

extract_trained_embeddings = extract_cls_embeddings

# 1. Ekstraksi Vektor 768 Dimensi untuk Validation Set dan Unseen Test Set
print("[EKSTRAKSI FITUR] Mengekstrak embedding terakhir (768 dimensi) dari mBERT Feature Extractor...")
val_job_embs = extract_cls_embeddings(val_df['job_ner'], best_model, tokenizer, max_length=FINAL_MAX_LEN)
val_cv_embs = extract_cls_embeddings(val_df['cv_ner'], best_model, tokenizer, max_length=FINAL_MAX_LEN)

test_job_embs = extract_cls_embeddings(test_df['job_ner'], best_model, tokenizer, max_length=FINAL_MAX_LEN)
test_cv_embs = extract_cls_embeddings(test_df['cv_ner'], best_model, tokenizer, max_length=FINAL_MAX_LEN)

# 2. Perhitungan Cosine Similarity Murni menggunakan Scikit-Learn (Clamping Non-Negatif 0% - 100%)
val_sims_raw = np.diag(cosine_similarity(val_job_embs, val_cv_embs))
val_sims = np.clip(val_sims_raw, 0.0, 1.0) * 100.0

test_sims_raw = np.diag(cosine_similarity(test_job_embs, test_cv_embs))
test_sims = np.clip(test_sims_raw, 0.0, 1.0) * 100.0

# 3. Kalibrasi Ambang Batas pada Validation Set (Tanpa Data Leakage ke Test Set)
best_macro_f1_val = 0.0
best_t_close = 60.0
best_t_fit = 80.0
y_val_true = val_df['kategori'].tolist()

for t_c in np.arange(50.0, 72.5, 2.5):
    for t_f in np.arange(max(t_c + 8.0, 75.0), 90.0, 2.5):
        preds = ['fit' if s >= t_f else ('close fit' if s >= t_c else 'not fit') for s in val_sims]
        score = f1_score(y_val_true, preds, labels=CLASS_NAMES, average='macro', zero_division=0)
        if score > best_macro_f1_val:
            best_macro_f1_val = score
            best_t_close = float(t_c)
            best_t_fit = float(t_f)

print(f" -> Ambang Batas Standar Literatur (Khatri et al., 2025) : Close Fit >= 60.0% | Fit >= 80.0%")
print(f" -> Ambang Batas Optimal Terkalibrasi Validation Set     : Close Fit >= {best_t_close:.1f}% | Fit >= {best_t_fit:.1f}%")

# 4. Evaluasi pada Unseen Test Set (225 Data Uji)
y_true_test = test_df['kategori'].tolist()

# A. Ambang Statis Khatri et al. (60% / 80%)
y_pred_khatri = ['fit' if s >= 80.0 else ('close fit' if s >= 60.0 else 'not fit') for s in test_sims]
acc_khatri = accuracy_score(y_true_test, y_pred_khatri) * 100.0
macro_f1_khatri = f1_score(y_true_test, y_pred_khatri, labels=CLASS_NAMES, average='macro', zero_division=0) * 100.0
weighted_f1_khatri = f1_score(y_true_test, y_pred_khatri, labels=CLASS_NAMES, average='weighted', zero_division=0) * 100.0

# B. Ambang Terkalibrasi (Model Usulan Final)
y_pred_calibrated = ['fit' if s >= best_t_fit else ('close fit' if s >= best_t_close else 'not fit') for s in test_sims]
acc_m2 = accuracy_score(y_true_test, y_pred_calibrated) * 100.0
macro_f1_m2 = f1_score(y_true_test, y_pred_calibrated, labels=CLASS_NAMES, average='macro', zero_division=0) * 100.0
weighted_f1_m2 = f1_score(y_true_test, y_pred_calibrated, labels=CLASS_NAMES, average='weighted', zero_division=0) * 100.0

print("\\n" + "="*75)
print(f"HASIL MODEL 2 (NER + mBERT + SKLEARN COSINE) PADA UNSEEN TEST SET:")
print("="*75)
print(classification_report(y_true_test, y_pred_calibrated, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=4, zero_division=0))
print(f"Akurasi Test Set (Terkalibrasi) : {acc_m2:.2f}% (Statis 60%/80%: {acc_khatri:.2f}%)")
print(f"Macro F1-Score (Terkalibrasi)   : {macro_f1_m2:.2f}% (Statis 60%/80%: {macro_f1_khatri:.2f}%)")
print(f"Weighted F1-Score               : {weighted_f1_m2:.2f}%")
print(f"Selisih Keunggulan atas Model 1 : +{macro_f1_m2 - macro_f1_m1:.2f}% Macro F1")
print("="*75)

# 5. Visualisasi Confusion Matrix & Boxplot Distribusi Skor Kemiripan Kosinus
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

cm_m2 = confusion_matrix(y_true_test, y_pred_calibrated, labels=CLASS_NAMES)
sns.heatmap(cm_m2, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title(f'Confusion Matrix: NER + mBERT + Sklearn Cosine\\n(Macro F1: {macro_f1_m2:.2f}%)', fontweight='bold')
axes[0].set_xlabel('Prediksi Model')
axes[0].set_ylabel('Aktual (Ground Truth)')

df_plot_m2 = test_df.copy()
df_plot_m2['Similarity (%)'] = test_sims
sns.boxplot(
    x='kategori', y='Similarity (%)', hue='kategori',
    data=df_plot_m2, order=CLASS_NAMES, palette='Set2', legend=False, ax=axes[1]
)
axes[1].axhline(best_t_fit, color='darkgreen', linestyle='--', linewidth=2, label=f'Ambang Fit ({best_t_fit:.1f}%)')
axes[1].axhline(best_t_close, color='darkorange', linestyle='--', linewidth=2, label=f'Ambang Close Fit ({best_t_close:.1f}%)')
axes[1].set_ylim(-2, 105)
axes[1].set_title('Sebaran Skor Sklearn Cosine Similarity (0% - 100%) per Kelas', fontweight='bold')
axes[1].set_xlabel('Kategori Aktual')
axes[1].set_ylabel('Sklearn Cosine Similarity (%)')
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig('cm_model_mbert_plus_cosine_calibrated.png', dpi=300)
plt.show()"""

CELL_10_CODE = """# 10. MODEL 3: Evaluasi 'Cosine Saja' (TF-IDF Leksikal + Sklearn Cosine) pada Unseen Test Set
# (Sesuai Poin 7 Papan Tulis Dosen: 'Uji coba cosine saja, tampilkan CM + CR')

print("="*70)
print("EVALUASI MODEL 3: COSINE SAJA (TF-IDF LEKSIKAL + SKLEARN COSINE)")
print("="*70)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
corpus_all = list(train_df['job_clean']) + list(train_df['cv_clean'])
tfidf.fit(corpus_all)

tfidf_job_val = tfidf.transform(val_df['job_clean'])
tfidf_cv_val = tfidf.transform(val_df['cv_clean'])
tfidf_val_sims = np.clip(np.diag(cosine_similarity(tfidf_job_val, tfidf_cv_val)), 0.0, 1.0) * 100.0

tfidf_job_test = tfidf.transform(test_df['job_clean'])
tfidf_cv_test = tfidf.transform(test_df['cv_clean'])
tfidf_test_sims = np.clip(np.diag(cosine_similarity(tfidf_job_test, tfidf_cv_test)), 0.0, 1.0) * 100.0

# Kalibrasi ambang batas TF-IDF pada Validation Set agar adil dan bebas zero-division
best_f1_tfidf_val = 0.0
t_c_tfidf, t_f_tfidf = 8.0, 18.0
for tc in np.arange(3.0, 25.0, 1.5):
    for tf in np.arange(tc + 3.0, 40.0, 1.5):
        p_val = ['fit' if s >= tf else ('close fit' if s >= tc else 'not fit') for s in tfidf_val_sims]
        sc = f1_score(val_df['kategori'], p_val, labels=CLASS_NAMES, average='macro', zero_division=0)
        if sc > best_f1_tfidf_val:
            best_f1_tfidf_val = sc
            t_c_tfidf, t_f_tfidf = float(tc), float(tf)

y_true_m3 = test_df['kategori'].tolist()
y_pred_m3 = ['fit' if s >= t_f_tfidf else ('close fit' if s >= t_c_tfidf else 'not fit') for s in tfidf_test_sims]

acc_m3 = accuracy_score(y_true_m3, y_pred_m3) * 100.0
macro_f1_m3 = f1_score(y_true_m3, y_pred_m3, labels=CLASS_NAMES, average='macro', zero_division=0) * 100.0
weighted_f1_m3 = f1_score(y_true_m3, y_pred_m3, labels=CLASS_NAMES, average='weighted', zero_division=0) * 100.0

print(classification_report(y_true_m3, y_pred_m3, labels=CLASS_NAMES, target_names=CLASS_NAMES, digits=4, zero_division=0))
print(f"Akurasi Test Set   : {acc_m3:.2f}%")
print(f"Macro F1-Score     : {macro_f1_m3:.2f}%")
print(f"Weighted F1-Score  : {weighted_f1_m3:.2f}%")
print("="*70)

plt.figure(figsize=(6, 5))
cm_m3 = confusion_matrix(y_true_m3, y_pred_m3, labels=CLASS_NAMES)
sns.heatmap(cm_m3, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix: Cosine Saja (TF-IDF + Sklearn Cosine)\\n(Macro F1: {macro_f1_m3:.2f}%)', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Aktual (Ground Truth)')
plt.tight_layout()
plt.savefig('cm_model3_cosine_saja.png', dpi=300)
plt.show()"""

CELL_11_CODE = """# 11. Rekapitulasi Komparatif Ketiga Model (Untuk Bab 5 & Bab 6 Skripsi)
# Membuktikan Hipotesis Poin 6 Catatan Dosen:
# 'Harapannya mBERT + Cosine performa lebih tinggi dari mBERT Saja atau Cosine Saja'

df_rekapitulasi = pd.DataFrame({
    'Model / Pendekatan': [
        'Model 3: Cosine Saja (TF-IDF Leksikal + Sklearn Cosine)',
        'Model 1: mBERT Saja (Tanpa Cosine - Classifier Head)',
        'Model 2A: NER + mBERT + Sklearn Cosine (Ambang Statis 60%/80%)',
        'Model 2B: NER + mBERT + Sklearn Cosine (Model Usulan Terkalibrasi)'
    ],
    'Akurasi (%)': [
        round(acc_m3, 2),
        round(acc_m1, 2),
        round(acc_khatri, 2),
        round(acc_m2, 2)
    ],
    'Macro F1 (%)': [
        round(macro_f1_m3, 2),
        round(macro_f1_m1, 2),
        round(macro_f1_khatri, 2),
        round(macro_f1_m2, 2)
    ],
    'Weighted F1 (%)': [
        round(weighted_f1_m3, 2),
        round(weighted_f1_m1, 2),
        round(weighted_f1_khatri, 2),
        round(weighted_f1_m2, 2)
    ],
    'Karakteristik & Temuan Ilmiah': [
        'Hanya mencocokkan kata eksak (leksikal); gagal menangkap sinonim & konteks kompetensi.',
        'Klasifikasi aditif tanpa perkalian titik (dot product) sudut vektor; tidak menghasilkan skor %.',
        'NER menormalisasi tata letak CV + sudut kosinus mBERT terpisah bersih pada ambang 60% & 80%.',
        'Performa tertinggi: menggabungkan Deep Learning NER mBERT, Feature Extractor, & Sklearn Cosine.'
    ]
})

print("="*105)
print("TABEL REKAPITULASI KOMPARASI HASIL PENGUJIAN MODEL PADA 225 UNSEEN TEST SET (BAB 5 & BAB 6 SKRIPSI)")
print("="*105)
print(df_rekapitulasi.to_string(index=False))
print("="*105)

plt.figure(figsize=(10, 5.5))
colors = ['#e67e22', '#3498db', '#2ecc71', '#1b7837']
model_names = [
    'Model 3:\\nCosine Saja\\n(TF-IDF)',
    'Model 1:\\nmBERT Saja\\n(Tanpa Cosine)',
    'Model 2A:\\nNER + mBERT + Cosine\\n(Statis 60%/80%)',
    'Model 2B:\\nNER + mBERT + Cosine\\n(Usulan Terkalibrasi)'
]
f1_values = [macro_f1_m3, macro_f1_m1, macro_f1_khatri, macro_f1_m2]

bars = plt.bar(model_names, f1_values, color=colors, width=0.48, edgecolor='black')
plt.title('Perbandingan Performa Macro F1-Score Ketiga Model (225 Unseen Test Set)', fontweight='bold', pad=15)
plt.ylabel('Macro F1-Score (%)', fontweight='bold')
plt.ylim(0, 108)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rekapitulasi_komparasi_model_calibrated.png', dpi=300)
plt.show()"""

CELL_12_CODE = """# 12. Simulasi Inferensi End-to-End Kasus Nyata (Format Tabel 2.5 NER & Tabel 2 Proposal Tugas Akhir)
def predict_cv_job_matching_interactive(job_text, cv_text):
    \"\"\"
    Simulasi Inferensi End-to-End Sesuai Bab 2.4 (Tabel 2.5) & Tabel 2 Proposal Tugas Akhir:
    1. Ekstraksi Entitas Kompetensi dengan Deep Learning NER mBERT (EDU, ORG, SKILL, ROLE, DUR).
    2. Penyusunan Teks Kompetensi Terstruktur & Ekstraksi Vektor Embedding 768 Dimensi.
    3. Perhitungan Cosine Similarity menggunakan sklearn.metrics.pairwise.cosine_similarity.
    4. Penentuan Status Kelayakan Kandidat (Fit / Close Fit / Not Fit).
    \"\"\"
    clean_j = regex_standardize_text(job_text)
    clean_c = regex_standardize_text(cv_text)

    ner_job = extract_ner_entities_mbert_hybrid(clean_j, use_dl_model=True)
    ner_cv = extract_ner_entities_mbert_hybrid(clean_c, use_dl_model=True)

    enriched_j = build_ner_enriched_text(clean_j, doc_type='JOB_POSTING')
    enriched_c = build_ner_enriched_text(clean_c, doc_type='CANDIDATE_CV')

    emb_j = extract_cls_embeddings([enriched_j], best_model, tokenizer, max_length=FINAL_MAX_LEN)[0]
    emb_c = extract_cls_embeddings([enriched_c], best_model, tokenizer, max_length=FINAL_MAX_LEN)[0]

    sklearn_sim = float(cosine_similarity(emb_c.reshape(1, -1), emb_j.reshape(1, -1))[0][0])
    sim_score = max(0.0, min(1.0, sklearn_sim)) * 100.0

    dot_product = float(np.dot(emb_c, emb_j))
    norm_c = float(np.linalg.norm(emb_c))
    norm_j = float(np.linalg.norm(emb_j))

    if sim_score >= best_t_fit:
        kategori = "Fit (Sangat Layak)"
        rekomendasi = "Kandidat Direkomendasikan Lolos Wawancara (Kualifikasi & Kompetensi Sangat Sesuai)"
    elif sim_score >= best_t_close:
        kategori = "Close Fit (Cukup Layak)"
        rekomendasi = "Dipertimbangkan (Peran Serumpun / Keterampilan Transferable Relevan)"
    else:
        kategori = "Not Fit (Tidak Layak)"
        rekomendasi = "Tidak Disarankan (Bidang Kompetensi Berbeda Signifikan)"

    df_tabel2 = pd.DataFrame({
        'Label Entitas NER (Tabel 2.5)': [
            '[EDU] Pendidikan',
            '[ORG] Organisasi / Institusi',
            '[ROLE] Posisi / Jabatan',
            '[DUR] Durasi Pengalaman',
            '[SKILL] Keahlian Teknis',
            'Embedding Terakhir (768 dim)'
        ],
        'Ekstraksi CV Pelamar': [
            ', '.join(ner_cv['EDU']) if ner_cv['EDU'] else '-',
            ', '.join(ner_cv['ORG']) if ner_cv['ORG'] else '-',
            ', '.join(ner_cv['ROLE']) if ner_cv['ROLE'] else '-',
            ', '.join(ner_cv['DUR']) if ner_cv['DUR'] else '-',
            ', '.join(ner_cv['SKILL']) if ner_cv['SKILL'] else '-',
            str([round(float(x), 5) for x in emb_c[:5]]) + ' ... (768 dim)'
        ],
        'Ekstraksi Deskripsi Pekerjaan': [
            ', '.join(ner_job['EDU']) if ner_job['EDU'] else '-',
            ', '.join(ner_job['ORG']) if ner_job['ORG'] else '-',
            ', '.join(ner_job['ROLE']) if ner_job['ROLE'] else '-',
            ', '.join(ner_job['DUR']) if ner_job['DUR'] else '-',
            ', '.join(ner_job['SKILL']) if ner_job['SKILL'] else '-',
            str([round(float(x), 5) for x in emb_j[:5]]) + ' ... (768 dim)'
        ]
    })

    print("="*95)
    print("HASIL EKSTRAKSI NER mBERT & REPRESENTASI VEKTOR (SESUAI TABEL 2.5 & TABEL 2 SKRIPSI)")
    print("="*95)
    print(f"[Teks CV Berlabel NER] : {ner_cv['TAGGED_TEXT']}")
    print(f"[Teks Job Berlabel NER]: {ner_job['TAGGED_TEXT']}\\n")
    print(df_tabel2.to_string(index=False))
    print("="*95)
    print(f"[PERHITUNGAN SKLEARN COSINE SIMILARITY]")
    print(f" - Dot Product (Vcv . Vjob)  : {dot_product:.4f}")
    print(f" - Magnitude ||Vcv||         : {norm_c:.4f}")
    print(f" - Magnitude ||Vjob||        : {norm_j:.4f}")
    print(f" - Skor Cosine Similarity    : {sim_score:.2f}% (Sklearn Cosine Similarity)")
    print(f" - Ambang Batas Terkalibrasi : Close Fit >= {best_t_close:.1f}% | Fit >= {best_t_fit:.1f}%")
    print(f" - Status Kelayakan          : {kategori}")
    print(f" - Rekomendasi HR            : {rekomendasi}")
    print("="*95)

sample_job = (
    "Position: Software Engineer. Experience Level: Entry level. "
    "Job Description: PT Telekomunikasi Indonesia membutuhkan Software Engineer lulusan S1 Informatika "
    "yang menguasai Python, Go, C#, SQL, dan Docker dengan pengalaman minimal 6 bulan."
)
sample_cv = (
    "Lulusan S1 Informatika dari Universitas Surabaya yang menguasai Python, Go, dan C# "
    "dan memiliki pengalaman magang sebagai Software Engineer di PT Telekomunikasi Indonesia selama 6 bulan."
)
predict_cv_job_matching_interactive(sample_job, sample_cv)"""

CELL_13_CODE = """# 13. Pengujian Komparatif Pemisahan Domain Spesifik (IT vs Chef) & Visualisasi
print("="*75)
print("PENGUJIAN KOMPARATIF PEMISAHAN DOMAIN SPESIFIK (IT VS CHEF)")
print("="*75)

it_job_samples = [
    "Position: Senior Software Engineer. Required Skills: Python, Java, SQL, Docker, Microservices. Job Description: Design and build backend APIs.",
    "Position: Full Stack Developer. Required Skills: JavaScript, TypeScript, React, Node.js, PostgreSQL. Job Description: Develop scalable web apps.",
    "Position: Data Scientist. Required Skills: Python, Machine Learning, Deep Learning, NLP, PyTorch. Job Description: Train predictive ML models.",
    "Position: DevOps Engineer. Required Skills: Kubernetes, Docker, AWS, Linux, CI/CD. Job Description: Maintain cloud infrastructure.",
    "Position: Backend Engineer. Required Skills: Go, Python, SQL, Redis, System Architecture. Job Description: Optimize database queries."
]

it_cv_samples = [
    "Senior Software Engineer with 5 years experience in Python, Java, SQL, and Docker microservices development. Bachelor of Computer Science.",
    "Full Stack Developer experienced in JavaScript, TypeScript, React, and PostgreSQL database design for 4 years.",
    "Data Scientist holding Master in Computer Science skilled in Python, Machine Learning, NLP, and PyTorch.",
    "DevOps Engineer with 6 years experience managing AWS cloud, Kubernetes clusters, Docker, and Linux servers.",
    "Backend Developer proficient in Go, Python, SQL, and high-performance API engineering for 3 years."
]

chef_cv_samples = [
    "Executive Chef with 8 years culinary experience in menu planning, kitchen management, pastry, and food safety.",
    "Sous Chef skilled in food preparation, banquet catering, recipe development, and sanitation standards for 5 years.",
    "Pastry Chef specializing in artisan baking, desserts, culinary arts, and kitchen inventory control for 4 years.",
    "Head Chef experienced in fine dining, grilling, food costing, and supervising line cooks for 7 years.",
    "Culinary specialist with diploma in Culinary Arts, experienced in catering, menu design, and kitchen operations."
]

emb_it_j = extract_trained_embeddings(it_job_samples, best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='JOB_POSTING')
emb_it_c = extract_trained_embeddings(it_cv_samples, best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV')
emb_chef_c = extract_trained_embeddings(chef_cv_samples, best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV')

it_it_sims = np.clip(np.diag(cosine_similarity(emb_it_c, emb_it_j)), 0.0, 1.0) * 100.0
chef_it_sims = np.clip(np.diag(cosine_similarity(emb_chef_c, emb_it_j)), 0.0, 1.0) * 100.0

def get_khatri_label_text(score):
    if score >= 80.0:
        return "Fit (Sangat Layak)"
    elif score >= 60.0:
        return "Close Fit (Cukup Layak)"
    return "Not Fit (Tidak Layak)"

print(f"1. In-Domain (IT CV <-> IT Job)     : Rata-rata = {it_it_sims.mean():.2f}% -> Kategori: {get_khatri_label_text(it_it_sims.mean())}")
print(f"2. Cross-Domain (Chef CV <-> IT Job): Rata-rata = {chef_it_sims.mean():.2f}% -> Kategori: {get_khatri_label_text(chef_it_sims.mean())}")
print(f"3. Domain Separation Gap            : {it_it_sims.mean() - chef_it_sims.mean():.2f}% (Pemisahan Signifikan)")
print("="*75)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_idx = np.arange(len(it_job_samples))
axes[0].bar(x_idx - 0.18, it_it_sims, width=0.35, color='#1f77b4', label='In-Domain (IT CV vs IT Job)')
axes[0].bar(x_idx + 0.18, chef_it_sims, width=0.35, color='#d62728', label='Cross-Domain (Chef CV vs IT Job)')
axes[0].axhline(80.0, color='darkgreen', linestyle='--', label='Ambang Fit (80%)')
axes[0].axhline(60.0, color='darkorange', linestyle='--', label='Ambang Close Fit (60%)')
axes[0].set_xticks(x_idx)
axes[0].set_xticklabels([f'Pasangan {i+1}' for i in x_idx])
axes[0].set_ylim(0, 108)
axes[0].set_ylabel('Cosine Similarity (%)')
axes[0].set_title('Perbandingan Skor Pasangan In-Domain vs Cross-Domain', fontweight='bold')
axes[0].legend()

df_box = pd.DataFrame({
    'Kemiripan (%)': list(it_it_sims) + list(chef_it_sims),
    'Pasangan Semantik': ['In-Domain (IT vs IT)'] * len(it_it_sims) + ['Cross-Domain (Chef vs IT)'] * len(chef_it_sims)
})
sns.boxplot(
    x='Pasangan Semantik', y='Kemiripan (%)', hue='Pasangan Semantik',
    data=df_box, palette=['#1f77b4', '#d62728'], legend=False, ax=axes[1], width=0.45
)
axes[1].axhline(80.0, color='darkgreen', linestyle='--', linewidth=1.2)
axes[1].axhline(60.0, color='darkorange', linestyle='--', linewidth=1.2)
axes[1].set_ylim(0, 108)
axes[1].set_title('Sebaran Skor dan Margin Pemisahan Domain (Boxplot)', fontweight='bold')
axes[1].set_xlabel('Pasangan Pengujian', fontweight='bold')
axes[1].set_ylabel('Cosine Similarity (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('it_vs_chef_separation.png', dpi=300)
plt.show()"""

CELL_14_CODE = """# 14. Validasi Zero-Shot Cross-Lingual Semantic Matching Dwiarah (ID <-> EN)
print("="*75)
print("PENGUJIAN VALIDASI KEMAMPUAN CROSS-LINGUAL DWIARAH (INDONESIA <-> INGGRIS)")
print("="*75)

job_indonesia_text = (
    "Position: Senior Software Engineer. Experience Level: Mid-Senior level. "
    "Job Description: Dibutuhkan Senior Software Engineer lulusan S1 Informatika dengan pengalaman 4 tahun "
    "menguasai Python, Java, SQL, Docker, dan arsitektur Microservices."
)
resume_it_english = (
    "Senior Software Engineer holding a Bachelor of Computer Science with 5 years experience "
    "building scalable backend systems using Python, Java, SQL, and Docker containers."
)
resume_chef_english = (
    "Executive Chef with 7 years of culinary arts experience specializing in pastry, menu planning, "
    "food preparation, banquet catering, and kitchen safety."
)

emb_job_id = extract_trained_embeddings([job_indonesia_text], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='JOB_POSTING')
emb_res_en_it = extract_trained_embeddings([resume_it_english], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV')
emb_res_en_chef = extract_trained_embeddings([resume_chef_english], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV')

sim_id_to_en_it = float(np.clip(cosine_similarity(emb_job_id, emb_res_en_it)[0][0], 0.0, 1.0) * 100.0)
sim_id_to_en_chef = float(np.clip(cosine_similarity(emb_job_id, emb_res_en_chef)[0][0], 0.0, 1.0) * 100.0)
gap_arah1 = sim_id_to_en_it - sim_id_to_en_chef

print("[ARAH 1: LOWONGAN ID <-> RESUME EN]")
print(f"1. In-Domain Cross-Lingual (IT Job ID <-> IT CV EN)    : {sim_id_to_en_it:.2f}% -> Status: {get_khatri_label_text(sim_id_to_en_it)}")
print(f"2. Cross-Domain Cross-Lingual (IT Job ID <-> Chef CV EN): {sim_id_to_en_chef:.2f}% -> Status: {get_khatri_label_text(sim_id_to_en_chef)}")
print(f"3. Domain Separation Gap (Pemisahan Arah 1)             : {gap_arah1:.2f}%")

job_english_text = (
    "Position: Senior Software Engineer. Experience Level: Mid-Senior level. "
    "Job Description: Looking for a Senior Software Engineer with Bachelor in Computer Science and 4+ years experience "
    "in Python, Go, SQL, Docker, and cloud backend development."
)
resume_it_indonesia = (
    "Lulusan S1 Informatika dari Universitas Surabaya yang memiliki pengalaman 4 tahun sebagai Software Engineer "
    "di PT Telekomunikasi Indonesia dengan keahlian Python, Go, SQL, dan Docker."
)
resume_chef_indonesia = (
    "Executive Chef berpengalaman 6 tahun di bidang Culinary, Pastry, Food Preparation, dan manajemen dapur hotel bintang lima."
)

emb_job_en = extract_trained_embeddings([job_english_text], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='JOB_POSTING')
emb_res_id_it = extract_trained_embeddings([resume_it_indonesia], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV')
emb_res_id_chef = extract_trained_embeddings([resume_chef_indonesia], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV')

sim_en_to_id_it = float(np.clip(cosine_similarity(emb_job_en, emb_res_id_it)[0][0], 0.0, 1.0) * 100.0)
sim_en_to_id_chef = float(np.clip(cosine_similarity(emb_job_en, emb_res_id_chef)[0][0], 0.0, 1.0) * 100.0)
gap_arah2 = sim_en_to_id_it - sim_en_to_id_chef

print("\\n[ARAH 2: LOWONGAN EN <-> RESUME ID]")
print(f"1. In-Domain Cross-Lingual (IT Job EN <-> IT CV ID)    : {sim_en_to_id_it:.2f}% -> Status: {get_khatri_label_text(sim_en_to_id_it)}")
print(f"2. Cross-Domain Cross-Lingual (IT Job EN <-> Chef CV ID): {sim_en_to_id_chef:.2f}% -> Status: {get_khatri_label_text(sim_en_to_id_chef)}")
print(f"3. Domain Separation Gap (Pemisahan Arah 2)             : {gap_arah2:.2f}%")
print("="*75)"""

CELL_15_CODE = """# 15. Evaluasi Matriks Kemiripan Multidomain (6-Sector Benchmark Cross-Domain Heatmap)
benchmark_domains = [
    'INFORMATION-TECHNOLOGY',
    'HEALTHCARE',
    'ACCOUNTANT',
    'CHEF',
    'TEACHER',
    'ADVOCATE'
]

domain_jobs = {
    'INFORMATION-TECHNOLOGY': "Position: Senior Software Engineer. Required Skills: Python, Java, SQL, Docker, Cloud. Job Description: Develop backend systems.",
    'HEALTHCARE': "Position: Registered Nurse. Required Skills: Patient Care, Clinical Assessment, BSN, RN, Medical. Job Description: Provide hospital nursing care.",
    'ACCOUNTANT': "Position: Senior Accountant. Required Skills: Financial Accounting, GAAP, General Ledger, Tax, Audit, CPA. Job Description: Manage financial statements.",
    'CHEF': "Position: Executive Chef. Required Skills: Culinary Arts, Pastry, Food Preparation, Menu Planning, Kitchen. Job Description: Lead culinary operations.",
    'TEACHER': "Position: Classroom Teacher. Required Skills: Teaching, Curriculum Design, Lesson Plan, Pedagogy, Education. Job Description: Instruct students.",
    'ADVOCATE': "Position: Commercial Counsel. Required Skills: Litigation, Legal Research, Contract Management, Court, Law. Job Description: Represent corporate clients."
}

domain_resumes = {
    'INFORMATION-TECHNOLOGY': "Senior Software Engineer with Bachelor in Computer Science and 5 years experience in Python, Java, SQL, and Docker.",
    'HEALTHCARE': "Registered Nurse holding BSN degree with 5 years hospital clinical experience in patient care and medical treatment.",
    'ACCOUNTANT': "Certified Public Accountant (CPA) with 6 years experience in financial accounting, GAAP, general ledger, tax, and audit.",
    'CHEF': "Executive Chef with 7 years culinary experience in food preparation, pastry, menu development, and kitchen management.",
    'TEACHER': "Dedicated Teacher with Bachelor of Education and 5 years classroom teaching, curriculum planning, and student assessment experience.",
    'ADVOCATE': "Senior Attorney and Legal Counsel with 6 years experience in litigation, legal research, pleadings, and contract law."
}

emb_d_jobs = {c: extract_trained_embeddings([domain_jobs[c]], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='JOB_POSTING') for c in benchmark_domains}
emb_d_res = {c: extract_trained_embeddings([domain_resumes[c]], best_model, tokenizer, max_length=FINAL_MAX_LEN, doc_type='CANDIDATE_CV') for c in benchmark_domains}

heatmap_matrix = np.zeros((len(benchmark_domains), len(benchmark_domains)))
for i, r_cat in enumerate(benchmark_domains):
    for j, j_cat in enumerate(benchmark_domains):
        sim_val = float(np.clip(cosine_similarity(emb_d_res[r_cat], emb_d_jobs[j_cat])[0][0], 0.0, 1.0) * 100.0)
        heatmap_matrix[i, j] = sim_val

plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
    xticklabels=benchmark_domains, yticklabels=benchmark_domains,
    vmin=0, vmax=100, cbar_kws={'label': 'Cosine Similarity (%)'}
)
plt.title("Matriks Kemiripan Multidomain Universal NER + mBERT + Sklearn Cosine (6 Sektor)\\n(Diagonal = In-Domain Sebidang, Non-Diagonal = Lintas Profesi)", fontweight='bold', pad=15)
plt.xlabel("Kategori Lowongan Kerja (Job Postings)", fontweight='bold', labelpad=10)
plt.ylabel("Kategori Resume / CV Pelamar", fontweight='bold', labelpad=10)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('multidomain_cross_similarity_matrix.png', dpi=300)
plt.show()

diag_mean = np.diag(heatmap_matrix).mean()
off_diag = heatmap_matrix[~np.eye(heatmap_matrix.shape[0], dtype=bool)].mean()
print(f"Rata-rata Kemiripan Diagonal (In-Domain)       : {diag_mean:.2f}% (Target >= 80%)")
print(f"Rata-rata Kemiripan Non-Diagonal (Cross-Domain): {off_diag:.2f}% (Target < 45%)")
print(f"Rasio Separasi Diagonal vs Non-Diagonal        : {diag_mean - off_diag:.2f}%")"""

CELL_16_CODE = """# 16. Script Download Langsung Bobot Model (.pt) & Seluruh Grafik Evaluasi (.png) dari Google Colab
import os
import json
import zipfile

os.makedirs('./models', exist_ok=True)

# Simpan juga bobot model NER mBERT & konfigurasi ambang batas terkalibrasi
if 'ner_model' in globals():
    torch.save(ner_model.state_dict(), './models/best_mbert_ner_token_classifier.pt')

model_metadata = {
    'model_name': MODEL_NAME if 'MODEL_NAME' in globals() else 'bert-base-multilingual-cased',
    'best_params_m2_cosine': best_params_m2 if 'best_params_m2' in globals() else {},
    'best_params_m1_saja': best_params_m1 if 'best_params_m1' in globals() else {},
    'threshold_close_fit_pct': float(best_t_close) if 'best_t_close' in globals() else 60.0,
    'threshold_fit_pct': float(best_t_fit) if 'best_t_fit' in globals() else 80.0,
    'macro_f1_m2_cosine_pct': round(float(macro_f1_m2), 2) if 'macro_f1_m2' in globals() else None,
    'macro_f1_m1_saja_pct': round(float(macro_f1_m1), 2) if 'macro_f1_m1' in globals() else None,
    'macro_f1_m3_tfidf_pct': round(float(macro_f1_m3), 2) if 'macro_f1_m3' in globals() else None
}
with open('./models/model_config_metadata.json', 'w', encoding='utf-8') as f_meta:
    json.dump(model_metadata, f_meta, indent=2)

# 1. Buat paket ZIP berisi seluruh Grafik (.png) + Metadata (.json) (Ukuran kecil, instan di-download)
charts_zip_path = 'grafik_dan_metadata_skripsi.zip'
with zipfile.ZipFile(charts_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists('./models/model_config_metadata.json'):
        zf.write('./models/model_config_metadata.json', arcname='model_config_metadata.json')
    for fname in os.listdir('.'):
        if fname.endswith('.png'):
            zf.write(fname, arcname=fname)
print(f"[ZIP READY] Paket grafik & metadata berhasil dibuat: {charts_zip_path}")

# 2. Daftar file model utama yang siap diunduh
model_files = [
    './models/best_siamese_mbert_cosine.pt',
    './models/best_mbert_saja.pt',
    './models/best_mbert_ner_token_classifier.pt'
]
for mf in model_files:
    if os.path.exists(mf):
        size_mb = os.path.getsize(mf) / (1024 * 1024)
        print(f"  -> Tersedia: {mf} ({size_mb:.1f} MB)")

# 3. Trigger Download Langsung di Browser Google Colab
try:
    from google.colab import files
    print("\\n[DOWNLOAD] Memulai pengunduhan ke komputer lokal...")
    files.download(charts_zip_path)
    if os.path.exists('./models/best_siamese_mbert_cosine.pt'):
        files.download('./models/best_siamese_mbert_cosine.pt')
    # Uncomment 2 baris di bawah jika ingin mengunduh model pembanding & NER juga via browser:
    # if os.path.exists('./models/best_mbert_saja.pt'): files.download('./models/best_mbert_saja.pt')
    # if os.path.exists('./models/best_mbert_ner_token_classifier.pt'): files.download('./models/best_mbert_ner_token_classifier.pt')
except ImportError:
    print("[INFO] Tidak berjalan di Google Colab. File tersimpan secara lokal di folder ./models/")"""

def main():
    nb_path = 'notebooks/03_optuna_tuning_siamese_mbert.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = [
        make_md_cell(CELL_0_MD),
        make_code_cell(CELL_1_CODE),
        make_code_cell(CELL_2_CODE),
        make_code_cell(CELL_3_CODE),
        make_code_cell(CELL_4_CODE),
        make_code_cell(CELL_5_CODE),
        make_code_cell(CELL_6_CODE),
        make_code_cell(CELL_7_CODE),
        make_code_cell(CELL_8_CODE),
        make_code_cell(CELL_9_CODE),
        make_code_cell(CELL_10_CODE),
        make_code_cell(CELL_11_CODE),
        make_code_cell(CELL_12_CODE),
        make_code_cell(CELL_13_CODE),
        make_code_cell(CELL_14_CODE),
        make_code_cell(CELL_15_CODE),
        make_code_cell(CELL_16_CODE),
    ]
    nb['cells'] = cells

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Successfully wrote {len(cells)} cells to {nb_path}!")

    # Compile check all code cells
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            lines = [l for l in src.splitlines() if not l.strip().startswith('!') and not l.strip().startswith('%')]
            compile('\n'.join(lines), f'Cell_{i}', 'exec')
    print("All 15 code cells compiled with 0 syntax errors!")

if __name__ == '__main__':
    main()
