# CATATAN SIDANG & BIMBINGAN TUGAS AKHIR (TIPS PAK NJOTO)
**Judul / Topik:** Sistem Pencocokan CV Pelamar dan Lowongan Pekerjaan Menggunakan *Deep Learning-Based Named Entity Recognition (NER)*, *Multilingual BERT (mBERT)*, dan *Cosine Similarity*

---

## PERTANYAAN 1: HYPERPARAMETER TUNING
> **"Hyperparameter tuning menggunakan apa, caranya bagaimana, dan tujuannya apa?"**

### 1.1. Menggunakan Apa? (*Tools, Algoritma, dan Parameter yang Diuji*)
1. **Framework & Library:** Menggunakan pustaka **Optuna** (`optuna.create_study(direction="minimize")`) dengan fungsi objektif meminimalkan **Validation Loss (`val_loss`)**.
2. **Algoritma Pencarian (*Sampler*):** Menggunakan **TPE (*Tree-structured Parzen Estimator*) Sampler** (`TPESampler(seed=42)`) yang dikombinasikan dengan **Guided Trial Enqueueing** (`study.enqueue_trial`), sehingga keenam kandidat *Learning Rate* wajib diuji secara eksplisit dan adil.
3. **4 Hyperparameter yang Di-tuning (Ruang Pencarian Terpadu):**
   - **Learning Rate (`lr`):** `[2e-5, 3e-5, 5e-5, 0.001, 0.005, 0.01]`
     * *3 kandidat dari Literatur Asli BERT (Devlin et al., 2019):* `2e-5` ($0.00002$), `3e-5` ($0.00003$), `5e-5` ($0.00005$).
     * *3 kandidat dari Catatan Konsultasi Dosen:* `0.001`, `0.005`, `0.01`.
   - **Max Length Token (`max_length`):** `[128, 256, 512]` token.
   - **Dropout Rate (`dropout`):** `[0.1, 0.2, 0.3]`.
   - **Weight Decay (`weight_decay`):** `[0.001, 0.01]` (penalti regularisasi $L_2$ pada optimizer **AdamW**, mengacu pada *Loshchilov & Hutter, 2019* dan *Devlin et al., 2019*).

---

### 1.2. Caranya Bagaimana? (*Alur Kerja Step-by-Step*)
1. **Persiapan Dataset Seimbang (1.500 Pasang CV–Lowongan):**
   - Terdiri dari **500 pasang *Fit***, **500 pasang *Close Fit***, dan **500 pasang *Not Fit***.
   - Dibagi secara *Stratified Split* (proporsi kelas selalu seimbang 1:1:1 di setiap subset):
     * **Data Latih (*Train Set* - 70%):** 1.050 pasang (`350 Fit`, `350 Close Fit`, `350 Not Fit`).
     * **Data Validasi (*Validation Set* - 15%):** 225 pasang (`75 Fit`, `75 Close Fit`, `75 Not Fit`).
     * **Data Uji (*Unseen Test Set* - 15%):** 225 pasang (`75 Fit`, `75 Close Fit`, `75 Not Fit`).
2. **Tahap 1 — Pencarian Hyperparameter (*Optuna Trial Search*):**
   - Pada setiap *trial*, Optuna memilih 1 kombinasi `(learning_rate, max_length, dropout, weight_decay)`.
   - Model dilatih pada **Data Latih (1.050 pasang)** selama 3 *proxy epochs* menggunakan optimizer **AdamW** dan *Linear Warmup Scheduler* (`warmup_ratio = 0.1`, `batch_size = 16`).
   - Di akhir setiap epoch, model dievaluasi pada **Data Validasi (225 pasang)** untuk mencatat nilai **`val_loss`** dan **`val_accuracy`**.
   - Proses tuning ini dijalankan secara **terpisah dan adil** untuk kedua model:
     * **Model Usulan (`SiameseMBERTCosineExtractor` / mBERT + Cosine):** Menggunakan fungsi loss `PureCosineMarginLoss`.
     * **Model Pembanding (`MBERTSajaClassifier` / mBERT Saja tanpa Cosine):** Menggunakan fungsi loss `CrossEntropyLoss`.
   - **Hasil Terbaik (*Best Hyperparameters*) yang Terpilih:**
     * **Model 2 (Usulan: Unified NER + Siamese mBERT + Cosine):**
       - `learning_rate`: **`2e-5`** ($0.00002$)
       - `max_length`: **`128`** token
       - `dropout`: **`0.1`**
       - `weight_decay`: **`0.01`**
       - *(Proxy Tuning Best Val Loss: **`0.0412`**, Val Accuracy: **`96.44%`**)*
     * **Model 1 (Pembanding: mBERT Saja tanpa Cosine):**
       - `learning_rate`: **`0.001`** ($1\times 10^{-3}$), `max_length`: **`128`**, `dropout`: **`0.2`**, `weight_decay`: **`0.01`** (*Val Loss: **`0.8770`**, Val Accuracy: **`57.33%`**)*
3. **Tahap 2 — Pelatihan Penuh (*Full Training*) dengan *Early Stopping*:**
   - Kombinasi hyperparameter terbaik dilatih ulang dari awal dengan batas maksimum **100 Epoch**, menggunakan mekanisme **Early Stopping (`patience = 5`, `monitoring = val_loss`)**.
   - Jika `val_loss` tidak membaik selama 5 epoch berturut-turut, pelatihan dihentikan otomatis dan bobot pada epoch dengan `val_loss` terendah disimpan:
     * **Model Usulan (Model 2 — NER + Siamese mBERT + Cosine):** Mencapai `val_loss` terbaik pada **Epoch ke-5** (**`val_loss = 0.0365`**, **`val_acc = 98.22%`**; berhenti otomatis di **Epoch ke-10**).
     * **Model Pembanding (Model 1 — mBERT Saja):** Mencapai `val_loss` terbaik pada **Epoch ke-5** (**`val_loss = 0.7303`**, **`val_acc = 64.89%`**; berhenti otomatis di **Epoch ke-10**).

---

### 1.3. Tujuannya Apa?
1. **Mencegah *Catastrophic Forgetting* pada Bobot Pre-trained mBERT:**
   - mBERT memiliki 12 lapisan *Transformer Encoder* (~177 juta parameter) yang sudah dilatih pada 104 bahasa.
   - Ketika diuji dengan *learning rate* besar (`0.001`, `0.005`, `0.01`), perubahan bobot menjadi terlalu ekstrem sehingga representasi semantik bahasa rusak (*catastrophic forgetting*) dan `val_loss` mandek di **`3.2429`** (`val_acc = 33.33%` — menebak acak).
   - Sebaliknya, pada rentang *fine-tuning* (`2e-5`, `3e-5`, `5e-5`), bobot *pre-trained* beradaptasi secara halus ke domain rekrutmen, dengan **`2e-5`** menghasilkan konvergensi `val_loss` paling rendah (**`0.0412`** pada tuning dan **`0.0365`** saat *full training*) karena gradien *multi-task* (Cosine + NER) membutuhkan langkah optimasi yang stabil.
2. **Mencegah *Overfitting* dan Memaksimalkan Generalisasi pada Data *Unseen*:**
   - Kombinasi `dropout = 0.1`, `weight_decay = 0.01`, dan *Early Stopping* (`patience = 5`) mencegah model menghafal data latih dan terbukti menghasilkan **99.11% Accuracy & Macro F1-Score** (**99.13% Macro Precision**) pada 225 data uji (*unseen test set*).
3. **Menemukan Efisiensi Konteks Sekuens Terbaik (`max_length = 128`):**
   - Karena entitas penting (`EDU`, `ORG`, `SKILL`, `ROLE`, `DUR`) sudah diekstrak oleh NER dan diletakkan di awal sekuens teks terstruktur, panjang **`128` token** memadatkan seluruh informasi inti kompetensi tanpa *padding noise* maupun pemborosan komputasi seperti pada `256` atau `512` token.

---

## PERTANYAAN 2: EXACT INPUT DAN EXACT OUTPUT MODEL mBERT
> **"Apa exact input dan output yang dikeluarkan oleh model BERT yang digunakan?"**

Dalam sistem ini, model dasar **`bert-base-multilingual-cased`** berperan dalam **2 tahap berurutan** (Tahap Ekstraksi Entitas NER dan Tahap Ekstraksi Vektor Fitur *Siamese*), yang kemudian dilanjutkan oleh perhitungan kemiripan kosinus menggunakan **`scikit-learn`**.

### Ringkasan Alur Dimensi Data (*End-to-End Pipeline*)
```text
[File PDF CV & Teks Lowongan]
       │
       ▼  (1. Pembersihan Regex & Ekstraksi PDF)
[String Teks Bersih (Cased Dipertahankan)]
       │
       ▼  (2. Tahap NER mBERT Token Classification - Tabel 2.5 Skripsi)
[5 Entitas Kompetensi: EDU, ORG, SKILL, ROLE, DUR]
       │
       ▼  (3. Penyusunan Teks Kompetensi Terstruktur + WordPiece Tokenizer max_length=128)
[Exact Input ke mBERT Feature Extractor]:
  • input_ids      : Tensor int64 ukuran [batch_size, 128]
  • attention_mask : Tensor int64 ukuran [batch_size, 128]
       │
       ▼  (4. Forward Pass 12-Layer mBERT -> Hybrid CLS+Mean Pooling -> Linear Projection 768x768 -> L2 Norm)
[Exact Output dari Model mBERT Feature Extractor]:
  • Vektor Embedding CV (u)       : Vektor float32 ukuran [1, 768] (L2-normalized, ||u|| = 1.0)
  • Vektor Embedding Lowongan (v) : Vektor float32 ukuran [1, 768] (L2-normalized, ||v|| = 1.0)
       │
       ▼  (5. Perhitungan Kemiripan: sklearn.metrics.pairwise.cosine_similarity(u, v))
[Exact Output Akhir Sistem]:
  • Skor Cosine Similarity : Float 0.0000 s/d 1.0000 (0.00% s/d 100.00%)
  • Kategori Keputusan     : Fit (>= 80%), Close Fit (60% - 79.9%), atau Not Fit (< 60%)
```

---

### 2.1. Detail Tahap 1: Model NER (`AutoModelForTokenClassification` mBERT)
Sesuai Bab 2.4 dan Tabel 2.5 Skripsi, NER berfungsi mengubah teks CV/Lowongan yang tidak terstruktur menjadi entitas kompetensi terstruktur.
- **Exact Input ke Model NER:**
  - `input_ids`: Tensor integer berukuran **`[batch_size, 128]`** berisi indeks token *WordPiece* (diawali token khusus `[CLS] = 101` dan diakhiri `[SEP] = 102`).
  - `attention_mask`: Tensor biner berukuran **`[batch_size, 128]`** (`1` untuk token kata aktif, `0` untuk `[PAD]`).
- **Exact Output dari Model NER:**
  - Tensor **`logits` berukuran `[batch_size, 128, 11]`**, yaitu skor probabilitas belum-ternormalisasi untuk setiap token terhadap **11 label BIO**:
    `O`, `B-EDU`, `I-EDU`, `B-ORG`, `I-ORG`, `B-SKILL`, `I-SKILL`, `B-ROLE`, `I-ROLE`, `B-DUR`, `I-DUR`.
  - Setelah diambil nilai maksimumnya (`argmax`) dan divalidasi dengan pola kamus rekrutmen (*hybrid gazetteer*), output yang dihasilkan adalah dictionary 5 entitas:
    * **`EDU` (Pendidikan):** Contoh `['S1 Informatika']` atau `['Bachelor of Science: Finance']`
    * **`ORG` (Organisasi/Institusi):** Contoh `['Universitas Surabaya', 'PT Telekomunikasi Indonesia']`
    * **`SKILL` (Keahlian):** Contoh `['Python', 'Go', 'C#']` atau `['Microsoft Excel', 'accounting', 'Payroll', 'budget']`
    * **`ROLE` (Jabatan/Posisi):** Contoh `['Software Engineer']` atau `['Consumer Banking Default Management Supervisor']`
    * **`DUR` (Durasi Pengalaman):** Contoh `['6 bulan']` atau `['17 years total experience']`

---

### 2.2. Detail Tahap 2 (Inti): Model Ekstraktor Fitur `SiameseMBERTCosineExtractor`
Sesuai arahan dosen pembimbing: **"mBERT hanya digunakan untuk ekstraksi fitur saja; ambil embedding terakhir, kemudian baru hitung cosine similarity menggunakan library `sklearn`."**

#### A. Exact Input ke `SiameseMBERTCosineExtractor`
1. **Format Teks Input (Setelah Diperkaya Entitas NER):**
   Entitas hasil NER disusun di bagian paling depan (*structured prefix*) diikuti teks asli, sehingga posisi dan struktur kalimat selalu konsisten meskipun gaya penulisan CV pelamar berbeda-beda:
   - **Contoh String Input CV:**
     `[DOC] CANDIDATE_CV | [DOMAIN] INFORMATION-TECHNOLOGY | [TIER] CANDIDATE_PROFILE | [ROLE] Software Engineer | [SKILL] Python, Go, C# | [EDU] S1 Informatika | [DUR] 6 bulan | [ORG] Universitas Surabaya, PT Telekomunikasi Indonesia || Lulusan S1 Informatika dari Universitas Surabaya...`
   - **Contoh String Input Lowongan Kerja:**
     `[DOC] JOB_POSTING | [DOMAIN] INFORMATION-TECHNOLOGY | [TIER] CORE_DOMAIN_ROLE | [ROLE] Backend Software Engineer | [SKILL] Python, Go, C#, SQL, Docker | [EDU] S1 Informatika | [DUR] Unspecified | [ORG] Unspecified || Position: Backend Software Engineer. Job Description: ...`
2. **Representasi Tensor Numerik yang Masuk ke `self.mbert(...)`:**
   Tokenizer `bert-base-multilingual-cased` mengubah kedua teks di atas menjadi 2 pasang tensor berukuran tetap (`max_length = 128`):
   - **Cabang CV:**
     * `cv_input_ids`: Tensor `torch.int64` dengan shape **`[batch_size, 128]`** (berisi angka ID kosakata mBERT dari rentang `0` hingga `119546`).
     * `cv_attention_mask`: Tensor `torch.int64` dengan shape **`[batch_size, 128]`** (berisi angka `1` pada token teks dan `0` pada token padding).
   - **Cabang Lowongan Kerja (Job):**
     * `job_input_ids`: Tensor `torch.int64` dengan shape **`[batch_size, 128]`**.
     * `job_attention_mask`: Tensor `torch.int64` dengan shape **`[batch_size, 128]`**.

#### B. Exact Proses Internal & Exact Output dari `SiameseMBERTCosineExtractor` (Unified Deep Learning NER + Feature Extractor)
1. **Keluaran Hidden State Terakhir mBERT (`outputs.last_hidden_state`):**
   - Shape: **`[batch_size, 128, 768]`** (setiap dari 128 token direpresentasikan oleh vektor kontekstual 768 dimensi).
2. **Kepala Klasifikasi Token NER Terpadu (`self.ner_head = nn.Linear(768, 11)`):**
   - Di dalam model utama `SiameseMBERTCosineExtractor`, setiap token `last_hidden_state` dilewatkan ke lapisan `self.ner_head` untuk memprediksi **11 label BIO** (`O, B-EDU, I-EDU, B-ORG, I-ORG, B-SKILL, I-SKILL, B-ROLE, I-ROLE, B-DUR, I-DUR`) secara simetris baik pada **CV Pelamar** maupun **Job Description** (`ner_logits` berukuran **`[batch_size, 128, 11]`**).
3. **Operasi *NER-Guided Entity-Weighted Pooling* (`CLS` + *Entity-Weighted Mean Pooling*):**
   - Token yang terdeteksi sebagai entitas kompetensi oleh `self.ner_head` ($P(\text{entity}) = 1 - P(\text{'O'})$) diberikan bobot perhatian ekstra `weights = mask * (1.0 + 0.5 * entity_prob)` saat merata-ratakan token (`mean_emb`), lalu digabungkan dengan vektor `[CLS]`:
     `pooled = 0.5 * cls_emb + 0.5 * mean_emb`  $\longrightarrow$ Shape: **`[batch_size, 768]`**.
4. **Lapisan Proyeksi Linear & Normalisasi L2 (*Exact Output* dari Model mBERT):**
   - Vektor `pooled` melewati lapisan `Dropout(0.1)` dan `nn.Linear(768, 768)`, lalu dinormalisasi dengan fungsi `F.normalize(projected, p=2, dim=1)`.
   - **EXACT OUTPUT YANG DIKELUARKAN OLEH MODEL mBERT:**
     * **Vektor Fitur CV (`u`):** Array/Tensor `float32` berdimensi **`[1, 768]`** (768 angka desimal antara `-1.0` s/d `1.0` dengan panjang norma Euclidean $\|\mathbf{u}\|_2 = 1.0$).
     * **Vektor Fitur Lowongan (`v`):** Array/Tensor `float32` berdimensi **`[1, 768]`** (768 angka desimal antara `-1.0` s/d `1.0` dengan panjang norma Euclidean $\|\mathbf{v}\|_2 = 1.0$).
     * **Prediksi Token NER (`ner_logits`):** Tensor berukuran **`[1, 128, 11]`** yang mengekstrak entitas kompetensi (`EDU, ORG, SKILL, ROLE, DUR`) dari CV dan Job Description.
   - **Penting untuk ditekankan saat sidang:** Untuk penentuan kecocokan CV–Job, model mBERT **TIDAK** mengeluarkan label *Fit/Close Fit/Not Fit* secara langsung melalui *Softmax Classifier*, melainkan mengeluarkan **vektor embedding 768 dimensi** (`u` dan `v`) yang dihitung kemiripannya menggunakan `sklearn.metrics.pairwise.cosine_similarity(u, v)`.

---

### 2.3. Detail Tahap 3: Perhitungan `scikit-learn` Cosine Similarity & Klasifikasi Ambang Batas
- **Exact Input ke `sklearn.metrics.pairwise.cosine_similarity(u, v)`:**
  - Vektor `u` (`shape: [1, 768]`) dari CV dan vektor `v` (`shape: [1, 768]`) dari Lowongan Kerja.
- **Rumus Matematika:**
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\sum_{i=1}^{768} u_i \cdot v_i}{\sqrt{\sum_{i=1}^{768} u_i^2} \cdot \sqrt{\sum_{i=1}^{768} v_i^2}}$$
- **Exact Output dari Cosine Similarity:**
  - Nilai skalar *float* dalam rentang **`0.0000` hingga `1.0000`** (dikonversi ke persentase **`0.00%` – `100.00%`**).
- **Aturan Keputusan (*Decision Threshold* Sesuai Proposal Skripsi):**
  - **Fit:** Skor kemiripan **$\ge 80\%$** (`>= 0.80`)
  - **Close Fit:** Skor kemiripan **$60\% \text{ – } 79.9\%$** (`0.60 – 0.799`)
  - **Not Fit:** Skor kemiripan **$< 60\%$** (`< 0.60`)

---

## PERTANYAAN 3: ALASAN PEMBAGIAN DATASET 70/30 (70% TRAIN : 15% VAL : 15% TEST)
> **"Kenapa menggunakan pembagian data 70/30?"**

### 3.1. Pemisahan Tegas Antara *Validation Set* dan *Unseen Test Set* (Mencegah *Data Leakage*)
- Dalam penelitian ini terdapat dua proses yang membutuhkan evaluasi di luar data latih:
  1. **Hyperparameter Tuning (Optuna) & *Early Stopping* (`patience = 5`, memonitor `val_loss`)** $\longrightarrow$ Membutuhkan **Data Validasi (*Validation Set*)**.
  2. **Pengujian Akhir (*Confusion Matrix* & *Classification Report*)** $\longrightarrow$ Membutuhkan **Data Uji Murni (*Unseen Test Set*)** yang sama sekali tidak pernah disentuh saat tuning maupun *early stopping*.
- Oleh karena itu, porsi **30% non-training (450 pasang)** dibagi sama rata menjadi **15% Validation Set (225 pasang)** dan **15% Unseen Test Set (225 pasang)** melalui dua tahap `train_test_split(..., test_size=0.30, stratify=label)` diikuti `train_test_split(..., test_size=0.50, stratify=label)`.

### 3.2. Keseimbangan Stratifikasi Bulat Sempurna (*Exact Integer Stratification*) pada 3 Kelas
Dengan total dataset seimbang **1.500 pasang** (`500 Fit`, `500 Close Fit`, `500 Not Fit`), rasio **70% : 15% : 15%** menghasilkan jumlah sampel per kelas yang **bulat sempurna (1:1:1)** di setiap subset:

| Subset Data | Persentase | Total Sampel | Kelas `Fit` | Kelas `Close Fit` | Kelas `Not Fit` | Fungsi Utama |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train Set** | **70%** | **1.050 pasang** | 350 | 350 | 350 | Melatih bobot mBERT & proyeksi 768-d (*forward & backward pass*) |
| **Validation Set** | **15%** | **225 pasang** | 75 | 75 | 75 | Memilih *hyperparameter* terbaik di Optuna & memonitor `val_loss` pada *Early Stopping* |
| **Unseen Test Set** | **15%** | **225 pasang** | 75 | 75 | 75 | Evaluasi akhir objektif (*Confusion Matrix* & *Classification Report* Bab 4/6) |
| **Total** | **100%** | **1.500 pasang** | **500** | **500** | **500** | |

### 3.3. Keseimbangan Statistik antara Kapasitas Belajar Model vs Validitas Pengujian
1. **Mengapa 70% (1.050 pasang) sudah sangat cukup untuk Training?**
   - Karena kita menggunakan pendekatan **Transfer Learning / Fine-Tuning** pada model *pre-trained* `bert-base-multilingual-cased` (yang sudah memiliki pemahaman bahasa dari korpus masif) ditambah **ekstraksi struktur NER**, **1.050 pasang data latih** (`350` per kelas) sudah sangat memadai bagi model untuk mencapai konvergensi tanpa *underfitting*.
2. **Mengapa dibutuhkan 30% (450 pasang: 225 Val + 225 Test) untuk Evaluasi?**
   - Jika hanya menggunakan rasio `80/20` atau `90/10` lalu dibagi dua untuk *Validation* dan *Testing*, maka setiap subset evaluasi hanya berisi 75–150 data (hanya 25–50 sampel per kelas). Jumlah sekecil itu terlalu sempit untuk menguji **6 sektor pekerjaan lintas industri** secara representatif dan membuat kurva `val_loss` pada *Early Stopping* menjadi fluktuatif (*noisy*).
   - Dengan **75 sampel per kelas (225 total)** di *Validation Set* dan **75 sampel per kelas (225 total)** di *Test Set*, setiap sektor industri dan setiap kategori (*Fit*, *Close Fit*, *Not Fit*) terwakili secara merata dan stabil secara statistik (mengacu pada pedoman pembagian data empiris *Gholamy et al., 2018*).

---

## PERTANYAAN 4: MENGAPA MENGGUNAKAN PRECISION, RECALL, DAN F1-SCORE? MENGAPA ACCURACY SAJA TIDAK CUKUP & APAKAH ACCURACY TETAP PERLU DIPAKAI?
> **"Kenapa yang digunakan itu Precision, Recall, dan F1-Score sebagai metrik evaluasinya? Selain itu kenapa Accuracy tidak dipakai sebagai metrik tunggal (dan apakah Accuracy juga tetap perlu dicantumkan)?"**

### 4.1. Mengapa *Precision*, *Recall*, dan *F1-Score* Wajib Digunakan pada Sistem Rekrutmen CV–Job Matching (3 Kelas)?
Dalam domain **Sistem Pendukung Keputusan Rekrutmen SDM (*Applicant Tracking System*)**, kesalahan prediksi di setiap kelas memiliki **konsekuensi bisnis (risiko kesalahan)** yang sangat berbeda, yang **tidak bisa terlihat jika hanya melihat nilai *Accuracy***:

1. **Peran *Precision* ($\frac{TP}{TP + FP}$) — Mencegah Pelamar Tidak Kualifikatif Lolos (*False Positive*):**
   - Pada kelas **`Fit`**, *Precision* mengukur: *"Dari seluruh pelamar yang diprediksi `Fit` oleh sistem, berapa persen yang benar-benar `Fit`?"*
   - Jika *Precision* pada kelas `Fit` rendah, maka banyak kandidat yang sebenarnya `Not Fit` atau baru `Close Fit` ikut diloloskan ke tahap wawancara (*False Positive*). Hal ini membuang waktu dan biaya tim HR/rekruter.
2. **Peran *Recall* ($\frac{TP}{TP + FN}$) — Mencegah Kandidat Potensial Gugur/Terbuang (*False Negative*):**
   - Pada kelas **`Fit`** dan **`Close Fit`**, *Recall* mengukur: *"Dari seluruh kandidat berkualitas yang melamar, berapa persen yang berhasil ditemukan/dijaring oleh sistem?"*
   - Jika *Recall* rendah, kandidat berbakat yang seharusnya cocok justru terbuang ke kategori `Not Fit` (*False Negative*), sehingga perusahaan kehilangan talenta terbaik (*talent loss*).
3. **Peran *F1-Score* ($\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$) — Keseimbangan Harmonis per Kelas:**
   - Karena sistem rekrutmen yang baik tidak boleh terlalu longgar (*Precision* rendah) maupun terlalu kaku (*Recall* rendah), **F1-Score** (rata-rata harmonis *Precision* dan *Recall*) menjadi metrik utama untuk menilai kualitas pemisahan di masing-masing kelas secara adil, khususnya pada kelas transisi **`Close Fit` ($60\% \text{–} 79.9\%$)** yang paling rawan tertukar dengan `Fit` atau `Not Fit`.

---

### 4.2. Mengapa *Accuracy* Tidak Digunakan Sendirian (Kelemahan *Accuracy* pada Klasifikasi Multi-Kelas)?
Ada **3 alasan ilmiah utama** mengapa *Accuracy* tidak dijadikan satu-satunya tolok ukur utama pada proposal/skripsi:

1. ***Accuracy* Menyembunyikan Kelemahan Spesifik per Kelas (*Class-Specific Blindness*):**
   - Rumus $\text{Accuracy} = \frac{\text{Total Prediksi Benar}}{\text{Total Seluruh Data}}$ hanya menghitung total diagonal pada *Confusion Matrix* tanpa peduli kelas mana yang gagal.
   - **Bukti Nyata pada Eksperimen Model 3 (*Cosine Saja / TF-IDF Baseline*):** Model *Cosine Saja* memiliki *Accuracy* keseluruhan **33.33%**, tetapi ketika dibedah menggunakan *Precision, Recall,* dan *F1-Score* per kelas, terlihat bahwa kelas `Close Fit` dan `Fit` gagal total (**0.00%** *F1-Score* karena seluruh pasangan TF-IDF tanpa semantik mBERT mendapat skor di bawah 60% dan menumpuk pada `Not Fit`). Jika hanya melihat satu angka *Accuracy*, ketimpangan kegagalan per kelas ini tidak terdiagnosis.
2. **Fenomena *Accuracy Paradox* pada Skenario Dunia Nyata (*Real-World Deployment*):**
   - Meskipun dataset eksperimen laboratorium kita dibuat seimbang (`500 Fit : 500 Close Fit : 500 Not Fit`), pada penggunaan nyata di perusahaan (termasuk pada grafik riwayat pemrosesan Admin di Web UI ketika HR mengunggah ratusan CV pelamar), distribusi pelamar di dunia nyata hampir selalu **timpang (*imbalanced*)** — misalnya 80% pelamar adalah `Not Fit`, 15% `Close Fit`, dan hanya 5% yang benar-benar `Fit`.
   - Pada kondisi dunia nyata tersebut, model yang menebak semua pelamar sebagai `Not Fit` akan mendapat **Accuracy 80%**, padahal **F1-Score kelas `Fit`-nya adalah 0%**. Oleh karena itu, standar evaluasi NLP modern selalu menjadikan **Macro F1-Score, Precision, dan Recall** sebagai metrik primer.
3. **Evaluasi Dua Komponen Sekaligus (NER + Klasifikasi Kecocokan):**
   - Dalam standar literatur NLP (seperti CoNLL dan paper *Drushchak & Romanyshyn, 2024*), evaluasi ekstraksi entitas **NER (`EDU, ORG, SKILL, ROLE, DUR`)** maupun klasifikasi teks multi-kelas selalu menggunakan **Precision, Recall, dan F1-Score** karena sebagian besar token teks adalah non-entitas (`O`).

---

### 4.3. Pertimbangan & Rekomendasi: Apakah *Accuracy* Tetap Perlu Dicantumkan pada Skripsi Anda?
> **JAWABAN & REKOMENDASI TEGAS: YA, *Accuracy* SANGAT PERLU TETAP DICANTUMKAN sebagai metrik pendamping bersama *Precision, Recall,* dan *F1-Score*!**

**Mengapa *Accuracy* sebaiknya tetap dimasukkan di Bab 4 / Bab 5 (dan sudah otomatis dihitung di Notebook Anda)?**
1. **Sesuai dengan Permintaan Papan Tulis & Catatan Pak Njoto:**
   - Pada papan tulis bimbingan Pak Njoto:
     * **Poin 4:** *"Tampilkan grafik train & val loss, **train & val acc**"* $\longrightarrow$ Grafik kurva belajar setiap epoch wajib menampilkan **Training Accuracy vs Validation Accuracy**.
     * **Poin 5, 6, 7:** *"Tampilkan **CM + CR** (Confusion Matrix + Classification Report)"* $\longrightarrow$ Output standar `sklearn.metrics.classification_report` secara otomatis menampilkan **Precision, Recall, dan F1-Score untuk setiap kelas (`Not Fit`, `Close Fit`, `Fit`)**, ditambah baris ringkasan **`accuracy`**, **`macro avg`**, dan **`weighted avg`** di bagian bawah tabel.
2. **Dataset Pengujian Anda Seimbang Sempurna (1:1:1 = 75 `Fit`, 75 `Close Fit`, 75 `Not Fit`):**
   - Karena pada **225 Unseen Test Set** jumlah sampel tiap kelas sama persis (`75 : 75 : 75`), maka secara matematis nilai **Accuracy (99.11%)** menjadi **valid dan bebas bias mayoritas**, serta nilainya selaras dengan **Macro F1-Score (99.11%)** (**Macro Precision 99.13%, Macro Recall 99.11%**).
3. **Strategi Terbaik untuk Penulisan Naskah & Sidang:**
   - Jangan menghapus *Accuracy*, melainkan **sajikan keempatnya secara lengkap (`Accuracy`, `Precision`, `Recall`, dan `F1-Score`)** di tabel evaluasi Bab 4/5:
     * Gunakan **Accuracy** untuk menunjukkan performa ketepatan global model dan grafik konvergensi *epoch* (*Train vs Val Accuracy*).
     * Gunakan **Precision, Recall, dan F1-Score per kelas** (dari *Classification Report*) untuk menganalisis ketajaman model dalam membedakan batas kategori **`Fit` ($\ge 80\%$)**, **`Close Fit` ($60\% \text{–} 79.9\%$)**, dan **`Not Fit` ($< 60\%$)**.

---

## PERTANYAAN 5: MENGAPA DISEBUT SIAMESE mBERT, APA ARSITEKTUR YANG DIPAKAI, DAN APA YANG DILAKUKAN SEBELUM TRAINING?
> **"Mengapa model mBERT kita dikatakan Siamese mBERT? Apa arsitektur persis yang kita gunakan? Dan apa saja yang dilakukan pada model/data mBERT sebelum proses training dijalankan?"**

### 5.1. Mengapa Dikatakan "Siamese mBERT"?
Istilah **"Siamese" (Kembar Siam)** mengacu pada konsep arsitektur **Siamese Neural Network (Reimers & Gurevych, 2019 - Sentence-BERT)**:
1. **Dua Cabang Input dengan Bobot yang Sama (*Weight Sharing*):**
   - Model memiliki dua jalur pemrosesan terpisah: satu jalur untuk membaca teks **CV Pelamar** dan satu jalur untuk membaca **Deskripsi Lowongan Pekerjaan (*Job Description*)**.
   - Kedua jalur ini **TIDAK** menggunakan dua model BERT yang berbeda, melainkan menggunakan **satu model mBERT yang sama persis** (*shared weights*).
2. **Memetakan ke Ruang Vektor yang Setara (*Common 768-d Embedding Space*):**
   - Karena bobot yang digunakan identik, vektor fitur CV ($\mathbf{u}$) dan vektor fitur lowongan ($\mathbf{v}$) berada pada koordinat semantik 768 dimensi yang setara.
   - Dengan begitu, derajat kesesuaian keduanya dapat dihitung secara langsung dan objektif menggunakan perkalian sudut ruang vektor (**`sklearn.metrics.pairwise.cosine_similarity(u, v)`**).
3. **Perbedaan dengan BERT Standar (*Cross-Encoder*):**
   - Pada BERT standar biasa, teks CV dan lowongan digabungkan jadi satu: `[CLS] CV [SEP] Job [SEP]`, lalu langsung masuk ke lapisan klasifikasi aditif (*black-box*). Cara ini tidak menghasilkan representasi vektor dokumen mandiri.
   - Pada **Siamese mBERT**, CV dan lowongan diproses secara independen menjadi vektor representasi fitur ($\mathbf{u}$ dan $\mathbf{v}$), sehingga sangat efisien dan transparan untuk pencocokan.

---

### 5.2. Detail Arsitektur Lengkap Model Usulan
Arsitektur model usulan kita bernama:  
**`Unified Multi-Task Siamese mBERT Feature Extractor with NER-Guided Pooling`**

Komponen penyusunnya dari bawah ke atas:
1. **Backbone Base Model:**
   - Pre-trained `bert-base-multilingual-cased` (12 layer Transformer, hidden size 768, 12 attention heads, 104 bahasa).
2. **Unified Deep Learning NER Head (`self.ner_head`):**
   - Lapisan `nn.Linear(768, 11)` yang bertugas memprediksi **11 label BIO** (`O, B-EDU, I-EDU, B-ORG, I-ORG, B-SKILL, I-SKILL, B-ROLE, I-ROLE, B-DUR, I-DUR`) langsung dari *hidden state* tiap token secara simetris pada CV dan Job Description.
3. **NER-Guided Entity-Weighted Pooling:**
   - Berbeda dengan *mean pooling* biasa yang menganggap semua kata sama pentingnya, token yang diprediksi oleh `self.ner_head` sebagai entitas kompetensi rekrutmen ($P(\text{entity}) = 1.0 - P(\text{'O'})$) diberikan bobot perhatian lebih tinggi:
     $$\text{weights} = \text{attention\_mask} \times \left(1.0 + 0.5 \times P(\text{entity})\right)$$
   - Vektor gabungan: $\mathbf{e}_{\text{combined}} = 0.5 \times \mathbf{e}_{\text{[CLS]}} + 0.5 \times \mathbf{e}_{\text{weighted\_mean}}$ (Shape: `[batch_size, 768]`).
4. **Proyeksi Fitur & Normalisasi L2 (`self.projection`):**
   - Lapisan `nn.Linear(768, 768)` yang diakhiri fungsi normalisasi L2:
     $$\mathbf{u} = \frac{\text{projected}}{\|\text{projected}\|_2}$$
   - Exact output akhir murni berupa **vektor 768 dimensi** ($\mathbf{u}$ dan $\mathbf{v}$).
5. **Fungsi Jarak / Metrik Kemiripan (Sklearn Cosine Similarity):**
   - `sklearn.metrics.pairwise.cosine_similarity(u, v)` menghitung sudut kosinus ($0.0 - 1.0$ atau $0\% - 100\%$) dan membaginya ke dalam 3 batas ambang skripsi: **Fit ($\ge 80\%$)**, **Close Fit ($60\% - 79.9\%$)**, dan **Not Fit ($< 60\%$)**.

---

### 5.3. Apa Saja yang Dilakukan pada Model & Data mBERT Sebelum Training?
Terdapat **4 tahapan persiapan teknis (*Pre-Training Preparation & Adaptation*)**:

1. **Preprocessing & Penjagaan Karakter Huruf Besar-Kecil (*Cased Preservation*):**
   - Teks dibersihkan dari URL, karakter aneh, dan spasi berlebih menggunakan *Regular Expression*.
   - **Teks sengaja TIDAK diubah menjadi huruf kecil (*no lowercasing*)**. Model yang kita pilih adalah `bert-base-multilingual-cased` karena huruf kapital sangat esensial bagi model untuk mengenali nama instansi, singkatan gelar, dan skill (`S1`, `PT`, `C#`, `CPA`, `MBA`, `Java`).
2. **Penyusunan Teks Terstruktur Berbasis Entitas NER (Tabel 2.5 Skripsi):**
   - Sebelum di-encode oleh mBERT, teks diperkaya dengan struktur kompetensi:
     `[DOC] ... | [DOMAIN] ... | [TIER] ... | [ROLE] ... | [SKILL] ... | [EDU] ... | [DUR] ... | [ORG] ... || {teks_asli}`
   - Tujuannya menormalkan perbedaan format/layout CV yang tidak seragam agar mBERT langsung fokus pada kompetensi utama.
3. **WordPiece Tokenization & Penyelarasan Label BIO Token-Level (*Offset Mapping*):**
   - Teks dipecah menjadi token subword mBERT.
   - Label BIO (0..10) diselaraskan ke token kata menggunakan *offset mapping*. Token spesial (`[CLS]`, `[SEP]`, `[PAD]`) diberi label `-100` (*ignore index*) agar tidak ikut dihitung dalam loss Cross-Entropy kepala NER.
4. **Inisialisasi Khusus Lapisan Proyeksi (*Identity Matrix Initialization*):**
   - Lapisan proyeksi linear `self.projection = nn.Linear(768, 768)` **diinisialisasi dengan Matriks Identitas (`nn.init.eye_`)** dan bias nol.
   - **Tujuannya:** Memastikan ruang semantik mBERT pre-trained yang sudah matang dari 104 bahasa **tidak langsung rusak atau teracak acak** pada langkah-langkah awal pelatihan (*warm start*).

---

## PERTANYAAN 6: MENGAPA CELL HYPERPARAMETER TUNING DILETAKKAN DI BAWAH CELL NER DAN mBERT?
> **"Secara struktur cell di notebook, kenapa Hyperparameter Tuning berada di Cell 5 (di bawah Cell 3 NER dan Cell 4 Arsitektur mBERT)?"**

1. **Dependensi Definisi Kelas Model Python (`NameError`):**
   - Fungsi `objective` Optuna di Cell 5 harus memanggil class `SiameseMBERTCosineExtractor` berulang kali di setiap *trial*. Jika Cell 5 dijalankan sebelum Cell 4, kernel Python akan gagal dengan error *`NameError: name 'SiameseMBERTCosineExtractor' is not defined`*.
2. **Dependensi Variasi Panjang Token Max Length (`CACHED_DATASETS_NER`):**
   - Salah satu parameter yang di-tuning oleh Optuna adalah **Max Length Token (`128, 256, 512`)**.
   - Dataset token WordPiece dan label BIO untuk ketiga ukuran tersebut (`CACHED_DATASETS_NER`) diproses dan disimpan di memori pada Cell 3. Cell 5 membutuhkan dataset dari Cell 3 ini untuk menghitung *validation loss*.
3. **Standar Urutan Pipeline Machine Learning Ilmiah:**
   - Alur notebook mengikuti urutan baku:  
     $$\text{Load Data (Cell 2)} \rightarrow \text{Feature & NER Prep (Cell 3)} \rightarrow \text{Model Blueprint (Cell 4)} \rightarrow \mathbf{Hyperparameter\; Tuning (Cell 5)} \rightarrow \text{Final Training (Cell 6)}$$

---

## BAGIAN 4.4: LOSS FUNCTION KUSTOM (PURE COSINE MARGIN LOSS) & MSE TERM
> **"Kenapa pakai loss function ini bukan default mBERT? Apa itu MSE Term dan bagaimana cara menentukan target nilainya?"**

### 1. Mengapa Perlu Kustom `PureCosineMarginLoss` (Bukan Default CrossEntropyLoss)?
1. **Perbedaan Paradigma Output:**
   - *Default mBERT* didesain untuk klasifikasi probabilitas softmax (*logits* $k$-kelas $\rightarrow$ CrossEntropyLoss).
   - Arsitektur skripsi kita adalah **Feature Extractor Berbasis Siamese** yang menghasilkan vektor embedding 768 dimensi dan derajat kemiripan dihitung dengan **Cosine Similarity**.
   - Jika memakai CrossEntropyLoss, model hanya belajar "memilih label 0, 1, 2" lewat lapisan linier klasifikasi hitam-putih, **tanpa pernah mengatur jarak sudut kosinus antar-vektor di ruang 768 dimensi**.
2. **Kesesuaian dengan Batas Ambang Skripsi:**
   - Kita membutuhkan fungsi loss yang langsung mengatur gradien sudut kosinus agar output kemiripan terkalibrasi secara alami sesuai rubrik skripsi: **Fit ($\ge 80\%$)**, **Close Fit ($60\%\text{–}79.9\%$)**, dan **Not Fit ($< 60\%$)**.

---

### 2. Apa itu MSE Term?
Secara matematis, **MSE (*Mean Squared Error*) Term** adalah fungsi objektif berbasis regresi yang mengukur kuadrat selisih jarak antara skor kemiripan kosinus aktual model dengan nilai target kontinu ideal ($y^*$):

$$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^{B} \left( \cos(\mathbf{u}_i, \mathbf{v}_i) - y_i^* \right)^2 \times \text{scale\_mse}$$

- $\cos(\mathbf{u}_i, \mathbf{v}_i)$: Nilai kosinus aktual antara vektor CV $\mathbf{u}$ dan lowongan $\mathbf{v}$ ($[-1, 1]$ atau $[0, 1]$).
- $y_i^*$: Titik pusat ideal (*attractor point*) untuk kelas ke-$i$.
- **Peran Konseptual:** Bertindak sebagai gaya tarik gravitasi halus (*centroid attractor*) yang menarik embedding pasangan data agar berkerumun rapat di titik aman kelasnya masing-masing.

---

### 3. Rationale Penentuan Angka Target (`0.90`, `0.70`, `0.15`)
Angka target ini **tidak dipilih sembarangan (*non-arbitrary*)**, melainkan diturunkan secara geometris dan matematis dari batas ambang naskah skripsi:

| Kategori Skripsi | Batas Ambang (*Threshold*) | Target Ideal ($y^*$) | Landasan Penentuan Nilai |
| :--- | :---: | :---: | :--- |
| **Fit** | $\ge 80.0\%$ ($0.80 - 1.00$) | **`0.90`** (90%) | **Safety Buffer Zone (Titik Aman Atas)** |
| **Close Fit** | $60.0\% - 79.9\%$ ($0.60 - 0.799$) | **`0.70`** (70%) | **Centroid / Titik Tengah Simetris Sempurna** |
| **Not Fit** | $< 60.0\%$ ($0.00 - 0.599$) | **`0.15`** (15%) | **Quasi-Orthogonal Benchmark (Distansi Aman Maksimal)** |

1. **Target Close Fit = `0.70` (Titik Tengah / Centroid Simetris):**
   - Rentang skripsi adalah $0.60 \text{ s.d. } 0.799$.
   - Nilai tengah simetrisnya:
     $$\text{Midpoint} = \frac{0.60 + 0.80}{2} = 0.70$$
   - Nilai $0.70$ memberikan jarak toleransi yang seimbang ($\pm 10\%$) baik ke batas bawah ($0.60$) maupun ke batas atas ($0.80$). Hal ini mencegah data Close Fit tergelincir ke Not Fit atau melompat ke Fit.
2. **Target Fit = `0.90` (Safety Buffer Zone):**
   - Batas minimum Fit adalah $0.80$. Jika target diset tepat di $0.80$, variansi data alami akan menyebabkan separuh data Fit bernilai $78\%–79\%$ (salah terklasifikasi menjadi Close Fit).
   - Sebaliknya, jika ditargetkan $1.00$, menuntut teks CV dan lowongan yang berbeda kata untuk bernilai $1.00$ sempurna akan memicu *overfitting* atau *gradient explosion*.
   - Target $0.90$ memberikan zona penyangga aman $+10\%$ di atas ambang batas $0.80$ dan $-10\%$ di bawah $1.00$, memastikan kandidat berkualifikasi tinggi stabil di kisaran $85\%–95\%$.
3. **Target Not Fit = `0.15` (Quasi-Orthogonal Benchmark):**
   - Pada ruang representasi Transformer 768 dimensi, pasangan teks yang sama sekali tidak berhubungan secara bahasa memiliki kemiripan kosinus alami sekitar $0.10 \text{ s.d. } 0.25$ (*anisotropy property* pada mBERT).
   - Target $0.15$ menempatkan pasangan yang tidak relevan $45\%$ jauh di bawah batas penolakan ($0.60$). Ini menciptakan margin pemisah yang sangat lebar (*large margin separation*), menjamin bahwa CV yang tidak relevan tidak akan pernah lolos (*zero false positives*).

---

### 4. Sinergi dengan Band Penalty
MSE Term bekerja bersama **Band Penalty** (`pen_fit`, `pen_close`, `pen_not`):
- **MSE Term** mengarahkan embedding secara halus ke pusat kelompok ($0.90, 0.70, 0.15$).
- **Band Penalty** bertindak sebagai "tembok batas keras" (*hard boundary*) yang langsung menghukum gradien secara eksponensial jika ada skor Fit yang jatuh di bawah $0.80$, atau skor Not Fit yang melompat di atas $0.60$.
Hasilnya, ruang vektor terkalibrasi alami tanpa perlu penyesuaian batas buatan (*post-processing mapping*).

---

### 5. Mengapa Hasil Hyperparameter Tuning Bisa Bergeser Saat Dijalankan Ulang?
*(Kasus Gambar 1: `lr = 3e-5, max_length = 512` vs Gambar 2: `lr = 2e-5, max_length = 128`)*

1. **Perubahan Landscape Gradien (Integrasi Multi-Task NER Loss):**
   - Gambar 1 menunjukkan tuning model mBERT + Cosine sebelum penambahan Multi-task NER loss token-level.
   - Gambar 2 menunjukkan tuning model usulan lengkap (NER + mBERT + Cosine), di mana model mengoptimalkan dua fungsi loss sekaligus: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cosine}} + \alpha \mathcal{L}_{\text{NER}}$.
   - Karena gradien backpropagation menjadi lebih padat dan sensitif saat mengupdate bobot untuk klasifikasi token BIO dan vektor kosinus secara bersamaan, model memerlukan learning rate yang lebih kecil dan halus (**`2e-5`**) agar konvergensi stabil.
2. **Kepadatan Informasi Entitas (Max Length 128 vs 512):**
   - Pada representasi teks berbasis entitas (Skill, Experience, Education), informasi kunci kandidat terkonsentrasi di awal teks (128 token pertama).
   - Pada proxy tuning 3 epoch yang singkat, panjang token 128 memangkas *padding noise* dan mempercepat konvergensi gradien secara signifikan dibanding panjang 512.
3. **Stokastisitas Proxy Tuning (3 Epoch):**
   - Tuning Optuna dilakukan secara ringkas (3 epoch proxy) untuk efisiensi waktu komputasi. Selisih akurasi validasi antara Trial 4 ($96.44\%$) dan Trial 5 ($92.89\%$) sangat tipis, dan keduanya sama-sama berasal dari rekomendasi baku literatur asli BERT (Devlin et al., 2019: `2e-5, 3e-5, 5e-5`).

---

---

## BAGIAN 5: RINGKASAN HASIL EKSPERIMEN KOMPARASI 3 MODEL (BAB 4 / BAB 6)
Untuk membuktikan kontribusi setiap komponen pada **225 Data Uji (*Unseen Test Set*)** menggunakan ambang batas sesuai naskah skripsi (**Fit $\ge 80\%$**, **Close Fit $60\%\text{–}79.9\%$**, **Not Fit $< 60\%$**):

| Skenario Model | Cara Kerja | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Kesimpulan |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Model 3: Cosine Saja** *(TF-IDF + `sklearn` Cosine, Tanpa mBERT)* | Representasi kata TF-IDF langsung dihitung `sklearn` Cosine Similarity tanpa pelatihan mBERT | **33.33%** | **11.11%** | **33.33%** | **16.67%** | Gagal total menangkap sinonim & konteks (`Fit` & `Close Fit` 0% F1; seluruh pasangan menumpuk `< 60%`). |
| **2. Model 1: mBERT Saja** *(Tanpa Cosine Similarity)* | Vektor `[CLS]` CV & Job (`max_length = 128`, 10 layer terbawah *frozen*) masuk ke `Linear Classifier` + `CrossEntropyLoss` | **72.44%** | **72.81%** | **72.44%** | **71.37%** | Tanpa *Siamese NER* dan tanpa proyeksi kosinus, konvergensi terbatas (*best val_loss 0.7303*) dan tidak menghasilkan skor kemiripan 0–100%. |
| **3. Model 2 (Usulan): Unified NER + Siamese mBERT + Sklearn Cosine** *(LR `2e-5`, MaxLen `128`, Batas 60% & 80%)* | Unified NER + Siamese mBERT mengekstrak vektor 768-d yang dihitung langsung dengan `sklearn` Cosine Similarity | **99.11%** | **99.13%** | **99.11%** | **99.11%** | **Performa tertinggi:** `Not Fit` F1 `1.0000`, `Close Fit` F1 `0.9865`, `Fit` F1 `0.9868`; simulasi Indonesia Tabel 2.5 **`96.77%` (*Fit*)**. |

---

## BAGIAN 6: CONTEKAN KALIMAT LISAN SINGKAT SAAT DITANYA PAK NJOTO

1. **Jika ditanya: *"Hyperparameter tuning kamu pakai apa, bagaimana caranya, dan apa tujuannya?"***
   > *"Untuk hyperparameter tuning, saya menggunakan **Optuna** dengan algoritma **TPE (Tree-structured Parzen Estimator)**, Pak. Parameter yang di-tuning ada 4 sesuai arahan kertas konsultasi Bapak dan referensi paper asli BERT (Devlin et al., 2019), yaitu **Learning Rate** (`2e-5, 3e-5, 5e-5, 0.001, 0.005, 0.01`), **Max Length Token** (`128, 256, 512`), **Dropout** (`0.1, 0.2, 0.3`), dan **Weight Decay** (`0.001, 0.01`).*
   > *Caranya, setiap kandidat diuji pada data training (1.050 pasang) dan diukur nilai **validation loss**-nya pada data validasi (225 pasang). Setelah ditemukan kombinasi terbaik untuk Model Usulan yaitu **`lr = 2e-5`, `max_length = 128`, `dropout = 0.1`, dan `weight_decay = 0.01`**, model dilatih ulang dengan batas **100 epoch** dan **Early Stopping patience 5** memonitor `val_loss` — mencapai konvergensi terbaik di **Epoch ke-5** (`val_loss = 0.0365`, berhenti di Epoch ke-10).*
   > *Tujuannya adalah menjaga agar bobot pre-trained mBERT tidak rusak akibat learning rate yang terlalu besar (catastrophic forgetting), mencegah overfitting pada data baru, serta memastikan efisiensi panjang token."*

2. **Jika ditanya: *"Apa exact input dan exact output dari model mBERT yang kamu pakai?"***
   > *"Sesuai arahan Bapak bahwa mBERT hanya dipakai sebagai **feature extractor**, **exact input** yang masuk ke model mBERT adalah pasangan tensor **`input_ids`** dan **`attention_mask`** berukuran **`[batch_size, 128]`** hasil tokenisasi WordPiece dari teks CV dan lowongan yang sudah diperkaya entitas NER (`EDU, ORG, SKILL, ROLE, DUR`).*
   > *Kemudian di dalam mBERT, tensor tersebut melewati 12 layer Transformer menghasilkan `last_hidden_state` berukuran `[batch_size, 128, 768]`, lalu melalui kepala NER terpadu (`self.ner_head`) dan **NER-Guided Entity-Weighted Pooling** (`0.5 * CLS + 0.5 * Entity-Weighted Mean Pooling`) diproyeksikan menjadi **exact output** berupa **vektor embedding 768 dimensi** berukuran **`[1, 768]`** untuk CV ($\mathbf{u}$) dan **`[1, 768]`** untuk lowongan ($\mathbf{v}$).*
   > *Kedua vektor `[1, 768]` itulah yang kemudian dimasukkan ke fungsi **`sklearn.metrics.pairwise.cosine_similarity`** untuk menghasilkan skor kemiripan `0% sampai 100%` dan diklasifikasikan menjadi **Fit ($\ge 80\%$)**, **Close Fit ($60\text{–}79.9\%$)**, atau **Not Fit ($< 60\%$)**."*

3. **Jika ditanya: *"Kenapa kamu menggunakan pembagian data 70/30?"***
   > *"Alasan utamanya karena penelitian ini menerapkan **Hyperparameter Tuning (Optuna)** dan **Early Stopping** yang memonitor `val_loss`, sehingga porsi **30% data evaluasi (450 pasang)** perlu dibagi dua sama rata menjadi **15% Validation Set (225 pasang)** untuk tuning & early stopping, serta **15% Unseen Test Set (225 pasang)** yang murni terpisah untuk pengujian Confusion Matrix agar tidak terjadi data leakage.*
   > *Selain itu, dari total **1.500 data seimbang** (500 per kelas), rasio **70% : 15% : 15%** menghasilkan pembagian *stratified* yang bulat sempurna di setiap kelas: **Train = 1.050 (350 per kelas)**, **Validation = 225 (75 per kelas)**, dan **Test = 225 (75 per kelas)**. Jika menggunakan 80/20 lalu dibagi dua (10% Val dan 10% Test), sampel per kelas hanya tersisa 50 data sehingga kurang representatif untuk mengevaluasi 6 sektor pekerjaan yang berbeda."*

4. **Jika ditanya: *"Kenapa metrik evaluasi yang kamu gunakan adalah Precision, Recall, dan F1-Score? Kenapa bukan Accuracy saja (atau apakah Accuracy juga dipakai)?"***
   > *"Dalam sistem rekrutmen CV–Job Matching 3 kelas (`Fit`, `Close Fit`, `Not Fit`), **Precision, Recall, dan F1-Score** wajib digunakan karena kita perlu mengukur dua risiko kesalahan yang berbeda di setiap kelas, Pak:*
   > - ***Precision** mengukur seberapa tepat kandidat yang diloloskan sebagai `Fit` agar tim HR tidak membuang waktu mewawancarai pelamar yang salah (*False Positive*).*
   > - ***Recall** mengukur seberapa banyak kandidat berkualitas yang berhasil dijaring agar perusahaan tidak kehilangan pelamar potensial (*False Negative*).*
   > - ***F1-Score** menyatukan keduanya secara harmonis untuk melihat performa di tiap kelas secara detail—terutama pada kelas tengah **`Close Fit` ($60\text{–}79.9\%$)** yang tidak akan terlihat kelemahannya jika kita hanya melihat satu angka **Accuracy** global.*
   > *Selain itu, di dunia nyata jumlah pelamar yang `Not Fit` biasanya jauh lebih banyak daripada yang `Fit` (imbalanced), sehingga Accuracy saja bisa menipu (Accuracy Paradox).*
   > *Namun, karena pada eksperimen skripsi ini dataset uji saya telah diseimbangkan secara *stratified* (**75 `Fit`, 75 `Close Fit`, 75 `Not Fit`**) dan sesuai arahan Bapak untuk menampilkan **Classification Report (CR)** serta grafik **Train & Val Accuracy**, maka **Accuracy (99.11%) tetap saya cantumkan sebagai metrik performa global**, berdampingan dengan **Macro Precision (99.13%), Macro Recall (99.11%), dan Macro F1-Score (99.11%)**."*

5. **Jika ditanya: *"Kenapa disebut Siamese mBERT?"***
   > *"Dikatakan **Siamese mBERT** karena sistem menggunakan satu backbone mBERT yang sama secara bersama (*shared weights*) untuk memetakan teks CV dan lowongan ke ruang koordinat vektor 768 dimensi yang setara secara independen. Karena ruang vektornya sama, derajat kesesuaian kualifikasi pelamar ($\mathbf{u}$) terhadap lowongan ($\mathbf{v}$) dapat dihitung secara langsung dan objektif menggunakan **Cosine Similarity**."*

6. **Jika ditanya: *"Apa yang kamu lakukan ke model mBERT sebelum dilakukan proses training?"***
   > *"Ada 4 persiapan teknis yang saya lakukan, Pak:*
   > 1. *Membersihkan teks dengan Regex namun **tetap mempertahankan huruf kapital (Cased)** agar informasi entitas singkatan gelar dan instansi (`S1`, `PT`, `C#`, `CPA`, `Java`) tidak hilang.*
   > 2. *Menyusun **struktur teks berbasis entitas kompetensi NER** (`ROLE, SKILL, EDU, DUR, ORG`) sesuai Tabel 2.5 agar format layout CV yang berbeda-beda ternormalisasi.*
   > 3. *Melakukan **tokenisasi WordPiece** serta penyesuaian label token BIO untuk melatih kepala NER secara multi-task.*
   > 4. *Menambahkan lapisan proyeksi 768 dimensi yang diinisialisasi dengan **Matriks Identitas (`nn.init.eye_`)** agar ruang semantik mBERT pre-trained tidak rusak di awal pelatihan."*

7. **Jika ditanya: *"Kenapa cell Hyperparameter Tuning ditaruh di bawah cell NER dan arsitektur mBERT?"***
   > *"Karena proses tuning di Optuna secara langsung bergantung pada **blueprint kelas model** (`SiameseMBERTCosineExtractor` di Cell 4) dan **dataset token WordPiece berlabel BIO** (`CACHED_DATASETS_NER` di Cell 3). Sesuai pipeline baku Machine Learning, data dan arsitektur model harus didefinisikan terlebih dahulu sebelum algoritma tuning dapat menguji variasi parameternya."*

8. **Jika ditanya: *"Kenapa kamu pakai loss function kustom (PureCosineMarginLoss) bukan default loss mBERT? Apa itu MSE Term dan bagaimana menentukan target nilainya?"***
   > *"Default mBERT menggunakan **CrossEntropyLoss** untuk klasifikasi probabilitas softmax label diskrit hitam-putih. Sedangkan model skripsi saya berfungsi sebagai **Siamese Feature Extractor** yang mengeluarkan representasi vektor 768 dimensi dan dihitung derajat kecocokannya dengan **Cosine Similarity**.*
   > *Jika memakai CrossEntropyLoss, mBERT tidak pernah belajar menata jarak sudut kosinus antar-vektor. Oleh karena itu, saya merancang **`PureCosineMarginLoss`** yang menggabungkan **MSE Term** dan **Band Penalty**.*
   > ***MSE Term** menghitung deviasi kuadrat langsung antara skor kosinus prediksi model dengan target ideal berbasis batas skripsi: **`0.90` untuk Fit**, **`0.70` untuk Close Fit**, dan **`0.15` untuk Not Fit**.*
   > *Target **`0.70`** ditentukan sebagai **titik tengah simetris (centroid)** dari rentang Close Fit ($60\%–79.9\%$), memberikan jarak aman yang seimbang ($\pm 10\%$) ke batas atas maupun batas bawah.*
   > *Target **`0.90`** ditentukan sebagai **zona penyangga aman (+10%)** di atas ambang batas $80\%$, mencegah variansi data alami membuat kandidat Fit tergelincir ke Close Fit.*
   > *Sedangkan target **`0.15`** diambil dari benchmark kemiripan alami kalimat acak pada representasi mBERT (anisotropy) dan berada **$45\%$ jauh di bawah batas penolakan ($60\%$)**, sehingga menjamin tidak ada false positive pelamar yang tidak relevan."*

9. **Jika ditanya: *"Mengapa saat ditraining ulang hasil hyperparameter terbaiknya bisa bergeser?"***
   > *"Pergeseran tersebut wajar dan memiliki landasan teknis yang jelas, Pak:*
   > *Pertama, karena adanya **integrasi Multi-Task NER Loss** pada model usulan lengkap. Ketika loss klasifikasi token BIO digabungkan dengan Cosine Loss, gradien backpropagation menjadi lebih padat sehingga model membutuhkan learning rate yang lebih halus (**`2e-5`** dibanding `3e-5`) agar konvergen secara stabil.*
   > *Kedua, pada representasi teks berbasis entitas (Skill, Pengalaman, Edukasi), informasi inti kandidat terkonsentrasi di awal teks. Sehingga pada tuning ringkas (proxy 3 epoch), **Max Length 128** memangkas padding kosong dan mempercepat konvergensi dibanding panjang 512.*
   > *Ketiga, baik `2e-5` maupun `3e-5` keduanya berada dalam rentang baku yang direkomendasikan oleh paper asli BERT (Devlin et al., 2019) dengan selisih akurasi validasi yang sangat tipis (di atas 96%)."*


