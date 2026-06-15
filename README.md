# ⚖️ Sistem CBR Prediksi Putusan Perlindungan Anak

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Sentence--Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Tesseract](https://img.shields.io/badge/Tesseract_OCR-5A5A5A?style=for-the-badge&logo=google&logoColor=white)

**Tugas Penalaran Komputer — SubCPMK-3 | Universitas Muhammadiyah Malang**

</div>

---

## 👨‍💻 Informasi Tim

| Atribut | Detail |
| :--- | :--- |
| **Tim** | Wahyu Andika & Ahmad Nur Mu'minin |
| **Mata Kuliah** | Penalaran Komputer — Semester Genap 2025/2026 |
| **Domain** | Pidana Khusus Anak — PN Denpasar |
| **Corpus** | 43 dokumen putusan (case_047 – case_100) |

---

## 📖 Deskripsi Proyek

Proyek ini mengimplementasikan sistem **Case-Based Reasoning (CBR)** untuk memprediksi amar putusan (verdict) dari sebuah kasus pidana anak baru berdasarkan kemiripannya dengan kasus-kasus historis dari Direktori Putusan Mahkamah Agung RI (putusan3.mahkamahagung.go.id).

Tantangan utama proyek ini adalah sumber PDF yang **terproteksi** — pdfminer hanya mengekstrak watermark/disclaimer, bukan isi putusan. Solusinya adalah pipeline dua tahap: pdfminer dahulu, jika konten non-disclaimer < 300 karakter maka fallback ke OCR menggunakan **Tesseract 5.x dengan bahasa Indonesia (ind)** melalui pdf2image (Poppler) pada 200 DPI.

---

## ⚙️ Arsitektur Sistem & Siklus CBR

Pipeline terdiri dari 5 notebook yang berjalan secara berurutan, mencakup siklus CBR **Retrieve → Reuse → Revise → Retain**.

```
43 Protected PDFs
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 · Case Base Construction                   │
│  pdfminer → OCR fallback (Tesseract 200 DPI, ind)   │
│  Output: data/raw/*.txt (43 dokumen)                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 · Case Representation                      │
│  Regex metadata extraction (offline, no API)        │
│  Fields: no_perkara, pasal, amar_putusan, vonis     │
│  Output: data/processed/cases.csv + cases.json      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 · Case Retrieval                           │
│  ┌─────────────────┐  ┌────────────────────────────┐│
│  │ TF-IDF (43×3616)│  │ SentenceTransformer 384-dim││
│  │ + Linear SVM    │  │ paraphrase-multilingual-   ││
│  │                 │  │ MiniLM-L12-v2              ││
│  └─────────────────┘  └────────────────────────────┘│
│  retrieve(query, k=3, mode="tfidf"|"embedding")     │
│  Output: models/*.pkl + embeddings.npy              │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4 · Solution Reuse                           │
│  Majority vote + Weighted similarity                │
│  predict_outcome(query) → predicted_solution        │
│  Output: results/predictions.csv + predictions.json │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5 · Evaluation                               │
│  Leave-One-Out Retrieval (P/R/F1 @ K=3)             │
│  SVM Cross-Validation (Accuracy, macro-F1)          │
│  Error analysis + Rejection analysis                │
│  Output: data/eval/retrieval_metrics.csv            │
│          data/eval/full_report.json                 │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Hasil Eksperimen

| Metrik | TF-IDF | Embedding |
| :--- | :---: | :---: |
| Precision @ K=3 | 0.977 | **1.000** |
| Recall @ K=3 | 0.977 | **1.000** |
| F1-score @ K=3 | 0.977 | **1.000** |

| Metrik SVM (Cross-Validation) | Nilai |
| :--- | :---: |
| Accuracy | 0.907 |
| Precision (macro) | 0.453 |
| Recall (macro) | 0.500 |
| F1-score (macro) | 0.476 |

- **Error rate SVM (train set):** 0.00% (0 / 43 misklasifikasi)
- **Rejection rate retrieval:** 2.3% (1 / 43 kasus — `case_086`, score = 0.0444)
- **Distribusi label:** terbukti = 39 kasus, lainnya = 4 kasus

---

## 📁 Struktur Folder

```
Penalaran-Komputer-Sub-3/
│
├── STEP 1/                          # Tahap 1 — Case Base Construction
│   ├── 01_build_case_base_ocr_fixed.ipynb
│   ├── data/
│   │   ├── pdf/                     # PDF putusan asli (case_001.pdf, dst.)
│   │   ├── raw/                     # Hasil ekstraksi teks (case_XXX.txt)
│   │   └── metadata_with_pdf.json   # Metadata hasil scraping
│   └── logs/
│       └── cleaning.log
│
├── STEP 2/                          # Tahap 2 — Case Representation
│   ├── 02_case_representation.ipynb
│   └── data/
│       └── processed/
│           ├── cases.csv            # Metadata terstruktur (CSV)
│           └── cases.json           # Metadata terstruktur (JSON)
│
├── STEP 3/                          # Tahap 3 — Case Retrieval
│   ├── 03_retrieval.ipynb
│   ├── models/
│   │   ├── tfidf_vectorizer.pkl     # TF-IDF vectorizer (fitted)
│   │   ├── tfidf_matrix.pkl         # TF-IDF matrix (43 × 3616)
│   │   ├── svm_model.pkl            # Linear SVM classifier
│   │   ├── label_encoder.pkl        # Label encoder
│   │   ├── embeddings.npy           # Sentence embeddings (43 × 384)
│   │   └── cases_index.csv          # Index kasus sinkron dengan matrix
│   └── data/
│       └── eval/
│           ├── queries.json         # 5 query uji
│           └── metrics_awal.json    # Metrik retrieval awal
│
├── STEP 4/                          # Tahap 4 — Solution Reuse
│   ├── 04_solution_reuse_refactored.ipynb
│   └── results/
│       ├── predictions.csv          # Hasil prediksi (query_id, predicted_solution, top_5_case_ids)
│       └── predictions.json         # Hasil prediksi lengkap
│
├── STEP 5/                          # Tahap 5 — Evaluation
│   ├── 05_evaluation_refactored.ipynb
│   └── data/
│       └── eval/
│           ├── retrieval_metrics.csv
│           ├── prediction_metrics.csv
│           ├── full_report.json
│           ├── plot_perbandingan_model.png
│           └── plot_confusion_matrix.png
│
└── README.md
```

---

## 🔧 Prerequisites

### A. Python & Library

```bash
# Core
pip install scikit-learn pandas numpy tqdm joblib matplotlib seaborn

# NLP
pip install sentence-transformers

# PDF Extraction
pip install pdfminer.six pdf2image pillow pytesseract

# Opsional (Groq LLM — tidak wajib, sistem jalan tanpa API key)
pip install groq
```

### B. Tesseract OCR (wajib untuk Stage 1)

**Windows:**
1. Download installer dari https://github.com/UB-Mannheim/tesseract/wiki
2. Saat install, centang **"Additional language data"** → pilih **Indonesian**
3. Catat path install (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
4. Set di `CONFIG` notebook 01:
   ```python
   "TESSERACT_PATH": r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

**Linux/Mac:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-ind   # Ubuntu
brew install tesseract tesseract-lang               # Mac
```

### C. Poppler (wajib untuk pdf2image)

**Windows:**
1. Download dari https://github.com/oschwartz10612/poppler-windows/releases
2. Extract, set path folder `bin`-nya di `CONFIG`:
   ```python
   "POPPLER_PATH": r"C:\poppler\Library\bin"
   ```
   Atau tambahkan ke PATH sistem (kosongkan string jika sudah di PATH).

**Linux/Mac:**
```bash
sudo apt install poppler-utils   # Ubuntu
brew install poppler             # Mac
```

---

## 🚀 Cara Menjalankan

> **Penting:** Jalankan notebook **secara berurutan** dari Stage 1 hingga Stage 5. Setiap stage menggunakan output stage sebelumnya.

### Stage 1 — Case Base Construction

```bash
cd "STEP 1"
jupyter notebook 01_build_case_base_ocr_fixed.ipynb
```

Sebelum jalankan, sesuaikan di **Cell 6 (CONFIG)**:
```python
CONFIG = {
    "JENIS_PERKARA"  : "Pidana Khusus Anak",
    "MIN_DOKUMEN"    : 30,           # minimal dokumen
    "TESSERACT_PATH" : r"C:\...",    # path Tesseract kamu
    "POPPLER_PATH"   : r"C:\...",    # path Poppler kamu (atau "" jika di PATH)
    "OCR_DPI"        : 200,          # 150 = cepat, 200 = seimbang, 300 = akurat
    "OCR_LANG"       : "ind",
    "MIN_VALIDITAS"  : 0.05,
    "MIN_KATA"       : 50,
}
```

Jalankan dari **Cell 6** (jika PDF sudah ada di `data/pdf/`) atau dari **Cell 1** untuk scraping ulang.

**Output:** `STEP 1/data/raw/case_XXX.txt`

---

### Stage 2 — Case Representation

```bash
cd "STEP 2"
jupyter notebook 02_case_representation.ipynb
```

Cell 11 sudah diset `GROQ_AKTIF = False` — sistem berjalan **100% offline** menggunakan regex. Tidak perlu Groq API key.

Fungsi ekstraksi yang berjalan:
- `ekstrak_via_regex()` — no_perkara, tanggal, pengadilan, terdakwa, jaksa, pasal_dakwaan, pasal_terbukti, amar_putusan, vonis_penjara, vonis_denda
- `hitung_fitur()` — jumlah_kata, jumlah_kalimat, kata_kunci_pa

**Output:** `STEP 2/data/processed/cases.csv` + `cases.json`

---

### Stage 3 — Case Retrieval

```bash
cd "STEP 3"
jupyter notebook 03_retrieval.ipynb
```

Jalankan **semua cell berurutan**. Cell 15 akan otomatis download model SentenceTransformer (~90 MB) saat pertama kali dijalankan.

Konfigurasi di **Cell 7**:
```python
CONFIG = {
    "TFIDF_MAX_FEATURES" : 5000,
    "TFIDF_NGRAM_RANGE"  : (1, 2),   # unigram + bigram
    "TFIDF_MIN_DF"       : 1,         # 1 untuk corpus kecil
    "SVM_C"              : 1.0,
    "TEST_SIZE"          : 0.2,
    "ST_MODEL"           : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "TOP_K"              : 3,
    "RETRIEVAL_MODE"     : "tfidf",   # "tfidf" atau "embedding"
}
```

Hasil model disimpan di `STEP 3/models/`:
| File | Keterangan |
| :--- | :--- |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (fitted) |
| `tfidf_matrix.pkl` | Matrix sparse (43 × 3616) |
| `svm_model.pkl` | Linear SVM |
| `label_encoder.pkl` | Label encoder |
| `embeddings.npy` | Embeddings (43 × 384) |
| `cases_index.csv` | Index kasus (sinkron dengan matrix) |

**Output:** `STEP 3/data/eval/queries.json` + semua artefak model di `models/`

---

### Stage 4 — Solution Reuse

```bash
cd "STEP 4"
jupyter notebook 04_solution_reuse_refactored.ipynb
```

Sistem prediksi berjalan **tanpa Groq** (Cell 10: `GROQ_AKTIF = False`). Dua algoritma prediksi:

```python
# Majority vote — amar putusan paling sering muncul di top-K
majority_vote_solution(top_k_results)

# Weighted similarity — ambil solusi dari kasus dengan skor tertinggi
weighted_solution(top_k_results)
# Returns: {best_case_id, best_score, amar_putusan, pasal_terbukti, vonis_penjara}
```

Fungsi utama:
```python
predict_outcome(query, k=3, mode="tfidf")
# Returns: {query, top_k_cases, top_k_scores, predicted_solution,
#           predicted_pasal, predicted_vonis, best_case_id, best_score}
```

**Output:** `STEP 4/results/predictions.csv`
```
query_id | predicted_solution | top_5_case_ids | top_5_scores | best_case_id | best_score
```

---

### Stage 5 — Evaluation

```bash
cd "STEP 5"
jupyter notebook 05_evaluation_refactored.ipynb
```

> **Penting:** Pastikan `cases_index.csv` dari Stage 3 tersedia. Stage 5 memuat `df` dari `cases_index.csv` (bukan `cases.json`) agar sinkron dengan dimensi `X_tfidf`.

Evaluasi yang dilakukan:
1. **SVM Classification** — Stratified K-Fold CV (fold = min(5, minority_class_count))
2. **Retrieval Leave-One-Out** — Setiap dokumen jadi query, relevan jika cosine similarity ≥ 0.05
3. **Error Analysis** — misklasifikasi SVM + rejection cases (similarity < 0.05)
4. **Visualisasi** — bar chart P/R/F1 perbandingan model + radar chart SVM + confusion matrix

**Output:**
| File | Keterangan |
| :--- | :--- |
| `retrieval_metrics.csv` | P/R/F1 TF-IDF vs Embedding |
| `prediction_metrics.csv` | Coverage solution reuse |
| `full_report.json` | Laporan lengkap semua metrik |
| `plot_perbandingan_model.png` | Bar chart + radar chart |
| `plot_confusion_matrix.png` | Confusion matrix SVM + distribusi amar |

---

## 🔍 Penggunaan Fungsi Retrieval

Setelah Stage 3 selesai, fungsi `retrieve()` bisa dipakai langsung:

```python
import joblib, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load artefak
tfidf      = joblib.load("STEP 3/models/tfidf_vectorizer.pkl")
X_tfidf    = joblib.load("STEP 3/models/tfidf_matrix.pkl")
embeddings = np.load("STEP 3/models/embeddings.npy")
st_model   = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def retrieve(query: str, k: int = 3, mode: str = "tfidf") -> list:
    """Retrieve top-K kasus paling mirip dengan query."""
    if mode == "tfidf":
        q_vec  = tfidf.transform([query])
        scores = cosine_similarity(q_vec, X_tfidf).flatten()
    else:  # embedding
        q_emb  = st_model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(q_emb, embeddings).flatten()
    top_idx = scores.argsort()[::-1][:k]
    return [{"case_id": f"case_{i+47:03d}", "score": round(float(scores[i]), 4)}
            for i in top_idx]

# Contoh
hasil = retrieve("anak melakukan persetubuhan pasal 81 perlindungan anak", k=3)
for h in hasil:
    print(f"  {h['case_id']} — score: {h['score']}")
```

---

## ⚠️ Troubleshooting

| Error | Penyebab | Solusi |
| :--- | :--- | :--- |
| `Tesseract not found` | Path Tesseract salah | Set `TESSERACT_PATH` di Cell 6 NB01 |
| `poppler not found` | Poppler belum di PATH | Set `POPPLER_PATH` di Cell 6 NB01 |
| `cases.json not found` | Stage 2 belum jalan | Jalankan NB02 dulu |
| `IndexError: index (N) out of range` | `df` tidak sinkron dengan `X_tfidf` | NB05 harus load dari `cases_index.csv`, bukan `cases.json` |
| `NameError: retrieve is not defined` | Cell dijalankan tidak urut | Jalankan semua cell dari atas secara berurutan |
| `SVM is None` | Semua kasus berlabel sama | Normal untuk corpus kecil. Retrieval cosine similarity tetap berjalan. |
| Teks hasil OCR masih disclaimer | Pipeline NB01 lama belum dihapus | Jalankan Cell 22 (hapus `.txt` lama) sebelum Cell 23 (pipeline) |

---

## 📦 Requirements

```
# requirements.txt
scikit-learn>=1.2.0
sentence-transformers>=2.2.0
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.64.0
joblib>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
pdfminer.six>=20221105
pdf2image>=1.16.0
pillow>=9.4.0
pytesseract>=0.3.10
groq>=0.4.0          # opsional
```

Install sekaligus:
```bash
pip install -r requirements.txt
```

---

<div align="center">
  <p>Made with ❤️ by Wahyu Andika & Ahmad Nur Mu'minin — Informatika UMM 2025</p>
</div>
