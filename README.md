# ⚖️ Sistem Prediksi Putusan Hukum (Perlindungan Anak) Berbasis Case-Based Reasoning (CBR)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Sentence--Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/Groq_LLM-f55036?style=for-the-badge&logo=groq&logoColor=white)

**Tugas / Proyek Penalaran Komputer — SubCPMK-3**  

</div>

---

## 👨‍💻 Informasi Pengembang

| Atribut | Detail |
| :--- | :--- |
| **Tim Pengembang** | **Ahmad Nur Mu'minin** & **Wahyu Andika** |
| **Mata Kuliah** | Penalaran Komputer |
| **Topik** | Case-Based Reasoning (CBR) |
| **Domain Kasus** | Hukum Perlindungan Anak |
| **Jumlah Kasus** | 86 dokumen PDF putusan pengadilan |

---

## 📖 Latar Belakang & Studi Kasus

Proyek ini menerapkan paradigma **Case-Based Reasoning (CBR)** untuk memprediksi hasil akhir (amar putusan dan vonis) dari sebuah kasus hukum Perlindungan Anak yang baru, berdasarkan kemiripannya dengan kasus-kasus hukum lampau. 

Sistem ini mengekstraksi representasi dari setiap dokumen putusan pengadilan, mencari (retrieve) *top-K* kasus historis yang paling ekuivalen menggunakan teknik pemrosesan bahasa alami (NLP), dan menggunakan analisis mayoritas (*majority voting*) atau perhitungan kemiripan terbobot (*weighted similarity*) untuk merekomendasikan vonis bagi kasus baru. Kami juga mengintegrasikan LLM (Groq) untuk memproduksi rangkuman prediksi yang lebih naratif dan mudah dimengerti.

---

## ⚙️ Metodologi & Siklus CBR

Proyek terbagi dan terstruktur menjadi 5 tahap (notebook) yang mencakup 4 langkah utama siklus (R4) dalam CBR: **Retrieve, Reuse, Revise, dan Retain**.

### 1. Build Case Base & Representation (Tahap 1 & 2)
- **Ekstraksi Data:** Memproses metadata putusan pengadilan menjadi format JSON yang terstruktur.
- **Preprocessing:** Pembersihan noise, stopword removal (bahasa Indonesia), standarisasi teks.
- **Representasi Kasus:** Memadukan field vital (Pasal Dakwaan, Amar Putusan, Fakta) dengan pembobotan tertentu (*weighted representation*) menjadi representasi dokumen.

### 2. Case Retrieval (Tahap 3)
Sistem menggunakan dua pendekatan berbeda untuk mencocokkan kemiripan teks kasus:
*   **TF-IDF + SVM:** Pendekatan statistik klasik berbasis frekuensi kata yang sangat efisien untuk menangkap kata-kata kunci hukum/pasal.
*   **SentenceTransformer (Embedding):** Menggunakan representasi *multilingual* (`paraphrase-multilingual-MiniLM-L12-v2`) untuk menangkap kedekatan makna dan semantik dari rincian fakta secara kontekstual.

### 3. Solution Reuse & Prediction (Tahap 4)
Prediksi hasil putusan (reuse) diturunkan dari evaluasi Top-K dokumen kasus yang paling relevan:
- **Majority Vote:** Memilih label putusan dominan yang paling banyak muncul di antara kasus-kasus temuan Top-K.
- **Weighted Similarity:** Pembobotan *outcome* secara proporsional berbasis probabilitas skor kemiripan Cosine.
- **Groq LLM Summary:** Memanfaatan Generative LLM untuk merangkum hasil prediksi putusan dan solusi akhir berdasarkan detail Top-K rujukan.

### 4. Evaluation (Tahap 5)
Mengevaluasi secara analitik kualitas deteksi NLP. Menghitung Precision@K, F1 Score, visualisasi Matriks Konfusi (Confusion Matrix) dengan pendekatan `Leave-One-Out`, dan perbandingan utuh efektivitas *TF-IDF vs Embeddings* terhadap domain terminologi hukum.

---

## 📈 Arsitektur Folder

```text
📁 notebooks/
├── 📁 STEP 1/                          : Pembuatan Case Base (Parsing metadata ke DataFrame)
│   ├── 01_build_case_base_revised.ipynb
│   ├── 01_build_case_base_ocr_fixed.ipynb
│   ├── 📁 data/
│   │   ├── metadata_raw.json
│   │   ├── metadata_with_pdf.json
│   │   ├── 📁 pdf/                     : File PDF putusan pengadilan
│   │   ├── 📁 processed/               : Data hasil preprocessing
│   │   ├── 📁 raw/                     : Data mentah sebelum diolah
│   │   └── 📁 eval/                    : Data untuk evaluasi
│   └── 📁 logs/
│       ├── cleaning.log
│       ├── debug_hal1.html
│       ├── representation.log
│       └── stats_tahap1.json
│
├── 📁 STEP 1 MANUAL/                   : Ekstraksi teks manual fallback (Opsional)
│   ├── 01_build_case_base_manual.ipynb
│   ├── 📁 data/
│   └── 📁 logs/
│
├── 📁 STEP 2/                          : Case Representation (JSON structure & framing teks)
│   ├── 02_case_representation.ipynb
│   └── 📁 data/
│       ├── 📁 pdf/
│       ├── 📁 processed/
│       └── 📁 raw/
│
├── 📁 STEP 3/                          : Case Retrieval (TF-IDF vs Sentence Embedding)
│   ├── 03_retrieval.ipynb
│   ├── 📁 data/
│   │   └── 📁 eval/                    : Output metrik retrieval awal
│   └── 📁 models/                      : Pre-trained files hasil training
│       ├── cases_index.csv
│       ├── embeddings.npy
│       ├── label_encoder.pkl
│       ├── svm_model.pkl
│       ├── tfidf_matrix.pkl
│       └── tfidf_vectorizer.pkl
│
├── 📁 STEP 4/                          : Solution Reuse (Prediksi majority/weighting + Groq LLM)
│   ├── 04_solution_reuse.ipynb
│   ├── 04_solution_reuse_refactored.ipynb
│   └── 📁 data/
│       └── 📁 results/                 : Output prediksi (predictions.csv, predictions.json)
│
└── 📁 STEP 5/                          : Evaluation (Akurasi, Recall, Confusion Matrix, Plot)
    ├── 05_evaluation.ipynb
    └── 05_evaluation_refactored.ipynb

📁 data/                                : Dataset utama (putusan PDF, metadata, hasil eval)
├── metadata_raw.json
├── metadata_with_pdf.json
├── 📁 pdf/                             : File PDF putusan pengadilan
├── 📁 processed/                       : Data terproses siap pakai
├── 📁 raw/                             : Data mentah
└── 📁 eval/                            : Hasil evaluasi & metrik
    ├── cases.csv / cases.json
    ├── queries.json
    ├── retrieval_results.json
    ├── retrieval_metrics.csv
    ├── prediction_metrics.csv
    ├── metrics_awal.json
    ├── full_report.json
    ├── plot_confusion_matrix.png
    └── plot_perbandingan_model.png
```

---

## 🚀 Cara Instalasi & Menjalankan

1. **Clone Repository**
   ```bash
   git clone <URL_REPO_ANDA>
   cd Penalaran-Komputer-Sub-3
   ```

2. **Install Dependencies**

   Install semua library sekaligus:
   ```bash
   pip install scikit-learn sentence-transformers pandas numpy tqdm joblib matplotlib seaborn groq \
               requests beautifulsoup4 pdfminer.six lxml undetected-chromedriver selenium
   ```

   Atau install per tahap sesuai kebutuhan:

   ```bash
   # STEP 1 — Build Case Base (scraping & ekstraksi PDF)
   pip install requests beautifulsoup4 pdfminer.six lxml tqdm undetected-chromedriver selenium

   # STEP 1 (versi OCR) — tambahan untuk ekstraksi teks via OCR
   pip install pdf2image pytesseract pillow PyMuPDF

   # STEP 2 — Case Representation (via Groq LLM)
   pip install groq pandas tqdm

   # STEP 3 — Case Retrieval (TF-IDF & Sentence Embedding)
   pip install scikit-learn sentence-transformers numpy pandas joblib tqdm

   # STEP 4 — Solution Reuse & Prediksi
   pip install scikit-learn sentence-transformers numpy pandas joblib groq

   # STEP 5 — Evaluation & Visualisasi
   pip install scikit-learn sentence-transformers numpy pandas joblib matplotlib seaborn
   ```

   > **Catatan:** STEP 1 versi OCR membutuhkan instalasi **Tesseract OCR** secara terpisah di sistem operasi. Unduh dari [tesseract-ocr.github.io](https://tesseract-ocr.github.io/tessdoc/Installation.html).

3. **Jalankan Pipeline per Notebook**
   Sangat disarankan menjalankan proyek ini secara urut dari `STEP 1` hingga `STEP 5`:
   - Buka dan jalankan **STEP 1** (`01_build_case_base_revised.ipynb`) untuk membangun case base dari data putusan PDF.
   - Jalankan **STEP 2** (`02_case_representation.ipynb`) untuk membuat representasi kasus. Masukkan `Groq API Key` saat diminta.
   - Jalankan **STEP 3** (`03_retrieval.ipynb`) untuk melatih model TF-IDF & Embedding dan menyimpan ke folder `models/`.
   - Jalankan **STEP 4** (`04_solution_reuse.ipynb`) untuk memprediksi putusan kasus baru. Masukkan `Groq API Key` saat diminta.
   - Buka **STEP 5** (`05_evaluation.ipynb`) untuk melihat laporan evaluasi performa dan visualisasi Confusion Matrix.

---

<div align="center">
  <p>Made with  by Wahyu Andika & Ahmad Nur Mu'minin</p>
</div>
