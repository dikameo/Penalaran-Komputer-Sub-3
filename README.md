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
| **Tim Pengembang** | **Wahyu Andika** & **Ahmad Nur Mu'minin** |
| **Mata Kuliah** | Penalaran Komputer |
| **Topik** | Case-Based Reasoning (CBR) |
| **Domain Kasus** | Hukum Perlindungan Anak |

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
📁 STEP 1/        : Pembuatan Case Base (Parsing metatada ke array DataFrame)
📁 STEP 1 MANUAL/ : Ekstraksi teks manual fallback (Opsional)
📁 STEP 2/        : Case Representation (JSON structure & framing teks)
📁 STEP 3/        : Case Retrieval (Retrieval Algoritma via TF-IDF vs Embedding)
📁 STEP 4/        : Solution Reuse (Prediksi majority/weighting + Groq LLM)
📁 STEP 5/        : Evaluation (Hitung akurasi, Recall, Analisis Error, Plot visual)
📁 data/          : Penyimpanan repositori dataset mentah, JSON putusan, dan metadata
📁 logs/          : Pencatatan history (logging) performa eksekusi
📁 models/        : Output penyimpanan pre-trained files (SVM model, Tfidf matrix, npy)
```

---

## 🚀 Cara Instalasi & Menjalankan

1. **Clone Repository**
   ```bash
   git clone <URL_REPO_ANDA>
   cd Penalaran-Komputer-Sub-3
   ```

2. **Install Dependencies**
   Library esensial untuk ML and NLP:
   ```bash
   pip install scikit-learn sentence-transformers pandas numpy tqdm joblib matplotlib seaborn groq
   ```

3. **Jalankan Pipeline per Notebook**
   Sangat disarankan menjalankan proyek ini secara urut dari `STEP 1` hingga `STEP 5`:
   - Lakukan setup data dan representasi kasus di **Tahap 1 & 2**.
   - Buka dan jalankan **Tahap 3** (`03_retrieval (1).ipynb`) untuk mentraining matriks kemiripan ke memori `/models`.
   - Lanjut le **Tahap 4** (`04_solution_reuse (1).ipynb`) untuk mengeksekusi prediksi penyelesaian masalah di kasus baru. (Bisa menaruh `Groq API Key` pada prompt yang diminta)
   - Buka **Tahap 5** (`05_evaluation.ipynb`) untuk melihat laporan grafis evaluasi performa klasifikasi CBR.

---

<div align="center">
  <p>Made with ❤️ by Wahyu Andika & Ahmad Nur Mu'minin</p>
</div>
