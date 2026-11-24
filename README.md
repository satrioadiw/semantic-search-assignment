- Nama: SATRIO ADI WIBOWO
- Absen: AI-70
- Program: REWORK AI ENGINEERING BOOTCAMP

# E-Commerce FAQ Semantic Search & RAG Assistant

- Prototype aplikasi tanya-jawab cerdas (*Question-Answering System*) berbasis arsitektur **Retrieval-Augmented Generation (RAG)** untuk layanan pelanggan *e-commerce*. Sistem ini menggabungkan pencarian berbasis makna semantik (**Gemini Embeddings + FAISS**) dengan generasi jawaban kontekstual (**Gemini LLM**).

---

## 1. Problem Statement & Knowledge Base

### Problem Statement
- Sistem pencarian FAQ konvensional yang mengandalkan pencocokan kata kunci (*keyword-based search*) sering gagal memahami pertanyaan pelanggan yang menggunakan sinonim, bahasa sehari-hari (*slang*), atau struktur kalimat yang berbeda dari dokumen resmi. 
- Akibatnya, pelanggan kesulitan menemukan solusi mandiri dan beban tiket *customer service* meningkat. Proyek ini membangun sistem pencarian berbasis makna (*semantic search*) yang mampu memahami maksud (*intent*) pengguna dan menghasilkan jawaban natural yang akurat tanpa halusinasi.

### Knowledge Base
* **Sumber Data:** `ecommerce_faq_knowledge_base.csv`
* **Volume & Struktur Data:** Terdiri dari **59 entri FAQ** yang mencakup **8 kategori** utama layanan *e-commerce* (seperti *Pemesanan*, *Pengiriman*, *Pembayaran*, *Akun*, *Promo*, dll.).
* **Atribut Dataset:** `faq_id`, `category`, `question`, dan `answer`.

---

## 2. Arsitektur & Alur Sistem (RAG Pipeline)

Sistem bekerja melalui lima tahapan utama secara *end-to-end*:

```text
[CSV Knowledge Base] ──▶ (1. Ingest) ──▶ [Unit Retrievable (Q+A)]
                                                │
                                          (2. Embed) ──▶ Gemini Embedding API (3072-d)
                                                │
                                          (3. Store) ──▶ FAISS Vector Index (IndexFlatL2)
                                                │
[User Query] ──────────▶ (4. Retrieve) ─────────┴──▶ Top-K Relevant Contexts
                                                            │
                                                      (5. Generate) ──▶ Prompt + Gemini LLM ──▶ [Final Answer]
```

1. **Ingest:** Memuat data CSV menggunakan `pandas` dan menggabungkan kolom `question` serta `answer` menjadi satu unit teks utuh (`Q: {question} A: {answer}`) agar representasi maknanya kaya konteks.
2. **Embed:** Mengonversi setiap unit teks FAQ menjadi vektor numerik berdimensi **3072** menggunakan model **`gemini-embedding-001`** melalui Google GenAI SDK.
3. **Store:** Menyimpan seluruh vektor embedding ke dalam *in-memory vector database* **FAISS** menggunakan indeks **`IndexFlatL2`** (perhitungan jarak *Euclidean / L2 Distance*).
4. **Retrieve:** Ketika pengguna memasukkan *query*, sistem mengubah *query* menjadi vektor menggunakan model embedding yang sama, lalu mencari **Top-3 (`top_k=3`)** entri FAQ dengan jarak vektor terdekat.
5. **Generate:** Menyisipkan hasil *Top-3* FAQ ke dalam *structured prompt* (berisi *Persona*, *Task*, dan *Grounding Rules*), lalu mengirimkannya ke model **`gemini-3.5-flash-lite`** (`temperature=0.2`) untuk merangkum jawaban yang ramah dan faktual.

---

## 3. Hasil Pengujian Query & Analisis Relevansi

Pengujian dilakukan menggunakan beberapa kategori *query* untuk mengevaluasi ketahanan sistem:

| Kategori Uji | Contoh Query Pengguna | Hasil Retrieval (Top-1) | Evaluasi & Analisis Relevansi |
| :--- | :--- | :--- | :--- |
| **Direct Match** | *"Apakah saya bisa membayar dengan cicilan?"* | *Apakah saya bisa membayar dengan cicilan?* | **Sangat Relevan.** Sistem langsung menangkap entri yang tepat di peringkat #1 dan merangkum syarat kartu kredit dengan akurat. |
| **Semantic Paraphrase** | *"Kenapa paket saya belum sampai?"* / *"Barang belum nyampe lewat jadwal"* | *Bagaimana jika paket saya belum sampai melebihi estimasi?* (`Dist: 0.3335`) | **Sangat Relevan.** Meskipun pengguna tidak memakai kata *"estimasi"*, vektor embedding berhasil memetakan kesamaan makna keterlambatan pengiriman. |
| **Out-of-Domain (OOD)** | *"Apakah tersedia layanan pinjaman tunai / paylater motor?"* | *Metode pembayaran apa saja yang tersedia?* (Konteks terdekat yang dipaksakan) | **Grounding Berhasil.** LLM mendeteksi bahwa informasi tidak ada dalam konteks FAQ, menolak mengarang jawaban, dan mengarahkan pengguna ke CS. |

**Kesimpulan Analisis:** Kombinasi `gemini-embedding-001` dan `IndexFlatL2` sangat efektif menangkap variasi bahasa alami pengguna, sementara pengaturan `temperature=0.2` dan instruksi *strict context grounding* berhasil menekan risiko halusinasi LLM.

---

## 4. Keterbatasan Sistem (System Limitations)

1. **Ketiadaan *Distance Threshold* (Forced Top-K):** Fungsi `semantic_search` selalu mengembalikan 3 dokumen teratas tanpa filter batas jarak maksimum. Pada *query* di luar domain (*out-of-domain*), dokumen yang tidak relevan tetap ikut terkirim ke dalam *context window* LLM sehingga memboroskan token.
2. **Penyimpanan *In-Memory* (Non-Persisten):** Indeks FAISS disimpan di RAM runtime. Setiap kali sesi notebook di-restart, proses *embedding* ulang terhadap seluruh knowledge base harus dijalankan kembali.
3. **Pencarian Murni *Dense Vector* (Tanpa *Hybrid Search*):** Sistem belum menggabungkan pencarian leksikal (seperti BM25). Untuk istilah yang sangat spesifik seperti kode promo unik atau nomor seri tertentu, *pure dense search* terkadang kalah presisi dibanding pencarian kata kunci eksak.
4. **Stateless (Tanpa Memori Percakapan):** Sistem memproses setiap *query* secara independen (*single-turn*) dan belum menyimpan riwayat percakapan (*chat history*) untuk pertanyaan lanjutan (*follow-up questions*).

---

## 5. Cara Menjalankan Notebook

### Prasyarat
* Akun Google dengan akses ke **Google Colab**.
* **Gemini API Key** aktif yang didapatkan dari [Google AI Studio](https://aistudio.google.com/).
* File dataset `ecommerce_faq_knowledge_base.csv`.

### Langkah-Langkah Eksekusi
1. **Buka Notebook:** Unggah dan buka file `Copy of starter_notebook_semantic_search.ipynb` di Google Colab.
2. **Unggah Dataset:** Buka panel **Files** (ikon folder di bilah kiri Colab), lalu unggah file `ecommerce_faq_knowledge_base.csv` ke direktori utama (`/content/`).
3. **Konfigurasi API Key (Secrets):**
   * Klik ikon **Secrets (Kunci)** di bilah navigasi kiri Colab.
   * Tambahkan *secret* baru dengan nama `GEMINI_API_KEY` dan masukkan nilai API key Anda.
   * Aktifkan *toggle* **Notebook access**.
4. **Jalankan Sel Berurutan:**
   * Klik menu **Runtime > Run all** (atau jalankan setiap *code cell* secara berurutan dari Section 1 hingga Section 8 menggunakan `Shift + Enter`).
   * Sel pertama akan otomatis menginstal pustaka yang dibutuhkan (`faiss-cpu` dan `google-genai`) sebelum membangun pipeline pencarian.
