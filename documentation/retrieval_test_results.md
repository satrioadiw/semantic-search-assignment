from google.colab import files
files.download('retrieval_test_results.md')

=== TEST CASE #1 ===
Query: Bagaimana cara membatalkan pesanan yang sudah dibuat?
Jawaban LLM:
 Halo! Pesanan dapat dibatalkan melalui halaman 'Pesanan Saya' selama status masih 'Diproses'. Namun, setelah status berubah menjadi 'Dikirim', pembatalan sudah tidak dapat dilakukan. 

Ada lagi yang bisa saya bantu?

Top-3 Retrieved Contexts (L2 Distance):
  [1] (Dist: 0.2392) [Pemesanan] Bagaimana cara membatalkan pesanan yang sudah dibuat?
  [2] (Dist: 0.5069) [Pengembalian] Bagaimana prosedur pengembalian barang yang tidak sesuai?
  [3] (Dist: 0.5349) [Produk] Bagaimana jika produk yang saya pesan ternyata kehabisan stok?
==================================================

=== TEST CASE #2 ===
Query: Bagaimana cara menghapus akun secara permanen?
Jawaban LLM:
 Halo! Untuk menghapus akun secara permanen, pengajuan dapat dilakukan melalui menu 'Pengaturan Akun > Hapus Akun' atau dengan menghubungi customer service kami. Ada hal lain yang bisa saya bantu?

Top-3 Retrieved Contexts (L2 Distance):
  [1] (Dist: 0.1889) [Akun] Bagaimana cara menghapus akun secara permanen?
  [2] (Dist: 0.6851) [Akun] Apakah bisa memiliki lebih dari satu akun dengan email yang sama?
  [3] (Dist: 0.6976) [Akun] Saya lupa password akun, bagaimana cara mengaturnya ulang?
==================================================

=== TEST CASE #3 ===
Query: Barang belanjaan saya kok belum nyampe-nyampe ya padahal sudah lewat jadwalnya?
Jawaban LLM:
 Halo! Jika paket melebihi estimasi pengiriman lebih dari 3 hari, silakan hubungi customer service dengan menyertakan nomor resi untuk ditelusuri lebih lanjut.

Top-3 Retrieved Contexts (L2 Distance):
  [1] (Dist: 0.3777) [Pengiriman] Bagaimana jika paket saya belum sampai melebihi estimasi?
  [2] (Dist: 0.4145) [Pengiriman] Kenapa status pengiriman saya tidak berubah selama beberapa hari?
  [3] (Dist: 0.4830) [Pengiriman] Berapa lama estimasi waktu pengiriman barang?
==================================================

=== TEST CASE #4 ===
Query: Salah masukin alamat rumah nih, masih bisa diubah nggak kalau sudah terlanjur bayar?
Jawaban LLM:
 Halo! Perubahan alamat hanya bisa dilakukan sebelum pesanan berstatus 'Dikirim'. Silakan hubungi customer service secepatnya jika ingin mengubah alamat.

Top-3 Retrieved Contexts (L2 Distance):
  [1] (Dist: 0.2304) [Pemesanan] Apakah saya bisa mengubah alamat pengiriman setelah pesanan dibuat?
  [2] (Dist: 0.5809) [Pengembalian] Saya menerima barang yang salah, apa yang harus saya lakukan?
  [3] (Dist: 0.5873) [Pemesanan] Bagaimana cara membatalkan pesanan yang sudah dibuat?
==================================================

=== TEST CASE #5 ===
Query: Apakah platform ini menyediakan layanan pinjaman dana tunai atau paylater untuk beli motor?
Jawaban LLM:
 Halo! Mohon maaf, saya belum memiliki informasi tersebut. Untuk informasi lebih lanjut, silakan menghubungi customer service kami.

Top-3 Retrieved Contexts (L2 Distance):
  [1] (Dist: 0.6429) [Pembayaran] Apakah saya bisa membayar dengan cicilan?
  [2] (Dist: 0.7287) [Pembayaran] Metode pembayaran apa saja yang tersedia?
  [3] (Dist: 0.7339) [Pembayaran] Apakah bisa membayar menggunakan lebih dari satu metode sekaligus?
==================================================

=== TEST CASE #6 ===
Query: Siapa pemenang Piala Dunia tahun 2022?
Jawaban LLM:
 Halo! Maaf sekali, saya belum memiliki informasi tersebut. Silakan hubungi customer service kami untuk bantuan lebih lanjut ya. Ada hal lain seputar pesanan atau e-commerce kami yang bisa saya bantu?

Top-3 Retrieved Contexts (L2 Distance):
  [1] (Dist: 0.8974) [Pembayaran] Metode pembayaran apa saja yang tersedia?
  [2] (Dist: 0.9164) [Pembayaran] Kenapa pembayaran saya berhasil dipotong tapi status pesanan masih belum terbayar?
  [3] (Dist: 0.9258) [Pembayaran] Berapa lama batas waktu pembayaran sebelum pesanan dibatalkan?
==================================================

---

# Catatan Analisis Relevansi Sistem RAG (Semantic Search + LLM)



1. **Evaluasi Retrieval Quality pada Direct Query & Semantic Paraphrase (Sinonim)**:
	- Hasil: Sistem berhasil melakukan dense retrieval dengan akurasi sangat tinggi baik pada query eksplisit maupun query parafrase/bahasa sehari-hari (misalnya frasa "barang belum nyampe lewat jadwal" tetap berhasil menarik FAQ "Bagaimana jika paket saya belum sampai melebihi estimasi?" di peringkat #1 dengan skor L2 Distance terkecil/paling dekat).
	- Penjelasan : Model gemini-embedding-001 memetakan teks ke dalam high-dimensional vector space (3072 dimensi) berdasarkan makna semantik (semantic intent), bukan pencocokan leksikal (exact keyword matching seperti TF-IDF atau SQL LIKE). Karena unit dokumen menggabungkan format Q: ... A: ..., vektor dokumen menjadi lebih kaya konteks (context-rich representation).
	-  Analogi Sederhana: Bayangkan seorang pustakawan cerdas. Pustakawan biasa (keyword search) hanya mencari buku berdasarkan judul yang persis sama dengan kata yang Anda ucapkan. Sebaliknya, pustakawan cerdas (semantic search) memahami maksud di balik keluhan user—meski user memakai bahasa santai, ia langsung tahu rak mana yang menyimpan solusinya.

---

2. **Evaluasi Grounding & Hallucination Control pada Out-of-Domain Query**:
	- Hasil: Ketika diuji dengan pertanyaan di luar knowledge base (seperti pertanyaan soal paylater motor atau Piala Dunia), LLM (gemini-3.5-flash-lite) mematuhi system instruction dengan menolak menjawab secara mengarang dan menyarankan pengguna menghubungi customer service.
	- Penjelasan: Ini membuktikan efektivitas prompt grounding dan pengaturan temperature=0.2 (rendah/deterministik). Meskipun FAISS tetap mengembalikan Top-3 dokumen terdekat secara matematis (forced retrieval), LLM bertindak sebagai reasoning filter yang mengevaluasi bahwa konteks yang diberikan tidak relevan dengan query pengguna, sehingga mencegah halusinasi (hallucination).
	- Analogi Sederhana: Bayangkan seorang staf CS yang dibekali buku panduan SOP. Saat pelanggan bertanya hal di luar buku SOP, mesin pencari tetap menyodorkan 3 halaman buku yang paling menyerupai pertanyaan tersebut. Namun, karena staf CS dilatih disiplin (temperature=0.2 dan aturan grounding), ia membaca halaman itu, sadar bahwa jawabannya tidak ada di sana, lalu dengan jujur berkata "Maaf, informasi tersebut belum tersedia di panduan kami" alih-alih mengarang aturan sendiri.

---

3. **Temuan Potensial Noise pada Top-K Retrieval**:
	- Hasil: Karena fungsi semantic_search selalu mengambil top_k=3 tanpa batas ambang jarak (distance threshold), untuk pertanyaan yang topiknya sangat spesifik (misalnya penghapusan akun), peringkat #2 dan #3 terkadang berisi FAQ yang agak berjauhan topiknya namun masih satu kategori.
	- Analogi Sederhana: Seperti meminta asisten membawakan 3 berkas paling mirip di meja. Jika hanya ada 1 berkas yang benar-benar cocok, asisten tersebut tetap terpaksa mengambil 2 berkas tambahan di sebelahnya sekadar untuk memenuhi kuota 3 berkas.

