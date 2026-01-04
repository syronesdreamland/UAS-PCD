# Dokumentasi Proyek: Aplikasi Analisis Citra Digital

## 1. Ikhtisar Proyek
Aplikasi ini adalah alat berbasis web untuk melakukan **Deteksi Tepi (Edge Detection)** dan **Operasi Morfologi** pada citra digital. Dibangun menggunakan Python dengan framework **Streamlit**, aplikasi ini menyediakan antarmuka interaktif untuk mengunggah gambar, memilih algoritma, menyesuaikan parameter, dan mengunduh hasil pemrosesan.

## 2. Arsitektur Sistem

### Struktur File
Kode program telah dipecah menjadi beberapa modul untuk modularitas dan kemudahan pemeliharaan:

- **`app.py`**: Titik masuk utama aplikasi. Mengatur tata letak halaman, alur logika utama, dan interaksi pengguna.
- **`utils.py`**: Berisi logika inti pemrosesan citra. Semua algoritma (Sobel, Canny, Dilasi, dll.) diimplementasikan di sini menggunakan OpenCV dan NumPy.
- **`ui.py`**: Menangani komponen antarmuka pengguna (UI) seperti sidebar, slider, dan pemilihan algoritma.
- **`styles.py`**: Mengelola styling CSS kustom untuk memastikan tampilan yang konsisten dan responsif (termasuk dukungan Dark Mode).

### Teknologi yang Digunakan
- **Streamlit**: Framework utama untuk membuat aplikasi web data interaktif. Streamlit menjalankan server web (berbasis Tornado) di latar belakang.
- **OpenCV (opencv-python-headless)**: Pustaka visi komputer untuk operasi pemrosesan citra yang efisien.
- **NumPy**: Digunakan untuk manipulasi array matriks citra.
- **Pillow (PIL)**: Digunakan untuk membaca dan menyimpan format file gambar.

## 3. Penjelasan Kode (Line-by-Line)

### `app.py` (Main Application)
File ini adalah "otak" dari aplikasi.
- **Imports**: Mengimpor modul yang diperlukan (`streamlit`, `PIL`, `numpy`) dan modul lokal kita (`styles`, `ui`, `utils`).
- **`main()`**: Fungsi utama yang dijalankan saat aplikasi dimulai.
  - `st.set_page_config(...)`: Mengatur judul tab browser, ikon, dan layout lebar.
  - `inject_custom_css()`: Memanggil fungsi dari `styles.py` untuk menyuntikkan CSS.
  - `render_controls()`: Memanggil fungsi dari `ui.py` untuk menampilkan sidebar dan mendapatkan parameter dari user.
  - **Layout Kolom**: `st.columns(2)` membagi layar menjadi dua bagian (kiri untuk upload, kanan untuk hasil).
  - **Logika Pemrosesan**:
    - Jika tombol "Proses Gambar" ditekan, aplikasi memanggil `process_image()` dari `utils.py`.
    - Hasil ditampilkan dengan `st.image()`.
    - Tombol unduh dibuat menggunakan `convert_image_for_download()`.

### `utils.py` (Core Logic)
Berisi implementasi matematis dan algoritmik.
- **`ALGORITHM_INFO`**: Dictionary besar yang menyimpan metadata (nama, deskripsi, kernel) untuk setiap algoritma. Ini digunakan untuk menampilkan info di UI.
- **Fungsi Deteksi Tepi**:
  - `apply_sobel_edge_detection`: Menghitung gradien X dan Y, lalu menggabungkannya.
  - `apply_canny_edge_detection`: Menggunakan implementasi Canny dari OpenCV dengan threshold yang bisa diatur.
  - `apply_roberts_edge_detection`, `apply_prewitt_edge_detection`, dll.: Implementasi manual menggunakan konvolusi kernel (`cv2.filter2D`).
- **Fungsi Morfologi**:
  - `apply_dilation`, `apply_erosion`: Menggunakan kernel matriks 1 (ones) untuk memperbesar/memperkecil area terang.
  - `apply_opening`, `apply_closing`: Kombinasi erosi dan dilasi untuk menghilangkan noise.
  - `apply_region_filling`: Algoritma kompleks menggunakan Flood Fill untuk mengisi lubang pada objek biner.
- **`process_image`**: Fungsi wrapper yang menerima gambar mentah dan nama algoritma, lalu mengarahkannya ke fungsi spesifik yang sesuai.

### `ui.py` (User Interface)
Memisahkan kode tampilan dari logika bisnis.
- **`render_controls()`**:
  - Membuat dropdown (`selectbox`) untuk memilih algoritma.
  - Menampilkan deskripsi algoritma secara dinamis berdasarkan pilihan.
  - Menampilkan slider parameter secara kondisional (misalnya, slider "Threshold" hanya muncul jika "Canny" dipilih).
  - Mengembalikan dictionary `params` yang berisi semua pengaturan user.

### `styles.py` (Styling)
- **`inject_custom_css()`**: Menggunakan `st.markdown(..., unsafe_allow_html=True)` untuk menyuntikkan blok `<style>`.
  - Mengatur warna teks agar kontras di mode terang/gelap.
  - Mempercantik tombol dan header.
  - Menghilangkan elemen bawaan Streamlit yang tidak diinginkan jika perlu.

## 4. Cara Kerja Server (Under the Hood)
Saat Anda menjalankan `streamlit run app.py`:
1. **Bootstrap**: Streamlit memutar server web lokal (biasanya di port 8501).
2. **Reactivity**: Tidak seperti web server tradisional (Flask/Django), Streamlit menjalankan ulang **seluruh skrip Python** dari atas ke bawah setiap kali ada interaksi user (klik tombol, geser slider).
3. **Session State**: Untuk variabel yang perlu diingat antar-rerun, Streamlit menggunakan mekanisme caching dan session state (meskipun di aplikasi ini kita sebagian besar stateless).

## 5. Pengembangan Lebih Lanjut
Kode ini sekarang sudah modular. Jika Anda ingin menambahkan algoritma baru (misalnya "Hough Transform"):
1. Tambahkan fungsi logika barunya di `utils.py`.
2. Tambahkan entri metadata di `ALGORITHM_INFO` di `utils.py`.
3. (Opsional) Tambahkan kontrol parameter khusus di `ui.py` jika algoritma tersebut butuh input khusus.
4. Update `process_image` di `utils.py` untuk memanggil fungsi baru tersebut.
`app.py` tidak perlu diubah sama sekali!
