# Dokumentasi Lengkap Proyek Digital Image Edge Detection

Dokumen ini berisi penjelasan menyeluruh mengenai proyek aplikasi web "Digital Image Edge Detection", mencakup penjelasan kode baris demi baris, arsitektur sistem, algoritma yang digunakan, hingga lingkungan pengembangan dan deployment.

---

## 1. Arsitektur Sistem & Teknologi

### A. Arsitektur
Aplikasi ini menggunakan arsitektur **Monolithic Web Application** berbasis Python.
- **Frontend (Tampilan)**: Dibangun menggunakan **Streamlit**. Streamlit merender komponen UI (tombol, slider, upload) menjadi HTML/CSS/JS yang interaktif.
- **Backend (Logika)**: Logika pemrosesan citra berjalan di sisi server menggunakan **Python**.
- **Communication**: Tidak ada pemisahan API (REST/GraphQL) secara eksplisit. Streamlit menangani komunikasi antara browser dan server melalui WebSocket (menggunakan Tornado Web Server).

### B. Teknologi Stack
1.  **Bahasa Pemrograman**: Python 3.x.
2.  **Web Framework**: Streamlit (v1.28+).
3.  **Computer Vision Library**: OpenCV (`opencv-python-headless`). Versi "headless" digunakan agar kompatibel dengan lingkungan server cloud yang tidak memiliki GUI (menghindari error `libGL.so.1`).
4.  **Matrix Processing**: NumPy (Citra digital diproses sebagai matriks angka).
5.  **Image I/O**: Pillow (PIL) untuk membaca dan menyimpan format gambar (JPG, PNG).

### C. Web Server & Build System
- **Web Server**: Streamlit berjalan di atas **Tornado**, sebuah web server Python yang *asynchronous* dan *non-blocking*.
- **Build Server**: Python adalah bahasa *interpreted*, jadi tidak ada proses "build" atau kompilasi (seperti `npm build` pada React). Server langsung menjalankan skrip `app.py`.
- **Deployment**: Aplikasi dirancang untuk berjalan di container (Docker) atau PaaS seperti Streamlit Community Cloud.

---

## 2. Penjelasan Kode (`app.py`)

Berikut adalah bedah kode dari baris pertama hingga selesai.

### Bagian 1: Import Library
```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
```
- `streamlit as st`: Library utama untuk membuat web app. Alias `st` adalah standar industri.
- `cv2`: OpenCV, otak dari pemrosesan citra.
- `numpy as np`: Digunakan karena OpenCV merepresentasikan gambar sebagai array multidimensi (matriks).
- `PIL.Image`: Digunakan untuk membuka file gambar yang diupload pengguna sebelum dikonversi ke format OpenCV.
- `io`: Modul Input/Output, digunakan untuk menangani buffer byte saat pengguna mendownload hasil gambar.

### Bagian 2: Informasi Algoritma (`ALGORITHM_INFO`)
Variabel ini adalah sebuah **Dictionary** Python yang menyimpan metadata untuk setiap algoritma.
- **Key**: Nama algoritma (misal 'Sobel').
- **Value**: Dictionary berisi 'name', 'description', dan parameter kernel.
- **Tujuan**: Memisahkan data teks dari logika kode, sehingga mudah untuk menampilkan penjelasan di UI tanpa mengotori fungsi logika.

### Bagian 3: Styling Kustom (`inject_custom_css`)
```python
def inject_custom_css(theme='light'):
    ...
    st.markdown(f"<style>...</style>", unsafe_allow_html=True)
```
- **Fungsi**: Menyuntikkan kode CSS (Cascading Style Sheets) ke dalam halaman web.
- **Parameter `theme`**: Menerima 'light' atau 'dark' untuk menentukan palet warna.
- **Variabel Warna**: `bg_color`, `text_color`, dll. diatur berdasarkan tema yang dipilih.
- **`unsafe_allow_html=True`**: Memberi izin kepada Streamlit untuk merender HTML mentah (tag `<style>`). Ini diperlukan karena Streamlit secara default membatasi kustomisasi tampilan demi keamanan.
- **CSS Rules**: Mengatur font (Inter), warna background, bentuk tombol (border-radius), dan menghilangkan elemen bawaan Streamlit (seperti hamburger menu) agar aplikasi terlihat profesional.

### Bagian 4: Fungsi Pemrosesan Citra (Core Logic)

Setiap fungsi di bawah ini menerima input `gray_image` (citra grayscale dalam bentuk NumPy array) dan mengembalikan citra hasil.

#### a. Edge Detection (Deteksi Tepi)
1.  **`apply_sobel_edge_detection`**:
    - Menggunakan `cv2.Sobel`.
    - Menghitung gradien arah X (vertikal) dan Y (horizontal) secara terpisah.
    - `cv2.addWeighted`: Menggabungkan hasil X dan Y dengan bobot 50:50.
    - **Kenapa Sobel?** Tahan terhadap noise karena kernelnya memberikan pembobotan lebih pada piksel tengah.

2.  **`apply_roberts_edge_detection`**:
    - Menggunakan kernel manual 2x2 (diagonal).
    - `cv2.filter2D`: Fungsi generik untuk menerapkan kernel kustom (konvolusi).
    - **Karakteristik**: Sangat sensitif terhadap noise, tapi cepat.

3.  **`apply_prewitt_edge_detection`**:
    - Mirip Sobel tapi bobot kernelnya seragam (semua 1).
    - Deteksi tepinya sedikit lebih kasar dibanding Sobel.

4.  **`apply_laplacian_edge_detection`**:
    - `cv2.Laplacian`.
    - Menggunakan turunan kedua. Mencari "zero-crossing" (titik di mana kurva berbalik arah).
    - Mendeteksi tepi ke segala arah sekaligus.

5.  **`apply_frei_chen_edge_detection`**:
    - Menggunakan kernel dengan nilai akar 2 (`np.sqrt(2)`).
    - **Tujuan**: Membuat deteksi tepi lebih "isotropik" (sama kuat ke segala arah sudut), lebih akurat secara matematis dibanding Sobel.

6.  **`apply_canny_edge_detection`**:
    - `cv2.Canny`.
    - Algoritma paling kompleks (multi-stage): Gaussian Blur -> Sobel -> Non-max suppression -> Hysteresis Thresholding.
    - **Parameter**: `low_threshold` dan `high_threshold` menentukan sensitivitas tepi.

#### b. Morphological Operations (Morfologi)
Semua fungsi ini menggunakan `cv2.morphologyEx` atau fungsi spesifik (`dilate`, `erode`).
- **Parameter `kernel_size`**: Menentukan ukuran "Structuring Element". Semakin besar, semakin kuat efeknya.
- **`apply_dilation`**: Menambah piksel (mempertebal objek putih).
- **`apply_erosion`**: Mengurangi piksel (menipiskan objek putih).
- **`apply_opening`**: Erosi lalu Dilasi (hilangkan noise putih kecil).
- **`apply_closing`**: Dilasi lalu Erosi (tutup lubang hitam kecil).
- **`apply_region_filling`**:
    - Teknik pengisian lubang.
    - Menggunakan `cv2.floodFill` dari titik (0,0).
    - Logika: Jika background putih, invert dulu -> Flood fill background -> Invert hasil flood fill -> Gabungkan dengan citra asli.

### Bagian 5: Orchestrator (`process_image`)
Fungsi ini adalah "Manager" yang mengatur alur pemrosesan.
1.  **Konversi Input**: `image.convert('RGB')` -> `np.array` (Mengubah objek Gambar Python menjadi Matriks).
2.  **Grayscale**: `cv2.cvtColor` (Algoritma deteksi tepi bekerja pada intensitas cahaya, bukan warna, jadi harus hitam putih).
3.  **Noise Reduction**: `cv2.GaussianBlur` (Kecuali untuk Region Filling). Mengaburkan gambar sedikit untuk menghilangkan bintik-bintik noise yang bisa dianggap sebagai tepi palsu.
4.  **Routing**: Memilih fungsi algoritma mana yang dijalankan berdasarkan input string `algorithm`.
5.  **Normalisasi**: `np.clip` dan `.astype(np.uint8)` memastikan nilai piksel valid (0-255).

### Bagian 6: UI Controls (`render_controls`)
Fungsi ini membuat komponen input di layar.
- `st.columns`: Membagi layar menjadi kolom-kolom.
- `st.radio`: Pilihan algoritma.
- `st.slider`: Pengatur parameter (Threshold Canny atau Kernel Size).
- Mengembalikan dictionary berisi nilai-nilai input user.

### Bagian 7: Main Application (`main`)
Titik masuk (Entry Point) aplikasi.
1.  **`st.set_page_config`**: Mengatur judul tab browser dan layout "wide".
2.  **Session State Management**:
    - `if 'theme' not in st.session_state`: Menginisialisasi variabel global untuk menyimpan status tema (Light/Dark). Ini penting karena Streamlit me-rerun script setiap kali ada interaksi; session state menjaga data agar tidak hilang.
3.  **Header & Theme Toggle**: Tombol untuk mengubah nilai `st.session_state.theme` dan memicu `st.rerun()`.
4.  **Display Logic**:
    - Jika gambar diupload (`if uploaded_file is not None`), tampilkan kolom "Original" dan "Result".
    - Jika tombol "Detect" ditekan, panggil `process_image`, simpan hasilnya ke `st.session_state['result_image']`.
    - Tampilkan tombol Download.

---

## 3. Penjelasan Algoritma Secara Teori

### 1. Sobel
Menggunakan dua matriks konvolusi 3x3. Satu mendeteksi perubahan intensitas horizontal (Gx), satu lagi vertikal (Gy). Nilai akhir adalah kombinasi keduanya. Sangat populer karena sederhana dan efektif.

### 2. Canny (Si Standar Emas)
Algoritma cerdas yang tidak hanya melihat gradien, tapi juga menyambungkan garis.
- **Hysteresis Thresholding**: Menggunakan dua ambang batas. Garis di atas ambang tinggi pasti tepi. Garis di antara ambang rendah dan tinggi hanya dianggap tepi jika tersambung dengan tepi kuat.

### 3. Morfologi (Dilation, Erosion, dll)
Bukan deteksi tepi dalam arti tradisional, tapi operasi bentuk.
- Bekerja dengan menggeser "kernel" di atas gambar.
- **Dilasi**: Mengambil nilai piksel maksimum di bawah kernel (membuat terang melebar).
- **Erosi**: Mengambil nilai piksel minimum di bawah kernel (membuat gelap melebar/terang menyempit).

---

## 4. Alur Kerja Pengguna (User Flow)
1.  User membuka web.
2.  Script `app.py` dijalankan server -> UI muncul.
3.  User upload gambar -> Gambar disimpan di RAM server.
4.  User memilih "Canny" dan geser slider -> Streamlit mendeteksi perubahan input.
5.  User klik "Detect Edges" -> Server menjalankan fungsi `process_image` -> Hasil disimpan di Session State.
6.  Halaman di-refresh otomatis oleh Streamlit untuk menampilkan gambar hasil dari Session State.
