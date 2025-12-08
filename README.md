Image Enhancer – Single & Batch Processing (Tkinter GUI)

Aplikasi ini merupakan tool peningkatan kualitas gambar berbasis Python dengan antarmuka GUI Tkinter. Aplikasi mampu melakukan peningkatan kualitas gambar secara single maupun batch, lengkap dengan preview Before–After secara real-time.

Tool ini dirancang untuk memproses ratusan foto sekaligus dan menghasilkan output gambar berkualitas lebih baik menggunakan pipeline enhancement modern.

✨ Fitur Utama

1. Enhancement Pipeline

Pipeline terdiri dari beberapa tahap yang dapat diaktifkan atau dinonaktifkan sesuai kebutuhan:

- Auto White Balance (Gray World)

- Auto Exposure Correction (Contrast Stretching)

- Denoise (Bilateral Filter)

- Face Beauty Retouching

- HDR-like Enhancement (Detail + CLAHE)

- Sharpening (Unsharp Mask)

- Final Color Tone & Saturation Adjustment

Setiap tahap dapat digabungkan untuk menghasilkan output yang optimal.

2. GUI Berbasis Tkinter

Fitur antarmuka mencakup:

- Preview BEFORE dan AFTER

- Daftar file gambar dari folder input

- Checkbox untuk mengatur pipeline enhancement

- Tombol:

  - Enhance Selected

  - Enhance All (Batch)

  - Save Result

- Progress bar untuk proses batch

3. Batch Image Processing

Semua gambar pada folder input dapat diproses otomatis dengan output:

`*_final.jpg` → hasil enhancement

`*_before_after.jpg` → gambar gabungan Before–After

Aplikasi mampu menangani ratusan gambar dalam satu kali proses.

📦 Struktur Direktori
project/
│── enhancer.py # file utama berisi GUI & pipeline
│── README.md # dokumentasi

⚙️ Dependensi

Pastikan Python 3 telah terpasang.

Instal modul yang diperlukan:

pip install opencv-python pillow numpy

Tkinter biasanya sudah termasuk dalam instalasi Python standar. Jika belum, install sesuai OS masing-masing.

▶️ Cara Menjalankan

Simpan file utama sebagai enhancer.py

Pastikan seluruh dependensi telah ter-install

Jalankan aplikasi:

python enhancer.py

🧭 Cara Menggunakan Aplikasi

1. Pilih Folder Input

Klik Pilih Folder Input lalu pilih folder berisi gambar.

2. Pilih Folder Output

Tentukan lokasi penyimpanan hasil.

3. Pilih Gambar

Klik salah satu gambar dari daftar untuk melihat preview BEFORE.

4. Atur Pipeline

Centang atau hapus centang fitur enhancement sesuai kebutuhan.

5. Enhance Selected

Memproses satu gambar dan menampilkan hasil AFTER.

6. Save Result

Menyimpan dua file:

- `nama_final.jpg` → hasil akhir setelah enhancement

- `nama_before_after.jpg` → gabungan Before–After

7. Enhance All (Batch)

Memproses semua gambar dalam folder input dan menyimpan hasilnya ke folder output.

📘 Penjelasan Singkat Fungsi Utama
**auto_white_balance_grayworld**

Melakukan white balance otomatis dengan asumsi Gray World.

**auto_exposure_stretch**

Mengatur brightness dan contrast menggunakan percentile 1–99.

**bilateral_denoise**

Mengurangi noise tanpa menghilangkan detail tepi.

**face_beauty_filter**

- Mendeteksi wajah menggunakan Haar Cascade

- Melakukan smoothing lokal pada area wajah

**hdr_like_local_contrast**

Menggabungkan detailEnhance + CLAHE untuk menghasilkan efek pseudo-HDR.

**unsharp_mask**

Memberikan efek sharpening pada gambar.

**make_before_after_image**

Membuat gambar gabungan Before–After lengkap dengan label dan border.

📝 Catatan Penting

- Pastikan folder input berisi file dengan format: .jpg, .jpeg, .png, .bmp

- File rusak akan dilewati otomatis

- Haar Cascade di-load dari cv2.data.haarcascades

- Preview otomatis di-resize agar aplikasi tetap ringan

📄 Lisensi

Proyek ini bebas digunakan untuk keperluan penelitian, tugas kuliah, maupun modifikasi pribadi.
