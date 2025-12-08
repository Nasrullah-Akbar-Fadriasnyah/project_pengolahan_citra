Image Enhancer – Single & Batch Processing (Tkinter GUI)

Aplikasi ini merupakan tool image enhancement berbasis Python dengan GUI menggunakan Tkinter, dilengkapi pipeline peningkatan kualitas gambar yang meliputi:

Auto White Balance (Gray World)

Auto Exposure Correction (Contrast Stretching)

Denoise (Bilateral Filter)

Face Beauty (Smoothing area wajah)

HDR-like Enhancement (Detail + CLAHE)

Sharpening (Unsharp Mask)

Final Color Tone Adjustment

Pembuatan gambar Before–After

Mode single processing & batch processing

Aplikasi dapat memperbaiki ratusan gambar sekaligus dan menampilkan preview Before/After secara real-time.

✨ Fitur Utama
1. Enhancement Pipeline

Pipeline terdiri dari beberapa langkah yang dapat diaktifkan/di-nonaktifkan:

Auto White Balance (Gray World)

Auto Exposure (Kontras otomatis)

Denoising (Bilateral)

Face Beauty Retouch

HDR-like Local Contrast

Sharpening

Final Tone/Saturation

2. GUI Berbasis Tkinter

Preview BEFORE dan AFTER

Menampilkan daftar file gambar dari folder input

Checkbox untuk mengatur pipeline

Tombol:

Enhance Selected

Enhance All (Batch)

Save Result

Progress bar untuk batch processing

3. Batch Image Processing

Semua gambar dalam folder dapat diproses otomatis dengan:

Hasil akhir (*_final.jpg)

Before–After (*_before_after.jpg)

📦 Struktur File
project/
│── enhancer.py          # file utama (kode GUI dan pipeline)
│── README.md            # dokumentasi

⚙️ Dependensi

Pastikan Python 3 sudah terpasang.
Instal modul yang diperlukan:

pip install opencv-python pillow numpy


Tkinter biasanya sudah tersedia di Python standar. Jika tidak, install sesuai OS.

▶️ Cara Menjalankan Aplikasi

Simpan file utama sebagai enhancer.py

Pastikan dependensi telah di-install

Jalankan aplikasi:

python enhancer.py

🧭 Cara Menggunakan
1. Pilih Folder Input

Klik Pilih Folder Input → pilih folder yang berisi foto.

2. Pilih Folder Output

Klik Pilih Folder Output → lokasi penyimpanan hasil.

3. Pilih Gambar

Klik salah satu file di list untuk melihat preview BEFORE.

4. Atur Pipeline

Centang/uncentang opsi enhancement sesuai kebutuhan.

5. Enhance Selected

Memproses satu gambar dan menampilkan hasil AFTER.

6. Save Result

Menyimpan:

nama_final.jpg → hasil enhancement

nama_before_after.jpg → gabungan BEFORE & AFTER

7. Enhance All (Batch)

Memproses seluruh gambar dalam folder input dan menyimpan semuanya ke output folder.

📘 Penjelasan Singkat Fungsi Utama
auto_white_balance_grayworld

Melakukan white balance menggunakan asumsi Gray World.

auto_exposure_stretch

Meningkatkan brightness/contrast berdasarkan percentile 1–99.

bilateral_denoise

Mengurangi noise tanpa menghilangkan tepi.

face_beauty_filter

Mendeteksi wajah dengan Haar Cascade.

Melakukan smoothing lokal pada area wajah.

hdr_like_local_contrast

Menggabungkan detailEnhance + CLAHE → efek pseudo-HDR.

unsharp_mask

Menambah ketajaman (sharpening).

make_before_after_image

Membuat file Before–After lengkap dengan label dan border.

📝 Catatan Penting

Pastikan folder input berisi file gambar format: .jpg, .jpeg, .png, .bmp

Program otomatis melewati file rusak atau tidak dapat dibaca

Haar cascade di-load otomatis dari cv2.data.haarcascades

Preview pada GUI di-resize agar tidak membebani kinerja

📄 Lisensi

Proyek ini bebas digunakan untuk keperluan penelitian, tugas kuliah, atau modifikasi pribadi.