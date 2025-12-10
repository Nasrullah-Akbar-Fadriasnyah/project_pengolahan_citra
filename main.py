import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ============================================================
#                FUNGSI-FUNGSI PENDUKUNG GAMBAR
# ============================================================

def auto_white_balance_grayworld(img_bgr):
    """
    Melakukan koreksi Auto White Balance menggunakan metode Gray-World.
    Konsep: rata-rata nilai R, G, B harus sama (abu-abu).
    
    Cara kerja:
    - Hitung rata-rata channel R, G, B.
    - Hitung rata-rata keseluruhan.
    - Skala tiap channel agar mendekati nilai rata-rata keseluruhan.
    
    Hasil: warna gambar lebih netral dan tidak dominan warna tertentu.
    """
    img = img_bgr.astype(np.float32)
    r_avg = img[:, :, 2].mean()
    g_avg = img[:, :, 1].mean()
    b_avg = img[:, :, 0].mean()
    avg = (r_avg + g_avg + b_avg) / 3.0

    img[:, :, 2] = np.clip(img[:, :, 2] * (avg / (r_avg + 1e-8)), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg / (g_avg + 1e-8)), 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg / (b_avg + 1e-8)), 0, 255)
    return img.astype(np.uint8)


def auto_exposure_stretch(img_bgr, low_perc=1, high_perc=99):
    """
    Melakukan koreksi exposure dengan contrast stretching.
    
    Cara kerja:
    - Konversi gambar ke YCrCb.
    - Ambil channel Y (luminance / terang).
    - Ambil nilai percentile rendah & tinggi.
    - Rentang pixel diregangkan sehingga kontras meningkat.
    
    Efek: gambar tampak lebih cerah & jelas tanpa merusak warna.
    """
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = img_yuv[:, :, 0].astype(np.float32)

    low = np.percentile(Y, low_perc)
    high = np.percentile(Y, high_perc)

    Y_stretch = (Y - low) * (255.0 / (high - low + 1e-8))
    Y_stretch = np.clip(Y_stretch, 0, 255).astype(np.uint8)
    img_yuv[:, :, 0] = Y_stretch

    return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)


def bilateral_denoise(img_bgr, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Denoise menggunakan Bilateral Filter.
    
    Keunggulan:
    - Menghaluskan noise
    - Tepi tetap tajam (tidak blur seperti Gaussian)
    
    Cocok untuk potret atau gambar detail.
    """
    return cv2.bilateralFilter(img_bgr, d, sigmaColor, sigmaSpace)


def face_beauty_filter(img_bgr, face_cascade, strength=0.6):
    """
    Filter smoothing khusus area wajah (beauty filter).
    
    Cara kerja:
    - Deteksi wajah dengan Haar Cascade.
    - Tambah padding agar smoothing terlihat natural.
    - Terapkan bilateral + edgePreservingFilter pada area wajah.
    - Blend (campur) hasil smoothing dengan gambar asli sesuai strength.
    
    Hasil: kulit wajah lebih halus tapi tetap natural.
    """
    img_out = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    for (x, y, w, h) in faces:
        pad = int(0.2 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        y2 = min(img_bgr.shape[0], y + h + pad)

        roi = img_bgr[y1:y2, x1:x2].copy()

        sm1 = cv2.bilateralFilter(roi, 9, 90, 90)

        try:
            sm2 = cv2.edgePreservingFilter(roi, flags=1, sigma_s=60, sigma_r=0.4)
        except Exception:
            sm2 = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)

        smooth = cv2.addWeighted(sm1, 0.5, sm2, 0.5, 0)
        blended = cv2.addWeighted(roi, 1.0 - strength, smooth, strength, 0)

        img_out[y1:y2, x1:x2] = blended

    return img_out


def clahe_on_luminance(img_bgr, clipLimit=2.0, tileGridSize=(8,8)):
    """
    CLAHE hanya pada channel luminance Y.
    
    Kelebihan:
    - Meningkatkan kontras lokal
    - Tidak merusak warna (Cr/Cb tidak diubah)
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    Yc = clahe.apply(Y)

    ycrcbc = cv2.merge((Yc, Cr, Cb))
    return cv2.cvtColor(ycrcbc, cv2.COLOR_YCrCb2BGR)


def hdr_like_local_contrast(img_bgr):
    """
    Efek pseudo-HDR.
    
    Cara kerja:
    1. detailEnhance → menonjolkan detail.
    2. CLAHE → menambah local contrast.
    
    Hasil: gambar terlihat lebih “pop” dan dramatis seperti HDR.
    """
    try:
        de = cv2.detailEnhance(img_bgr, sigma_s=12, sigma_r=0.15)
    except Exception:
        de = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    clahe = clahe_on_luminance(de, clipLimit=2.2)
    return clahe


def unsharp_mask(img_bgr, amount=1.5, sigma=1.0):
    """
    Sharpening menggunakan teknik Unsharp Mask.
    
    Cara kerja:
    - Blur gambar
    - Gabungkan gambar asli dengan gambar blur → meningkatkan edge
    
    Hasil: gambar lebih tajam tapi tetap natural.
    """
    blur = cv2.GaussianBlur(img_bgr, (0,0), sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def final_color_tone(img_bgr):
    """
    Pengaturan tone akhir:
    - Naikkan saturasi (warna lebih vivid)
    - Turunkan brightness sedikit agar tidak overexposed.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.06, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.98, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)



# ============================================================
#        PIPELINE UTAMA – MENGGABUNGKAN SEMUA LANGKAH
# ============================================================

def enhancement_pipeline(img_bgr, face_cascade,
                         do_awb=True,
                         do_exposure=True,
                         do_denoise=True,
                         do_face_beauty=True,
                         do_hdr=True,
                         do_sharpen=True,
                         do_final_tone=True):
    """
    Pipeline lengkap peningkatan kualitas gambar.
    
    Urutan proses:
    1. Auto White Balance
    2. Exposure Correction
    3. Denoise
    4. Beauty Face Filter
    5. HDR-like Enhancement
    6. Sharpening
    7. Tone Adjustment
    
    Setiap langkah bisa dinyalakan/dimatikan dengan checkbox di GUI.
    """
    img = img_bgr.copy()

    if do_awb:
        img = auto_white_balance_grayworld(img)
    if do_exposure:
        img = auto_exposure_stretch(img)
    if do_denoise:
        img = bilateral_denoise(img)
    if do_face_beauty:
        img = face_beauty_filter(img, face_cascade)
    if do_hdr:
        img = hdr_like_local_contrast(img)
    if do_sharpen:
        img = unsharp_mask(img, amount=0.8, sigma=1.2)
    if do_final_tone:
        img = final_color_tone(img)

    return img



# ============================================================
#            MEMBUAT GAMBAR BEFORE–AFTER UNTUK DISIMPAN
# ============================================================

def make_before_after_image(before_bgr, after_bgr, label_font=None):
    """
    Membuat gambar gabungan (Before & After) dalam satu file.
    
    Langkah:
    - Samakan tinggi kedua gambar
    - Gabungkan secara horizontal
    - Tambahkan strip hitam semi-transparan di atas
    - Tulis label BEFORE dan AFTER
    - Beri border putih
    
    Hasil akhir: cocok untuk perbandingan visual.
    """
    H = max(before_bgr.shape[0], after_bgr.shape[0])

    def resize_to_h(img, H):
        if img.shape[0] == H:
            return img
        scale = H / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*scale), H))

    b = resize_to_h(before_bgr, H)
    a = resize_to_h(after_bgr, H)
    combined = np.hstack((b, a))

    pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    W, H_ = pil.size
    mid = W // 2

    overlay = Image.new('RGBA', (W, 48), (0,0,0,140))
    pil.paste(overlay, (0,0), overlay)

    if label_font is None:
        try:
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except Exception:
            label_font = ImageFont.load_default()

    draw.text((12, 8), "BEFORE", font=label_font, fill=(255,255,255))
    draw.text((mid+12, 8), "AFTER", font=label_font, fill=(255,255,255))

    border = 6
    w0, h0 = pil.size
    bg = Image.new('RGB', (w0 + 2*border, h0 + 2*border), (255,255,255))
    bg.paste(pil, (border, border))

    return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)



# ============================================================
#                     FUNGSI FILE & DIREKTORI
# ============================================================

def ensure_dir(path):
    """
    Membuat folder jika belum ada.
    Berguna untuk memastikan folder output tersedia.
    """
    if not os.path.exists(path):
        os.makedirs(path)



# ============================================================
#                   KELAS GUI TKINTER – APLIKASI
# ============================================================

class EnhancerGUI:
    """
    Kelas inti GUI untuk Image Enhancer.

    Fitur utama:
    - Memilih folder input & output
    - Preview gambar BEFORE & AFTER
    - Enhancement single file
    - Enhancement batch (semua gambar)
    - Opsi pipeline lengkap (checkbox)
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Image Enhancer - Single + Batch (Before/After)")
        self.root.geometry("1000x700")

        self.input_folder = None
        self.output_folder = None
        self.current_before = None
        self.current_after = None
        self.current_filename = None

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self._build_ui()
        self.file_list = []


    # --------------------------------------------------------
    #             MEMBANGUN UI (TOMBOL / LIST / PREVIEW)
    # --------------------------------------------------------
    def _build_ui(self):
        """
        Membuat seluruh tampilan GUI:
        - Tombol folder input/output
        - List file gambar
        - Preview BEFORE–AFTER
        - Tombol Enhance + Save
        - Progressbar batch
        - Checkbox pipeline
        """
        frm_top = tk.Frame(self.root)
        frm_top.pack(fill=tk.X, padx=8, pady=6)

        btn_in = tk.Button(frm_top, text="Pilih Folder Input", command=self.choose_input_folder, width=18)
        btn_in.pack(side=tk.LEFT, padx=4)
        btn_out = tk.Button(frm_top, text="Pilih Folder Output", command=self.choose_output_folder, width=18)
        btn_out.pack(side=tk.LEFT, padx=4)
        btn_refresh = tk.Button(frm_top, text="Refresh Daftar", command=self.load_file_list)
        btn_refresh.pack(side=tk.LEFT, padx=4)

        left_panel = tk.Frame(self.root)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        lbl = tk.Label(left_panel, text="Daftar Gambar (klik untuk pilih):")
        lbl.pack(anchor='w')

        self.lst = tk.Listbox(left_panel, width=40, height=30)
        self.lst.pack(side=tk.LEFT, fill=tk.Y)
        self.lst.bind("<<ListboxSelect>>", self.on_select)

        sb = tk.Scrollbar(left_panel, orient="vertical", command=self.lst.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.lst.config(yscrollcommand=sb.set)

        right_panel = tk.Frame(self.root)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        preview_frame = tk.Frame(right_panel)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_before = tk.Label(preview_frame, text="BEFORE", compound=tk.TOP)
        self.canvas_before.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4, pady=4)

        self.canvas_after = tk.Label(preview_frame, text="AFTER", compound=tk.TOP)
        self.canvas_after.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4, pady=4)

        ctrl_frame = tk.Frame(right_panel)
        ctrl_frame.pack(fill=tk.X, pady=6)

        self.btn_save = tk.Button(ctrl_frame, text="Simpan Hasil", command=self.save_result, width=14)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.btn_enhance_sel = tk.Button(ctrl_frame, text="Enhance Selected", command=self.enhance_selected, width=18)
        self.btn_enhance_sel.pack(side=tk.LEFT, padx=4)

        self.btn_enhance_all = tk.Button(ctrl_frame, text="Enhance All (Batch)", command=self.enhance_all_batch, width=18)
        self.btn_enhance_all.pack(side=tk.LEFT, padx=4)

        self.progress = ttk.Progressbar(ctrl_frame, orient='horizontal', mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        opt_frame = tk.LabelFrame(right_panel, text="Opsi Pipeline (centang/uncentang)")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.var_awb = tk.BooleanVar(value=True)
        self.var_exposure = tk.BooleanVar(value=True)
        self.var_denoise = tk.BooleanVar(value=True)
        self.var_face = tk.BooleanVar(value=True)
        self.var_hdr = tk.BooleanVar(value=True)
        self.var_sharp = tk.BooleanVar(value=True)
        self.var_final = tk.BooleanVar(value=True)

        tk.Checkbutton(opt_frame, text="Auto White Balance", variable=self.var_awb).pack(anchor='w')
        tk.Checkbutton(opt_frame, text="Auto Exposure", variable=self.var_exposure).pack(anchor='w')
        tk.Checkbutton(opt_frame, text="Denoise (Bilateral)", variable=self.var_denoise).pack(anchor='w')
        tk.Checkbutton(opt_frame, text="Face Beauty (smooth wajah)", variable=self.var_face).pack(anchor='w')
        tk.Checkbutton(opt_frame, text="HDR-like (detail+CLAHE)", variable=self.var_hdr).pack(anchor='w')
        tk.Checkbutton(opt_frame, text="Sharpen (Unsharp)", variable=self.var_sharp).pack(anchor='w')
        tk.Checkbutton(opt_frame, text="Final Tone", variable=self.var_final).pack(anchor='w')


    # ============================================================
    #                       EVENT HANDLER
    # ============================================================

    def choose_input_folder(self):
        """Memilih folder berisi gambar input."""
        p = filedialog.askdirectory(title="Pilih folder yang berisi gambar input")
        if p:
            self.input_folder = p
            self.load_file_list()

    def choose_output_folder(self):
        """Memilih folder tujuan output dan memastikan folder tersedia."""
        p = filedialog.askdirectory(title="Pilih folder output (akan dibuat jika kosong)")
        if p:
            self.output_folder = p
            ensure_dir(self.output_folder)
            messagebox.showinfo("Folder Output", f"Folder output diset ke:\n{self.output_folder}")

    def load_file_list(self):
        """Meload daftar gambar dari folder input."""
        self.lst.delete(0, tk.END)
        self.file_list = []
        if not self.input_folder:
            return

        files = sorted([f for f in os.listdir(self.input_folder)
                        if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])

        for f in files:
            self.file_list.append(f)
            self.lst.insert(tk.END, f)

    def on_select(self, event):
        """Menampilkan preview gambar ketika dipilih di listbox."""
        sel = self.lst.curselection()
        if not sel:
            return

        idx = sel[0]
        fname = self.file_list[idx]
        path = os.path.join(self.input_folder, fname)
        img = cv2.imread(path)

        if img is None:
            messagebox.showerror("Error", f"Gagal membuka {fname}")
            return

        self.show_image_on_label(img, self.canvas_before)
        self.canvas_after.config(image='', text='AFTER')

    def show_image_on_label(self, img_bgr, label_widget, maxsize=(480,480)):
        """
        Menampilkan gambar pada label widget Tkinter.
        - Resize otomatis agar muat di preview.
        - Konversi BGR → RGB → PhotoImage (Tkinter)
        """
        h0, w0 = img_bgr.shape[:2]
        maxw, maxh = maxsize
        scale = min(maxw / w0, maxh / h0, 1.0)

        if scale < 1.0:
            img_rs = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)))
        else:
            img_rs = img_bgr.copy()

        img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        tkimg = ImageTk.PhotoImage(pil)

        label_widget.image = tkimg
        label_widget.config(image=tkimg, text='')

    def enhance_selected(self):
        """
        Enhancing satu gambar yang dipilih.
        Menjalankan pipeline sesuai opsi checkbox.
        """
        sel = self.lst.curselection()
        if not sel:
            messagebox.showwarning("Pilih file", "Silakan pilih satu file dari daftar.")
            return

        idx = sel[0]
        fname = self.file_list[idx]
        in_path = os.path.join(self.input_folder, fname)
        img = cv2.imread(in_path)

        if img is None:
            messagebox.showerror("Error", f"Gagal membuka {fname}")
            return

        self.current_filename = fname
        self.current_before = img.copy()

        out = enhancement_pipeline(
            img, 
            self.face_cascade,
            do_awb=self.var_awb.get(),
            do_exposure=self.var_exposure.get(),
            do_denoise=self.var_denoise.get(),
            do_face_beauty=self.var_face.get(),
            do_hdr=self.var_hdr.get(),
            do_sharpen=self.var_sharp.get(),
            do_final_tone=self.var_final.get()
        )

        self.current_after = out.copy()
        self.show_image_on_label(out, self.canvas_after)

        messagebox.showinfo("Berhasil", "Proses enhancement selesai.\nKlik 'Simpan Hasil' untuk menyimpan file.")

    def save_result(self):
        """Menyimpan hasil enhancement dan versi Before–After."""
        if self.current_after is None or self.current_before is None:
            messagebox.showwarning("Tidak ada hasil", "Lakukan Enhance terlebih dahulu.")
            return

        if not self.output_folder:
            messagebox.showwarning("Belum ada output", "Pilih folder output terlebih dahulu.")
            return

        fname = self.current_filename
        base = os.path.splitext(fname)[0]

        out_path = os.path.join(self.output_folder, f"{base}_final.jpg")
        cv2.imwrite(out_path, self.current_after)

        ba = make_before_after_image(self.current_before, self.current_after)
        ba_path = os.path.join(self.output_folder, f"{base}_before_after.jpg")
        cv2.imwrite(ba_path, ba)

        messagebox.showinfo("Disimpan", f"Hasil tersimpan:\n{out_path}\n{ba_path}")

    def enhance_all_batch(self):
        """
        Memproses semua gambar dalam folder input secara batch.
        
        Mekanisme:
        - Loop setiap file gambar
        - Jalankan enhancement pipeline
        - Simpan hasil final + before-after
        - Progress bar diperbarui tiap iterasi
        
        GUI tetap responsif karena update UI menggunakan event loop Tkinter.
        """
        if not self.input_folder:
            messagebox.showwarning("Folder input", "Tentukan folder input terlebih dahulu.")
            return

        if not self.output_folder:
            messagebox.showwarning("Folder output", "Tentukan folder output terlebih dahulu.")
            return

        files = sorted([f for f in os.listdir(self.input_folder)
                        if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])

        total = len(files)
        if total == 0:
            messagebox.showinfo("Kosong", "Folder input tidak berisi gambar.")
            return

        self.progress['maximum'] = total
        self.progress['value'] = 0

        for i, fname in enumerate(files, start=1):
            path = os.path.join(self.input_folder, fname)
            img = cv2.imread(path)

            if img is None:
                print("Lewati:", fname)
                self.progress['value'] = i
                self.root.update_idletasks()
                continue

            out = enhancement_pipeline(
                img, self.face_cascade,
                do_awb=self.var_awb.get(),
                do_exposure=self.var_exposure.get(),
                do_denoise=self.var_denoise.get(),
                do_face_beauty=self.var_face.get(),
                do_hdr=self.var_hdr.get(),
                do_sharpen=self.var_sharp.get(),
                do_final_tone=self.var_final.get()
            )

            base = os.path.splitext(fname)[0]

            out_path = os.path.join(self.output_folder, f"{base}_final.jpg")
            ba_path = os.path.join(self.output_folder, f"{base}_before_after.jpg")

            cv2.imwrite(out_path, out)
            ba = make_before_after_image(img, out)
            cv2.imwrite(ba_path, ba)

            self.progress['value'] = i
            self.root.update_idletasks()

        messagebox.showinfo("Selesai", f"Semua gambar telah diproses dan disimpan di:\n{self.output_folder}")
        self.progress['value'] = 0



# ============================================================
#                         MAIN PROGRAM
# ============================================================

def main():
    """Menjalankan aplikasi Tkinter."""
    root = tk.Tk()
    app = EnhancerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
