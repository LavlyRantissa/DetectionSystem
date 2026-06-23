# Rancangan Sistem Deteksi Konten Buatan AI pada Gambar dan Video Tanpa Audio Berbasis Routing Modalitas dengan Visualisasi Grad-CAM

Skripsi Teknik Komputer, Universitas Indonesia, 2026

---

Sistem ini mendeteksi konten buatan AI berdasarkan tipe file input gambar dan video menggunakan dua model. Jalur deteksi gambar menggunakan model EfficientNet-B0 untuk klasifikasi tiga kelas, yaitu Real, Full-Synthetic, dan Tampered. Jalur deteksi video menggunakan model MobileNetV2 untuk klasifikasi biner, yaitu Real dan Fake dengan subtype head yang mengidentifikasi delapan subtype manipulasi seperti Change of Style, Faceswap, Extrapolation, Interpolation, Inpainting, Outpainting, T2V, TI2V.

Dataset:
[SID-Set](https://huggingface.co/datasets/saberzl/SID_Set)
[FakeParts](https://huggingface.co/datasets/hi-paris/FakeParts)

## Hasil Penelitian
Evaluasi dilakukan menggunakan sistem deteksi yang sudah terintegrasi dengan mekanisme routing modalitas.

| Tipe Input | Jumlah Sampel | Akurasi |
|---|---|---|
| Image | 1.000 (500 Real + 500 Fake) | 84.90% |
| Video | 1.500 (750 Real + 750 Fake) | 95.93% |
| Mixed (gambar + video) | 1.500 (750 gambar + 750 video) | 90.73% |

Split data yang digunakan: 70% train / 15% validasi / 15% test. File CSV split disertakan di repo sehingga angka tersebut dapat direproduksi tanpa melatih ulang model dari awal.

## Arsitektur

![Arsitektur Sistem](placeholder)

Backbone dipilih melalui eksperimen yang membandingkan tiga kandidat backbone untuk masing-masing jalur, kemudian backbone dengan validation F1 tertinggi dipilih sebagai backbone terbaik. EfficientNet-B0 dan MobileNetV2 adalah backbone yang terpilih, keduanya kemudian dilanjutkan ke eksperimen selanjutnya, yaitu pemilihan learning rate terbaik melalui perbandingan tiga variasi learning rate pada setiap model, yaitu learning rate 1e-3, 1e-4, dan 1e-5. Learning rate terbaik untuk kedua model adalah 1e-4.

## Cara Menjalankan

Notebook dirancang untuk Google Colab dengan GPU. Pastikan runtime sudah diset ke GPU sebelum mulai.

### 1. Clone repo ke Google Drive

Buka Google Colab, jalankan cell berikut:

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run([
    'git', 'clone',
    'https://github.com/LavlyRantissa/DetectionSystem.git',
    '/content/drive/MyDrive/skripsi'
])
```

### 2. Sesuaikan path project

Di bagian konfigurasi setiap notebook, ada satu variabel yang perlu disesuaikan:

```python
PROJECT_DIR = "/content/drive/MyDrive/skripsi"
```

Ubah variabel ini jika repo di-clone ke lokasi lain di Drive.

### 3. Jalankan notebook secara berurutan

Setiap notebook menghasilkan file yang dibutuhkan notebook berikutnya, sehingga notebook harus dijalankan secara berurutan.

```
image_model.ipynb → video_model.ipynb → detection_system.ipynb
```

Instalasi dependensi dan download dataset dilakukan secara otomatis di cell pertama masing-masing notebook.

**Jika tidak ingin melatih ulang dari awal**, unduh checkpoint dari Google Drive (link di bawah), letakkan di path yang sesuai, kemudian dapat langsung menjalankan `detection_system.ipynb`.

```
MyDrive/skripsi/experiments/image_1e-4/checkpoints/best.pth
MyDrive/skripsi/experiments/video_1e-4/checkpoints/best.pth
```
## Model Checkpoint

Checkpoint model dapat diakses di:

Google Drive: **https://drive.google.com/drive/folders/16DLMhZVcZ9yOXi8x3AgcmSHLYdH-4jEk?usp=sharing**

## Struktur Repo

```
skripsi/
├── image_model.ipynb
├── video_model.ipynb
├── detection_system.ipynb
│
├── shared_splits/               # split CSV
│   ├── image_split.csv
│   ├── image_binary_input.csv
│   ├── video_split.csv
│   └── mixed_input.csv
│
├── cleaning_reports/            # ringkasan cleaning dataset
│   ├── image/
│   └── video/
│
├── experiments/
│   ├── selection_image/         # hasil seleksi backbone gambar
│   ├── selection_video/         # hasil seleksi backbone video
│   ├── image_1e-4/
│   │   ├── reports/             # metrik dan classification report
│   │   ├── visualizations/
│   │   └── checkpoints/        
│   ├── video_1e-4/
│   │   ├── reports/
│   │   ├── visualizations/
│   │   └── checkpoints/         
│   └── detection_system/
│       ├── reports/
│       └── visualizations/
│
├── visualizations/              
├── requirements.txt
└── .gitignore
```

## Dependensi
```
torch >= 2.0 (cu118)   torchvision   timm
opencv-python-headless   Pillow   scikit-learn
numpy   pandas   matplotlib   seaborn
datasets   huggingface_hub   pyarrow
grad-cam   psutil   tqdm
```

---

Lavly Rantissa Zunnuraina Rusdi · lavly.rantissa@gmail.com · https://www.linkedin.com/in/lavlyrantissa/

