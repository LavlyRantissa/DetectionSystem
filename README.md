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

## Struktur Repo

## Dependensi
