<div align="center">

![HoaxLens — Analisis berita bahasa Indonesia](docs/assets/banner.svg)

**Analisis teks berita Indonesia dengan model terlatih, dalam antarmuka gelap yang sederhana.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1.3-15263F?logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20ROS-2563EB)
![Inference](https://img.shields.io/badge/Workflow-Inference%20only-164E63)

[Memulai](#memulai) · [Fitur](#fitur) · [Evaluasi](#evaluasi-model) · [Struktur](#struktur-repositori) · [API](#api-prediksi)

</div>

---

## Tentang HoaxLens

HoaxLens merupakan aplikasi Flask untuk menganalisis indikasi hoaks pada teks berita berbahasa Indonesia. Aplikasi memuat **vectorizer TF-IDF dan model XGBoost + Random Oversampling (ROS) yang sudah dilatih**, kemudian menjalankan prediksi terhadap masukan pengguna.

Aplikasi **tidak melatih ulang model** dan tidak membaca dataset penelitian saat melayani permintaan. Hasil prediksi merupakan pertimbangan analisis, bukan verifikasi kebenaran sebuah berita.

## Fitur

| Fitur | Keterangan |
| --- | --- |
| Analisis teks | Prediksi kelas berita dan probabilitas hoaks menggunakan model ROS. |
| Preprocessing Indonesia | Pembersihan URL dan karakter, stemming Sastrawi, serta penghapusan stopword lokal. |
| Dashboard evaluasi | Perbandingan metrik baseline dan ROS, ROC, precision–recall, serta confusion matrix. |
| Antarmuka responsif | Tema navy gelap, aksen biru, card bertekstur halus, dan layout untuk desktop maupun ponsel. |
| Validasi masukan | Pesan error untuk JSON tidak valid, tipe data salah, teks kosong, dan teks yang habis setelah dibersihkan. |
| Arsip terpisah | Dataset, model pembanding, dan prototipe lama terpisah dari kebutuhan runtime. |

## Memulai

### 1. Clone repositori

```sh
git clone https://github.com/bangisa/hoax-flask-app.git
cd hoax-flask-app
```

### 2. Siapkan lingkungan Python

Gunakan Python **3.11** seperti lingkungan pengujian proyek, lalu buat virtual environment:

```sh
python -m venv venv
```

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```sh
source venv/bin/activate
```

### 3. Instal dependensi dan jalankan

```sh
python -m pip install -r requirements.txt
python app.py
```

Buka **[http://127.0.0.1:8080](http://127.0.0.1:8080)**. Port dapat diatur melalui variabel lingkungan `PORT`.

Model dan data dicari relatif terhadap lokasi file kode, sehingga aplikasi dapat dijalankan dari direktori kerja lain. Server bawaan Flask digunakan untuk pengembangan lokal.

## Alur analisis

1. Pengguna memasukkan isi berita, bukan hanya tautannya.
2. API memvalidasi payload dan teks.
3. Teks dibersihkan, diubah ke bentuk dasar, dan disaring menggunakan stopword.
4. Vectorizer TF-IDF mengubah teks menjadi fitur.
5. Model XGBoost + ROS menghasilkan kelas prediksi dan probabilitas hoaks.

Dashboard membaca **hasil evaluasi tersimpan dalam JSON**. Angka dashboard tidak dihitung ulang dari berita yang dimasukkan pengguna.

## Evaluasi model

Nilai berikut berasal dari [`metrics.json`](metrics.json) yang disertakan di repositori.

| Metrik | XGBoost baseline | XGBoost + ROS |
| --- | ---: | ---: |
| Akurasi | 89,00% | 89,25% |
| Precision hoaks | 88,37% | 88,79% |
| Recall hoaks | 90,91% | 90,91% |
| F1-score hoaks | 0,8962 | 0,8983 |
| ROC-AUC | 0,9591 | 0,9632 |

Angka ini menggambarkan evaluasi yang tersimpan, bukan jaminan akurasi pada setiap berita baru. Kode pelatihan asli tidak disertakan. Preprocessing mengikuti pipeline penelitian yang tersedia di arsip; kesesuaian dengan pelatihan asli belum dapat diverifikasi sepenuhnya.

## Struktur repositori

```text
hoax-flask-app/
├── app.py                       # Aplikasi Flask dan endpoint prediksi
├── preprocessing.py             # Preprocessing teks Indonesia
├── models/
│   ├── tfidf_vectorizer.pkl      # Vectorizer aktif
│   └── xgboost_ros_model.pkl     # Model prediksi final
├── data/
│   ├── stopwords_indonesian.txt  # Stopword yang digunakan saat inferensi
│   └── ...                      # Dokumentasi sumber stopword
├── templates/                   # Halaman evaluasi, prediksi, navigasi, footer
├── static/theme.css             # Tema gelap dan layout responsif
├── metrics.json                 # Metrik evaluasi tersimpan
├── roc_data.json                # Data kurva ROC
├── pr_data.json                 # Data kurva precision–recall
├── conf_matrix.json             # Confusion matrix
├── tests/                       # Pengujian preprocessing dan validasi
├── archive/
│   ├── research/                # Dataset, model baseline, pipeline offline
│   └── legacy/                  # Prototipe antarmuka lama
├── docs/assets/                 # Aset dokumentasi repositori
└── requirements.txt
```

**Untuk runtime:** pertahankan kode aplikasi, `models/`, stopword di `data/`, template, CSS, dan empat JSON evaluasi. Folder `archive/`, `tests/`, dan `docs/` tidak diperlukan untuk melayani permintaan web.

Lihat [panduan arsip](archive/README.md) untuk penggunaan pipeline evaluasi offline dan [sumber stopword](data/README.md) untuk asal data preprocessing. Dependensi pemuat model tetap diperlukan meskipun aplikasi tidak melakukan pelatihan.

## API prediksi

**`POST /predict`** dengan `Content-Type: application/json`:

```json
{
  "text": "Isi berita berbahasa Indonesia yang ingin dianalisis."
}
```

Respons berhasil (`200`) memiliki bentuk berikut. Nilai ini hanya ilustrasi kontrak API, bukan hasil pengujian teks contoh di atas:

```json
{
  "prediction": 1,
  "probability": 0.81
}
```

- `prediction`: `1` untuk hoaks, `0` untuk berita asli menurut klasifikasi model.
- `probability`: probabilitas kelas hoaks dalam rentang `0`–`1`.

Masukan tidak valid menghasilkan status **`400`**, misalnya:

```json
{
  "error": "Teks tidak boleh kosong"
}
```

## Pengujian

```sh
python -m unittest discover -s tests -v
```

Pengujian mencakup urutan preprocessing, kesesuaian hasil endpoint dengan model, validasi payload, teks kosong, dan masukan dengan spasi. Pengujian runtime tidak memerlukan file penelitian di `archive/`.

## Konteks penelitian

> Analisis Pengaruh Random Oversampling (ROS) terhadap Kinerja XGBoost dalam Memprediksi Berita Hoaks Berbahasa Indonesia

**Muhammad Isa Maulana · 211240001099**<br>
Teknik Informatika · **UNISNU Jepara**
