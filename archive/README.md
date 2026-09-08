# Arsip penelitian dan antarmuka lama

Folder ini tidak dibaca atau diimpor oleh aplikasi web maupun pengujian runtime.
Tidak perlu disertakan dalam deployment aplikasi.

- `research/dataset_terjemahan.csv`: dataset penelitian, bukan input runtime.
- `research/xgboost_baseline_model.pkl`: model pembanding untuk evaluasi offline.
- `research/model_pipeline.py`: fungsi evaluasi offline dan prediksi pembanding.
  Menggunakan vectorizer dan model ROS aktif dari `models/` serta preprocessing
  bersama. Dari direktori proyek, impor dengan
  `from archive.research.model_pipeline import run_pipeline`.
  Fungsi tersebut menerima DataFrame, bukan membaca dataset secara otomatis.
  Untuk pekerjaan berbasis DataFrame, pasang pandas terpisah di lingkungan riset.
- `legacy/test.html`: prototipe lama yang mengirim ke `/train`. Endpoint tersebut
  tidak tersedia; file ini disimpan sebagai referensi, bukan halaman aplikasi.

Isi dataset dan model dipertahankan tanpa perubahan. File arsip masih dapat
disimpan di Git untuk dokumentasi dan reproduksi penelitian.
