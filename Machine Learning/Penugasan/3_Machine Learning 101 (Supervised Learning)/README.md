# Overview

Selamat datang di **Customer Churn Prediction Challenge**! Dalam challenge ini, Anda ditantang untuk membangun model machine learning yang mampu memprediksi apakah seorang pelanggan akan keluar dari bank (churn) atau tetap menjadi pelanggan aktif. Data yang disediakan mencakup informasi demografis, finansial, dan perilaku pelanggan.

Link Kompetisi: https://www.kaggle.com/t/eae59a87b88945f1a17384afd23e1eac

## Tujuan

Tujuan dari challenge ini adalah untuk memprediksi variabel target `Exited` dalam dataset uji. Variabel ini menunjukkan apakah pelanggan telah keluar dari bank (`1`) atau tetap menjadi pelanggan aktif (`0`). Anda akan membangun model machine learning berdasarkan dataset pelatihan yang telah disediakan.

## Description

![Ilustrasi Churn Bank](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16173485%2Ff1c1cc4fbb6d9e04f2e51e6ffbed15f8%2F1698802120424.png?generation=1737262122445269&alt=media)

_Sumber: [Bank Customers Churn Analysis](https://www.linkedin.com/pulse/bank-customers-churn-analysis-deborah-olatayo-koeof/) oleh Deborah Olatayo Koeof_

## Latar Belakang

Di sebuah kota yang ramai bernama **Arithma**, terdapat sebuah bank bernama **Sentinel Bank** yang telah melayani masyarakat selama puluhan tahun. Dengan teknologi perbankan modern dan layanan pelanggan yang prima, mereka selalu berusaha memberikan pengalaman terbaik bagi nasabahnya. Namun, dalam beberapa tahun terakhir, sesuatu yang aneh mulai terjadi.

Setiap bulan, sejumlah pelanggan **menghilang** dari sistem tanpa jejak—bukan secara fisik, tetapi dalam bentuk akun yang tiba-tiba ditutup. Tidak ada peringatan, tidak ada keluhan sebelumnya, hanya angka yang perlahan turun di laporan keuangan. Para eksekutif bank mulai panik. Apakah ini sabotase? Apakah ada pesaing yang diam-diam mencuri pelanggan mereka? Atau mungkin ini hanya pola alami yang bisa diprediksi?

Untuk mengungkap misteri ini, Sentinel Bank memutuskan untuk **mencari para pemecah kode terbaik—mereka yang mampu menelusuri jejak digital pelanggan dan menemukan pola tersembunyi dalam data**. Mereka menggelar **Customer Churn Prediction Challenge**, sebuah kompetisi bagi para data scientist dan machine learning engineer untuk membangun model yang mampu **memprediksi siapa saja yang berisiko pergi sebelum semuanya terlambat**.

Peserta akan diberikan akses ke data yang mencakup **informasi demografis, finansial, dan perilaku pelanggan**. Dengan data ini, tantangannya adalah menemukan pola yang dapat membantu Sentinel Bank mengambil tindakan pencegahan sebelum pelanggan benar-benar pergi.

Apakah Anda mampu mengungkap **rahasia pelanggan yang hilang** dan membantu Sentinel Bank menyelamatkan bisnis mereka?

---

## Dataset yang Disediakan

1. **`train.csv`**: Data pelatihan yang mencakup informasi pelanggan dan label target (`Exited`).
2. **`test.csv`**: Data uji tanpa label target. Anda diminta untuk memprediksi kolom `Exited` untuk dataset ini.
3. **`sample_submission.csv`**: Contoh format file untuk mengunggah prediksi Anda.

## Evaluation Metric

Submisi peserta akan dievaluasi menggunakan **F1-Score**, sebuah metrik yang mempertimbangkan keseimbangan antara **Precision** dan **Recall**. F1-Score sangat cocok untuk dataset yang memiliki ketidakseimbangan kelas, seperti masalah churn ini.

Berikut ini adalah contoh kode dari scikit-learn untuk menggunakan metrik F1-Score:

```python
from sklearn.metrics import f1_score

y_pred = [0,1,1,0,0,1,1,1,0,0] # anggap ini merupakan hasil prediksi dari data validasi
y_val = [0,1,1,1,0,0,1,0,1,0] # anggap ini adalah nilai asli target dari data validasi

print(f'F1-Score result: {f1_score(y_pred, y_val)}')
```

Lebih lanjut mengenai informasi tentang F1-Score bisa dilihat [disini](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

### Rumus F1-Score

F1-Score dihitung menggunakan rumus berikut:

<div class="formula">
        <code>F1 = 2 * (Precision * Recall) / (Precision + Recall)</code>
</div>
<br>
<b>Precision</b>
<div class="formula">
        <code>Precision = TP / (TP + FP)</code>
</div>
<br>
<p><strong>TP</strong>: True Positives, <strong>FP</strong>: False Positives</p>

<b>Recall</b>

<div class="formula">
    <code>Recall = TP / (TP + FN)</code>
</div>
<br>
<p><strong>TP</strong>: True Positives, <strong>FN</strong>: False Negatives</p>

## Persyaratan Format Submisi

1. File harus dalam format **CSV (Comma-Separated Values)**.
2. Kolom pertama harus berisi **ID** sesuai dengan yang terdapat di `test.csv`.
3. Kolom kedua harus berisi prediksi biner untuk variabel target **Exited**, dengan nilai 0 atau 1.
4. Pastikan tidak ada nilai yang hilang (missing values) dalam file submission.

Submissions yang tidak mengikuti format di atas dapat menyebabkan diskualifikasi atau kegagalan evaluasi.
File yang diunggah harus berisi header dan memiliki format sebagai contoh berikut:
| ID | Exited|
| --- | ------ |
| 2001| 0 |
| 2002 | 1 |
| 2002| 1 |
| ... | ... |

Contoh format submission yang benar adalah seperti file `sample_submission.csv`.

Pastikan untuk memeriksa kembali format file submission sebelum diunggah untuk menghindari kesalahan dalam proses evaluasi.
