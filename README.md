# Simulasi dan Resampling — Ridge Regression

## Project-Based Learning · S2 Statistika Terapan 2026

Dashboard interaktif untuk mengevaluasi Ridge Regression pada dataset **Wine Quality** dengan pendekatan simulasi dan resampling.

## Tujuan

Mengevaluasi kestabilan dan informasi hasil Ridge Regression ketika diterapkan pada data Wine Quality melalui analisis multikolinearitas, pemilihan lambda menggunakan cross-validation, serta bootstrap resampling.

## Dataset

Dataset yang digunakan:

- **Wine Quality Red**
- **Wine Quality White**

Data berisi karakteristik kimia wine dan nilai kualitas wine.

## Metode

Analisis mencakup:

1. Ordinary Least Squares (OLS)
2. Ridge Regression
3. Variance Inflation Factor (VIF)
4. 10-fold Cross-Validation untuk pemilihan lambda
5. Bootstrap Resampling
6. Bootstrap Confidence Interval 95%

Standardisasi prediktor dilakukan berdasarkan data training.

## Isi Dashboard

### Tab 1 — Analisis Ridge

Menampilkan:

- jumlah observasi
- nilai maksimum VIF
- lambda optimal hasil 10-fold CV
- lambda yang dipilih secara interaktif
- MSE OLS
- MSE Ridge
- perbandingan koefisien OLS dan Ridge
- tabel VIF
- tabel koefisien
- interpretasi hasil dan temuan utama

### Tab 2 — Simulasi Resampling

Menampilkan:

- jumlah bootstrap (B = 500 atau 1000)
- koefisien Ridge hasil bootstrap
- Mean dan Standard Error
- Bootstrap Confidence Interval 95%
- RMSE
- lambda hasil 5-fold CV pada bootstrap
- status informatif setiap prediktor

## Cara Menjalankan

Pastikan **R** dan **RStudio** telah terpasang.

Buka folder project, kemudian jalankan:

```r
shiny::runApp()
```

Dashboard akan terbuka melalui browser.

## Struktur Folder

```text
Dashboard_Ridge_Regression/
├── app.R
├── winequality-red.csv
├── winequality-white.csv
├── cache_results/
│   ├── red_results.rds
│   └── white_results.rds
└── README.md
```

## Catatan

Folder `rsconnect/`, `.RData`, dan `.Rhistory` merupakan file konfigurasi/lingkungan kerja lokal dan tidak diperlukan untuk menjalankan analisis utama maupun dashboard.
