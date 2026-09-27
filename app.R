# ============================================================
# PROJECT ANAREG - DASHBOARD RIDGE REGRESSION
# Wine Quality: Red vs White
# Fokus: Prediction, Stability, Informativeness
#
# Cara pakai:
# 1. Simpan file ini sebagai app.R
# 2. Letakkan winequality-red.csv dan winequality-white.csv
#    di folder yang sama dengan app.R
# 3. Jalankan: shiny::runApp()
#
# CATATAN:
# - Analisis mengikuti pipeline script Red/White:
#   train-test 80:20, standardisasi berdasarkan train,
#   OLS, VIF, Ridge + 10-fold CV, test evaluation,
#   bootstrap B=1000 dengan re-tuning lambda.
# - Hasil analisis disimpan otomatis ke folder cache_results/
#   sehingga bootstrap tidak perlu diulang setiap kali app dibuka.
# ============================================================

required_packages <- c(
  "shiny", "bslib", "ggplot2", "dplyr", "tidyr",
  "glmnet", "car", "plotly", "DT", "scales"
)

library(shiny)
library(bslib)
library(ggplot2)
library(dplyr)
library(tidyr)
library(glmnet)
library(car)
library(plotly)
library(DT)
library(scales)


options(scipen = 999)
options(digits = 4)

# Versi cache dashboard. Naikkan jika struktur hasil bootstrap berubah.
CACHE_VERSION <- 7L

# ============================================================
# TAMPILAN DASHBOARD
# ============================================================
dashboard_css <- tags$head(
  tags$style(HTML("
    html, body {
      min-height: 100%;
      height: auto;
      overflow-y: auto !important;
      overflow-x: hidden;
    }

    body {
      background: #f7f9fc;
      color: #243447;
    }

    .navbar {
      margin-bottom: 0;
    }

    /* Tab boleh memanjang; browser yang melakukan scroll, bukan card */
    .tab-content,
    .tab-pane,
    .bslib-page-navbar .tab-content {
      height: auto !important;
      min-height: 0 !important;
      overflow: visible !important;
    }

    .bslib-sidebar-layout {
      min-height: 0 !important;
      height: auto !important;
      overflow: visible !important;
    }

    .bslib-sidebar-layout > .main {
      padding: 22px 28px 60px 28px;
    }

    .bslib-sidebar-layout > .sidebar {
      background: #ffffff;
      border-right: 1px solid #e2e8f0;
      padding: 22px 18px;
    }

    .card {
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(31, 41, 55, 0.05);
      margin-bottom: 18px;
      overflow: visible !important;
    }

    .card-body {
      overflow: visible !important;
    }

    .card-header {
      background: #ffffff;
      border-bottom: 1px solid #edf1f5;
      font-weight: 600;
      color: #243447;
      padding: 13px 16px;
    }

    .kpi-card {
      min-width: 0 !important;
      width: 100% !important;
      min-height: 112px !important;
      height: 112px !important;
      border-radius: 10px;
      border: 1px solid #e2e8f0;
      background: #ffffff;
      box-shadow: 0 2px 8px rgba(31, 41, 55, 0.04);
      padding: 22px 16px !important;
      display: flex !important;
      flex-direction: column !important;
      justify-content: center !important;
      align-items: flex-start !important;
      box-sizing: border-box !important;
      overflow: hidden !important;
    }

    .kpi-label {
      font-size: .82rem !important;
      line-height: 1.15 !important;
      color: #5b6777;
      margin-bottom: 8px !important;
      white-space: nowrap !important;
    }

    .kpi-number,
    .kpi-number *,
    .kpi-number span,
    .kpi-number div {
      display: block !important;
      width: max-content !important;
      max-width: 100% !important;
      font-size: 1.02rem !important;
      line-height: 1.1 !important;
      font-weight: 600 !important;
      white-space: nowrap !important;
      word-break: normal !important;
      overflow-wrap: normal !important;
      hyphens: none !important;
      letter-spacing: -0.01em !important;
      margin: 0 !important;
    }

    .kpi-optimal-btn {
      text-align: left !important;
      color: #243447 !important;
      cursor: pointer;
      appearance: none !important;
      -webkit-appearance: none !important;
    }

    .kpi-optimal-btn:hover {
      background: #f7fbfb !important;
    }

    /* Academic Premium palette: restrained accents by meaning */
    .kpi-row > .kpi-card {
      border-top: 3px solid #e2e8f0;
    }
    .kpi-row > .kpi-card:nth-child(2) {
      border-top-color: #2a6f97;
    }
    .kpi-row > .kpi-card:nth-child(3) {
      border-top-color: #00a6a6;
    }
    .kpi-row > .kpi-card:nth-child(4) {
      border-top-color: #2a6f97;
    }
    .kpi-row > .kpi-card:nth-child(5) {
      border-top-color: #e76f51;
    }
    .kpi-row > .kpi-card:nth-child(6) {
      border-top-color: #00a6a6;
    }

    .lambda-optimal-label {
      display: block !important;
      color: #5b6777 !important;
      font-size: .75rem !important;
      line-height: 1.15 !important;
      margin-bottom: 8px !important;
      white-space: nowrap !important;
    }

    .lambda-optimal-value {
      display: block !important;
      width: max-content !important;
      max-width: 100% !important;
      font-size: 1.02rem !important;
      line-height: 1.1 !important;
      font-weight: 600 !important;
      margin: 0 !important;
      color: #243447 !important;
      white-space: nowrap !important;
      word-break: normal !important;
      overflow-wrap: normal !important;
      letter-spacing: -0.01em !important;
    }

    /* Ikon/showcase sengaja disembunyikan agar tidak menutupi angka */
    .value-box .value-box-showcase,
    .value-box > .showcase,
    .value-box .showcase {
      display: none !important;
    }

    /* Tabel koefisien harus memanjang mengikuti seluruh isi, bukan terpotong
       oleh tinggi default DTOutput. */
    .coef-table-card {
      height: auto !important;
      min-height: 0 !important;
      overflow: visible !important;
    }

    .coef-table-card .card-body {
      height: auto !important;
      min-height: 0 !important;
      overflow-x: auto !important;
      overflow-y: visible !important;
    }

    .coef-table-card .dataTables_wrapper {
      height: auto !important;
    }

    .coef-table-card table.dataTable {
      width: 100% !important;
      margin: 0 !important;
    }

    /* Area interpretasi: lebih tinggi dan tidak membuat scrollbar internal */
    .interpretation-card {
      min-height: 165px !important;
      height: auto !important;
      overflow: visible !important;
    }

    .interpretation-card .card-body {
      min-height: 105px !important;
      overflow: visible !important;
      padding: 16px 18px !important;
    }

    .interpretation-text {
      font-size: 0.98rem;
      line-height: 1.6;
      margin: 0;
    }

    .interpretation-card .section-note {
      margin: 0;
    }

    h3 {
      margin-top: 4px;
      margin-bottom: 18px;
      font-weight: 650;
      color: #243447;
    }

    .section-note {
      background: #f7f9fc;
      border-left: 4px solid #00a6a6;
      padding: 12px 15px;
      border-radius: 6px;
      margin-top: 2px;
      line-height: 1.55;
    }

    .sidebar-title {
      font-weight: 650;
      margin-bottom: 12px;
    }

    .selectize-input,
    .form-control {
      border-radius: 8px !important;
    }

    .dataTables_wrapper {
      font-size: 0.9rem;
    }

    .plotly {
      border-radius: 8px;
    }

    @media (max-width: 1000px) {
      .bslib-sidebar-layout > .main {
        padding: 16px;
      }

      .value-box {
        min-height: 105px;
      }
    }
  "))
)


# CSS tambahan untuk header project dan layout 2-tab
project_css <- tags$head(
  tags$style(HTML("
    .project-brand {
      display: flex;
      align-items: center;
      padding: 14px 8px 16px;
      white-space: normal;
      width: 100%;
    }
    .brand-text {
      display: flex;
      flex-direction: column;
      line-height: 1.08;
    }
    .brand-project {
      font-size: 1.56rem;
      font-weight: 650;
      letter-spacing: .01em;
      color: #2a6f97;
    }
    .brand-title {
      font-size: 2.24rem;
      font-weight: 800;
      margin-top: 8px;
      color: #00a6a6;
    }
    .dashboard-tab-nav {
      margin: 4px 0 18px;
    }
    .dashboard-tab-nav .nav-link {
      font-weight: 700;
      font-size: 1rem;
      padding: 12px 22px;
      margin-right: 8px;
      border-radius: 10px 10px 0 0;
    }
    .dashboard-tab-nav .nav-link.active {
      background: #ffffff;
      box-shadow: 0 -1px 0 rgba(0,0,0,.04), 0 3px 10px rgba(31,41,55,.08);
    }
    .page-wrap {
      max-width: 1450px;
      margin: 0 auto;
      padding: 24px 28px 60px;
    }
    .page-wrap h2 {
      margin: 18px 0 16px;
      font-size: 1.55rem;
      font-weight: 700;
      color: #243447;
    }
    .control-card .card-body {
      padding: 14px 16px 10px;
    }
    .control-card .card-header {
      padding: 10px 15px;
      font-size: .9rem;
    }
    .helper-text {
      color: #667085;
      font-size: .82rem;
      line-height: 1.4;
      margin-top: 4px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    .lambda-optimal-btn {
      width: 100%;
      text-align: left;
      border-radius: 10px !important;
      border: 1px solid #e1e7ef !important;
      background: #ffffff !important;
      color: #243447 !important;
      box-shadow: 0 2px 8px rgba(31, 41, 55, 0.04);
      padding: 12px 14px !important;
      min-height: 78px;
    }
    .lambda-optimal-btn:hover {
      background: #f8fafc !important;
    }
    .lambda-optimal-label {
      display: block;
      color: #5b6777;
      font-size: .75rem;
      line-height: 1.15;
    }
    .lambda-optimal-value {
      display: block;
      font-size: 1.25rem !important;
      line-height: 1.15 !important;
      font-weight: 600 !important;
      margin-top: 3px;
      color: #243447;
      white-space: nowrap !important;
      word-break: keep-all !important;
      overflow-wrap: normal !important;
      letter-spacing: -0.02em;
    }
    .summary-item {
      background: #f7f9fc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
    }
    .summary-value {
      font-size: 1.1rem;
      font-weight: 700;
      margin-top: 4px;
    }
    .value-box {
      min-height: 78px !important;
      height: 78px !important;
    }
    .value-box .value-box-value {
      font-size: 1.25rem !important;
      line-height: 1.15 !important;
      white-space: nowrap !important;
      overflow: visible !important;
    }
    .value-box .value-box-title {
      font-size: .75rem !important;
    }
    .interpretation-card {
      min-height: 130px !important;
    }
    .interpretation-card .card-body {
      min-height: 85px !important;
    }
    .insight-card {
      border-left: 4px solid #00a6a6;
    }

    .insight-card .card-header {
      font-size: 1.05rem;
    }

    @media (max-width: 900px) {
      .project-brand { white-space: normal; }
      .brand-project { font-size: 1.05rem; }
      .brand-title { font-size: 1.45rem; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .page-wrap { padding: 18px 14px 45px; }
    }
  "))
)



# Finishing layer: Academic Premium v2
finishing_css <- tags$head(
  tags$style(HTML("
    /* --- Global visual hierarchy --- */
    body {
      background: #f5f7fa !important;
      color: #243447 !important;
    }

    .page-wrap h2 {
      color: #243447 !important;
      letter-spacing: -0.015em;
    }

    /* --- Cards: subtle premium depth --- */
    .card {
      border: 1px solid #dfe6ee !important;
      box-shadow: 0 3px 12px rgba(36, 52, 71, 0.055) !important;
    }

    .card-header {
      color: #243447 !important;
      background: #ffffff !important;
      font-weight: 700 !important;
    }

    /* --- KPI cards: meaningful accent colors --- */
    .kpi-row > .kpi-card:nth-child(1) { border-top: 3px solid #2a6f97 !important; }
    .kpi-row > .kpi-card:nth-child(2) { border-top: 3px solid #e0a458 !important; }
    .kpi-row > .kpi-card:nth-child(3) { border-top: 3px solid #00a6a6 !important; }
    .kpi-row > .kpi-card:nth-child(4) { border-top: 3px solid #2a6f97 !important; }
    .kpi-row > .kpi-card:nth-child(5) { border-top: 3px solid #e76f51 !important; }
    .kpi-row > .kpi-card:nth-child(6) { border-top: 3px solid #00a6a6 !important; }

    .kpi-number {
      font-size: 1.12rem !important;
      font-weight: 700 !important;
      color: #243447 !important;
    }

    .kpi-row > .kpi-card:nth-child(5) .kpi-number {
      color: #c9573d !important;
    }

    .kpi-row > .kpi-card:nth-child(6) .kpi-number,
    .kpi-row > .kpi-card:nth-child(3) .kpi-number {
      color: #008f8f !important;
    }

    /* --- Tabs --- */
    .dashboard-tab-nav .nav-link {
      color: #667085 !important;
      transition: all .15s ease;
    }

    .dashboard-tab-nav .nav-link:hover {
      color: #008f8f !important;
      background: #f4fbfb !important;
    }

    .dashboard-tab-nav .nav-link.active {
      color: #008f8f !important;
      background: #ffffff !important;
      border-top: 3px solid #00a6a6 !important;
      box-shadow: 0 -1px 0 rgba(0,0,0,.03), 0 4px 12px rgba(36,52,71,.08) !important;
    }

    /* --- Temuan Utama --- */
    .insight-card {
      border-left: 4px solid #00a6a6 !important;
      background: linear-gradient(90deg, #f0fbfa 0%, #ffffff 22%) !important;
    }

    .insight-card .card-header {
      background: #f0fbfa !important;
      color: #006f70 !important;
      border-bottom-color: #d7eeee !important;
    }

    .insight-card .interpretation-text {
      color: #243447 !important;
      font-size: 1rem !important;
      line-height: 1.65 !important;
    }

    /* --- Interpretation and conclusion --- */
    .interpretation-card .card-header {
      background: #f8fafc !important;
    }

    /* --- Tables --- */
    .dataTables_wrapper table.dataTable thead th {
      background: #eef5f7 !important;
      color: #243447 !important;
      border-bottom: 2px solid #9bcaca !important;
      font-weight: 700 !important;
    }

    .dataTables_wrapper table.dataTable tbody tr:hover {
      background: #f4fbfb !important;
    }

    /* --- Control panels --- */
    .control-card .card-header {
      color: #2a6f97 !important;
    }

    /* --- Brand --- */
    .brand-project {
      color: #2a6f97 !important;
    }

    .brand-title {
      color: #008f8f !important;
    }
  "))
)



# Finishing layer: Academic Premium v3
finishing_css_v3 <- tags$head(
  tags$style(HTML("
    /* Academic Premium v3: visual-only changes.
       OLS/Ridge Plotly colors, header, and tab menu are untouched. */

    .card {
      border-color: #d9e2ea !important;
      box-shadow: 0 4px 14px rgba(36, 52, 71, 0.065) !important;
    }

    /* Control panels */
    .control-card {
      background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%) !important;
      border-top: 3px solid #d6e8eb !important;
    }

    .control-card .card-header {
      background: transparent !important;
      color: #2a6f97 !important;
      letter-spacing: .01em;
    }

    /* KPI cards: soft tint by meaning */
    .kpi-row > .kpi-card {
      border: 1px solid #dfe7ee !important;
      box-shadow: 0 4px 12px rgba(36, 52, 71, 0.055) !important;
      position: relative;
      transition: transform .15s ease, box-shadow .15s ease;
    }

    .kpi-row > .kpi-card:nth-child(1) {
      background: linear-gradient(145deg, #ffffff 0%, #f4f8fb 100%) !important;
      border-top: 3px solid #2a6f97 !important;
    }

    .kpi-row > .kpi-card:nth-child(2) {
      background: linear-gradient(145deg, #fffdf8 0%, #fff7e8 100%) !important;
      border-top: 3px solid #e0a458 !important;
    }

    .kpi-row > .kpi-card:nth-child(3) {
      background: linear-gradient(145deg, #ffffff 0%, #eefafa 100%) !important;
      border-top: 3px solid #00a6a6 !important;
    }

    .kpi-row > .kpi-card:nth-child(4) {
      background: linear-gradient(145deg, #ffffff 0%, #f1f7fb 100%) !important;
      border-top: 3px solid #2a6f97 !important;
    }

    .kpi-row > .kpi-card:nth-child(5) {
      background: linear-gradient(145deg, #fffafa 0%, #fff1ed 100%) !important;
      border-top: 3px solid #e76f51 !important;
    }

    .kpi-row > .kpi-card:nth-child(6) {
      background: linear-gradient(145deg, #ffffff 0%, #eefafa 100%) !important;
      border-top: 3px solid #00a6a6 !important;
    }

    .kpi-row > .kpi-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(36, 52, 71, 0.09) !important;
    }

    .kpi-label {
      color: #667085 !important;
      font-weight: 600 !important;
    }

    .kpi-optimal-btn {
      background: linear-gradient(145deg, #ffffff 0%, #eefafa 100%) !important;
      border-top: 3px solid #00a6a6 !important;
    }

    .kpi-optimal-btn:hover {
      background: #eaf8f8 !important;
    }

    .lambda-optimal-label {
      color: #007f80 !important;
      font-weight: 650 !important;
    }

    /* VIF and beta panels */
    .analysis-card {
      box-shadow: 0 4px 14px rgba(36, 52, 71, 0.06) !important;
    }

    .vif-card {
      border-top: 3px solid #2a6f97 !important;
      background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%) !important;
    }

    .vif-card .card-header {
      background: #f1f6fa !important;
      color: #2a6f97 !important;
    }

    .beta-card {
      border-top: 3px solid #00a6a6 !important;
    }

    .beta-card .card-header {
      background: #f0fbfa !important;
      color: #007f80 !important;
    }

    /* Table panels */
    .vif-card table.dataTable tbody tr:nth-child(even),
    .coef-table-card table.dataTable tbody tr:nth-child(even) {
      background: #f8fafc !important;
    }

    .vif-card table.dataTable tbody tr:hover,
    .coef-table-card table.dataTable tbody tr:hover {
      background: #edf8f8 !important;
    }

    .coef-table-card {
      border-top: 3px solid #2a6f97 !important;
    }

    .coef-table-card .card-header {
      background: #f3f7fa !important;
      color: #243447 !important;
    }

    .dataTables_wrapper table.dataTable thead th {
      background: #eaf3f5 !important;
      color: #243447 !important;
      border-bottom: 2px solid #9bcaca !important;
      font-weight: 700 !important;
    }

    /* Temuan utama */
    .insight-card {
      border: 1px solid #b9dddd !important;
      border-left: 5px solid #00a6a6 !important;
      background: linear-gradient(105deg, #eefafa 0%, #ffffff 48%, #ffffff 100%) !important;
      box-shadow: 0 5px 16px rgba(0, 166, 166, 0.08) !important;
    }

    .insight-card .card-header {
      background: rgba(238, 250, 250, 0.82) !important;
      color: #006f70 !important;
      font-weight: 750 !important;
    }

    .insight-card .card-body {
      padding: 18px 20px !important;
    }

    /* Interpretation and conclusion */
    .interpretation-card {
      border-top: 3px solid #9aa9b8 !important;
      background: #ffffff !important;
    }

    .interpretation-card .card-header {
      background: #f7f9fb !important;
      color: #334155 !important;
    }

    .interpretation-card .card-body {
      color: #3f4d5d !important;
    }

    .page-wrap h2 {
      color: #243447 !important;
      font-weight: 750 !important;
    }
  "))
)


# ============================================================
# 1. FUNGSI ANALISIS
# ============================================================

run_wine_analysis <- function(file_name, wine_name, B = 1000) {

  if (!file.exists(file_name)) {
    stop(
      paste0(
        "File ", file_name,
        " tidak ditemukan. Letakkan file CSV di folder yang sama dengan app.R."
      )
    )
  }

  df <- read.csv(file_name, header = TRUE, sep = ";")

  # -----------------------------
  # Train-test split
  # -----------------------------
  set.seed(123)
  train_indices <- sample(
    1:nrow(df),
    size = floor(0.8 * nrow(df))
  )

  train_data <- df[train_indices, , drop = FALSE]
  test_data  <- df[-train_indices, , drop = FALSE]

  # -----------------------------
  # X dan Y
  # -----------------------------
  predictor_names <- setdiff(names(df), "quality")

  X_train_raw <- as.matrix(train_data[, predictor_names, drop = FALSE])
  Y_train <- train_data$quality

  X_test_raw <- as.matrix(test_data[, predictor_names, drop = FALSE])
  Y_test <- test_data$quality

  # -----------------------------
  # Standardisasi berdasarkan TRAIN
  # -----------------------------
  mean_train <- colMeans(X_train_raw)
  sd_train <- apply(X_train_raw, 2, sd)

  X_train <- scale(
    X_train_raw,
    center = mean_train,
    scale = sd_train
  )

  X_test <- scale(
    X_test_raw,
    center = mean_train,
    scale = sd_train
  )

  train_scaled <- as.data.frame(X_train)
  names(train_scaled) <- predictor_names
  train_scaled$quality <- Y_train

  test_scaled <- as.data.frame(X_test)
  names(test_scaled) <- predictor_names
  test_scaled$quality <- Y_test

  # ==========================================================
  # OLS
  # ==========================================================

  ols <- lm(quality ~ ., data = train_scaled)

  # VIF
  vif_values <- car::vif(ols)

  vif_table <- data.frame(
    Variable = names(vif_values),
    VIF = as.numeric(vif_values)
  )

  # ==========================================================
  # RIDGE + 10-FOLD CV
  # ==========================================================

  set.seed(456)

  # X sudah distandardisasi dari training data,
  # sehingga standardize = FALSE agar tidak distandardisasi ulang.
  cv_ridge <- cv.glmnet(
    x = X_train,
    y = Y_train,
    alpha = 0,
    nfolds = 10,
    standardize = FALSE
  )

  lambda_min <- cv_ridge$lambda.min
  lambda_1se <- cv_ridge$lambda.1se

  ridge_final <- glmnet(
    x = X_train,
    y = Y_train,
    alpha = 0,
    lambda = lambda_min,
    standardize = FALSE
  )

  # Jalur koefisien Ridge untuk interaksi lambda
  ridge_path <- glmnet(
    x = X_train,
    y = Y_train,
    alpha = 0,
    standardize = FALSE
  )

  # ==========================================================
  # KOEFISIEN
  # ==========================================================

  beta_ols <- coef(ols)
  beta_ridge <- as.matrix(coef(ridge_final))

  coef_table <- data.frame(
    Variable = names(beta_ols),
    OLS = as.numeric(beta_ols),
    Ridge = as.numeric(beta_ridge)
  )

  # ==========================================================
  # TEST PERFORMANCE
  # ==========================================================

  pred_ols <- as.numeric(
    predict(ols, newdata = test_scaled)
  )

  pred_ridge <- as.numeric(
    predict(ridge_final, newx = X_test)
  )

  res_ols <- Y_test - pred_ols
  res_ridge <- Y_test - pred_ridge

  mse_ols <- mean(res_ols^2)
  rmse_ols <- sqrt(mse_ols)
  r2_ols <- 1 -
    sum(res_ols^2) /
    sum((Y_test - mean(Y_test))^2)

  mse_ridge <- mean(res_ridge^2)
  rmse_ridge <- sqrt(mse_ridge)
  r2_ridge <- 1 -
    sum(res_ridge^2) /
    sum((Y_test - mean(Y_test))^2)

  performance <- data.frame(
    Model = c("OLS", "Ridge"),
    MSE_Test = c(mse_ols, mse_ridge),
    RMSE_Test = c(rmse_ols, rmse_ridge),
    R2_Test = c(r2_ols, r2_ridge)
  )

  # ==========================================================
  # BOOTSTRAP
  # ==========================================================

  set.seed(789)

  n_train <- nrow(X_train)
  var_names <- c("Intercept", colnames(X_train))

  boot_beta_ols <- matrix(
    NA_real_,
    nrow = B,
    ncol = length(var_names)
  )

  boot_beta_ridge <- matrix(
    NA_real_,
    nrow = B,
    ncol = length(var_names)
  )

  # Simpan lambda hasil 5-fold CV untuk SETIAP bootstrap.
  # Ini membuat dashboard dapat menampilkan lambda yang benar-benar
  # dipilih pada bootstrap ke-b yang sedang ditampilkan.
  boot_lambda <- rep(NA_real_, B)

  colnames(boot_beta_ols) <- var_names
  colnames(boot_beta_ridge) <- var_names

  # Bootstrap mengikuti script proyek:
  # - resample training data
  # - fit OLS
  # - retune lambda dengan 5-fold CV
  # - fit Ridge
  for (b in seq_len(B)) {

    boot_indices <- sample(
      1:n_train,
      size = n_train,
      replace = TRUE
    )

    X_boot <- X_train[
      boot_indices, ,
      drop = FALSE
    ]

    Y_boot <- Y_train[boot_indices]

    fit_ols_boot <- lm(Y_boot ~ X_boot)

    boot_beta_ols[b, ] <- coef(fit_ols_boot)

    cv_boot <- cv.glmnet(
      x = X_boot,
      y = Y_boot,
      alpha = 0,
      nfolds = 5,
      standardize = FALSE
    )

    boot_lambda[b] <- cv_boot$lambda.min

    fit_ridge_boot <- glmnet(
      x = X_boot,
      y = Y_boot,
      alpha = 0,
      lambda = cv_boot$lambda.min,
      standardize = FALSE
    )

    boot_beta_ridge[b, ] <-
      as.vector(coef(fit_ridge_boot))
  }

  # Bootstrap SE
  se_boot_ols <- apply(
    boot_beta_ols,
    2,
    sd
  )

  se_boot_ridge <- apply(
    boot_beta_ridge,
    2,
    sd
  )

  # Bootstrap CI
  ci_ols_lower <- apply(
    boot_beta_ols,
    2,
    quantile,
    probs = 0.025
  )

  ci_ols_upper <- apply(
    boot_beta_ols,
    2,
    quantile,
    probs = 0.975
  )

  ci_ridge_lower <- apply(
    boot_beta_ridge,
    2,
    quantile,
    probs = 0.025
  )

  ci_ridge_upper <- apply(
    boot_beta_ridge,
    2,
    quantile,
    probs = 0.975
  )

  stability <- data.frame(
    Variable = var_names,
    SE_OLS = se_boot_ols,
    SE_Ridge = se_boot_ridge,
    Ratio_SE_Ridge_vs_OLS =
      se_boot_ridge / se_boot_ols,
    CI_Ridge_Lower = ci_ridge_lower,
    CI_Ridge_Upper = ci_ridge_upper
  )

  # Data bootstrap long untuk grafik
  boot_ols_df <- as.data.frame(
    boot_beta_ols
  )
  boot_ols_df$Model <- "OLS"

  boot_ridge_df <- as.data.frame(
    boot_beta_ridge
  )
  boot_ridge_df$Model <- "Ridge"

  boot_long <- bind_rows(
    boot_ols_df,
    boot_ridge_df
  ) |>
    pivot_longer(
      cols = all_of(var_names),
      names_to = "Variable",
      values_to = "Beta"
    )

  # ==========================================================
  # DIAGNOSTICS
  # ==========================================================

  residual_df <- data.frame(
    Fitted = fitted(ols),
    Residual = resid(ols)
  )

  cook_values <- cooks.distance(ols)

  cook_df <- data.frame(
    Observation = seq_along(cook_values),
    Cook = as.numeric(cook_values)
  )

  cook_threshold <- 4 / nrow(train_scaled)

  # ==========================================================
  # RETURN
  # ==========================================================

  list(
    wine_name = wine_name,
    data = df,
    train_data = train_data,
    test_data = test_data,
    n_total = nrow(df),
    n_train = nrow(train_data),
    n_test = nrow(test_data),
    n_predictors = length(predictor_names),
    predictor_names = predictor_names,

    X_train = X_train,
    X_test = X_test,
    Y_train = Y_train,
    Y_test = Y_test,

    ols = ols,
    vif_table = vif_table,

    cv_ridge = cv_ridge,
    lambda_min = lambda_min,
    lambda_1se = lambda_1se,
    ridge_final = ridge_final,
    ridge_path = ridge_path,

    coef_table = coef_table,
    performance = performance,

    boot_beta_ols = boot_beta_ols,
    boot_beta_ridge = boot_beta_ridge,
    boot_lambda = boot_lambda,
    boot_long = boot_long,
    cache_version = CACHE_VERSION,
    stability = stability,

    residual_df = residual_df,
    cook_df = cook_df,
    cook_threshold = cook_threshold,

    B = B
  )
}

# ============================================================
# 2. LOAD / CACHE ANALYSIS
# ============================================================

dir.create(
  "cache_results",
  showWarnings = FALSE
)

load_or_run <- function(
    file_name,
    wine_name,
    cache_name
) {

  cache_file <- file.path(
    "cache_results",
    cache_name
  )

  if (file.exists(cache_file)) {
    cached <- readRDS(cache_file)

    # Cache lama belum menyimpan lambda untuk setiap bootstrap.
    # Karena dashboard sekarang menampilkan lambda sesuai bootstrap terpilih,
    # cache tersebut harus dihitung ulang sekali.
    if (!is.null(cached$boot_lambda) &&
        length(cached$boot_lambda) >= 1000L &&
        identical(cached$cache_version, CACHE_VERSION)) {
      message("Membaca hasil cache: ", cache_file)
      return(cached)
    }

    message("Cache perlu diperbarui ke versi ", CACHE_VERSION, ": ", cache_file)
  }

  message(
    "Menjalankan analisis ",
    wine_name,
    ". Bootstrap B=1000 dapat memerlukan beberapa menit..."
  )

  result <- run_wine_analysis(
    file_name = file_name,
    wine_name = wine_name,
    B = 1000
  )

  saveRDS(
    result,
    cache_file
  )

  result
}

red <- load_or_run(
  "winequality-red.csv",
  "Red Wine",
  "red_results.rds"
)

white <- load_or_run(
  "winequality-white.csv",
  "White Wine",
  "white_results.rds"
)

wine_data <- list(
  "Red Wine" = red,
  "White Wine" = white
)


# ============================================================
# 3. METRIK SIMULASI BOOTSTRAP
# ============================================================
add_boot_metrics <- function(result) {
  B_available <- nrow(result$boot_beta_ridge)
  Xtest_aug <- cbind(Intercept = 1, result$X_test)

  # Lambda referensi untuk simulasi: 5-fold CV pada training set utama.
  set.seed(456)
  cv5_main <- cv.glmnet(
    x = result$X_train, y = result$Y_train, alpha = 0,
    nfolds = 5, standardize = FALSE
  )
  result$lambda_5fold <- cv5_main$lambda.min

  result$boot_rmse <- vapply(seq_len(B_available), function(b) {
    beta_b <- result$boot_beta_ridge[b, ]
    pred_b <- as.numeric(Xtest_aug %*% beta_b)
    sqrt(mean((result$Y_test - pred_b)^2))
  }, numeric(1))

  result$boot_nfolds <- 5L
  result
}

red <- add_boot_metrics(red)
white <- add_boot_metrics(white)

# Simpan ulang cache agar metrik RMSE tersedia pada pembukaan berikutnya.
saveRDS(red, file.path("cache_results", "red_results.rds"))
saveRDS(white, file.path("cache_results", "white_results.rds"))

wine_data <- list("Red Wine" = red, "White Wine" = white)




# Academic Premium v4: stronger panel styling + Unpad logo.
finishing_css_v4 <- tags$head(
  tags$style(HTML("
    /* Logo: add identity without changing the existing header typography */
    .project-brand {
      gap: 16px;
    }

    .brand-logo {
      width: 68px !important;
      height: 68px !important;
      object-fit: contain !important;
      flex: 0 0 68px !important;
      background: #ffffff;
      border-radius: 12px;
      padding: 5px;
      box-shadow: 0 3px 12px rgba(36,52,71,.10);
    }

    /* More visible but restrained card differentiation */
    .control-card {
      border: 1px solid #cfdde5 !important;
      box-shadow: 0 5px 16px rgba(36,52,71,.07) !important;
      background: #fbfdfe !important;
    }

    .control-card .card-header {
      background: #eef6f8 !important;
      color: #285b73 !important;
      border-bottom: 1px solid #d8e7ec !important;
      font-weight: 700 !important;
    }

    /* KPI: clearly different soft backgrounds */
    .kpi-row > .kpi-card:nth-child(1) {
      background: #f5f8fb !important;
      border: 1px solid #d6e1ea !important;
      border-top: 4px solid #2a6f97 !important;
    }

    .kpi-row > .kpi-card:nth-child(2) {
      background: #fff8ea !important;
      border: 1px solid #eadbbd !important;
      border-top: 4px solid #e0a458 !important;
    }

    .kpi-row > .kpi-card:nth-child(3) {
      background: #eefafa !important;
      border: 1px solid #c9e8e8 !important;
      border-top: 4px solid #00a6a6 !important;
    }

    .kpi-row > .kpi-card:nth-child(4) {
      background: #f1f7fb !important;
      border: 1px solid #d0e1ec !important;
      border-top: 4px solid #2a6f97 !important;
    }

    .kpi-row > .kpi-card:nth-child(5) {
      background: #fff3ef !important;
      border: 1px solid #efd3ca !important;
      border-top: 4px solid #e76f51 !important;
    }

    .kpi-row > .kpi-card:nth-child(6) {
      background: #eefafa !important;
      border: 1px solid #c9e8e8 !important;
      border-top: 4px solid #00a6a6 !important;
    }

    .kpi-row > .kpi-card .kpi-number {
      font-weight: 700 !important;
    }

    .kpi-row > .kpi-card:nth-child(5) .kpi-number {
      color: #c9573d !important;
    }

    .kpi-row > .kpi-card:nth-child(6) .kpi-number,
    .kpi-row > .kpi-card:nth-child(3) .kpi-number {
      color: #008f8f !important;
    }

    .kpi-row > .kpi-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(36,52,71,.11) !important;
    }

    /* VIF panel */
    .vif-card {
      background: #f8fbfd !important;
      border: 1px solid #cddfe9 !important;
      border-top: 4px solid #2a6f97 !important;
      box-shadow: 0 5px 16px rgba(42,111,151,.07) !important;
    }

    .vif-card .card-header {
      background: #eaf3f8 !important;
      color: #245d7d !important;
      border-bottom: 1px solid #d3e3ec !important;
      font-weight: 750 !important;
    }

    /* Beta panel frame only: Plotly colors remain exactly as they are */
    .beta-card {
      background: #fbffff !important;
      border: 1px solid #c9e5e5 !important;
      border-top: 4px solid #00a6a6 !important;
      box-shadow: 0 5px 16px rgba(0,166,166,.07) !important;
    }

    .beta-card .card-header {
      background: #eaf8f8 !important;
      color: #007f80 !important;
      border-bottom: 1px solid #cfeaea !important;
      font-weight: 750 !important;
    }

    /* Main finding: intentionally visible focal panel */
    .insight-card {
      background: #f0fbfa !important;
      border: 1px solid #a9d9d9 !important;
      border-left: 6px solid #00a6a6 !important;
      box-shadow: 0 7px 20px rgba(0,166,166,.10) !important;
    }

    .insight-card .card-header {
      background: #dff4f3 !important;
      color: #006f70 !important;
      border-bottom: 1px solid #bfe2e1 !important;
      font-weight: 800 !important;
    }

    .insight-card .card-body {
      background: transparent !important;
    }

    /* Coefficient table */
    .coef-table-card {
      background: #ffffff !important;
      border: 1px solid #cfdae4 !important;
      border-top: 4px solid #2a6f97 !important;
      box-shadow: 0 5px 16px rgba(42,111,151,.06) !important;
    }

    .coef-table-card .card-header {
      background: #edf4f8 !important;
      color: #243447 !important;
      border-bottom: 1px solid #d6e2ea !important;
      font-weight: 750 !important;
    }

    .coef-table-card table.dataTable thead th {
      background: #e4eff4 !important;
      color: #243447 !important;
      border-bottom: 2px solid #8fb9c8 !important;
    }

    /* Interpretation / conclusion */
    .interpretation-card {
      background: #fbfcfd !important;
      border: 1px solid #d5dde5 !important;
      border-top: 4px solid #7b8b99 !important;
      box-shadow: 0 4px 14px rgba(36,52,71,.055) !important;
    }

    .interpretation-card .card-header {
      background: #f0f3f6 !important;
      color: #334155 !important;
      border-bottom: 1px solid #dce3e9 !important;
      font-weight: 750 !important;
    }
  "))
)


# Academic Premium v6: restrained two-color KPI system + unified Tab 2.
finishing_css_v6 <- tags$head(
  tags$style(HTML("
    /* ==========================================================
       V6 COLOR SYSTEM
       Coral group = static/reference KPIs
       Teal group  = lambda/Ridge KPIs
       Plotly OLS/Ridge colors are untouched.
       Header and tab menu are untouched.
       ========================================================== */

    /* ---- TAB 1 KPI: only two visual families ---- */

    /* Coral family:
       Observasi, Max VIF, MSE OLS */
    .kpi-row > .kpi-card:nth-child(1),
    .kpi-row > .kpi-card:nth-child(2),
    .kpi-row > .kpi-card:nth-child(5) {
      background: #fff3ef !important;
      border: 1px solid #efd3ca !important;
      border-top: 4px solid #e76f51 !important;
    }

    .kpi-row > .kpi-card:nth-child(1) .kpi-number,
    .kpi-row > .kpi-card:nth-child(2) .kpi-number,
    .kpi-row > .kpi-card:nth-child(5) .kpi-number {
      color: #c9573d !important;
    }

    /* Teal family:
       Lambda Optimal, Lambda Terpilih, MSE Ridge */
    .kpi-row > .kpi-card:nth-child(3),
    .kpi-row > .kpi-card:nth-child(4),
    .kpi-row > .kpi-card:nth-child(6) {
      background: #eefafa !important;
      border: 1px solid #c9e8e8 !important;
      border-top: 4px solid #00a6a6 !important;
    }

    .kpi-row > .kpi-card:nth-child(3) .kpi-number,
    .kpi-row > .kpi-card:nth-child(4) .kpi-number,
    .kpi-row > .kpi-card:nth-child(6) .kpi-number {
      color: #008f8f !important;
    }

    /* Keep labels neutral and consistent */
    .kpi-row > .kpi-card .kpi-label {
      color: #667085 !important;
      font-weight: 600 !important;
    }

    .kpi-row > .kpi-card:hover {
      transform: translateY(-1px);
      box-shadow: 0 7px 18px rgba(36,52,71,.09) !important;
    }

    /* ---- TAB 2: same visual language as the VIF panel ---- */
    .tab2-vif-card {
      background: #f8fbfd !important;
      border: 1px solid #cddfe9 !important;
      border-top: 4px solid #2a6f97 !important;
      box-shadow: 0 5px 16px rgba(42,111,151,.07) !important;
    }

    .tab2-vif-card .card-header {
      background: #eaf3f8 !important;
      color: #245d7d !important;
      border-bottom: 1px solid #d3e3ec !important;
      font-weight: 750 !important;
    }

    .tab2-vif-card .card-body {
      background: transparent !important;
    }

    .tab2-vif-card table.dataTable thead th {
      background: #e4eff4 !important;
      color: #243447 !important;
      border-bottom: 2px solid #8fb9c8 !important;
      font-weight: 700 !important;
    }

    .tab2-vif-card table.dataTable tbody tr:hover {
      background: #edf5f8 !important;
    }

    /* Keep Tab 2 explanatory text in the same academic blue family */
    .tab2-vif-card .interpretation-text {
      color: #3f4d5d !important;
    }
  "))
)


# Academic Premium v7: explicit semantic selectors for KPI colors.
# This avoids dependence on layout_columns()/nth-child DOM structure.
finishing_css_v7 <- tags$head(
  tags$style(HTML("
    /* ==========================================================
       V7 KPI COLOR SYSTEM
       Coral: Observasi + Max VIF + MSE OLS
       Teal:  Lambda Optimal + Lambda Terpilih + MSE Ridge
       ========================================================== */

    .kpi-row > .kpi-card.kpi-observasi,
    .kpi-row > .kpi-card.kpi-vif,
    .kpi-row > .kpi-card.kpi-mse-ols {
      background: #fff3ef !important;
      border: 1px solid #efd3ca !important;
      border-top: 4px solid #e76f51 !important;
    }

    .kpi-row > .kpi-card.kpi-observasi .kpi-number,
    .kpi-row > .kpi-card.kpi-vif .kpi-number,
    .kpi-row > .kpi-card.kpi-mse-ols .kpi-number {
      color: #c9573d !important;
    }

    .kpi-row > .kpi-card.kpi-lambda-optimal,
    .kpi-row > .kpi-card.kpi-lambda-selected,
    .kpi-row > .kpi-card.kpi-mse-ridge {
      background: #eefafa !important;
      border: 1px solid #c9e8e8 !important;
      border-top: 4px solid #00a6a6 !important;
    }

    .kpi-row > .kpi-card.kpi-lambda-optimal .kpi-number,
    .kpi-row > .kpi-card.kpi-lambda-selected .kpi-number,
    .kpi-row > .kpi-card.kpi-mse-ridge .kpi-number {
      color: #008f8f !important;
    }

    .kpi-row > .kpi-card.kpi-observasi .kpi-label,
    .kpi-row > .kpi-card.kpi-vif .kpi-label,
    .kpi-row > .kpi-card.kpi-mse-ols .kpi-label,
    .kpi-row > .kpi-card.kpi-lambda-optimal .kpi-label,
    .kpi-row > .kpi-card.kpi-lambda-selected .kpi-label,
    .kpi-row > .kpi-card.kpi-mse-ridge .kpi-label {
      color: #667085 !important;
      font-weight: 600 !important;
    }

    /* Make the clickable Lambda Optimal card visually match the teal family */
    .kpi-row > .kpi-card.kpi-lambda-optimal:hover {
      background: #e6f7f7 !important;
    }

    /* Tab 2: every card uses the VIF blue treatment */
    .tab2-vif-card {
      background: #f8fbfd !important;
      border: 1px solid #cddfe9 !important;
      border-top: 4px solid #2a6f97 !important;
      box-shadow: 0 5px 16px rgba(42,111,151,.07) !important;
    }

    .tab2-vif-card .card-header {
      background: #eaf3f8 !important;
      color: #245d7d !important;
      border-bottom: 1px solid #d3e3ec !important;
      font-weight: 750 !important;
    }

    .tab2-vif-card .card-body {
      background: transparent !important;
    }
  "))
)


# Academic Premium v8: final CSS override after all previous style layers.
finishing_css_v8 <- tags$style(HTML("
  /* ==========================================================
     V8: explicit KPI classes + descendant selectors.
     bslib layout_columns() may insert wrapper elements.
     Coral: Observasi + Max VIF + MSE OLS
     Teal:  Lambda Optimal + Lambda Terpilih + MSE Ridge
     ========================================================== */

  .kpi-row .kpi-card.kpi-observasi,
  .kpi-row .kpi-card.kpi-vif,
  .kpi-row .kpi-card.kpi-mse-ols {
    background: #fff1ed !important;
    border: 1px solid #efd3ca !important;
    border-top: 5px solid #e76f51 !important;
  }

  .kpi-row .kpi-card.kpi-observasi .kpi-number,
  .kpi-row .kpi-card.kpi-vif .kpi-number,
  .kpi-row .kpi-card.kpi-mse-ols .kpi-number {
    color: #c9573d !important;
  }

  .kpi-row .kpi-card.kpi-lambda-optimal,
  .kpi-row .kpi-card.kpi-lambda-selected,
  .kpi-row .kpi-card.kpi-mse-ridge {
    background: #eaf9f8 !important;
    border: 1px solid #c9e8e8 !important;
    border-top: 5px solid #00a6a6 !important;
  }

  .kpi-row .kpi-card.kpi-lambda-optimal .kpi-number,
  .kpi-row .kpi-card.kpi-lambda-selected .kpi-number,
  .kpi-row .kpi-card.kpi-mse-ridge .kpi-number {
    color: #008f8f !important;
  }

  .kpi-row .kpi-card .kpi-label {
    color: #667085 !important;
    font-weight: 600 !important;
  }

  /* Tab 2: all cards use the same VIF-blue treatment */
  .tab2-vif-card {
    background: #f7fbfd !important;
    border: 1px solid #cddfe9 !important;
    border-top: 5px solid #2a6f97 !important;
    box-shadow: 0 5px 16px rgba(42,111,151,.07) !important;
  }

  .tab2-vif-card .card-header {
    background: #e7f1f7 !important;
    color: #245d7d !important;
    border-bottom: 1px solid #c7dce8 !important;
    font-weight: 750 !important;
  }

  .tab2-vif-card .card-body {
    background: transparent !important;
  }
"))


# Academic Premium v9: normalize Lambda Optimal typography and
# give Tab 2 KPI cards the same restrained KPI treatment as Tab 1,
# with teal accent only.
finishing_css_v9 <- tags$style(HTML("
  /* ==========================================================
     V9 TYPOGRAPHY
     Lambda Optimal uses the same label/value scale as other KPIs.
     ========================================================== */

  .kpi-row .kpi-card.kpi-lambda-optimal .lambda-optimal-label {
    font-size: .82rem !important;
    line-height: 1.15 !important;
    font-weight: 600 !important;
    color: #667085 !important;
    margin-bottom: 8px !important;
    white-space: nowrap !important;
  }

  .kpi-row .kpi-card.kpi-lambda-optimal .lambda-optimal-value {
    font-size: 1.02rem !important;
    line-height: 1.1 !important;
    font-weight: 600 !important;
    color: #008f8f !important;
    margin: 0 !important;
    white-space: nowrap !important;
  }

  /* Remove button typography inheritance that can make the
     Lambda Optimal card look different from the other KPIs. */
  .kpi-row .kpi-card.kpi-lambda-optimal {
    font-size: inherit !important;
    font-weight: inherit !important;
  }

  /* ==========================================================
     V9 TAB 2 KPI
     Same card shape as Tab 1 KPI, teal accent only.
     The whole card stays white/neutral; only the accent line
     and value are teal.
     ========================================================== */

  .tab2-kpi-card {
    width: 100%;
  }

  .tab2-kpi-card .value-box {
    min-height: 112px !important;
    height: 112px !important;
    background: #ffffff !important;
    border: 1px solid #dfe7ec !important;
    border-top: 4px solid #00a6a6 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(31,41,55,.04) !important;
  }

  .tab2-kpi-card .value-box .value-box-title,
  .tab2-kpi-card .value-box .value-box-title *,
  .tab2-kpi-card .value-box [class*='title'] {
    color: #667085 !important;
    font-weight: 600 !important;
  }

  .tab2-kpi-card .value-box .value-box-value,
  .tab2-kpi-card .value-box .value-box-value *,
  .tab2-kpi-card .value-box .value {
    color: #008f8f !important;
  }

  .tab2-kpi-card .value-box .shiny-text-output {
    color: #008f8f !important;
    font-weight: 600 !important;
  }

  .tab2-kpi-card .value-box:hover {
    box-shadow: 0 6px 14px rgba(31,41,55,.07) !important;
  }
"))

# ============================================================
# 4. UI
# ============================================================
brand_header <- div(
  class = "project-brand",
  tags$img(
    class = "brand-logo",
    src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAJQCAIAAADdV43mAAAACXBIWXMAAC4jAAAuIwF4pT92AAAgAElEQVR4nOy9f1Qc533/Oz92Z5cYLYImFSVAEuF7vxW7qpVT5CMMnFbpNciSI8fOQUhWU6sRTSQ5sg+cpJFVu8Z1KitxDpwmrZBdlMrt15IQJ04j25LA90bf7wGMTsS5lqNdlO85Rk6AEnT7LYiFhN3ZnZn7xyibNTvzzPPM88yv3ed18kcsLbsjdmbe8/n1/rCKojAUCoVCoRQqnNMHQKFQKBSKk1AhpFAoFEpBQ4WQQqFQKAUNFUIKhUKhFDRUCCkUCoVS0FAhpFAoFEpBQ4WQQqFQKAUNFUIKhUKhFDRUCCkUCoVS0FAhpFAoFEpBQ4WQQqFQKAUNFUIKxUmmpqenpqedPgoKpaChQkihOMPZc/07v/DoZ/9482f/ePOfbP2zk6+8ShWRQnEElm6foFDs5EY0evKVVy9euhyPx3P/tuGB+j27d29/aFtJSYn9x0ahFCZUCCkUO1hcXLx46fLJV16NxmIwr9/dtmvH9oe2P/SQ1QdGoVCoEFIo1gIOAcGEQqE9u9v27G7bGIlYcWwUCoWhQkihWMfZc/3wISCYSDisKiJNmVIoxKFCSKEQZmp6+uQrr549128iBDSEpkwpFOJQIaRQiHH2XP/Zc+dG3x2z+oOqKiv37Nm9Z3dbdVWV1Z9FoeQ9VAgpFFympqfPnus/e/bc9MyMzR/90LZtO7Y/tGd3m82fS6HkE1QIKRTzjI6+e+bcuXP95509DLWn5sBXv0IDRArFBFQIKRRk1FmIb3/nZfwQsLJUYRhmZoElcVxMwwP1B776FVpBpFCQoEJIoSBAsBGmtU7atTm9Zb3MMMzELDcwzl+O8kQUkVYQKRQkqBBSKFCMjr7b+8qrly5fxnyfcIW8vym9LSKvCWpceldvceev+QZjXHyFgCLubtv1+O7dDQ0P4L8VhZLHUCGkUAw4e64fPwsaKlJawnJ7U7q2QoZ5/cA4PxjjB6M8zoeqVFVWfvOvv0Ft2ygUPagQUijaqL2gJ195FTMLCg4BwSwl2PPX+IFxPjaL649PG2ooFD2oEFIoq1FN0TB7QVFDQDBqEfH8OI+fMn1o27aDX/0KzZdSKBmoEFIov+PipUsnX3kVcyK+slTpbE6ZCwENIZUyjYTDB776FTqASKEwVAgpFIbcOER2IygkaoQXKkK7DGcW2IFx/vw1H2aXaSgUOvDVrxz4yl/R8iGlkKFCSCloFhcXT776z5iFwMpSZdfmdHuThBQCDkb58+P8UIxnGKY5LG2LSC1hGVURh2L85Sg/MI4bIO5u2/XNv/4GLR9SChMqhJQCZWp6+tvfeRmzEFhfI7c3pZvDEvyPzCyw56/5Bsa1RwZb66SWsNQSQXhD5rc9NX3DuAEiLR9SChMqhJSCA98XTW2E6WxOqb4wkGSHgGDWBJVtEXl/UzqM2GgzFOPPj+NWEKk9DaXQoEJIKSBGR9/99ssv4/TCmGiEmVlg+4Z956/xSwnkcK2yVNnflN4WkZAUl0gFUZ0+pN00lEKACiGlIMAfiq+vkTubU0iNMINRvm/Yd/UW7ggg89siYmsdWsp0YJwfGPeNTZo/gKrKSrW5lHbTUPIYKoSUPAdTAk1kQcFVQBzMpUwnZrm+YR9OQw1tLqXkN1QIKfkJfjtoZanS3pTetRmhFxS+CohJbYXc3pRG6jJdSrB9w1j5UiqHlHyFCiEl38CXwPoaubUuDZ+HjK+w58f5U4hNmxnrNYZhLkc5c2PyrXVSa126vgYhQMTMl4ZCoe0PbaOzFpR8ggohJX/Al0DUifjYLHcKMesIsF4z7Syq9tTsqpPgA0R1zQVOvpSOHlLyBiqElHwAc02giUKg2paJ1AgDH2jOLLCnhn0m1hO21klIFUT1g3AsTKkcUvIAKoQUb4M5Fx8qUtqbEExh1EaYvmGEWYhQkbKrTtrflEYagVAxt54QtYKIP49P5ZDiaagQUrwKpgSqE4HwhcCxSW5gHC2XiFpr1GMpwV6OcqiFvTVBZddmqR1FgAfG+e4hP5VDSqFBhZDiPTAlEFWfULOg5nxnYDCXMm0OS+1NCA01V29x3UN+0900VA4pnoMKIcVL4Esg/FB8fIVVx+/gVQdnBy8SJqzU1AgYPl+K2U3z19/4Oh20oHgFKoQUb7C4uHj02edMS2BrnQS/I3dmge0e8l+OcvCFQBMLmPAxYaWGmi9VfxXm5JDOHVK8AhVCitvBHIporZPgs5Rjk1zfsA9+It7cAibimAgQkQYQqRxS8hsqhBT3giOBqIW6gXG+b9g3AT3AR6oRhiBqgNg37INvMd2yXt61GfZfYeL9M4RCoWPfepFaeFPcCRVCihvBlED4iQhUUxjrGmEIguodg1Q+VK3azMkh3WhBcSdUCCmu4+y5/qPPPmeDBPYNI0wEmrAedRbVaxt+BnFNUGlvktqb0lbLYSQcPvatF+n6X4p7oEJIcRGmN0UgSSBqxcuFWVB4UIfl1QUXkCEvjhw2PFD/9996cWMkgvqDFApxqBBSXMHo6LtPHn7KaglEHYpH6jV1OUMxvm8YIV8K32SEI4d06JDiBqgQUhzG9NZ4VAnsHvJDDsWr79xah7YX3hOg7ibcsl7ubE7BNJfiyCEdOqQ4CxVCimNMTU8f/ZvnLl2+jPqD1kmg2jZiw0S8s6D2fyLJYfeQr2/Yh3pIoVDom9/4+oGvfgX1BykUfKgQUhzA9HQ8kgQiOWd6uhBoDtXCFP5XBC+HpucOqyor/+n736N9NBSboUJIsRXTcxHWSWBLRGpvstsUxlUgjVvYIIe0j4ZiM1QIKfZhuim0szllhQQimc7kPUhe2/C7OyZmua4LZiy8d7ftOvatF2nhkGIDVAgpdjA6+u7RZ5+LxmKoPwivVfASmMe9MPgghXHwcmhuowUtHFLsgQohxVqmpqe/dvgpE02h1kmg49ag7gdJDuGTpUMx/vkfI+87pIVDitVQIaRYxeLi4rdf/u4rr/4z6g/W18hdO1Mw03vwElgg7aBkQZqIgJfDgXG+64Ifdcqi4YH6f/z+9+jEIcUKqBBSLOHkK69+++XvonbEwO8LRJXAgmoHJYsVcmh66JBOHFKsgAohhTDmPGIqS5UXHkk1h43likqgI1gkh8//GLmttKqy8tjfv7j9oYeQfopCAUCFkEIMcwPyoSKlayeUXMFLYH2N3N6UhpFVChJIcghZ5Z1ZYDv7BdQ+GpoppRCECiGFAOp04Hde/i7ST8G3rgxG+a4LsBIImVylmMYKObx6i+s4J6D20dBMKYUIVAgpuJibDmytk154JGUogfAGaVQCbQZJDjseTMMseDo17Ot+B61wSDOlFHyoEFLMcyMa/Ztnn0MdjaivkbvbRMMQgUqgJ4CXQ3XfYWdzyvANTbiV0kwpBQcqhBQzmBuNqCxVenaLhooFP8RGJdAlwMshZAeTucLhX3/j69/8xteRfoRCYagQUkxgYoM8ZEcMlUBPAy+HtRVy107jtlIThUM6fU8xARVCCgLmcqEwTqHxFbZv2Nc3zC8lyIQUFKeAf5rZsl7u2W2cJO95x4c6cUh9SilIUCGkQGGuLxSyHNg37Ose8lEJzCfg5bC1TuramQL30ZhYZBEKhY5968U9u9vgf4RSsFAhpBhz8dKlo3/zHFJfKGQ5EHIugkqgR4FcPQHZR3P1Ftf1Y39sFqFwSJtoKDBQIaSAMDEjHypSOh9M729Kg18G2RSqzhp2PGjwbhQ3A7l6orJU6dqZaokYPO6gjliEQqEDX/0KbaKhAKBCSNHFhF8ozHQgZJqLborIMyCNgbasl7seSYWBlutLCbaj3z8YRciURsLhf/z+P9BlvxRNqBBSNLgRjX7t8NNI6wPDFXLXIwZtnPAdMZDj9hTPAdn5sr8p3fmgwQC+iZ5S6kRD0YQKIeUjmBgQhMyFDozzz//YbyiBLRGpayfdGp/PQI7Mrwkqnc3pdqPzCrWnlM5XUHKhQkj5HSYWR8CEbmOTXNcF/4RRjwMdDSwoZhbYrgvG6U2YiUMT0/df/cpfffMbX6ehIUWFCiGFYRhmcXHxycNPIzXFwORCIcuB8DuYKHkGZCMozIjFUIzv6EfY90tDQ0oGKoQU5uKlS08efhq+KQYyF9o95DcsB0K+FSW/gdlZD5MpNeFTSkNDCkOFsMAxEQjC1PAgpwNhHGcoBYJqz9Y95Ae/DCZTitpEQ0NDChXCwgV1OgJmRn5mge04JxhOB9KOGIomkIVDmExpzzs+Q1nNhoaGhQwVwkJkanr6a4efQrIMNYze1NGInncMslIwlUVKgQMT0sFkSiF9bTLQ0LBgoUJYcKAGguEKubstVQsccIbJhUIuoKBQVGDmImCm71GdaGhoWIBQISwgUANBmE4WyFwoLQdSTLCUYJ//sXHXcceD6fYm0PQ96nwFDQ0LDSqEhQJqIAizOAKmLxRyAQWFogfMiEVlqdLdJoKbaFBDQ7rmt3CgQpj/LC4ufumJfUiBYE+bwVTf2CTX2W9QxaHTgRSCwMhYc1jqaQM10aCalFKH0gKBCmGegzoj2BKRetpATjHxFbbrgkG2iq6MoFgBTE/pmqDywiMGpWjU0fu/f/HvDnz1KwgHSvEaVAjzFtQZQZgADsYvlI5GUCwFpqd0y3q5ZzcoIY8aGtK9hvkNFcL8ZHT03T9/Yh98INjelO5sTgMCQZimGMhlvBQKPoZjgjDzFUihYSgU+qfv/8P2hx5CO1CKF6BCmG+gro+AUa/uIb/hgGBnc4rmQil2AtMLWlshd7eB5iuWEuz+0wgNpbvbdh371ot0uCLPoEKYV9yIRr/0F/vg10cYBoKxWa6z32BxBO0LpTgIjE9px4PpzuYU4AVIDaV0uCL/oEKYP3z75e9+5+XvQr7YMBCMr7Dd7/hOAf2LYfpLKRSrgSn4GYaGqLOGdLgin6BCmA+gTsobLhGEmY4wjCYpFDuBaaKBCQ27LsA6lNIOmryBCqHnQRqQMIzhYAJB6hdKcScwa5gMQ8OJWa6z33hFogrtoMkPqBB6mMXFxaPPPneu/zzk6w1nBGECQdoUQ3E5ME40hqEh0vIK2kHjdagQehWkvhiYQLCj3z8UA1VZaFMMxUMYKplhaIi015B60HgaKoSeBKkvxlDAxia5/acFwJg83SNP8SIwSU5waIg0dx8Khb75ja9TDxovQoXQY6Aah3btTAEEDKYiaJhQpVDcjGFoaGhDgzR3/9C2bf/0/X+gaVJvQYXQSyD5xRjuETSsCNLpCEp+YBgaGtrQIA1XVFVW/tu/nqZpUg9BhdAzIKVD25vSz+/UTfjQQJBSgMCEhqf2iYDlFUgdNNSq20NQIfQAU9PTX/qLfdFYDObFoSLl1D7QpHxslms/TQNBSiECExr2tKVaIronP1IHDU2TegUqhG4HaUzQMIwzdA2lgSAl7zEM7FrrpK6dunsNkTpoaJrUE1AhdDVHn30O3j4b3Bdj6BpKA0FK4WAYGlaWKn37RMBwBZI9KU2TuhwqhC5lcXFx5xceg0yHGvbF9A37uod8gAEJGghSCg0YGxrwcAWSBw0dunczVAjdCFJ3KNg41HBSngaClELGsOYHHq5YSrDP/9g/MA6VJqVD966FCqHrgO8ODRUpXTtTrXW6GjYY5Tv6QQvl62vkU/tEGghSChnDmp9hBw3MKiiVUCh07Fsv7tndZvJYKdZAhdBFIA3Lhyvkvn2gKeCuC37AgAQ1i6FQsjGcmgd30CClSb/6lb869q0XTR4oxQKoELoFJO9Q8JigYV+MoYhSKAWI4dQ82J4UNU164d/foCVDl0CF0BWcPdf/taeehnmlYUnPsC+Gro+gUAAYDlc8vzMF8KBBSpP+99dO0033boAKocMgrVICR3KGfTGVpcqpfSKguZRCoTAMMzHL7QeaTjSHpZ42MmlSOlnhBqgQOgmSZUxrndTdJur9raFfjOFWegqFksEwzwkeNERKk9LJCsehQugY8DMSht2hfcO+Fy7oJnPogASFYg7DDhrDNGlnvwDzQZFw+N/+9XR1VZWZo6RgQ4XQGU6+8urfPPe3MK8ED8vHV9j9p4Wrt3STMHSbLoWCw8wC235aAOQ5DdOk4CxrBloydBAqhHaDVBQEG74YLtSlfTGuQrnDKws89xnd/DbFtYA7aAzTpPDepLRk6AhUCG0FqSgI9g4F22dXlio9u0E7KCj2I38oJP+lzFf/G/9DUJ5BFFdx9Ra3/7RgOk0Kv8KJlgzthwqhfSAVBQGrlAy7Q6lxqDtR7vCJ7k8wDMOVp4Uvz7NB+pjiMZYS7P7ToEFDcJoUftM9LRnaDBVCm4CfFAxXyAMHdW3PDLtDwXEkxVlW/rZc/T/sWkl4/A5XruuKQHEt4NgOPHRvWHHMEAqFLvz7G9SY1B6oENrBk4efgiwKgmckwN2hdEzQ5chz/uSJ38v8JxtUAof+N7uWdvN6D3CadE1QeeER3TZvpMmKf/zeP1BjUhugQmgtSNuUuttEvYsnvsJ2XQBdPDQd6n7UGmH2n9AcqXcxTJOCH2pPDfu69B9qs9ndtuufvv89M4dIgYYKoYXA24eGipSBA7rBnKF3KE2HeoJcIWQYhv/DpPD4giPHQ8HHME16St8KyrD7JkPDA/X/9tpp2j5jHVQIreLipUtPHn4apjUGXBQEr1Ki6VAPId0MimfX5v658Ogi/9kV+4+HQgTDNOmpfWJ9DW7JkO4ytBQoNzwKKidfefVLT/wljAq21kmXO5J6Kth1wd/+mu6kYEtEGupMUhX0CvKc9rhL6lJISdAr0atsWS8PdiQBQ4S7Tgb6dBaiVZYqAwd1CyLZRGOxnV94bHT0XaxjpejAd3V1OX0M+caTh5/63vf/EeaV3W1iZ7N2SjO+wv55X+DN93WLgp3NqZceSwV0JwkprkN6r0iZ00qjpVkmzfL/R9L2I6KQIVTE/Hm9tJRg/98p7Qea//m/+JkFrr5GDuR8/wEf0xKRSoqY//G/DNpnksnk2f7+6qoqGhcSh6ZGSQLfGmNYFATMSICnDCmuJfmDMvkXus6TwaP/H+2a8TrgHUzgkiH8lCFtnyEOTcgQ40Y0+idb/wxGBcMV8mCHbkpzYJxv7dVVwXCFfPVokqqgFwGoIMMw6bGP2XYkFItorZMGDuh6rU3Mcs3dAb1G0+Yw6GezOdd/fucXHl1cXMQ6VkoWVAjJMDr67s4vPAbTINoSkQYO6j4Vdl3wd/brFgXBBcU8gC2u5ssb2eJqpw+EPPKHBlsIpPeK7DkSZ8njr1iltkIeOCi2RHSHCAElQ/Vn9Tprshl9d2znFx6bmp7GOlbKb6GpUQLAu8a0N6Wf36ltJgLeI2G4iSkPKPpilC3+iKeUPB9Vfv1LaW5EnhuW5284dWBESF0pTl8pBr8mcOi/8sxrhivbyJU38eWN7D2f4so+UtlSlqdXfpjPtS7wZEVrndS1U9eMrbNfgJm4p+4zpKBCiMvRZ5975dV/hnklYF4ePClYIDMSfHljoOVtvb9VlqfTk6+nP3hdWZ6y86hIkTzxcb2u0Qx548fNFlf77t3rq9m76skmm+TgDmluxM6jsh/wZAW4ZAi5yzAUCh371ovUfQYTKoTmgV+oBG6NAU8K1tfIp/bpThnmGcL9x30bDoJfk548k7r+krfkMGO3DYb7tBj48rwNx2MdrFDiv/+4r+Zx8MvSN3vFnx6x55CcBTwpuCaoDBzUrQvCT9zT5U2YUCE0CXyDaLhC7tN/7gPbhwJSqXkJK5QEPz8KCCNUFDGeev+l9MQJe44Kn/TYPalLa2BeWfR3c1YfjHX4ag/573uGFULglynL04k3GxSxUHo9DM1FAbmiiVmus98PM3FPW0lxoEJohhvR6NcOPw2jguB4DlAJKISioCbgBGk28u2R5E8e98T9FCYvqhL4y3kvbu5lhRKhsZev2gHz4kJIiuYCNhcFGJMamppmoE5spqFCiMyNaHTnFx6DdI3RO7njK2zrSUGvKAhOpeY9QmOvYW5NRZ6PiqMHXN5Hs2rpBBhhzx1+Q8LS4yEOK5QEWi6u6oXRIz15RhwxyH7nK+BJwS3r5VP7RMz2mUg4fOHf36BaiAodn0Dj7Ll+SBXsbhP1VDA2y205FtBTQXVSsGBVkGGY1E+PKCJUzwhXFgm0XOTKNlp9SDik30UYEIQMHN0DkgoqYjxVGKVBTcCTgldvca0ndauJ3W1iF0SVJBqLbfrjzTeiUawDLTyoECKgjkkYqmCoSAEk/QejfGtv4U4KwqCIi+mbsPU/VggFtp5lBZc+Ait3eOl63g4IIqkgwzDpmyc8kcq2DvCU4cQs19qrmwXd35QGhIwZ4vE4dSVFhQohLEeffQ5mWFDNauqpYN+wD2Ci3bUzBVhgVlAgNcKwxVWBlovWHQwOqZ8YzA6uQvkV1I46lyA09sKrIIP4teYra4JK3xNiu87eNHXiXi8LqsaUUFr46GNnz/XjHmvBQE23oXjy8FOv/eu/Gb5MXahU8/u6Wf4TV7QTX6Ei5cSfF2JrjC5Skvu9P+JK/k/Il7NFv8+wrOyyFgzlDp/6EVqoyhbLPo+sZPLVHvIbzbpkI02/nf7gjHXH4y3+9L/JVWXKYExb8AZjfDzB/ul/00iifmKN8sgm6eok959LBmMVFy9dLgmF6ur+mMDh5js0IjRgcXFx5xcehRkWVFVQc0wivsK29uo+5VWWKgMHxOYwVcGPgNpY6L/viNuKhajhoIdgi6v99z2D9CMF2CkKprVOGuxI6oV3p4Z9ekOE6vImGCe2v3nub588/BTugRYAVAhBqMOCo++OGb4SUNubWWBbT+p6p4UrZLpTUBNl/meoPyLcf9yKIzGHPOfP4+pgoLHXcF5wFSa+0LyntgK0y3Aoxree1NbCNUHl/IEkTA7pXP/5Jw8/RR26wVAh1EUdk4AZFmxvSgMaRJu7dRtEaWsMABNDEdy6Rr76YSsOxgSpi1AT9F6EL2/k1jWi/pTLp1ycQg3vAO0zW44FAK2knc3GraTn+s/v/MJjVAsBUCHUBl4Fu9tEPfMXdaESbY0xh7n2QtR8nUWkx+4BL13SwxPT9P5NZn7JBd4vCsCwfaa1VxiMahdWOh7UfQrPRl1wT7VQDyqEGkCOzIPHJFTPXE0VVH9wv855T1ExNxHBlUX4cuRghSxKgjNcNEEE6b2i5A/KbPigbNjiahPhIGP2Cy0cntd/Ml5KsO2v6Q7Ut9ZJAwd1a40Z6IghACqEq4EcmQePSXT2C3rO8eAfpGQw3fnC37uX7JGgknqjRNFJAxjClUM9Hkk3g4nuT4iILalE8NceMveDbmtlciFgSevsF/RM2rasl2HHKr7wGNXCXKgQfgT4kXk9C7T4CgswQwLvpqd8BGGtuZ/jq5wsE6bH7pF+HjD942ypwROSkuDEM6Xi2bXKHWPDLSvga8w+Z5j9QgsKVdL0PPpPDfv0nrDBfTcZVC28eOkS7oHmF1QIfwfkfl2ABZrqIApQQcBuesoqTAcQrBByqmVGucNjJkXBi3nlOX+y+xPZQmtzTZEr24jaLJr9s2QPJl+prZCHOnUlbWCcb+kJAMYqYLTwS0/8JR23z4YK4V0gjWNUMQOMSdAGUVLglPr4aqg1CMQRz5SaTooyDMN9GqRq8pxf/EEZzvvj48NIOzteu/UQ6p5CwG6m1pPCzIL2WAXgB7P52lNPUy3MQIWQYRjmycNPwWyZb4lIeioIHpMAzFdQNGGFEnMdGSr8uiaCBwOJ+EYJpmU2ILzTU0FlwdYEKU5Ux61rpP0y8KwJgnrxJma55m7tsQrwD2bztaee/vbL38U90LyACiHz5OGnYIxjWuukvid0VRAwJgGYr6DogdnwwhZX2XzPld4rwh+f5zckNf9cucPrxYI2Vwpxnk4YF/QxeQ7AEht1rELPoRtyxPA7L3+XWs8wBS6Ei4uL8CqodzoOjPPbegJ6YxKn9tEGUTP4N5hsTcxgZ0VKnvOnLpmsnGVg10p6BUJAxtVOIcT/leJ/rQWIevPR7AgFO3RDjhiq1jO4R+lxClcIVfs0GBXsbNad71GHBTX/Su0spQ6iJvBveoYtrsJ8E7bsj4gcjCFKghPPrMUv3ekVCNNj9wAyrsodXknYdBWzxZ/Cfocqc8P4BU5rHWjpBKBNHfAEnw21YStQIVRVENI4puNB7dGurgt+PRUMV8iFvGIeB7a42kcibrAtNSr+oIxIWOZ74De5fwgzmy/fND+tgQSRINu34RBbXI3/PoVGbQVorAIwuAy29s5Q4DZshSiESCoIGJk/Naz9nK52llIVNIcJN2dN7OlRxG+QUdHLi6bHPmYYa0ofmvFyMwERAWOFUKCxF/99ChDDsQrAiCHMuH0h27AVnBBCqiDYPg2Qi6iv0Z2voBji3/QMZjuGnaSuFJPaL8Hr7CCU3jN+f+l6kT3ZUY5QJMeta6QJUnOo0xF6C5gGxnm9zU1UC8EUlhBOTU9DqqCeC1p8hW3p0a1Ot9ZJ5w/QYUGT8NUP++87QurdrBZU6b0igoaimst4pZtByKQrjF7iw96DWyPM4L/viHv2hHgL8AImwOam2gr56lFj65nC1MICEsIb0eifbP0zSBUEGMcARubpsKBpuLKNQoNnMmbyhwJBn0/u0yK7VuO+Br+/Ij32MVIHAwC/gykboaGXes2YBjxiCNhiCGM9U4D23IUihPALJcypIN2phAMrlARaLhIpDdqAPOcXz5YSfENfvUabDMMw8q9gq4/KHT49dg+5I7IDVggFWi7SEXvTAEYMJ2a5lh7dcXtIG7aCsucuCCHEV8HYLNfSo2scQ3cq4eAtFVQSHFmfM3atxG9IaP4V0kbD9JVi2+YoSEG1EBNAFmpmgW3tFQBaqLcKOENBaaHHrhwTQKogYC+Eahyj6ezHAHMUFENUFeTKIpa8Oek2feIqyOiHg6goCTZ1cQ2Rt7ITrixCtRCH1jrp1D7dcXuAFvY9YXzjKhwtzFdthJUAACAASURBVHMhhFdBvb0QAPu0UJEycFC3ak0xxFIVZCwYJRR/UEZkWCIDG1T0+kVNIF0vki0bpbBuHIVqISbNYd1xe1UL9Zr7YB7iC0QL81kIkVQQ1URUzaNuWU+HBc0T+NwZ61SQYRh5/gbBdyM1MpgN/9kVNkjyFBLPllqUICX7y1wFVxYJfO6Mde+f9wCmI5YSoA2pVAtV8lYI8VVwMMqDVZCOzOMgNPZ6aWTwUojUyGA2vvpfk31DJcGKZyzZf6uI1vbTc+saBTpoj4GqhXpdMES0cHT0XdyjdCv5KYT4Kjgwzre/RlXQKoTGXl/N404fBSzSe0VWzCfwm1Y0pyYysKYGUuVfCClyA4524qt5nGohDrUVMqAjFKyFhk3v8Xh856OP5esKwzwUQiIqCDAR1VtPT4HEWypIdmQwG01z0WxY4LZ6AOkrxfaM2BOHaiEm4OmIzn6he8iv+VeQY9D5us4334TQahWk9mmYCPcft0cF5XkCJQ3iI4MZuE+LekuXfvca/T29hqQuheQ57VueaRTR4LIigq/mceH+4zZ8UL4C1sKed3wAe+6C1cK8EkKqgi7Hd+9e34aDNn1Y6g7mGygJLvVGCdlhiQz+rcuGr+F1FjPBoCRY8QdlZBtnlIWfEXw3AL4NB310hS8Ga4LK5Q7dhnbAXa5gtTB/hJCqoMvhqx8WGk7Y9nH4XY7imbXE20RVuPI0TLSHExEyFmih1f0y2QgNJ6gZKSaALhiqhavIEyGkKuhy7LcSxbxrp64UIxm7IAHfLMr/YRLng+Q5X+oNYgVOSycocqFmpPhQLYQkH4QQUgVbIpKennUP6a7Yba2TLnfQhRJYsEJJYOtZm03U5Llh8z/7oUBws8Qq2LUS/BA9ZlDIMIz084BISAttFkJqwEYEqoUweF4IIVWwtU7qe0JbBTv7hZ53tDNgdKEEEQKfO0N2cQEM8vKUuR9UEpxFDTIqSFYyejakSEjXi4g0kSrLv8R/EyRYIUQH7fGhWmiIt4UQXgX1vlHAbA1VQSI4smtXEeOKWSG0rkFGBclclF0rgWcNIRF/VCLdDGK+ic0RoQrd4ksEwKTgwDjf0hPQXNsEMDLNJg+00MNCCF8XfOER7T51qoJWw5c3Ety1C4/p/kbpZlD6eYDswWTDb0L2VOMwekezSf2oBH+gQr49QuRgkPDfd8Q6p9PCAXBPA6wwBBiZZuN1LfSqEOJ3x1AVtBpWKBEaTjry0dKcmfu1kuBS1szOZzAcos+F34DVL5NBSbCpN0owm0gdCQoZhhEaTtJiIT7mtBBgZJqNp7XQk0JIVdAT+Dc9Y39pUMVcp0z6SrGlSVF2rWQ4RJ8Lfr9MBvwmUqeEkC2uoglSIlAt1MR7QkhV0BPw5Y32zc7nYOJ+rdzhrTAUzcbc6kE2KHPlxNY+Sz8P4Oyyl+dtmqnPxbfhIE2QEsFqLfTingqPCSFVQa/g3/xtpz5aWZ42MUSY+onlRtWmW0AJBoUMw6QurTFdLJTnb9hjtKaJgydVnmGpFnpxZ5OXhHBqepqqICSd/dqnsj347t1r6aJBMNJt5Lyocoe3YstSNlx52nT/J6l+mQw4CVLbjNZy4coizlqvzSw4dk0Rxzot9OL+Qs8I4eLi4pf+Yh9VQRi6LvgHxvmOfsKey5CwQon/PifLOSbyojjZQkhwojqyESGjFgvNOgaYa0QihX/zcae6ZgbG+fpjwbFJz9wzDTGthT1tBqVuz2mhN77UxcXFnV94LBqLgV9GVZBhmMEof2rYxzDMUIzvG7bEKhOMr/aQUz0yKgp6HcuGpUU4UR0blIlME2Yjjd2j3NG+IsA41S+jwgohX+0h+z93ZoF9/sd+hmH2n3Yy10Ic0zMVUPsLvaOFHhBCfBXsHvIXiArOLLDZgWD3kM/+i9a3wYH7VDaod2r5Q8HSZlEV7g9MLhdUIS6ESoI1Vxa1319mFb4Nh+wPCjvO3V3TvZRgW09aZULrCOa0EObO6SEtdLsQ4qvgwDhfOA5q+0/fvVxVlhJs1wVbE6S+e/fa7CmaC2qnjPRzXMsVGDCVjHh2lGEY6XqRiaDQ2YiQYRhWCNm8mGJgnL9663e3yolZzubLymqoFrpaCCFVsLJUMbFTIv9UsOuCf2J29Rc6MM7bWeF3tjrImLI+kX9leQKZdatpu7mgkMjSYxxsPs1yt7qfGvblU7GQsV4LFxft2+FlAld/l0effc5QBUNFyql9VAV/VxrMJfcytgjfvXudrQ6aw7p1SwTBWdILQLpeZMZrBnvpMSZscZVt7aN6j5J5VixkjLRQr/kuP7TQvUL45OGnzvWfB78mVKQMHBBrKzTMGwtKBeMrLKBH9HKUs+eKdTwcZNB7GvHtN2GwoQZpGhONQqY3exDEtpNN7zlyKcHuP+2BRygkAPfGoRjWHTUai7lZC10qhFQFkVhVGlzFUoIdjFn+RfPVD3sxHFQWzHROmvkgPJNPgovmV2FCCE1v9iAIW1xlQ6VwbJIDVBau3uIcacy2FMAdEvO+6mYtdKMQUhVEom/Yl13J13uN1Yfhr3XMUA0Hec6mG5n8IVb0YN1xynM+61TWUmw45QbGDX7tL1zwx3IK816nALXQdV/hyVdetUgFwxVy/qlgbJbrHjK+RU7Mgh5s8WGLq+1fOqgJasuobRGhdBNru5PyKwtTuJgi7RTcuka2uNq694+vsJejxnfIToecKyzFtBa2Nxn44kZjsaPPPod7fKRxlxCePdf/N8/9reHLunamNFUwNsupQ6+5qPMVuMfnPjr7/YCkaDaXoxbe8f1OzDhrgjpNb26o3ATmZhVUlARn6ZZE1HDT3HIPK7D0xBuMcTAXV/5NU6iY08Lnd6Za6wwmhc71n3/y8FO4x0cUFwnh2XP9X3vqacOXdbeJmr/o2CzX2qtdKgNMGXqa7iGNeQk99CwFiMDXOOkA6RVM+3qnzdqhQWJpuGkplp548M+O+TdNodJaJ3U2axtBALRQ7xadjdu00C1f3sVLl6gKIhGb5fSMAjSxLjvKVz/s+BC9J5CuF5noTJHeK7J6P5Sbm1rBWDpcj6RtztrcW0fHg2k9VRsY1zVxhNRC9+RIXSGEN6LRJw9TFUSjSycJDMCih1a+eocVb2sOtuyPkF5v8xCh+KMSJC1Mj90j/shyOzHknK2w1poDMYNFp99glIcsOqjMLLDdKA+mHgKgai9c0HWvhNHCV179Z5cs8nVeCCFXDLbWSVQFM8B0iuZiUZmQr7LV7wqMU6sJ4BF/VJK6UmzYqCndDCZ/UJa6tMaGQ0IVQq5so0VHYgKLTr8x9Ovr1LAv/zpIVQCqBthn8MIjqbBWM0c2Lllq7/DXBq+CmmXb+Aqr1y0SKlK621J5qYIzCyxMp2guVkSEXs+LEjezhiF9pTjZ/YnUpZD8oZBRRPlDQf5QSI/dI75Rkji2Tjy71hOWN45jUXbU3MWSlx2kKia0cE1QGTgoekILnYzlFxcXv3b4aRwVbD0paHaLAOYr8oDnfwzbKbqKpQQbm+UMz0sk+HJXTE1kQO2nZ9dKtjWOZqMk2PTYx6wu/hUIfHmjNPUWwTeMr7DwbWjZTMxy3UN+vQYTr9PdJjKMtuZ19guhoNgSWa2Uqha29grgWPnos89FIuGNEce2eTsWEcKvlaAqmM3YJDcUM3/jJh4UcuuayL4hJpyVg2UUFbc9/RA/CXEuk75hW23ubQaQ7ezo1/YWgIkLHV9S4ZgQwi9X0vyrwlRBhmH0WpYhISuErFDClTn2EKcJW4rWLOPavRB2gro0mL3nUxYdiTm4sgjZ2rCJAmGGpQSrN82cBwBUbSnB6kV+a4JKd1sqVAS61pzVQmeE8MnDT2Gum9dLXPS0ac/a5wfdQ37Mh02yQuiqpgkVVgghZUdZvH25hYkLTWXJnoqYl8lQjM/LsUIVc1pYWyEPHBANtfBrh592xIDNgW8L0kq0T2e5EqBJqbtNbA470PtgD/EVtm8Yt5q1lGAJ5m24cnflRVV4lKPiyg0coSircFteVIXsqWiuQJgNZubG5YC1sF1nQRWMFjplRmq3EH775e9CWolWliKroOHYiqfpumCyR2YVsf8g9qW7MCJkGIZDuVNzNCJkGO4PEJ4G3Pn0Q/BUJBLMzSyw+beYIhtAtnNmgdVb5FtbIfe0GVxx0VjsS0/sI3KQ8NgqhGfP9X/n5e+CXwM21NZTwS4IgztPM7PAkvJIIzjq5LZakQrSYBm7VnJkgsJdBBGqCa4aG81A8FQkdYF0D/ny0msmAyDCAyy1bw4bL6kYfXfMZgM2+4QQ0kpUr8gHtjzfb2R57nU6zhHLtBCsXritU0YFdbAMtVUk/+ChfwNscbU7v3SCR4WfF1VZSuSt10wGsBbiLLW32YzUJiG8EY3C2MrpFfkKbcXgKsYmORM+MnrEZsk8pVq6AQcTpE11/IakdUfiCdhS2JjYPWtGciF1Qk7PEwvjTg378niUQgWQ7QQvtYcxI7Vt0N4OIYS0j9FLbwKWKxWCCjIM0z1Esht7KcESydi4eWKPW9cI39PBb0gU8hAFfHKYFUrcvGaE1AlJ8KGTIX3xuhNAthNzSYVtpjOWC+Hi4uKX/mIfjH2MZnoTbCX6wiP53+lANhxUIRIUssVuLBBm8G96Bv7FfP2vrTsSlwOfGfbVHnKznR6RE5K4WejAeD7P12cALy8ELKnINaNZxdeeetqG4UJrhVC1j5memQG/TO+XOLPAFqCh9iqseKIkcrW7OTXKMAy3rtEHncfzfXbF0oNxM5CZYba42rfBvXlRhtAJOUMuL5qhEIJCBpjtBCyp6GkzNua2YdDeWiH80hP7cEzU9p/WVsHKUqVAVNCKcJBhmEJ4RGUYxn/fM5A3R3at5Kv/jdXH40LYoMJvSMC8MtDY6+ZwkBRWrI8YGOfzdSvFKsDG3INa228gDdi+9Bf7LB0utPDrefLwU6PvjoFfo2eiBrYSPaUza59/WPQsSXCU0M2wQiiw9Syk+ZZv63IBVgp5uFBYuP84t86Nc/TEsegZ8VRezxRmA8h2AsxI+/YZDNpPz8xYOmhv1Q3x6LPPQQ3O6wR2+08XqJVoNhaFgwyhq92d0/Sr4MoigZaLMFrIBmX/ow54OzmLD6I46rt3r28DQheuUxA5IQm2jGZTIJVCFb1sJ8CArbJUgTGdsW7Q3pL77Nlz/a+8+s/g16iSpmeipicA+W0lugrrnCkgr0lWKOHLG/Wyi+5fgatyVwshcqT8hgT/hwU0SsFvWjHsF/XVHhIaTthzPJjonZBscTVf3gh5upIaLsqlQCqFjJEBW2e/X8905tQ+xwbtyQvhxUuXYAbn9QK7Lv2yan5bia5iZoHFWbdkCEzRgr93b6Dl7aIv3vjYE4sfe2IxuO3twOfO+GoPeSIWzIYriwQ/PwozZe9/bLFA3EfZoOLfvgR6gVAiNPYKm1+y7ZCIwJVt9NUeCnzuTHDb2+p5W/TFG4GWt/l7oQY/iLgYalJQQSHAgA1gOrNlvXa/SDbn+s9/28iezASEY44b0eiTh41VsLtN10RNL5me9yZqq7D6+TEOURta9RCtVon4qh0MwyjL09Ycl1WwQiiw9XX59khy5KCyPKX7sqAsPL6QPPFxxbIbokvwP7rI6jur8eWNQsNJF26ZAMDe86miL0b1jhkmIrS6paVv2Ne1M/8nvlRU0xlNzVNNZzTjv9Y6Kb6S6roAuvt95+XvVldV7dndRvBoSX7xU9PTMIPzep1FBW6ilk18hb0ctfaahLnmAelEtrjKW3dJFW5dY9EXbwQ+dwYQ1LJrJeHL8/ndOMNvWtFrFuXLG4Pb3g60vO257xd8TsLkxmGeDnE4f43Pb/fRVZgzndnflIYZtCc7UEHsbgs/OE/tYww5P85bl6JRgbkg3ewdYxpleVqaeluevwF4DVeeEr48n685Uq48rZcU5csb/Zu/nZcNojAns9UR4VKCHYwVRMN2BrDpDM6gPdnhQmLfCszGeT1Jo/Yxq7Ch2bpwyhUZlOVpcfTQyg8j6Q9eN3zxXS3MOz9urjwtfHleLykqzY0k3mxIDu6Qb4/YfGBuwIZwrXBaZjK01kmdzdr3cJxBe7JbfMkIIeTGeU1Ji6+w7TqD84VjH5PN2CRng0pZ1CbuThQxDi+BGdigHPjyvG/rsnUHZjPcp0WACmaQ5kYSl3cUoBzGoawFsJhZYPN4eb0eHQ/qZjs7+7UHKtTWU82ttBnULb5EjpDAVwIzMqgnaergvOZ9H7CkPr8ZGC+U2Vt7SL1/PIEogdn4ty4HDv1XHoSGvvrfBCBUMIMqh+LoIUU0qHfkDfYYTRTmBd7dJtbXaJ97esOFa4LKKaNB+2gsRmSgAveLhxwZ7G5LmRicBz8O5CU2tMmowIzqgwtp7keejybebExdf0kRsfInXHkq8OV54dFFj67wZddKgb+c9z9kRs/SH7ye+GEkfbOX+FHZjHtO5stRrqBaZjKc2qc7XNh+Wnej/cABOwYqsO65o6Pv4owMAgbnT+0rFPuYVQzGOKvbZODB1A9nSb1/PPFmA8HbH//ZlWDnf3pLDtmg4lMj2s+Yj2gVcVH86ZHk4A7PzcxkA3My25AaZQqyZUYFYKU2s8DqDRfW6phRZ/Odl7+Lua3J/PdxIxr9cwjDG72Rwe4h0OD8lvWFqIIMw1zW8qW1CMPHUsC8nZtRxHhycEfqOvIkePeQv+obRVXfKNp/WtsgmMnI4Z47LveguSuBnf/p37oMnw4FoPbRSNNv47+VI8CczKR20xty/lohZkcZoJXaxCynNz4IaLfJgDlQwSqKmfTj4uLin2z9M8P9Sl07U5rzf4CRwc7mVMeD+dm2bkh8hQ3/bdC2jzt/IKmXtVfhyxsDLQg3PkWMS9NvqXccvuphriyCe4joyPPR5JU9JiS8b9j3wkevw8pSpbVO2rU5rZeiVxKc9F6RdDMg/0L7ZHYE/g+T/IaEnpv2wDjfPeRfXGHam4xvLpr4ag85ZTcjTd+de+HKNqrGDvAkB3dIcwa9P1XfKDJ/cIiMHU0UYOlHZSjG7z+tO0So5znQ2S/oxU4qoVDof175f6qrzMy/mhFCdcug6WGJsUlu18kA0o8UCIDnAyswFEKGYT72BGx2VJp+Wxw5mJ2A4qsfFhps3d2TewzwtPYG9BL1zWFpW0R31xrDMEqCkz8U5F8I8oeCPOfAwz67VuI+LfKfEbkNSc34L77C9g37Vrl8mb7c+PJGYetZO7/ZXEsgrmyj0HAS/mHrN68ZO8vYKYTP70y1F5JJyCrMba7f1hMAz3pGwuEL//5GSQmyDbIZIXzy8FOGbaL1NfL5Axq5I8DIoN6PFA77TwuW+ouuAkYIg58fhbnXSNNvJ3/yeO6fc2Ubg5+3qQU/PXlGHDG/JAEghCprgsq2iNwSlsCjvkqCU37lk34hKAu8coe3KFhk10rsWon7jMitlbjPiICy5WCUH4zxeo/SlzuShmtRNeHKNgZaLtqjhfJ8NPFmQ+6fs0JJ8ItRmGPQe4dsbM7H1FbIgx0FfbsDRHh6tybA/ooMD23b9t//9TTqwSA/vUIOS2j6yIFHBg2tx/Ob+Iq1Ltuan2j4Gvn2sKEQKmJcT4Hk+Rup94/77zti5vhQwFRBhmHqa2SwEC4l2IFxfmCcXxNUdm2WWsKS5rXKBmX2M+Kq5hT5Q4FhGHnOr1qYqhoJeWDsWoktlRiGYYMKV55iihSu3DilGZvlBsb5waiB0fPYJGdOCOX5G8nB7fZoYfLKHs0/V8TF1LUjMMsx5NvDhq+xbu+EJhOzXHyFBc8G5DfdbWI8oV2M339a0NxfoXp567XVqFy6fPnJw0/90/e/h3QwaBHhjWj0Tz/3f4FfEypSBjuSuelv8K5dzR8pKAajfPtrtpaaOh5MG1aJ+OqHA1sNJvDAIsQWVxd90drOdXwVVGnpCSC1S0DGiHYCqX8qa4LKUCfWdWdDXCjfHklc1i0HskJJ0R7jenDyyl5p6i3wawAlG4sA5AALBPCGwsGOpOaDwtVbXGuvwTf1P37yf2+MIPQooHVJxRcNRpEA838d/X46Mghg0N5wEBJp6i3DeWpwZ4rVrafy7REiKsgwjN60qx5qjNj+mqA2mvYN+6w2q9RkZoFVKy61zwW39QRODfsgVdDQucMQNS60dOIe3OECWQ82VEFHsLNF3J0YDlRo/hTMtiZDqVoF4cJ+107txbmd/brVr4LatQvAtcZL0vRbvhqN+p8bUJanNWuT5lD9j/Rq2GCGYrx6hq8JKvU1svo/c1lHGMYmudgsNzHLmTPkA6xORUWevyFe2YPUXYwEzNYIMOnJM0SOhDiuveTtRB2o0NvW1NkvaGpea500MZsmuLqcpBDqrQxUKyuaP1JQu3YBzCywrnXBlqbeBgshX/UwYGgPZh2uORQxnryyh+zUv6qFnTrZCxiWEmxGFBmG2bJeDn9SrixVwhVyuEIxVxMam+TiK2xslovNsjMLLOa4G0EVVJHmRsRrz1g0U8GvawL9bbnxogxpyqWzj0sJdmySM2xYy3tqK+SunSnNJtKBcb62wqfZXvv8ztT5cWJrrUgKYfiTGt/oYBS0ZbDAU+QZ3JwkkabeUpanAZveuLIIX/2wXvbJf98zFh1Y6v2XrPDNClfIAwfE7nd8RHaAXL3FrerBqa2QQ0EmVKSEK7RFMZ64a3oZT5Af8VY9q4j3aKQnTviqd1ixv4ktrvLVHkpPaHfE+DcZnF3K8rQ786IqY5M8FUKGYVrrpJmFlOZqjhcu+KtKFc0yPEGnOstnnjr6tc0CWiIFPTK4CtssLcyRev8lcG+e0NCbXP5lriwJjb0WjdXLt0f0bo56sEKJf9Mz4k+NW1hDRUrXzlRLWOrs13aExyHzXQ8ZDOKSB6Y9SoUr24j6kJH8yeOQwwyoCJtfUuZ/llssFBp7DaU39b4zs/+Q0Oxoho4H09PznGbusKPfP2FxP5rlX4PesITe5uLCxOXXg2HLDCuEgp8f8W96JlPR4asfDn5+1LriYhK9QcZXe4gD5tlWUV8jjx1NdLflQydXbYV8/kAS3kpGuP84TNYxG3WYAf3QoAi0vC009nJlG9X/5KsfDm4zyNgzqtWRi8NBBs77vnB44RHtHYQ22C87Y3lXgFsGAbi5QKgCObDlv++IDSODDMOkJ8+gNqOyQolvwyFWCLFCCVJZUU3gq85kLv+aNFkTVDqb06gmJmzpHwkNJ1d+iBbNpz943bfhkEU5AF/N46jPValrR9xvHB+bNTnKmX+o1eva5wjYGjQ0PID0erTnkUgkDPhb+G1eVAWzsWcLGibpD16X58172pLFhKG2//7jatYuE1Ug0VonjR1NnD+Q9FBVe01Q6XgwffVoElUFubKNrBBii6t89+5F/dDUtW+i/ohFyPNRpCWUVWXO3JRcng2yGaekAe07AHu42bPEJP9wZPjMBC65x5kJB4urM8EEuIt1bJLrHvLr3Zvqa+TuNjH2dwmXdzurg1YTLyY6m1Or+mJgQluu/G4C2X/fM6yAZtsozY245IEJ9XR1KgHu8v6AAqFAt4G4CqceCVETMtLciD1+aQaHgb5rPtD4u72ygDJhxvyw5x3flvVye1Nas1ctVKSo+dL4Cjs2yQ3GeHPDfMSprZBb66RtEUnznp4tgX3DPGCCIlMdVDs2UePv9M0TMLZnlpJ6/7jhrgmXMD3v/JlDcUYIlxIszY5msNnkMIOJHvrU9Zec2q+koohx1BscX/1wdm8hVxZhi6tzY0rVnCzzn1dvcVdvCZWlyv6m9K46SfN3FSpSWiJ3/dVmFlh1yH1skrPzGb+2Qq6vkevXy/U1sp5Dx/lrvr5hPrvjQLW2unpU28Iqe8ORb8Oh9AevI4Xg0tRbjKNCKM9HTSTPGYbZst7Ab9YKaL9MNpq/DRvEwnIhXBNUcnt+YrPslvVUCBmGYeIrrHtW0sOQHNwe/PwoYKzQUqRptCZAVigRNh9f9Yd89cO5oxea5r8zC+wLF/wvXPAbbmJSlxe2MndfEJvlJmbZ6XlODfeJ3OzWBBV1JD9coVSVyZWlCngEbTDKnx/n9Ryd1D3puf+iValjVggJ9x9Hsu9RxEX59ogVM4VQn748nRzc7shHm4b2y4DRnLgl+wBhuRCGK5Srt7x0o7cZp8JB0yjiYvLKHttW8Kz+dMTqoH/TM7ma7avZmyuE4DuRahbz/I+VbRG5vkZqCWuHX9nvFq5gGOYjMpPJgY9NGvsnqFL323dDsKRR9y5djnKGD1ghre48vnq1wzVftQNgmKCJNOeMEFphNmQDce0lyhSTVFVWov4IrRE6jIO1pZDZLaSq1bJtiwY/8tFzxvt0MvDljb4NGuOGXFkkd2C8JSIZZsYym5gYhtmyXm6JSEiGopkAjriZiJqYvRzlxyaN9U9FPf5Vf8gKJZojCsLm44m5YXiBsdppXY/k4HYrzIashvrLkKW6GjlfhSyEDQ/Uj747hvpTFD2m5x2rEOBkY6xzECUFK5QIDSf1/tZXeyh3Z8WpfeKWYwFIIcl4p6ku2+EKpb5Ge0OhRajZ17FJM606tTrrP/W+Vra4yn//cfgtH8ryL5GOhxR89cM4QlhVply9RfBwYKH99o5DMiIcm+Q7HkSbWKJ4LjXKMAxXttGp3lG2+FMMAxWJCo29gEImX/UwK6yetg4VKSa2T/zWZZvpecfHMExthWqxrdTXSKEirKeNbOIrrOq4HZvlYv+x2r8UiTVB5ZTO7huAMayv5nF5bgR2Mk9Ya/rwcPDfd0Saesu0Fjo1QeGJSeL8hqZGHYagbywSW9abvEGzQklg61myBwMPV97IQNyL/ZueyW59zIUVQvy92pVCzO0TE7PcxCyT0UWGYSpLlcpSpapMUe+z4QqD+iLDMDMLrJoqINhrowIw3ebLG8E9UP7Nx+X5n8HIjDnXAiIEtp5NvNlgrkxYyPviXYLmnHzqMgAAIABJREFU/VDzeyF756RC6DBumD9DQrP9xDY0I7lV+O7dCxOw+jdo7zRQt090XfDr7Q5DRbXQcyTntormsNTTtnrEPoPhJgdWCAVaLiZ+GDGUGb7Kscw5W1wFaa2ei1Otm3SCIoNmhkyzaxSQS6uqQr5BWf4FaO5mKsBUgN5d1SkhNFfN0ms/sQ1WCIHv175790JOcwMsxEJFSneb2PdEPthtq6wJKs/vTOllRBmGYYurYfo8VS0E283w5Y0OTpoyDOPbcBDVMVzFKZc1Clmqq5GXOSMLUkNDA9LrNVu0C6o4PDDO1x8Laq7acpCqMmQhBLefqMjzUfn2iHx7BLytAgffhoPC/atHAxn18O4/juRpAt6V2BKRBjuSHQ+mvW7+sGW9PNRp4Diq+SvVhCuLBD8/qpf85Mo2CpZlzhUxrp5dhi5uQsNJVHM4xrkaoR7qfcOp6klBQTIy82Lfh6XEZrnW3oC6007zGnPQZdTENe+//zggKSrPR5ODOxJvNiQu70hc3pH4YST1Puy9FRXfhoPBbW9nWhxZocR3797g50dRo1XVQgzwglCR0tmcunrUq3K4Zb18/kBy4GAS/HXz5Y3gkuoq2OKqu1u3Pio2vnv3Wjdgmnr/eOKHkbtn15sNycEdADlU21xNfIrp2jkmmreCylKl5x3flmOBPhI7oikASP5+6ZNLNn3Dvhcu/C4K1My6ODhIi5oa9d27F7AER56PJge3Z5eOFHExdf0lZXnKIttJbl1jgMTItv++Z6QPXgcXvVQ57GxODYzz56/5PFHR2bJe7mxOQX7LhtVB7Z+674hvwyFp+i1leYoVSviqh60rHoujh1Y1rEpzI/Lg9kDLRb00LFqb628Jf9IBlzVG51ag5rGXEuwLF/wD43x3m/a6vrxHM58PGDwrCSE/itEHDfLEV9iOfv8qaytXZV1qES8nrmyjP8eoLBs9O4/0B69z5Y3WrefFRy06QvZWqF7bqnvnwDjvwkanylKlJSK1N6Xhz7dVXqxIsELIhi83PXlGU89Uk6OiL+p2scK3uWZw1XUarlCGYnf//8Qst60n0PFgGn67shfRNF3SlH/A1bcxglyiRhbCjcCVhLlo/hscnCK3mvgK23pSyG2+d1VntmYXlh7qvAQg3yVNvw1wEklPnHCzEDIM49twUJp6C97Lu7L0boA4s8BejvKDUd7xGFHVv9Y6CTViYIUSoaHX+HWOotncq6IsT0nTb+vldVkhhDpN4XKHl553fGOTHKDpiWIOZCEsCYFK0LlrJfQc8VE/1xPEZrn204Lmv85VOQ34iJAVSgItF8EpL/ATtycsr4SGkyaGzypLlfamu8vfxyY51eQlNmuTi7pquq3a2ZiOY/ybnnHEMxYJwxMMUOBki6sCLRdX5e0BuOo6ra+RMtOoGa7e4lpPCgWbJrUIwqnRQl4rEV9hUU1JnGqWgX/s5e/d62wrvD3gDJ+p1NfImd9qbJabmWdjs5zqBUNqK9OW9bI6lV9fIyHZcOvh+DCMPXBlEU3zBD0cWcaE1GAxMcu19gqAjZJ5hqYrMmCPY5UNXqMUTdSMKEAFNbXHqfYi+OtHmnqL2Wyw2o2vehiw/s39rqQqvg0HpbkRpB0LeqirJ7ItrVWDNCarBBJPaE/TZgxomKwFFFbk61ihxLo5B7KAd1/ADO8jfa2O9MvEZrlcD3TAXKO6UTL/tFBnoB6tRliNPlCPbrrd8ADS693WLWkR+09r1AXdCVKDuLI8ZbhbjiuL8OWNejU2f61nYg6hoTe5/EsrcrmhIqW+xipJM0fgc2fcnxRV8dce1FMymOF9+fYI0jaM+vXyKYQdJxYCzniDtyt7FKdiA8L37txw1W3zc1bQPeQHP0K6agQN9V4MkzAUGk6yxRpuDv5Nzzi1oNUErBAyN4jtOYTGXg99L9y6Rs0BD7a42tDhgYE7gbNxz8OKIUsJtvWkQOfWsgmhz04wxIUwX7tgAIxNcrkF7VUgdWlaTX2N7pp1TeT5G+mbBo2FbHFV8POj2Y5lbHF14HNnnFpSYRquLGJoIQbPzAI7MM4TuU8RvLLAI6HuxH/fkcDnzmQ/bKkWCoaTi+mbvaghfqhIcc8QhWFf28Qs13XBXa5VZNH8LgCBB+pcg4qZGmEkHI7GYsavKwDiK2xnv+D0UaBh4oFX/OkRbl0TOAfFCiGh4YTQcEK+PcLe8ykHjbkxUbUQvs8QjHp6NIfvrvBF+uWriwZjs9zYJDcxy02/TKCiAO/F6jb4qh1FVTuU5Wnl17+EDGfl+ai5Bqj6GpmU5TommhaVqxgY52srfGALPU+gKW/2PJSYEcKSErTYU7MLS891zFt0v+PzVhBs2kEqObgd5gGcYRgP5dz0IKWFmTN8KMZnDBbUrUwMw4Q/Ka+6zalLlxjL1hF4VwUzsMVVkM9YyvJ0cnC7uU+pr5FcIoSQdA/5tkWkPLijOgXhrlH43bx5IIQzC+wpOA9AvTjAfndW0/UP1cXDOidJt8GVRYJfjCYHtxPvnVG3MjGIaofvgSk09nouI2oaRYzruR3BYH+ZUO95GrKFdSnBdpwTBg4mSR+X82i2AgH2F6GuhVAx8+BpYttTXtJxDjcpan+VG7VAmI08fyM5uN3Q+D9vYIWQai2N8yZu6JNii6uD294uIBVcnsZ8gqksVVBtCDHRm4qDSY2qXL3FDUa9FMXmoilvmg0WxPcXmRFC1G1Pebnla2ySc9xYywSYj7qFpoXMb9s0TP+4431SalNJHuSrIZHno4k3G/DjeA/1jmbweteMg+v5CN/KNXN9minQTDnEoyDtF3TJoE9z2Hw4mEERF5OD2+XbsM6cXkeej4ojzo9C4tyXCySbzWhtQTFNvUP7mHCYWWDdtvcUH+3UqP4AXuMDaJPuKmbUCNCfWiATLbFZtHDQJe4PpB5yFXExcXlHetJ8nOQVpOm3Sd1YnSL9wevJwR3W7Ul2D+nJMyYMY/VwSUSIWsvoGyYzruMImssYNG+exP+NZoQQ7Ludi4ll6C4HskfGbZC9tsWRg/mthenJM8mfPI55Yw1/0vmTX5obSQ5uV5annT4QC0lPniEbuIeKFKeW9OKwlGAHY15NthHpwA8hDjWokP+V5TqN6aRGvVrXja+w3mqtVqksVYgHpuLIQfEaVi+JaxFHDxG5scI3O4DBfJqU528k3mzI1+KueO0ZK9LXueafniDPsqOajtuAypqJZYSMOSEE2406WPC0B48+cFmU6klPnBBHD1nxzk6hiPHcfeiOgz9rpBZ3pem3iRyPexBHD8FvlkDCJdlRVGYWWI+2j8I7bhPHjnu6dv+rZ323+9DzopoPNTbTQqJTRpP0B6/nTaihiPHk4HaCKuiquoAiLiZ/8ng+JbTl+ah1jyzhCtkN0y8mGIx5UgghK3+AJT+RsBl/Nca0EFZVVur9VW7Qitr242bMrZdzQ7OMdY+3vtpDq9zXFDGenjyTvLI3eWVv+mavVzo11J5DshP0LnSN8FZCWxHj6Zu9d8+lyTOrziWuLOKrtTAhsS3i8JVr7rIl5XDrOJrTnAAfElTXswwmmz6qq6umZ2bM/aynuezNnMOW9bJFIxy+e/cKH11YKM9Hk1f2ZBbfSFNvsddf8t9/3OUD3QQ7762A7HNMeuIEIy66324tPXkm9dMjmS9FmnqLLX4psPVs9oOXsPklRly0KC5sCXvMay3DYIxrrfNSjVOzDx+1xB5CbOTMQD4s02yB1ey/8pZLp4pHxx8tKvvneleqph6r1r8p4qI4ctDNo4fy7RGLVNANWXFN3D9WId8eEUcOrvpSlOWp3A5YoeFE9uYTgni0TMh49pEdBoC/2saNZjplGNNCCPBzg5c3LwrhkDeT71Zcz77aQ7khhXjtiJ6cJF0wlq5JevJM4vIOi2JBN2TF9bg7VuFWLdQ7YRRxUby2eqeE0HDCihypR4coGA/eqTRN5jRvXFb0Y9oU3+SHy5pHw0ErBieExt5VGVGGYRQxrrdJnFGX3buvp4b4/JkVWOd7qXrmuVAL5fkoYK28NPVW7jELm18SGg0WZ5rAo0MUjNfuV/BxEaD8aW4ZIWNaCKv1fbc1f/v54bLm0dlH4ley3h4DZeFnBj+ZukP2SDCxRwXx+2VIDSNq4lItNDpVNE82X83jxLVwm4eF0JP3q2w0n+ABjZaoZi8ZyAuhJi4x28TE/q1JRCDrmgjY5sPe8ymCH2Q1tsWCLmwcXYVLtRCI3slGXAsz+yM9h7ciDU3ZRhWOqmqTm5FM/qbANja5Awaawq7ZVuNmTAxOOM6aoEIwIhSAzZ9scRVfrrvlgBVK3LMDwRMZ0Qw2tGy4TQu5dY2soPt0z5c3Atbz+moeF+4/TvBgPJod9eJ6nFWg2sqgRmgZTP6mwDY2kMVMbzXLxFdYbx2wCsF7qO/evb4NBuIB2N7nJ3pvwsHmhRJe6Tx0mxYCThjDJZG+DQcJ9pFaZ0ZhNR4a18a3lQmFzG9ZMf9rAnxqbv+PdvOPp8xlPJoXJVXhYIur/ZuNlYxb1yg09uY+y/s3PeOSOUJ1XtDpo0DDNnsaef6GeGWPPZ9liK/m8VzBY4USobEXJrXg33ycLUbbnKpHfY1XLWYmvHPXym2B0fydA9JypjtlGNMD9eqnjr47pvlXkJGTh55WGK8dbYaWMJl7aKCxF3Ktna/mcX5dU+rmCWX+ZwzDsMWf8t37uEuSoooYT17ZY/PUPH7Lrp01KmluRBzVGIxxBP99R/jyxvQHZ5TlXzIMw5b9kX/DIUBSNBtWCAUaexOXdxA5km0R2YuT9dPzHMN4IJzVlAzU3fSmp+kZHCEEoNneWlsh54r5UoL1yqOWFy2LSBnK8OWNSErGFlflDle4ATHL8sY28L8Cm6fy0x+8zpVtNEyD2wO3rlEw+xTFrWvkyxulOQJODh61mPFKvwx81Ulz3FDF9DQ9g5MaBczUawZPmi3gHso3euhQM5Aq8huWZDyBeO0ZIvdEVDQfbBHfwe4qo/jTI252AoKH1KnrlULvKryyC0gzzND8nVvUqGHJ84Jm8c/rM/VejAiJFAjZ4mpuXaMixt3TRmECafpti5b1GBIqUnDSHk6lTJI/edzT37h6xnLrGolUCkNFSrMHW2a80ugOH2YAZg0aHwDtBwRjPjXa+MAD39H5K82IUG+m3isORp5rGa2tkInUlpTlqd+89rvkOyuUcOVNfHkjt65p1dIJ16IsTzs7LBGuUK7eMnn+4AeU5lDERfHKnkCLZ/YXyvNR+fawNDcizw1bUQbeFpE851vmFXQiQo0nD4vuw5bUCBmt4p+rFrOZwIovIPxJ2bpZH4uSOYq4KE29pVqpscXV/tpDfM1eyD4apxBHDzi7VgLniw5/0rELR5obSd/sdUmxUA9FjEuTr6cmTlhd/bU0O2qd5cjYJOf+vC58KyKgRgjeGA8Gp0YI+tTcUFcnIizoJyxLrbNsWMKiLE+JPz2S+GFEHD20aiGAe0jf7HWkNJgNTmjurK2J+NMjLnSIVVGWp8XRQ4kfRsSfHrGhB6qyVLHO9NWpuN8laFbTkGqEOEOEDGaNEPDZuaGu12uE3sIKo209FHEx/cHriTcbUu8fd1tVSVmeTl13vn8V57twfH+FOHrA2QPIRRHjqfePJ95sSH/wup2xvrc2/HkIyIgQkJbDGSJkMIUQ8NnQEaE3armeKxDanwxRxMXU9ZcSbza4qtvQ8aSoCs7X4XheS56/kb5JfquDaeTbI4k3G1LXX7L/m3X8uzCB+weglxIad1fUFbY4Q4QMdkSo+9l6o4Q4H+cggMS0O3HKFEpZnkpc3iFee8YNoaE0/bbjSdEM5prCXHLJpK6/5IbUtyLGxWvPJC7vsH8YVCVMqAHNTtzf7g7fMmrFSl4VvIhQ/7PhRwnzwBnWEJsDSoJG22OTXOZ/8D+VnjiRu0bcfsSfrl7f6iDmvhGXhCCau3DtPobl6eTgdqQZmNjs785eUtegRw243Qz8ECFgLNK03bYKVtco4LM1QyjN3jn3P7DgMz3Paj5I1tdIPe+Qb9zFuXuOTXKDMT72H5zeA8qW9XJ9jdwSkcC1K3n+RuLNhkDLRadGLFLvH3cqbtDE3JdCdoUWDtLUW/LtEaes8lSHWMNc6MwCOzbJXY7yE7PayldbIYcrlPoaqSVs0nSpJSydGiZ/zepdTV4pHuGgGRFqfjuANK9LhVDzLNQzl2nGKnNSVmNijn5mge0e8l+Ocpr5+myu3uKu3uJ63vFVliqtdVJ7U1rvhqKIi4k3GwD7C61DEeNOjc/roWbVkOISsiu08Eldf8mRsUKYnVmDUf78OG845zcxy03MMqpZWnNY2haRUPtfVANuw8sElfzY2GoOzRl5zScDQNRkehOhCtbjRgTYqJNraqA5IOm5rYTuB8loOzbLtfYG6o8FB8Z5pMt7ZoHtece35Vige8gPOEHFkYPpyTPwb0uE9M0TbuiRWcX+pjTS67dF3BIOqkhzI/Z3Qhmq4MA4X38s2P6agDrtPhTjO/sF9cxH+kG3fS9eR/PpUHPKwIpNhCpYIlRSAmrUyc3nanoHe64h0+UgGW13D/m39QRwyrRLibtyCLibiCMHbZ5Fc1s4qLKrTkLyS2utQxNOG7B5FgW8OTI2y7X0BDr7BZx7yMwC29kvtPQE4N/Eu+sJ3YlmajS3lgR4TI+EcZOKuNFYwwP1en+Vq96a0a4nzKwxexb0UttWtEJAJtNmFtiWngCpCuVSgu3sF1p7de8mycHttmlhevKMC8NBhmFCRUp7E+w9VK3FWno8JpDmRmz7HsGbI/uGfdt6AqS8NCdmueZu0MNcNlZ8L3pj1oXQQpH7b9RslgYoRUkJrrMV7mmEOkGRq/OF+U1bB0yBMDbLNXcTu4lkuHqLa+4OaKYvFHFRHD1gz0yFO8NBlfamNGRQ2NmcsvpgzJG+acevVxHjgBnQzn7hhQt+sp+oPsx1QbytFQbcelMZ+FOALq8+aqajNLtJgOZququQIMH9LaNOUGh+34UwQWEPlaWK4ZzTwDi/rSdAvNqvspRgd53UfrKW52+krG/Bl+ej8vwNqz/FNKEi5dQ+0fBlrXWSC8NBlfQHr9vwQJO6dkTve+zsF6xbDXhq2NfZLxg+ubr228nFcWciMEQWMGEWCBkCQqjfL6M3QZH7h3kfFALGX8hOTBvmRQfG+c5+geAnaqJ3q0p/8LrV3Rb2xCs41NfI3W0gLaytkLt2ujQcVJEmX7f0/eXbI+kPtD+ie8hv9YLcgXG+9aSBFhJZcJbBc0P6BHHD7ASDL4Ql+qlRTQHX/Mo9USbEUSyAIQJZ321wGd8eFVR5/sd+zRM3aeU6JEWMq2sxXE5rnXT+QFLzWmgOSwMHRJens1IWJ5/1TpKxSc6KudtcJmY5cI6UrAE3QAg9cW/EAX52YkY/NQqeX4ABu1kGuIMiN+ep0y/jgdSopZsiiLAmqAAyNrFZ7vkfEy6rAFhKsJ39Gh+nLE+l3j9u0YdK02+5s00ml/oaeexoou8JcX9Test6ect6uePB9OWO5Kl9bldBhmGU5SnrWmYATggwBTxSGD412pMdxc+WuXzbAfzsBEAmwPMLMBBQoKrKSr2/gtxB4YnUKM7tCZDdJng5Acab4its+2nBorqgHhOzXPeQxp0rPXHCoiKTNOWZLbIqLRGpa2dq4GBy4GCysznl8nJONhaloAFOCINR3uZ96wPj/GBUNw1LcBOFpSsnXZ531eytyz1mwC0UMLkAD4ETq1p/pD+fdlDgLAyzZ1ZS069ApeuCH+YY1gQVpEE3Q/qG+dynHEVclKbJJzC9khfNDyz6VQNi+vMWlwY16ejXvXDCFTKpiwWQbcrvRkIieyeqsAuEDBEhBLSuagazqP9Ol2BRwopgEKBnKDM2ycH0F9RWyKf2iWSjxqUEOxjTOAesmHCwuoODko0iLkrT5ONvwImBahxDhKUECygokLKYsS4Zbm7niW1oVkCRPWWqq/GPhERECHAc1Spvav473S+EmIql90WSugZqK3QNZTTzk6tYE1QGDohWfAuXtZJL8vwN4rsp9PoM3YZyh5c/FAD/UxLeCAKIJ6KV5Wm9kQkrkkZb1kOFdEMxXu/TATkYJPTuLe6/K2Ki2UWomTUElM8aHwD1qUBCoAULIITwo4Rjk5zLH140/eHwIVXKBoSDMNkVtUfDCt9XvQd5afot3wZiHaSKGHfn+KA851d+5ZPn/PKvfModXrkDG9awayV2rcT9QZorT7F/kObKXTdQIU29xTSQjOwBCXMr+ulmFtiBg2Jrr3HtvHvIP3AwmfvnVvfL4K9Bdfm8o6bSozZUYtptqxAQQnDj6MzC6g1EmruH3G+9jR0R8ponJalStt7DaR/EypjM+LZFxdrYLJf725PmRggKoRVFR9PIc375Q0G6GZB/YX5YRVXN7HfgPi3yG5LcZ0SXiKKaHeWrdpB6Q8AWZSv66WYW2JIi5dQ+cdfJAPiVV29xufcx5rdDFPgtPHpyhR8RurwDWTtS0m4ZtXCaniGSGmWAjaO5Z4lHU6OMZevC8bVQb3BiZoGFqaxY7eYVX9H4Q3lumOBHuKFfVJ7zpy6FEt2fSJ74vdSlNTgqqP3+vxBSl9YkT/xeovsTqUshec6+WQI9ANJlAsApYdH9YXqera+ROx40Njc/f037gdLSkAs/PHB5H7KmvOUe81KC1XsSItIyypASwgJpHMVRLMA/EF8I9a5GmN9qbYXsSIM12YE/B/tFlQSXHrtH1b/02Mfgk5/mP/EOnx77mKqI6bF7HKwpkv21A04J/CQhgM7mlOEloNnzxZDYnAwoCeHLP06vu9Voyhuq3TaRllGGlBAWSOMozlkFcFnDnyLSF0Ljm7KDlwopuzUr2hdhUO7w4hsliWO/n7q0xgb90zyA1KU1iWO/L75R4swBLE+RanoCnwwWTYVnbsSGqyInZjnNoAR/czIge4kp/2uCiptTo5DblxigMxeRllGGWERYGI2jOE1igEICvmcNTkSYfeZZlOfRe1tSQSHZBB0MdyWw+xPSdWt6qBCRrhcluj/hiBySqs6CTwaLkhYZnYDb2aJ9g8Ls8gM8iWL6q7k5HGR05E3zmAHSQKRllLFBCJEaR4kcjHVgnlh6XydmE/aaoOLm9mvALYxUnyfZciMYt0lgNo7IIamnEGebfitLja0k9PIrmI+PehdvfIXFHOr1Ysuo5s3Q6pZRhlxqFCTLucEQ6r/WJYSKjJccAdAL8DHTPgRbzkjNRWUDyB3pWUoiYefgROpKcfLEx10ogdlI14uSJz6eulJs08cRKhOCTwYbmj5MP+ZiHpte9hLfbtv1nTIa90PkaXpX1QgZhomEdf2/c+/I2vGvlfVwUuA8ZOkpPWbaR+8CNlFgsOIRsl2/+qIs/xL//eXbdoSD8odC8sTH01eKFXv9Ws2hJNj0leLkiY/LH9qxbIRIrRd8MljkxknkhMd8E70fxw8MXB4RQtYIrXYZVSEmhIDundx/cKhIIxHh/oiQwVzGpP+Ih1NmIBvGkV293RyWrE6N2lAgTF0pTv5LmTxnxwIggshzvuS/lNkQGhL5CsAnA9m1R5n3JPI+OIkiwD8Ks65RWerqTpmZBY2WUdQmykhEdy08KsS0B7CqXjO3rhnHuN9hFuchC9Avg/PAi1m5XCXPu8h56q8JKi88AppQJNIsY2mBULnDq4GgdR9hNWpoaGnVkMhXYHgyAFIL5iAYMJkWadAmQv1WSRhcHg5qe8po3QZtyIsyRJxlVBofeOA7On+lmaOrr5FzZW96nt2yntQRWYJqOW+uiK0+BGk+ppnO5gOe+yA94VadZy0Ract6ja/GBC88YjyeJd8e4dY1mv4ISwuE0s1g6kclxHOh3KdF7jMiV55mgzluO78QlF/5pZ8bGJ2gIs/5kic+7n90kd+gP8SD8/7YXwFMcrW1TuobJmDjkvWGH1FWnJpcuEIZipn8Qb2/wrwGraj3EwRy+xIDzBRuJBcREhNCQPfOzAK7lGBX5UKryjRu/ROzHMO4+vtjGGZbRIZZ5qDJ2CSn2TxSazaqAygNpLguJdhVFmhdj6RgDBjBdLeJMAvbMINCZeFnOD8OIHWlmGwgyG9a4TckwVLEfUZU/4/8oSD9QpDeKyIVySkJVjy71rd12b91mcgbfuTNxUV5PsqVmb8rQZ4G3W2pbT1knhIqSz9ixoTZoqlpGwn5g5p/XgAFQlcsps9A7PGquqoqFArp/W3u05bmrd8TZUKcR61BHcMz0xEh+HSH3Je2StfDFTI4pQlmTVDpewJKBRnsYMKiAqH4RgnxdKjvgd/AB2TcZ0T/1mWunHAyMH2lWHwDd5e3JjLeEwnkaRCukLvbRJwPyrDqfWBmtwAXqWlHfr2IEHOWrLIUq7/dBjTlTfNupicKVZWV+IvpM5AUno36+pz7vWqeVe4fJWTwHrUA/0Arlm9Alg/PX1u9Pre1Tjp/IGli72hrnTTUmYS328CMCInnRZUEJ54ptWJAArWBU0lwxHOkDMNI14vEM6XEXdlkvCcS+NOgtU7C18L9TelVV7HeE2o2gEK+uQdZQF1jUGt5GTz4fjdWkytvmsoNyA8T7JRhyAphgRit4TSwzSyw+mP1Zt4TfAVCvudSgs1dUlFfI189mux4MA0jh2uCSmudNHY00d0mIj2KKvOYkQTJ1KiS4MQflFkhPwy6EFo3+SD9PCD+oIysFmJ+EUingfqUZjriaa2TunZ+JOERX2EvRw1+G2uC5GMsvcszvsLiFgjdvdJO81+n7TKq3zEEaM80gU0RoWYgrPmERbAYbh16y/9g0FxUy5jNuII7pOHfs+cdX+7DSqhI6WxOTbyY6HtC3N+U3rJezj5Zt6yXm8NSx4Pp8weSEy8iSyA+ihgnMpJ/990SnPgDC2ckUJdRSDct0WMVec5HVgtt9oWpr5EHO2Cf0jKsCSrP70yLwIDzAAAgAElEQVTlBpR9wz7DAqEVJTe9yxM/MebyiBDeXA3YKUOsQMgQbJZhgLEqvNFabJZtJvkPtISWiMnaOMMwg1FesxHcmisNoce1s98/cEDUVNaWiGTRpYVT5CPbKSOeWWtCBdm1kn/rcvq9IkOdUxKscodn18L+GiGFU9hzR57zSWP3oHa3ynM+8czawJfnkX4K9IYY/TImTgP1Ka29KT0Y4y5HefC6scpSpbVOam9K557eMwts37BxHhLGjBQVvUseJk8LgOwosBVoyoHmY4E9Q4QMWSFU+2Xi8bjm3169tXoHvU6ZkIdZD+Ys4Qq5slQxl8W9eovTG6IwMbdgeAzwPa4Ts9z+04LmJm53QrBTRnyjBDVi4z4t+rcuZ/o8RYgflz8U+M9q7WbMAXKXPfdpkd+Q4Dcw/q3L0ntFqSvFSF2m8i8E8Y0S4TEy1ufyws9wGkfNESpSWusktTNrbJKL5eyICFfI4U+CFo11nINqkMZJAmkC6GfBjAit0GyyTGivIUQwVwuFQgSHCBmyQsgwzMZIePTdMc2/iv3HaiHU6RFye41QpbXOfFCoN0ShOVsJZnreYOCkJSzBD3tcvcXtPy30tKXs9KQwHUmQWl6RulKM1B3DrpWERxczEsgwDP/ZFRZChCRoIZRuQm0kyZ6F4D+7wn92BXXqQ7pelCqViMxUmE5Ty/NR/E9nGKa+RkZNq3QP+WGuuOawBL4i9NbGAgA4q2H2SRDXbOJodsrk/oYBZTKyeVGGbI2QQe+XyS2Qxld020lcxa7N5sNWvdSHifSj4XNDSwRkcpbLUIxvPSngfAWDUb6zHyW6St0x90GYjTYq0ntFSMrhq/9NsPM/s1VQBUZL4INOmE4ZdTA/9zACh/4LPgHLMEz6SrH0HoEuWfP+MmZPAEwGxnnIZ1lDuyUTj+8tOglM0zPKKoaa7TgInTL6v1WA0JiDsBAChFozHNZ8LPJEv0xlqWJ64EGvRS2Mviwe5nfV2Yw2FDgxyzV3B3L7SA2ZWWD3nxbaXxPQGkfNBnYEDE3m/KlLusOvq2CDivDoov8h7cw//9kV1qhxAzLhycBJpp70cuWpwKH/QppBTF0KyXN++NdrQipAhwH/Wblv2Af5uFZZqhg+oZoYgMbZIQrA/XlR+E4ZD0eEqP0y3p0mZDCCwqUEqzcnhBoUziywhhdhax1aUMgwzFKCfeGCv/5YEPL5VI0C648F1Z4FpA5Y03qG60qT4FJvwDqosUFF+PI8OLHJ1//a8H1gQj15zm94VPymldxwMAMblAOH/je/CSoNyzCMkmBTb5RgNpGa/h5N1Hpbe808qKnEV9j9p4UXLsAKP8xzJOrYn17cNrPA4oQB6hST6R+3B/hOGcCdjWynDENcCMH+MrlBsXf9ZRiGaa2TTIycq+hlR02cxzAXIWpQqDKzwHb2C7XPBTv7hYFxPvsBJb7Cjk1y6mN17XPB9teEbMnE3LAIA/7qn/SVYsg2UVUFuXKD36Gv/jeGQSFM8Q9mcML/OeNMrPDYIrwWynM+fDMdZXka8x0gqSxV1Ac1VAUaGOe3HAuAW0yz2bJeNrwkZxaQx/704ja92SpIdm12uwoyruyUYYg3yzDAfpmxScjGUW8IIcMwuzZLp0w9mV6Oct1tGn9uoh91YJw31LnWOun8NZNW2ksJdmCcRypd2DBQiGtJ86GQHvsY5Iv9jy4aqiDDMGxQ5ut/DZYTmJynbCSW/KYVyCqg8Nhi8g4PWZtMj32M/8MEINA0RPn1L9liwncoTdS2spkFVs3D729KbwPWwuMr7GCM6x7yo+ZUuyC8Bs9fQ74D6PWzYBYIiS/osAIXdsowxCNChpC/jCfKhAzGmUc2Owpz/fTsFk3Hr0ig2u6YS6nhFAiVBCf+CNalkN+0Au8RahgUKgkWnB1VEpxhnAoTDmYQHr8D3zsj/ggrQWpuoMXEV5l935xZuJvGrz/2/7P39lFxnHee71NV3dVNDN2CxDEhQGyhuxvR4JcZyYcWcHfkXQF6Mb62F2FJu9eeiCSC2Z09cHY2ku6dCK/nSpqdvc25s7ugmUUZ554rOYjjeCNLtkBzo3MPKHAsMpItGmV3hOQAJmicgNTCpru6Xu4fDyqVqp56+qmX7q5u+nN8chT6rWi661u/t+/Pu/9tNjTsDg27hyaZwQkmNOzuPutu6fMEfujtGjDcAta5jU/onRZZIRpDVJKkvGjNesMdBqnHmZ0yIBlCWLdli95N6DIpyl8mU4YoSgsl0+OretlRE+J65GfuhA3cpYVST6t5K21yfESd/wo4M02DVjxl+LGvEDatUF7JveM++TPDoBB/H+FXuDdITJQXJQ8H5UNi95K+w9JdhjxQRjzcXJhu/AOAXlOwRA2HmZ6Lrp6LrrYfs10DbM9F18kRl7lESENAICkohC4mdqVRodeDaiKyVJIZ4SBxpwxyiy0kMyJCzGoMuI9J9UOkLR7mXXAapj9/gxNqq2uIiX7U+1GqcyBx8b+xygbD4oSkpntbWv61yQfeZciLYa6ty9qtgQkekigoxJcAExYRDYWDELo47t5OKue8wal8JbYMtJBgetsDIZUlIslV49g0bbQygulBtZIXJWltdQLknTLIUiLE9k4ZkAwh9Pv9ZaWlerci1lCgHUczIyIEAAQrRNNzFENh9PtvordlOExUxpNtOJIH4coLi5i2lTG0h0j8jXHTNa9IfxuXSpXuMnrZ0YQbJ+gnOUPhoPy0hiy8Ta9qEu2zfsVjem0ZCZUlop7XoBLYfWr0yfW+fWPTlubozXXDpR5k/4fR7Uu2d8qAZAghAKC2Vj87qlE4pNVQeJ62uBg2lZj+FIaG0WFcsMJMuh/2diZ+0VbuSHNmfG1sR7zNGrJSM7e0KGHQxusMsCccbDdhASPc8MZCjxtaqSF+yprbfWGjB3q6IFfBlhOGl1cXeCW9BNLghPm8aEZMTQAA7kcRZinIAmHKti/JJEUIq/WPFZnzNFosdRqmg8K5JUqvRdacuHYNsHriqqStng+1pqh3JhmYnp2IG58QWF1aZCRbSK0TmG/jLFuFa+jV8/j6HLVOMNTSKUXp+Ic+7p11Rv24gak3CmKXX1pCktEY0hAQSFQwPE839nhMNLa01aPbZCIrRP1umKc1/dhUgjzXGRqcAHZvX5JJdURIHhpn0BAFsBAU6s0Ft2wSzG097Lnoaunz4GcxIyvU7GImxdwqzDVlGA0HHz5wwRXr/Zoh75WEnqLxn6uVRriKVkfy51QiLri5HxWZ7nwxHRSmzC8tGUIYWaHw19+RFSo07G7q8ZhIY2LCQdPmAPindRrI99boKD2mGdMKqY4IAWo0Qm8NhZ3HZCtakTYdFA6HGb0vVbfZBOb4Lbqpx7P/bVbbjzM2TXefddcc9Zh2DE9IWVHSxyfMzU7wY4+ZeBREilKx3q+Se3IyG6P4Yp5wLU+prDB6wz+ni1gI+bHHYr1ftbhb0dzbleLFhPYyfovefcLT0ufRfnHC81a/OJhw0OgABsnTOhCddCDKU0V/Hy8myrJCss6GtVuCmLF6VayTWWsoYJF8/HBM9fnraojvPmFmmWpo2I1s5gxWiA0BgdwFQ8VweHVJW4FXCpRIc0spcjM3eqluIrwz85C7jPXV89x7fuY2695xn6SVlNkYwwdk3I+KoGGNdJfhThfiE5jUOoGkTUa6y5hYKYVE+JXH0ALF1QMw/qdJWYsNIeO36PFbLHjwxYlEbRhrLi3UjduGwuZzM6WFkjYXFZ6ny1Aj6mlHe0ov8EraKAhzpqoKJGtXbbLSj9bH6iMrlsZLk8fYNH0/SmmzGcGKxG5MSAYndIPCnta49Ure/SgFbTgsPo9zMNGmbyUcVCJcy+N+VESSJqWfTFDPg1FmrPdr0dDjCaO3hM8GYF9M79dsUUGIiTfNRETo2BYb+MWx5SzU3YxebQZzraafFlmR6f6ZW68dPY1MaVZFAuNe20kKB0HyhBAz84gumaKGKJxZJoRC3j+CmALsajCpW50/QZ+8fHlJnII/0hyf/YuV2b9YWQtNpLZsGoKICy7uR0UJRYKwsYXU77QQd40FY0pzfTEYzLxppuwR0khlidj/Otf/OmeuJE9CyyZBb8ivf8Rl+gq1sgRx5T00yYzfoh1YV7KlOwRfdLNC0iJC/ZImMvJFj9WbsoRINvDvdD9KhTTVgtJCyVwH1/gtWu/P31gl7E9OMVzO1SR1KisZGB0iFG547VUIKUrFPyyIYbtJjU7imyZ+KT/W+zXriV8tUpQiXBEs47Q8Z0K6m+ONVUJjlTB4ICleE5Ulol6x32J1EPm03WfdwJEhBPkoPebgMy8i9Pv9mHyuNvhFlwn1S6ZpRJ5xOYm6mutqiJvrZ8NsR+tujpue2Sd5RaN7vTMOkn0OJhA/ZWO9XzM9aWAIaQlxxhSu5kVDj/OX8u2V+UdewuBb59g8px5ydm42CbWDAq8UakUnRQEA3WfdpquDLZsE7ddWLrLMLVEJPRdTjPVR+mQsnZBJotJgvNa0b4ovT9KmJlLW30GO6niQKU1zNmZzS7hqwck37E/dDE48XCXvfLteGYlDr8bFIGLtPa0gRSn+Un409Lg2i2hy/EAH8VdeuTApRWl+7LFo6HHuPb9pOzQAAEkjTPLeOocAVSo8T7f02fn3AgAUeKXBdk4v3TI2TZueHSzwStpwMLJCHfnZwxOIo5oNkWdyo6P0mCyjdZIohHX6/TK2TFamhdnFR/6cyJRmsEI0l8zsuejSXcGVJw0e4OyNCwu8kpyasFEIk33tIi0Z65QhzIvSxXzCbYK6h3SX4d7zR48+Ef/QJ9zwirdZ4Woe906huWdDv4SisyZ69OvxDwtMSyDllVzBL71dn7MvJ+7wNJMdtbwqMmXIX6jun5kPzpDgVTCyQmEyQAnpauC1UWb/yCP2344qEyL7X4wWCOuSlhcFSRVCrNEaomNYJ1/soD8nQIXtXQMsomtmG29OWva/jXg2iC9PGmyP2VUvbAgIw12xZDgzzS4669qFsIuS3bvk6frctXXZvBxGKX7sK9w762J/U8S9509GulJccFkMAV1blz1dn7u3R6BPDcnmXhvbUB3LYHusc5ttlXi8CgIAOgcMb0aUqVkvaicx5pYo1YBjhHR1WCpA50UNrltIXqcMSKoQlpeVYdy3tZF7RvjLaFVqbgkxSuHLk8wlSO9HqZYTuPNOd3P8zIGYlQCuZr145kDs5Buc8kmyuExIUuWCu40or+jeurwqh8a9rZ0M/STHvnzP2/W5+9F9GiS7LJJUYXUCZUUPvwJdDfGxw1Hrl4YNAWH8cAyjgv0jLtPDwQVeqec1xIlFW6NxVIMF+aQAdg1hZkaEwKDXGtJ924FlQi09F13aSDFYIZq7xpyap/Fpk2CFOHY4GmrlDMkhXOQ9djg62B7Typ6jrh/xGJpUk6I0SQjl2vKl/G8ohzBzSBcny7yKLuZdwS9dW5fl/+gnOdPBKO6FnuQ8f7jo+c4i0qGNWickDAqlu4wh23GTWwnTgeobVFoohVo5KIcm5qBKCyV4iYkZZh+aZN48a35w8M2XEL14gxOMtrTmnG/0/SiF3Eqv/UWQs4aQ2i3BpBzcA5LlLAOpq639ycAZ5E1j04xWJ4IVoraAPDZNO8dbXU+VuwbcQ51qn+WuhvjYNG1iL+jgBAMAi48p4UKl8Dw9NMmMTdPhefWuR+iLEfimWFooBStEzCVqeN7wWrU0Yug8S9KxQhfzdDGiE515boV5bkW8zfJjj9k1nEB5JSb4hes51HLdrQAAINzw8mNfsSUbyTy74n4hcWjr2vKlcC3BvKB4m2U2kp5ZxcXrTNlOwjunl6Ew3VZPqXQLymGoFQxNMmO36LHpBDP1BV6pqUps2cQnTKuE52mSvaF6IHeoqXpkZJzjRmLLBGEyttIrSe7pz1DjKAAgWCGghJBxjhCqmmVkpubp0LBba/Rw8g2u5qjHRB2eRAsBAIESnMKRMDTJWPlyOhyScXW8mTX9FMc+xUl3mfjP8xMKBh4YAuLnC5mNUWZjVLzNWukIJZRACF0cp9YJ+NcSF1zMRnPH4mim5umWE6zexgk4Xwj/HZ6nIyvwfx9+lwMlYmkRwiQMCWxMNd2SozeP2Dlgc5uP7eid6hH31I8ZkuS1LZNcIayuqvL5fJEIut99/BataoPMiDKhHj0XXY1Vgupb4cuTBts5c1+AwQnm3grboz+HZJ3+EZeVRI0tUPnlyXtyooiQwAKGWiewr9yTXlg2LYfsy/fI10fQT3Gejt9xPyoyapxNF/Pu7RFD25rgowS8EN5mYcCaDKj88jROH0ItVJXMtcDvtelSOrzcNK1YevOIQ5OM6XJjyrAlIsTEVLaQdI3BDH9kU5kQ0oUKrQIl4psvmfQwGw4zLSdY/E4lc8wtUS19nmSooNELFzqZQkgSVCHzokigHHq7Pidx/lRiSAVXX8srejp+a+iF3Nvvezp+a1QFAQDUNxK8A1ZaVRNi7gNgouKgx9Q83RDyDE0m63fsH3G1/dh8LAgA6GmNa+POyAqVEbkcwgIhxgSgKhDw+/1JObgHJF0IMcMfyE5ZhweFyjYzLVPzdDdKWlo2CeaaSOFzNvV4QsNuu6wioM9v8KjXxlOJY0l4BjcqaQAAap3g+c4iu+cuYW+Le/t9oyoow+69S9KwQ3klT8fvXMEvzL1KwpdIqhA6gftRqu3H7P63WXuvuW253Ay1ckir0v1vWxLX1IA8yRgvECY3LwpSEREaXtLr6GnChI2aJ0dcyEtLZKGbnJ6LrpqjntCw+fEjAEB4nu4aYJO6jNBRkOyIMN2oyWyMero+Tygh9JOcaX0CsIX1lQTNQZRX8nT8ljyuRb5KwvsY2kucoQyHmeBRb9eADTkYaBRl/XJT77zRP+LKiAtZ5MnQ6Ek+qaP0kKSfENdUmRDSOeAeRJXQQ60cAKxpX6X7UarnoqvnoqshIAQrRHwjqJKxaXoozAxN6i57qlkvwn7dtKSgqaKnjd2fPJNGEEMnzAriHusV2e8s4it57h33TT8/hC6OM8+uYAqT7J6lVEw9EickKNZYFosqehoYdFFPKoMTzOAEA3c7NFUJRmd2hyaZMxP2lO70MknheTrtpX1C7GkZTXKnDEiBEAIAdmxv0h+iUAshTB+rzsiwTOgEP0yS9ev3o1TXgBvZimZRCyGqjbtlRasJ97Ii0ed9mJEfm6ZJdooWeCU4+dTVEG/p81i/zDRqcmj0vEleUiKZfqOtSQjlFd3bI7G/KUI/+ZOclUBNxv3CMkYITRQFTUA+SkgXVRt6ZqMfAIDytTAHXGw7u0hrcyRT8/SbZ+k3z7rh9FFliRgoEeXvmpKxaTqyQoXnaXOzUnroqWBkhbLdFjVJkE8Q4pfxJrtACFIjhNk0TUgoxlPz9P632cF29WQhACDUyvny3LbM7d2PUuO3qPFblp7ErzjBBitE699ko3WL5HWNkrRcWo+lMDrkMlsaVEGtE+hiXu/XkaJ0ClY+JW+CwsQHwBZH6dJCaewwHI4U5pYovcvTBzeltDqD6SpoOUFUGkzeekVybJogTHo4CFJQIwSJyoSEpqNDzugSRjqDIxm/pWsQ090cN907YztzS9T+t1ePM3lzGhiMNg3ST9Ql6UjMgSme2Rir0frz7PEPCgw5vyQb6rFvGbp/UtuGMcgXtZEVylG7GjAq2DXAEk7K+xywNcSWLpAUFAhBaoQQbzqqfbMaA84tE/ryEHG9HoMTjN5mpZZNwpkDMXPr7G1n/NbqRpi0bOilCo3VCIGtQaRgzcNFXHDHf4rO21BeycbSHaYrR7iWFws9Hr+U75DeTirf2NI4Ex8AW5A7wLvPup3jw3JE/yo5NOwmr6ognTxTDHlEOBROZ4EQpEYIgcHeUeRuwsgK5ZAPq6Gh2p6LLr3PbrBCHO6KOSGDcaQ5bm/amfyqhcovp1if0eenDfbXJANxwc391B/r/apexpKyozr48NmwyU95LWI09Dh3ujB+KR9ug7LxAAhhig3H6xTrM3plY0tXF7yinVuiLkw64sRS4JX6X+e0myUggxOMoWbvtFzUKrHFYjQ1BUKQmhohSFAmRF81aGVvbJp2gmwgfeAwdA2wAHBIpSktlIY6Y6Fhd7rmGQq80psvPVRBnyUHMTMwxfWmHlUnzJyz5QCk35B234kLbrBCCZ+y0m/c4qds8pbCW0G6ywh3GfCoMypdzAOvCBSpWgZOT+ZJtvTyKDHaA7x6PMX1/M1T5Pe3cdVXaaE02M51DaQ5KKwsETHuNsod2sRPmOZsUwYVCEHKhHDH9ia9m+BuQlWSMLhePDmivudQmLFrG58VTNgsYbQQANDVEG+sElL/VSwtlPrfeGRrWuqvImnjAQQAgCnbBT46aMsBSHcZ8TYr3WXER/OKD8OpKG3U50x+ZuuHZwvy8cte3spvkb0pXBMRIYAfAyNCaAtyB3igRBw8wLWcIC2/2U7nNl5rUyxjQgULvKQOqMnjAmqCsDFg0GI0VUKYoj+83++vCuiaxWlTE0gnBYeUCUsLEZnbhHQNsHr1QgBAoEQc6owdaY6nrGrYEBCGOtVb0+zqSifPXDFlu0w8P5VfZu6Eq0VccME9uvylfOV/4qfs6n+mVBDYLYTJm2eXopTp31EFxfrN7Z0w9zGwiDL88uVJQ522bb0mp7JEvNAZs1cFgTPWi9oSEe7Yvt3OY9InddKCLRMiThk1qP3FDnGY1cvj4+m56MJ/ptvq+fHDsc5tfFLlEJYikFvTkP5wJiDMXLk27DNRIIS4nz1k7oGpxMYqXVoKfkZxVXaYeyDF+lwb9pHf35ZrYu1lX3dzvP91LjUXowVeKdTKaS9GlZhTQQBAEyqQSCXIucDKElF7zknjDkIlqRNCrOko4jCcHBQi+1pJGJxgGns8mMALDraPH44daUZs4LTO/np+/HAM+d52DVid9DeKa8Ne04+ln6hjyhOEERaH5a0j/Mq2HnZbNhRaIaEXHcX6XRtNCiGw9mEwR+eAW9vN0VgljB+ONaAyeHZR4JU6t/Hjh2P4DrWuAdacCgILJyi70HFWM9gvmuQdhEpSpyuYIHduCdERinzXkHnn1OPLk0y3WU7N0zVHPXgzQ1+e1FbPjx2OXuiM7a/nrStiaaHUuY0P//todzN6qZO9KkiyHdtV2WFxIpCt7cObmKTCeAyLcDXPlgk/4Wpe2ntzErbVeF44bTq+BwDQT9SZDijNcT9KtfQhbEV9edLJN7gzB2LIpJQVKkvEUCs39Va0qwG3Wy2yQrX0eUx/HxsCQloGgpUgy37oAiFmgjAlgxOQlDYr1m4JXv7FGPImbUdooER0stdaV0PcimtoU48HXyGHBErEQLPY3RyHBk5j0zTyokELdF+DfqTBCkRGQiayQu1/m7XXwDf8WYJno4uq3c9YzW1SrM/T+AF3aY/gJKdKJVKU4se+4t66bPF54pfybTmeJEGxfvfzx60bHbifOSQujIiL1xPe0y5TXKiFg+2cNj8ZrBAH22Nj0/TghAt+70y/iiHb0vA83WZtCcZuBzhwWS8Q+ny+lLWMghQL4Y7t23WF8Ba9X9NIj/RaG5p0RO9oaaHUssnYHIWKnouuoTAdQm0a0wI30bc9eIvgvuwH/1j9zsiWDeSl8rFpumvA5tUzCWHKd7G1fVaiByUSp7ucITUmnHiEscdcz61YiU0dMilPfUP3S8eU73JV2JDYXL2yudyecDbGxk8s1MKe1jiyXhCsEIMVHHzF8Gc0vB5NaOELp+VILkNVWJ+kKi2UkL9IKhm/hSj7IbPNmJ6P1MzRy6Q2ItRXeL1tHQjTUZRkpgUrQSEE7hrcX893beMNZTNk7TTdHhZZoUIXXbZYnhqFKd9plwryN3pJAog0IkUp7nQh+51Fc3agwg0v74xwEHP8/M1Trg17bbG+o1gfU77TriFRQlaXEdbz3c26GZrSQqm0UFBpjDaaCZRI5tKS4XnalgGqhEmmFGBLv2jKBicgKT0PVldVlZWWzs7NIW8dDjOqqwZkyTd5i6SNAgtv1mfhT464zlxh2uqFtnpjcmga6P2WvEAQ/8zcaDsAwHoMIXERfqoXfx/6SS6pbSbUOoFaJ0h3GUzQJi644h8UsInWCqIe6I6/h7PVYL4dcwW/oL7BQ5V6ZOR/wWVjHJlwfXH82jFP43nrL8RPn4Yfj9QDN4mGWjnyi0tbphQiK1T3WQPeaRhgmsr681iEvFMG0/OxY0eKBicgqQ4Iamu3YCxmVEIIvda0V0layUwXbfW8LWv84K7B/hGmrV7YvdmG7hg9ki2BkITPz422m545kxFmz2HyohB7+2Uor0Q/yVHfiDNPclShoHxyKUrHPyjQW5YkXMuLLbgNxYXC1TwOr4LPrqjEFfazyAlh6BUg3GbFT1mLopjwbRQWRsU7oxaDQmH2PIkKJq91fG6J2n3C0xAQ3nwpKT3bKiIrVP+Iq3+EsWvRvBPCQT1nNW0BCLN6qay0tLzMmGOtRVIthDt3bNcTwguTzBFNaqIxgPZac4gQ+vKkUCu3+4Qn8V0JUK7e3b1JnYexwtwS1T/iwuzmBQB0buODFcLcEnXkZ267vpkYuNF274uXjRo0KxEIvEjob/DCNdOvAMAD8aOf4uincMsFKa/IvnIvqi854oIrFnrc/fI9Rn+PBES6y8Q/8Am/SvChSrjjiVonMM+tMM+tAADEBbd4mxWu5pmbnaf1C4Qy/M3TrAUhlJZn0xULqoD7Pls2Cfvr+ST5s8Dv45krtkkgAKBmveiEcBBp3OpYZzWZlEeE+iVQZEdoY5WgzT0iJTNdBCvE/fW8vcU2+FUs8ErBCrGpSghWiOauT+Fu+rFpOmHtoWWTIF9OIveU2o7E3Ytd2uN90WTDp8RFSJpFTffL0MU8vTHKPMkZegaYJtW7VYpS3Dvr6GLeFfyC3hjTRofCDa9ww4PZwfvIs1Ci5JAAACAASURBVBkJ8ujiOF0cdwW/kKK0eMMj3PAmFNpHHk7wJggz50Btgkw1htilPQnje3uRTbeRt8p76tvq+caAgYYXDJEVaihMn7nisrdJG9L9kiNOichxCOTgBGaz3s7U5kVB6oXQ7/djhii0HaEOH6KAdDfHw5/ZuZwacj9KqZbRB74p+rwgoDBogJda8kI1uCl7bokKzxtb1qH8ngcrEBcfyUBcvM7f6HNtNBMHCLNE/RR0cZzySoRDePSTHP0NHsZ/5npbSMRJXHBx7/nBew+9sFd/brCWyV/NY4wv/qW8ohwmCje8/NhXEr4u5SXy5pa4e6azo/yNvhQ3PcnDS3j3lqn51a2ilSViY0AMVghG64Lwuzk2zdi7v15F57ZkBa9GQQ7IG44IU9syClIvhCC7hihkTr7BNfZ4kld7e7CMPllfpDNXmOB6ESZjU2lUGL92jKkwY7QmEs8O0k9yJNGPe/t9V/ALo4ehRLiaZyhKs+jwKX7KCje8CROtGJiNUZKW1ISdMjLCghkhlLhI/Nox8vvjzSgIkV0SWzYJ/SOI+ouKqXl6an41U1JZIpYWSnBHt2rNLLwSBQBEoiD8GY0pg9lIzXrRCdVBoOOXVrMeEU8jRywgtVuCqVm9pCQNQmh0iKIxgBiicMgmChlfntT/BtfSx6agupYMYAd5/+sc1MKa9aJF0SUM2SXuHn+j1/2M4VUSwoJmO4kO9FNEQijeZoEFIZSidOrH3uPv+U0HrwAAiWyxBnlyWFwYAcb/lPyNXkNJUevW8KWFD4ccIivU7KKxJ5yap6fmwXAYAADStUBNpsArnXwj/fOyEGTvK7LXAdP8n0pnNZk0WHfCIQq9W7UjlnphtdMkJ1AiDranyLE3GbQo2nOsB4XkJ5eEIxBaJC4iLc8Q3pkwZrIYn/HpGHuXohR3ep3ph4s3iMqE5EGnOYsfEx8Ai8gf76FJpuVEpl68AgAKvNJgO8I9P13YtHFCd2df8kiPhzUmKNROlvjyJGSPqEP2SivJXC2sWS+GWh9eV6oSPklF4u7x06eNPWTpE/I7U+sEujhx8kC6y5j2BRWu5vFjXzH3WIuIn7LxD026EwgEGy3oYt7QCIp4x5gW8tOnU9wjAxQf71kyw0LH0kPmS5Ua5pYMDE7o5bd9Pl91VVVSjg9Lej4EmKYgI9cUTpmsV5KJWgi3Yyt/EkjtemthxtgsttHIg7CphDBCUj9qwW1aimyBH/uKcJWo0VSFSLAcw2g/jtGeF6N/eluQzydt9bwTRg7MEWrl0u6mpsSWJUJpCQdBuoTQ6CYK5HotzP6O9BIoEccPx0ws700LNevFwQPq7MqZ1O5jEokLfhDyvCiEMLkn3DC8NUlccHM/KiLpSiUMTB+5/5NcwuVHEO49v1EtJNxoYbQZRzT4pzH6p7cONAKV/2+olcuUr6qSUCvnNAlHjkMEURs8HDU4AUmblmxv0lV+rcIhl8JHViiH7OnVAhdeO+2TqqVlkzDYHlOp4Ng0/aZNG3oJgZ33Bu6//GtDzw9FJeHdhF95DC2Cl6I0iQpSXondc9fb9bmn47eeP1wkeWa6mPd0/M7znUVP1+eu4JckD+He8xta3kvS2kM/yRm15pEWDWStxTujqc+LaitqgwcySQsLvNKZA048t2j7Xwq8aAdwXKdMygcnIGkTQozy6/SOGqu4phK9DulQK5eyhdcmONIcV9YFIeF5ev/bNphzGu1xN5TtNDFzltCHBRL/oIDwCQlVkC7mPR2/leMq+inORbCVyf3KPdgLSnlF9/YI+zKRWnDvFBIKOT/2GElrD+GbpsRQRJiW/VlT87SqudGXJw0e4BxiVoWntFAabDfghpoykDEJ8jixGyfSMDgBSZuQYPpl4Ei46ofIKwuH7Ont/InuVtvGKmG4y/4NnxYpLZQudMbaNPMn4XnargkQoz3uhrTNRBjBkO1CEj9l+bHHEt9twc39qChhoym1TmC/s6h6XRJrNNUAO/PcCokWSlGK+1FRQi2UojTJ+CB0aEt4N/WTGxFCc0P01i9/j/xMvZseLuN1YJilpCEgDHXGnNMdowR5KkaWtLAFwvTkRUEahbC8rKwqENC7VRsUQosZ1Q8Jt9SmgK4Btl/HZa20UBpsj/W/zjnECqdzG4/8OtmogsD4xjiROKUmLk4aPxwAiPs++Ev5+PZRUhX0Suzeu9ohP7iwAvNAZBbXRi2M/9RPVB00roKrx8BFCO9J/ke3F73d9KFWLtTqxBROgVc60hw/+QZ6UiKyQqU9N4bs2ECm8ZyzcUJJOt8+zK89hprmRgaFtqwvsYU3z7oxXk2NVcJQZ6xzG5/Gr1nNenHscLSrIa79OtmrgsDIHCFEWp4hPYHG75o5IABcwS9Jek+kKIVJkJJ3x7B7lvScyfBCSBWib2WeWyGpF+K1ULiaR2IvQHklwtok4gDIhlsMDYMqiZj30nmInha2bHJcCqdmvTjchcjfQMLzdGOPJ70t9EhDmcoShKHMFCrbB0n9xgklaRVC/U7ZoUmELzuyAckh2VHI4ASDMVrz5UldDfHxw2mQw5r14pkDscH2GDIqHZxgbPfEgd6nhjA0HWgCyisyZN4xwrU8ZAcpuQq6ti5jDFlML4dyb4+QdP3oaSH5pAcT/MK0YQ0hpv/cdiWB9LQQpnCONMfTHhoWeKVQK6f3tQUPrl9T4OKGBxmNIPPMmFb/NIaDIL1CiLeY0c7LN1YJ2o9matz8yJmapxtCHkxblCyHodakJ0sLvFLLJmHscHSwPaZXYO8+6+4asN9c436UMlwmJIsPjDboKyEMCgEA8ff8KiERruYRqiD9JOfGdsToxXwksHvvEsa1Ki0k7O4B1sJBQFzBNfd3tPfLrqeFAIC2ej4t16yQAq/UuY0fP4zrDh2cYJp6PE6wxSEvEGJOjDvTVyAE6RVCgM+OooL9pion7qxXZQCgb2fXAItRAl/eqkRd6Iztr7d5E2+BV2oICKFWbuqtKEZuYVLF3gVSjz6/4eyojXdDApswiV7lUSGBa3IJVYTdazJ5SwLlFdk9SyT3VP4K5CoIAHBtXbYSDhK2wJj7OxpNuSfkfpRq6vEgC/xpSeHIEogsYUAiK1TXAIspxKQSZCiiGtaU74kxlEn9DkIlaRbCOowBN7r66sQyIdKHZXCCqTmKCw0fPFbsbo5DRezcxpsuThR4pZr1Yuc2/syB2NRbUXwLXGSFCg27m3o8SW01Mlq3SM1sNfPcCuFg+6qQ3GYTLotX4n75XrKTivRTHGHEJmshSXcPhFonWNzCQYi5P3eSimFvnnXvfxt95apM4SR13BDaHE69ha7iy4Tn6ZYT6jb1NNqNoqfdMsRQRibN1uk7tm/3+XyRCOIiHc7Lq4Z7kO8vHLdIY0+m3gI/GBrWrBe7X0psCRgoEeX7hOfpqXlqdpEOzz9MMIbnKaXi+vIeLoIpK0Jcf+kxOMGEht0pyCcb7WSzkvM0hHt7JPY3RST3lKIU4T0hruCXJFYs0m+s+hW4t0fE2yyJtklRKtb7VfJnJpxZtI65P7eJ2jMhw2GmsYcOtaKn9GAKp2WTMLdEXZhkhiYZu3aiVZaILZuEpiqB5CscGnYjTzVpnKkwUiDUvYipS8fGCSVpFkIAwI7tTT8ZOIO86cKkWggBAA0BQTuSmd71hHhnzvFbdFOPB66AJ5SrQIkYKAEA2DzVRC6BcA+wxa+60YfDxlETuwmNAiMq222yqXUCybA8AIAwRYnH/co9QwpHgiv4JfnSJSuYbhlNag5jbonafcLTEBDefEn3q1paKLXV87CHc2yaHptm4BJsQ1eWNevFwDfF4HoxWIForUQyNk13Dej2xaTYHFiG3Gj7fpTCpMfWekQIANi5Y7ueECKzo01VCCEcnEinEML9GHi/t8EJZnCCadkk7K9P9S7pyAo1FKbJo8DSQmmoc9V3rWtA1yuAhLFp2pALhrT0CZVosyuVX57wDkz5Lor1AwDExevCDGKXvWvrsnDDY+/iJNampKi0RHRUdHHctXWZZDSekIRCTuWXWynQKjHXMpqazrjhMDM2TbfVC231PF6lghWi8uMdnqcjKw9386rw5UmBEtFQ8gYyt0Qd+Zkbc3pRrldMMUhtQ37lMcuCtjc1pctQRib9Qmg4O4oa0kx7dhQpz1qgHFaWiG31fGOA9GLQNOF5+uSI68Kksd2Nyu8VcisyOUNhxpAQkqw4p7FC6H72kGrNr7Q8y105qJJDyiuyL98zlPbEg5+XUCFhB97J5dm9dVm84bW4SVEGKeQU63dVdrgq9lH5qzNe/PTp+LVjGEVkihMvqTdnrmZxbLy0UGqsEnxeMBSm8ZHl/SjVc9HVP8KQyKGMfIFr11KIuSUqNOxO+AVMo+MaehMvqpMDmxdNZ5sMxBG2LBijVe1HX289YXp7R1s2EaX4IVPzdNcAG/ihd//b7OAEY33jtpLICjU0yXQNsMGj3qYez+AEYiITz9zSw8IkcmSFHKN/FJKGQ+qxb+ndxD5/XLvsnsov82w9xdb1qX5OaPtJAl3M4+clVNiSGoW4X7GnpIcUcrqo2vviZfczB2UVBAC4KvZ6X7xMF1VbeTlz5moWh4b73+C6m+NdDXHCle5QDmuOeroG0CMWyQMmQoNHvSSXoUjhSQH3o4jYV89oG9cpk9YJQkj6I0IAwM4d2z+8cAF504VJ5kiz2p7DgdlRAMD+et7o0obhMAN/kcoSMVghBkrEyhJEeh1PZIUKz1PheXpqnoblCkMP1zK3RLWcYOVVbcEK0fSWD1hCIP+NSDoJqfwyivVrh9WY8l2uje16j3JV7BUXRvmbp5Q/dG9dFm+z4qeW2tApr2RIjRJ6gRo6HlsSpMjBRyq/3NP4AbJkS7E+T+MH0XerkCODCWN6YLZl1Mq2GWXhylDq6H6UkhM55F0t5oCdOCdHXOQZYBjmJul48CCzncgJt+Gw7uV+VSCQRkMZGUcIIaZSCt1EVV3LzsyOttXzhj7BSqbmH8nVVJaIPi/AlBMiURD+jAbGG1LIj8f3wFmFMOurx8kRl3bHhR4Sd09cnKSLEqyoZsp3qSQNAMBuPo5/lHvzcWHmnOrcze69Sz5agH7a7RE9KzU0tiYAgOUEKbVOQA4+eur6MI1LFOtzP3+cG1VfeZDkRcXFSRO26RZTPkq1CA2b6dqdmqffPEu/edYNL1sNtbrggU03CRO2SNJ49Y/MdiLDU0wov+e1VjuPySyOEEK/37+9qUkvKBycYI40P6J8es0p6e0dBQCEWrndJ8xsOVcBvw/jt6w/k0lq1ovyiQN52UHOhUm6u5kiP1/w06fYomP4+7g27FUJIVO+S5m+Q0KxPmbDPn6q95EfekX3K/fIh81VMN+OGd7hTrApSbrLGLJhY/cuRUOPGzoMiJ4zOFNclzCwc1Xs1RYLmQ37Er6oeMdMOGhxWbR8gm7p81i8fISXrSdHAHgwOR6sEMuKxNJCKVCSuG8FpnBgsiT8GW3lYEoLpd1p2piB7ALVXUDoVGc1GUcIIciW7GiwQtxfzyfPqyU1FHilntcexnAkPbEY7kepoTBNvuBGmDkHNicQQvqJOldlhyxpFOt3bdhL8uTujR0qIQQA0MVx9juL5rRQ/JQVruYZ0kKSXhhpyYAQSlE6/oGZmRPKK7HfWUSGs67KDpJnYMp3Kd9PuqjaVZH4D4Hs48VjcQt3gVeSO0qCFaKNeRTYyKp9QqQzRniestcRrbsZN3qfVLIpLwqcI4TZkR0FAHQ3x2cXLX1p0452iKqsyNJbGhp2kwuhtDxDkh1lNx+ji6rFxet0fjlTljgchFD5ZfBRqp/TxXH39gi5fczDo41S3Ht+5jbr3nGfcHZC/A3JFDzpmVpccMd/6jeXF9VL6lKsnynbSfQMzxyiWD9/85S0PMMU17Fb30n4EImLmGgZtRgOKk/QXQ1xc0lIQySpZqGkISCkqzoIbMqLptdWTYkjukbBg+yo3q3a1iln9o5CelrjSbViSirQPkP5k8gKdeaKpXd1bokyNIPBT6vrf0hcFXvZzcdcG9sJVRBCF9cjf0648A+JcC0v1vtV4WoeyZ1JIkIyyxg6fik/1vtVcyrIvnxPL5DVe4u0UKzP/czBvFevf+X1e57G8yRmCMKs4XAQAGAxy6I6QYdajdR0HUlpodSTvt/CrryoQwqEwDlCCADYqZ8sJnc3T7vvKADAlycNHkiuLWGSqCwRuzVZ6NBFl/V8jqH2BEHTCGMjcNAeiRUtlO4y3Hv+aOhxfuwxfBXQ+hS/dJeJX8qPhR433SyKUUEAgMXRCDwm/riDE4zFOXpV/jBQImqrLRlEgVfq11nSmxpsyYuWlZZWVyVI/KQMBwlhwuyo6od62VEn7Kz35UlDnbgVKg6kskQcPKD+do1N07aUPA0FhRJ3j58+bf1FTcA8t+Lp+B3hqiYt0l0m/mFBrPersd6vCVfztBlO8TbRaITe3YSrebHer0VDj/OX8s1191BeydPxO9Pb5y0iLc+ayIuaa/JUol3U0FbPI1NKzqfAKw22c2k0FwU25UUd0iYDSb9myJjIjiKVxglBISTUyjlhvScJSBWE217seonQsJvcOiB+LUG/jGkSNu7D3hnTu3Mh4oKLe88fPfr12I+KuJ/645fy4X88WfoUACDdZcTbrHibFW5445fyudOF0aNPcO+ZLAdCqHWCXndMaoh/bPjPaj0cBA8sWlQ/zMQShhNUcG4p2/KiwFFCCLDZUWS13Og1SOppq+eHu2KmlyulBqQKAgD2v23n8uu5JQq59Q2JtDyTpKCQpGWRLo57On5HuK0Jj/gpK1zL4y/lw/+Ea0RCKH7KRkOPx/6mKPY3Rdw76/hL+cKvPBb9aOgnOU/H70hU0JztS0IkLmKiX9R6OAjpuehS2aBkXAmjskQc7oqlVwWBTh9GRudFgdOEEJMdRfZP6+2sd0J2VKa0UBpsT8U+enM0BASkCnYNsLZ3vmlPRhiSERSKd0YJbaMpr+jp+K2VRe2OwhX80vOdRcK+VmHmnMQRrS82BH+j1+gcvb37wroG1DmJDNLC/fX8UGfMCecQcn/RTMmLAqcJIT47qtMyg/gQk0ceKaNlkzDUGevcZvMyeisUeKUjzfGTqKp7/4grSRnmrgHSC3xpeSb+cQKzGKMYFVf39gi7567pkqEToLyS5w8X3duNCRt/Qz1taRFpedbomz+3RPWP2PkhnJqnuzUmiM7XwsoS8cyBmLaLLS0g9y5lel4UOE0IATY7Sr6zHvMHSCNw1fXY4WiyV12T0BAQhrtibSj/gcEJxqhpKjnIk5Ee8WvHxMVJu15amD1volOD2Rj1dH3OfDtm12GkEubbMU/X5yZWDPJTvfYGhdzlA0Yf0vkT1t7xcwDA4ASjLXvD1rb0enEgKfBKoVZuqDOWxv0SKrIyLwocKIQ7tjf5fOhpJL3sqDbGsuhDkWxgdHihM7a/Pg0BYs168cyB2Mk30KnargHWxgYZJCeNhJvc5QO2nJElLqI1xiSE8ors3iV2z12LHTSphFoneP5wkd27ZG5FosTd4y6bfLu08Df6jF6C9I+4kjSTDh20tT/vbo73v845pLWttFA60hwfP+y4zvOszIsCAJju7u50H8MjeL3ev//7v58Mh5G3elyIXV9zS9TVGfV3BnlPR/H1AukP/rHYVs83VomPF4BIFHx+P4lLR0sLpZbNQu+/4NrqeaRTTGSF6jjFvv+x7se3ISAcezVeWSJN/wMdsXap/otp+g++LX69IPFJR1r5B3H+b5mn/jnFmDdxlbhIbGiHxaWy9OM881wUuCRpwQ34pK+HNQ3llVz1X7CvROjHLYU40r3/QXnW0Y9vtng84uJk7BKRAZ5MeJ7uGnBzSXuTh8IMAJQ2zNrwdelfBgWPC4TnqeS9Op6GgPBHL/Ch1vjvfUv0JCsvY5KpeTp0UX1MpYXSsVcRaduuM+6Yznv4f/7Ff3ji61+3//gsQEmSI66AlHzw4Yf/8vU/1Lt16q2o6qotPE839SDOktp7Ohzoxjs2zcBtSra0CdSsF4MVYmOVgG82g6ceTJNRaaE0djgK/43MLxmlwCuNH44RDgXTRdV6K4ESAlXQxk5I6S4T/zlp/2eKYZ5dITd7I4Gt6yOxD9VDXJyMDe0w1CMTWaEaezwp2ETfsknQ24syt0SdueLqHzG8y9M0KdjxZJ03z7q17Rf763lt/XI4zOx/G32KKCstvfZ3E0k5Pgs4UQgBAE9t+EfInfUAgFArp00XBI96td8c5D1Tz/632UCJtHuzmSzo2DQdWVmtTsubLZHWvdAFH4DV5U2BErG0iHS1Yf+IKzScwD6mZr042P6wToZ8w42iN7aBhC6qZmtPJPQgVSHMnudG200s/UkIlEPxV14bt+yahvJKzHMrruAXyUjeujbsc28+buIqhL/Rx32k3pOcEOvbIQAAndt4uFm+a4DF5OFr1ovIZjGZwQnmwiSTpDoLXFsRrBAaA/asc0o2yG/9hU7ERAfmbf/+97579M/eSsrxWcChQvhH//qPfzJwBnlTY5XQ/7r6Oq5/xKXt7whWiGcOpL/HofusG5qzNASEtnreOXVvAEB4nu7+mZvwvKP8xNsSFAKDWgjgSfmZQyTmouKd0fi1Yya6YwwhRWnhah4/9hXrxmnmoNYJruCXzHMrNkaBiFfJL3c/e4g8NDT95uN1ixBl9iKyQgV+6MXfOdTKJfxWwpWBFlM1lSWrq5qCFQLJwiaZyAoViRrbJ2w747folj514k35Vsvcj1KVf6r7nl/95RWHbJxQ4lAhvHz5F80vv6J369jhqOozMbdEBY8i3nrtPVPP0CTT9uOHmlFaKO2v53dvEtJ7DRhZobrPuo2edJT5TFuCQmBcCwEAdFE1U76LKa6jCp9WRirS8qy49ImwMCrMnLNYETSKeJvlr+alLECkvBL97ahry5eptImh8svdlR2YXR8SFxFmzwk3T5m7/rBFBcGjqSC9M4MKOYIkfAmYnhmbfni04XkKNkkGvin6FC8YrBAAAIZkTwn0xLkwSb/5Ujy9+S3kX6dzG9/VoP4EYq6SqwKB/+/S/5uU47OGQ4UQAPDs722anZtD3tTdHNf2OiMzKl0N8c5tae6KRl6TFnilpipxfz2fep8I6PBy5oqZ+ocyQaoSeCuUFkr9b6TZOMouxNus8CuveJu14oWmB13M009xzLejJiYibITKL6fzy6mip2UHc3FhROLuWSnE2qWCcowyt0TNLlKhYdKEB2FomDKGJhll62zar+kDP/RqxyGQR9X2Y1ZvC9D/8da/P/D97yXl+KzhuMFzmR07tv/VX/9X5E3IBby7N/Pjt9Tn5TNXXGkXQuRi2/tRCrZxp7JIPjTJDIXRveOE7N788M1srBJq1tuz43RuiWrpY3ta4w5v9CWBfoqDKiVFaSiH4m1WWnCb9simiuP0UxyUwKTmP8mRlmeE5RlgX9rZLhUEAMgBytg0bSh7P7dE7T7hqVkvdjXE0yiHkRUK2lko0y1yB0C6QA4Fwkyv6ofIDU0yDhycgDg3Irw+OfkHL/wzvVuHOmOqmXS9YsBge/qtPkkqag0BIVgh2q6I4Xl6ap4am2YuTNIWW+C09QDCvBM5yA607ECK0tJvXFKUXo0Uo7R2Qy/9DR54RQAAXcxTXpH6Bu8Q5UsekRVq/9u2mfmpWroq/9Rr7jOfFjnEXKdiGlxTAzLIO9Ic1zpynBxx6TlmODYvCpwcEVZXVVUFAnoDhYMTzJHmRz6mcBmF9mN05oqrZn06P0MAgMaAWOCV8N/J4TAzHGbePOsuLZTgtENlCWnbpwxMB41NM/Af9o4kayWqtFDq3Mb3XLTtU3RyxDU2TYda49mRJlVCeUXqKQ4AwGxM96E4hoRDO0ZR1ava6gVzH87xW/TuE57SQqllk2Cu35uc8Dw9OMEMTeKWbCAn1lOGXpC32+DyH2cmRSHOjQgBACf+6q//tz/9IfImZLcSsmrly5PGD8fSPlAo944apbJE9HnVFXiZSBSEP6MBAHNLVFJHr1TX2koaezy2u5zvr+e7thloXsiRcZAM7RhCm06IrFA1Rz3WXwLOOVSWiIES0XTbi4qxaXoonED/IAVeaeot9bkulSCDvIaAcPINdYCBTxHd+vv/7vfrrsVOL44WwpnZ2ed+X9fY4uQbnHa1pmMHCm3PIqYYTK1+bolqCNlwulFR4JXa6gVDvXw5MoK5JarzJzbvNiktlIY6Ef4MoWG3jRkL+bVKCyV4bQqbQhMmUaFXBnSsDn9GG/rdkZ2ZqaSpx6M12kaeVHsuuvTWZm1vavp//u+3k3F4tuDc1CgAoLysbHtT04cXLiBvvTDJaIWwsUrQBl6DE660C6HtWcRUcqQ5jskOlRZKPa1xuzpIZe5HqZ6Lrv4Rpq0+6empHKkBdoIk41sQakVP4HQ1xG1Z7asEZl+gmKl+F207AtL+whDKDrXUMzVPI9dNIM+oZ67o/mUx2xScgONMt1Vg3r7BCcQAAHKdwti0PXZlFmlLh8W2daAPAP4+jVVCkrpzoRwGj3r3v80OTuj62edwOJEVKjTsrjnqSYYKdm7D+VSkss1k/Bat+s+iCqbFl18JsuaHXDcxfkv3NOvz+Zy2d0mF04UQs4wCAHBhUn38pYUScsORufqcvfjypPS2fpmgskTsaSVKy3Q1JHfgdzjMdA2wgR96W/o8oWG3Qy5uciRkbonqPrsqgcmw7mwICPjMYbBCTPsMlTkKvFJXuo/8DEoIkSurMOEgZuO6Q0i/PODx+/07tjfp2a2dHEHkPNvqee2swoVJ5ogD+vKDFeL+et4JqkxCaaFkyPMl1MoBYNtAmB7wQlsOLGTbqkCJ6MuT7OplSBl0UbWNhuDOITxPj03TgxOM7Y1USggv1Loa4mPTxipzTqCrIc0FcuT4ILQy1t4ZswXW4XlR4HwhBADsfe01fg+u2wAAIABJREFUPSEMz9NzS5QqdYCcVZhboobDiJpi6ulujkdWqGSrhXWg24vR72FqtFDJ1Dw9NQ+GH52yKfBKgRIJAAClUf55WRFiBBgDdDC36UjReLa+E7W8IsoJyAbx4XlqbNpqSpAEQxdqJ9/gWk6wSVVleyEpSSQb8nAQU7YoKy3dsT0nhJaprd1SVlqqZ7d2csSlCvV8eVJTlYgYKJxwhBACALqb49C9N90HoosJ/0+Z1GuhlvtRavzW6tdyGD2Jahg4xxKsEAMlYrDCnnUBTHEdlV/m2rAvfu2Y9WdLDeF5OrIC2yBpOL2T7NEdJAVeYxdqvjxp8ABnyzRFCiAvSSQPvfHBJpT905D+gg7HuskocfT4hMyf/8V//A9/8R+RNyEHCsem6d0nEBsK0+7XJ2PO8zo1NASEnta4xRM9ch9INmHLLhG47U/iItF3q5KxKwp27YNVpwXSCy954dfD54kCR123VZaI5lwX5pao/W87PS4s8EqD7em33rVrfNCZ6yZUZIYQ2jVQiHTrTiOmp+yTh41DS9DsMbtbWqxcNFD55XmvrlYH4x8ftysoDM/TQ5PM2DRtvXHfmVhJVwAAIiuUk3OkDlFBYGQmO0Nt1ZQ49NOgorysrHZLUO9W8kS2dr1yeulujve/zqXd9QZSWihd6IzZOLobrBCHOmNpn+BMKsNhpuUEa26ow/3sIfnfro0d8iYH04xN0y19nqYeT89Fl/XGfWfSskmwooLgQY7UmR/L0kKnqCByFkJvfBBzXnWyrZqSzBBCAMCe117Tu2loEjFQiPTBk8dgnUNjlTB+OM1qUeCVOrfxY4ejtn8D4cTImQNqh/RsYmqe7hwwnASm8suVe24p1ueq7LByGN1n3btP2LDb3bEUeKVQK6c3OG8I+LEMtTrlGhRSs14cQm17TwvIWYjdmxGnKcz4IMiEwQlIxnxt9rzWihkoPHNFHRTC5UeoezorKAQPvpZjh6Np6eVp2SQMd9kZCGqBoeGR5rijzjs2MhxmtHU1PJ66PtVP3M8cpPLLzR1AaNhxOXZ7qVkvDnfZfL0IP/lpX00DACjwSkea44PtCIu4tHA/SiFnIZBdrJgz6mutux1rLqoiY4QQYC8ukLG5njm6M1NGpYXSyTe4scPRlk1CCgQDWr6NHY6GWrnUNBC11fPjh2Od2/islEM9i0UkTPku+ok67c+16kjC2DSdodZ9JJQWSv2vc4PtsWR8SksLpcH2WMq+AkigHqd9UkLJhUnawPbBTB4flMmMZhkIfkMhcu8gst7rhLX1CYHLyawvEVRRs14MVohwzZONT2uIyAo1FKZDw+4s66Mh7Emm8su9L16mWHR6g7tyiJ/qNfS6LX3ZmRGFrusp85vuH3GdHHGl8jPZsklo2WS18TgZkLtsYzatlpWWXvu7iaQcXxLIJCEEAPyTrf9Ub0Mhcncl0nseOXHhWKBDx9Q8bW6/YM16saxIghtknPaVG5umBydctot9uiBccuJtOo8MB2Wi79eRe82E5+mmHsSkUEaTxsUjSboAVZKaNYemmZqnGzWfqAKvNH4YkblFSibk+9/77tE/eysph5gEMkwIMRsKAQBTb0VVaTe9ARfkxEVGkHAyDO6FAQSrYZzD0CQzdosm2c3mZEgmT9jnj7s2tuPvY2is0IETOFaoLBHb6nkntHSOTdNj04yNxmxOSMaQ0DWAcMNAhhlZMD4ok2FCeO/evfX/0z/WuxU5Jrj/bXZY43rQWCX0v55h/tdrgbklKvwZPXaLDn+W3DE4OYuOMVEzur4noRC6Nuxja4nSnuLiZGxoB4kWZkdetLRQaqwSHLueJTxPzy1S0NBxdpEicdKR92lDZ85MuSq9H6Vqjnq0BcILqHbWN8+69QYnarcEz/6395JyiMkhw64l/X7/a6279axH+0dcWiFsq+e1QgiDD2d+69YypYVSaaHQ+MDDSV5nCmPf8Dyl+orC4FjpJiojn3pkf1FfHjB0MW5vBwpTvotQBQEAdFGV+/nj3GiC2BEAkLkxdIFXClaIwQqxqUpw+DcxUCIGSkAjyloMGs5BjH7AHIhemwzy90IOcEMw027OJMOEEGA9uOGYoKplJlghlhZK2vPF4ATj/JaZNY4vTwpWwFNk+nNlVqCLqtlaYx2hcMowoRYaFUJt2Koc/JCvOaCJKPyhXRGnbNZaViRWlqA3GGQc2fFbyCDT7MiOVozLts/ny5TxQZnME0K8B/eZK66a9eqc5/56Xut72T/iyglhDhuRq7Mq6KJqT+MHem2iGEi0EHmRhwHG0Eo0WTvdaw5l9BOeR4QOKuBiLABAxu3GWpuM30Ivo28MIMR+cAK3fTBTxgdlMk8IAQAHvv89vZaZwQnmzZcoVcvM7k2CVgjhLiQnlOVzZAEw0af9uWkVhEAtjH90UK9eaFQIzbnBQZTRT6YUvXKQgxyNb6pCLFqZW6IwDhKZYqumJCPL7Htea8XcinSZQQoe5qImRw5DhaumKoQwuDbs8744aloFV5+kYq+n8QM9J9LAN3OClMMG7kfRS1KRps2YRuWqQKC6qsrOI0sJGSmEsGVG71ZkI1PLJsSfE87n2XlkObII8mxBgVfqblb3i5L3iCaELqryvniZLqrW3mQ0pZFLUeZA0j+CUEETbTKZGA6CDBVCAMBe/a4kpLN2sEJE+j47bR9FDufQVk/qBndSsyGWreuzSwUhVH6Zp/EDpnyX6ueBEtGQW2YupZkDCTIvuhbaZCCZKoSwZUbvVqS86f1Rs8PWJIft+PKkwfYECwoKvNKZAzGlulCs39t0XrlZwi4o1ufZeop9/rjq5z2vkW5RKPBKyPXiOdY4w2HEyKze0qUsa5OBZKoQAmwMjvQoaQyIyPMFMieQIwcAIFAi6hmFlxZK++v58cOPqCBdVO198TLeQc0iro3t3hcvK/dUlBZKPa1EhpxvvhR3+MRejrSAXlqAWrqUfW0ykAxzllFy7969Z39/cyQSQd6KdNZG+lFllvVojnQBXUXgv5F+NDYWBRMicZH4lYP8zVPyT8LzdNvbrF4HaYFXevOleK5HOocWPac0pIk80oANkinL6JFkcETo9/uNLmZCZkfnltDtUjlyKCktXHVCgRYNypso1m97URAPxfrY2l7PC6flbtJAiQiXarVsEuRyeIFXqlkvHmmO277ML0fWgNwg1hBA2P3gly5lbjgIMjoiBIkWMyG3ASCtR4MV4pkDMfuPL8cagC6qZmtP0EXpaRmXlme5yweEhdG0vHqOTEfPXLT/dU5rKYdZuuTz+a798kqGFghBRkeEAIDqqqqqQEDvVmRRFxkU5uYocpjDtWGfp/GDdKkgWO0mPa/toMmRgwSkuSj0QNfeGbN9OnPbZCAZf/bHxONIedPmtSC5OYocRmGfP87W9lqcl7cF2EGDHDTMkQMDUtuQQ/Tjt2iMh1FG50VBFgjhntdafT7dMxFS3pC7cnJzFDnIoVi/98XLCTcLphK6qMrT+IFrw750H0iOjEFvamI3qpyMHDSE1G4JZqKbjJKMF0KAdVxDyltujiKHFVZnJNKXDtUDdtCwdcbWXORYsyDjBD1zUUxHYcYtXdKSDUKIj8q18ubLk5AjMrnsaI6EMMV1nsYPqHznrt52Vez1Np3X8ybNkQOiNxGolzDTex6fz4c3f84IskEIy8vKtjfpzlGQWwfBfRR2HlmO7MK1YZ+n8bwTioJ46CfqMD7dOXIAnepgzXrDLRSZXh2EZIMQAgDa9f8YyKC+tBDtHoRpi8qxxknlvLx1YMkwp4U5kOjtmjBqLgoS7QLKFLJECPHWo8ilIch9FEjD7hw56KLqDFJBCF1U5c6NVeRAgeyH0JuawCxd2t7UVF7m3DIBOdlTFfvBv/uTf/XH/wZ5U3ieHr9Fq0z64T4K7XxFaNidG67PDqj8cqZ8lxwVSdw9afETcfG63pJbzPN4Gj9IwgEmHVfFXmHmvDBzztCjKNZPF1VTRU8r3zph5py0PJOEY8yRBpAFI2R1ELm2XgaTisssskcId2xv8vl8etajZ664atZzqh+21fNaowQ4fYjc2ZQdUPnlTHG97NpM5ZfTCgdnJbJfibh4HXB3SSTEVdkhLX4CABCXZ0ycNynW76rsAAAwxau+1UoDa4mLSEufAAAk7p64eF1anhEXPxEXr+s9m7Q8w0/1MsV17NZ3lIU9aXlWmD3H3zyFeawST12f8+uCerCbj6+QCSFdVO3asI8p26VsBZK4CHe5PaGUQu2ki+vhPwAAwL1O2VgrLk6C+F3w4IMhLoyYcMN5+OTsOsDdTfgMdFE1xfplRYf/F3lP5VMJM+cIPxgZyuAEemqiMYBaPag/NVFWWlpbu8Xmg0sTmW2xpuLw//6nf/XX/1XvVqSHbPCoV/uZaNkkhFrVqpk1UKzf/ewhczNwEhcR74wIC6N68QHF+t3PH1cuIRLvjELdEhdGEkopNO1kynYaPqSZ88LMOb0np/LLPVvf0Q48iIuT8Ss/wJ9PmeI6T+N58uNxINzlDqU9txamuM69+c+174/ERWJDOzCqwJTvYorrVNqZkIRPKx8VVfQ0nV9OF1VTj31LfglpeTZ2aQ/y4RTrp4vrmfKddOHT5uZbxMVJ7vKB7BZC5EmvcxuvjQj1/Lgh//kv/6/sKBCCLBPCmdnZ535/s96tbfX8Ec0a8dCwu+ci4pIHqZrZBFNcx9aesDIGIC5O8jd6kWdYTF8JjMaglOo9M1vXZ26fnzB7np/qRQobxfr1vND46dPcqO5lgbfpfFLXKqUAYfZ87Oe676feu42RKxi4uyr2mfj84FUQZrNd5Tv13nNxcTI2tEN7xUPll7ufPcSU7bISu/M3+uLXjhnNnGcW47folj6P9ufIM17PRZde/2Cmm4uqyCohBAD8i//1jQ8vXEDe5MuTxg/HVKP0kRWq5qhHO3SPVM0sw0T4pUW8Mxr7+V7tuSNhj6XERYTZc/Frx5CRpWkthIfEfXRQe6qlWL/31UnkiVJPC6n88rxXMz44kJZnV95Fh0dGVXBVAjd2mNMbXRlj/Uz5LtfGDnwYp/dwV2UHu/mYieORWTve5btPeLTjg8gcmJ4fN+T73/vu0T97KymHmA6yrUMSU7yNrFBnrpAO159ZA45rEncv9vO9sUv7JA5dWCVBb2SNv3mKu9yBeSDF+lwVe/Nevc7W9VGaIiU32o5/OP6QvC+Oup89pPq5xN2LDe1A/rKuir3a+wMA6KKnzR2Do9CL29zPHtK72ohfQVxJMMV13hcvu585aK8Kup895H11kq3tNaeCbF2fRRUUZs9H369dCyo4NU8jh+iR5qJIP26Z7BgflMk2Iayt3YLZR0G+pDCyQq0RxzVh5lz0/VpxcdL0M+iNrPE3T/E3Ett9QTnU6lBCKcXjfuagapM7AEBcvB6/chB9GBs7tL9CFttYU6zftRH99vI3+rQZb/ezhzyN503n0pEyBs3qSJRVWp5Fq+CjBWmjSFyEu9yBTGlkJcgTYM16MYDqDcQMVWfN1IRMtgkhwF6qGBquXzuOa9LyTPT9WhLR0gNqofbn3EcHhVmiThP3MwfZuj6VFFnUQrqoSruTgb95CnlIFOtj1pJjNbNhH1J+xMVJ7qNHrhUo1u954bT7GfQFBAlIFSTfYCVxkdilPcj0uxXfc3hU+DaibELPLxQZCSD9uGWyZmpCJguFEL+PAjkciswMrDXHNe6jg1bSpHRRFdLumRttl5ZnSZ7BVbFX+wz8zVP89GlzhwQAoFifp/EDpnzXI4f0Efqc7taJkLISvV82fuUHyv8Lm4ysFJKRKsiU7yLfYMVdbtfmaS1aHPDTp0k6V7MJ5KlPb4geEwZUBQJZMzUhk4VCCLBBIRyuV/0wUCKqxu0ha81xTZg5FxvaQahbWlwVe7VrgCTuXuzSHsJnYMp2anOk3Gi7VS3cekoZF0rLM8gnpPLL5PlF+Z6mX9c5aC9umOI6ZJKTnz6tLJVhWm3JX5q7fECbEWVrSdMP/I0+bYMxPDDTR8VdOcSNtq+RdCjkfpQ6g7qsRw7R65USIVlWHYRkqRB+77uYW5HyhvxA4JePZCXi4nUrJUO2tldbVxMXr5PnXd3PHFQFcACA+EcHrVQxAQCexg+UBxa/hm6vUMmwtPxrKy/qEKALgRJkZxAAgJ96JMZi6/osbpviNAN/FOtna08QxoISF0H+pTwvnDY9JsFd7lD9mmuB/hGEX6iJqlBZaWnWzA4qyU4h9Pv9r7Xu1rt1bBqxahk6rmnvjPHZy1Zgd6Vp4fFsfUfbdRK/dow86crW9qnUFIaVVrpbKdanPDBpeQb5C9JP1CmDwuzoJFRJEVNch5zSExcnlfdknz9ucbQm/vFx7Rvofv44ubjGrxxEdpmanuxM6C2QldyPUkhtQ6pggtWDezJ+9SCS7BRCAMAP/t2fYG5FBoXIojEylZr1SNw9Va2IHCq/TFvqk7h7wjTpCYhifWztCZWaSsszeg2f5AfmeeFhRjT+MTooZGtPKP+veCfjtVClRqpfUEb5hljsQwEAiHdGtcEcZmBDi8RFtElRprjOdNsOP316Daog0BmEKPBKyDMe5tLf5/Phk22ZS9ae4vFLCpFuey2bBKSbzFqrFEKEhVHTfaRM2U5oGaokbiQfRRdVKUULwt88ZVGW6Cfq5KygMHMO3T6aX6Y8eP6m+fKkE1DJiauyA1kdFGYfenNT+eXuzVbXVmhV0LVhnyENE2bVnnlUfjm79R1zxyNxkbhOk1TWo3PdL2g30euVEiE7tjdljZWMiqwVQpCoxxcZ/iMrhWPTazEoBDCfabZxht18TNt4YkhZ6SfqtJGlXm2PHPczB+UD40bbkQlS9zOH5AFEYeaclZRs2hFmH6oglV/ufgZRHRQXJ5XGOp5HPcpNveh5VRhKF1UbEletblGs38qBcZfXVneMDPKiHwCwezMiHESWEmXwabaMJpvP77W1W2q3BPVu7R9xab1jWjYJKg82yNoMCg01fGpht76jmmc3qqyuir2qyFJYGCUcTExwYKwfACBx97jLB7Q6R7E+zwMNlrh7/I0M7q1QXjog12ioGjtZIzU8JBIXUQ2o0EXVnsYPDGmYVrcMFRdVIFtP1wjIc5de9gvTJpN9Q/RKslkIAQB7XtMt7ep5x7TVIwrIazYoFBevmx5pV/WnAFM9L+zmY6qRDPLBRMyBybGmuHgd6btGP1Ena3D82jGLPavpIv7xcXkCxFXZoe0xUXmKMuW7LJYGAexwUYydmIjk+OnTKt0yVFxUobUIWDvohYPI1Bd+E332DdEryfKT+57XWjGb65FBYVs9jwwKMXu5shsrI+1axxmMyZkebG2vUgstxqkQ5cCinhaym4/JzavIwNHhiIuTcjhIF1VrDTlVKkjll5OP9+mhakiBA3+GjNnEO6MqA3SjxUUl8Hc099gswFA4iMl71W4JZt8QvZIsF0KAzWtHVqgLk+p3wJcnIYNCvWurtYCVkXat44wJ4zSVFlqJU2WUxUK96Uk5otUTS8cCbcngv2FMproDNHx5qILWKnDycyoLeybm8cXFSdXGqIRrTDBAFVybpUEAwPgtxJwY0DHSwp/fMKm17CD7hRBurte7VW+OIlcpVGFFC7Xeada1kL95iruCngo38Jxb35FjPml5Jja0Q1WAVI6CZJAWrgrAg/wkW9enismE2fMqgzErFTiIykrNpAo+qlvWVXBNmaipQJ6v9Cy2ByfW3BC9Eqa7uzvdx5BcvF5vLBa7/ItfIG+NRKmyIilQ8ojsedzg82Xq6oz6KmFqnt69WfDlJetQHY4wc54q+Ja5hQx0UTX91afFz/4WCDH4E3HxuvTFLFNuYGSbKd8pfTErn9rEz69I8QjzzX9m4nggFONhnvrnwv/40epRCTHh9rtSPEI//jzFrC4vpf3/iPKsEz77WwCAtPIPwqfvMsX/M5X3ddMvmmzExcnY+7UPVfDR/QwSF4n/XXf8o4PyHwJY2/748EWtqaAwe567tDengnYxfotGCmGolSsrUl/l690ZcvTP3qqusnSR5HyyPyIEABz43ndNBIXkd147cKPt8Y9NTpgxZTtV25r4m6ei79cZ651RxYVTvRZzpNCVW9ndyk/1Rt+vVQ4suja2yy8KN3XEPz7uzNAw/vHx6Pu1spyo5uLFO6PR92u1PmpWVfDOqEUV5G/0qXYhuSo7TKuguDgZfbdqLasg0A8HgxXGTJV9Pl/Wh4NgjQih3+/fsV13uN7Qbqa1XCmExK8diw3tNNe3SRdVeV+dVM4XiovXo+9WGRqT1+ZIo+/XWekj1W5rkpZnohd2xoZ2ygfG1vYqHVDj145F36+1YgVuO8Ls+ZV3q5XDEnDDA/y3uDgZG9oZvbBT1c9pXQXjHx+PXtgpaxhcMUiughIXiV3ax6kqiy+cNr1rl7/Rp7wUWJuM30K7ZiObRfXuDMlKi20tlCQhimHZx8zs7HO/v1nv1kCJeKEzpvrh3BIVPOrV3rllkxBq5Ww+vkyDYv3uZw+ZbrWPf3ycn+pV5cHcm4+TN2vw06eVvYUU63dVdljZmQcA4K4c0toxM8V1zIZ9TNkuAIA220bll7srO5gK9G6/FCBxEWH2HD/VqzowOLoHABBmzwk3T2k9P+miarb2hJW6oHhnlPvokUX2RucF+enT8Y8ecRNliuvY2hPm1v9CU7fssIe1yO4THq221awXB9vVZzm9O0N8Pt+1X17JVjcZJWtFCAEAf/Sv//gnA2f0bh1sj2k3MXUNsEgDmrHDUWT/8VqDyi93bdjnqthn4swlLc9yVw4qx8WgmLk2dhCeSYXZ86plOtZlSbwzGhttR25fYorrqPxv6ZlVMuW7mPKdzBP1pne4G0WYPS/MnBdm1D5kENeGfdLyr/VUwf3sISsXDcLseX6qV/Xkhkp6WtGi8stNDwvy06eRYr82Gb9Ft/R5tD8/cyCmzYvqXe5D/t2f/Nsf/Mm/tfn4HMkaEkJ8UBisEM8cyAWFJqGLqpnyXXprDTCId0bjU30qOVwVFYLVB+LiZOzSHq1uMcV1dHE9U1xHPfYtE8rET5+OXztmbhkhXVRNF9fTRdV04dMW+zBVSMuz4tIn4uJ1cWHE9EnftWGf+5lDJt4T8c6ouDwjLowKCyPad4Z9/jhJekBanhVmz/E3TynjSHMSKC3PCndGhJnz4sLIGk+EqjAUDupd60Nu/f1/XwvhIFhTQghyQWFKoIuq6aKnqfxyWAukCp/GxGfS8qz0xa+FBcSmAvhUVP636KJqprgO+TzinVFx8XpC3xD5eSjWD2uBeMEWFydB/K4q72cOprgOsOvgiz4sjrrX6WmkXJWUuHvw1cXF64C7a0u4w9b10fnl+N8d/kXgq0vLM9Lyr8XF63iloYuq2eePI6854B9IXJ6RFj9BPg+VXw7d7Ej+IvAZxIWRhIe0ZrExHHytdfd/+U9/afPxOZW1JYSXL/+i+eVX9G7NBYUOh8ovpx+0d9p1KkzGc2YK8MoA/tuuvKIs9rKQ50glNoaDV395JYvNRVWsLdswaMN9+RdjyFuhoagqKITto9qPy+AE09VA5YLCVCItzwim0pUpfs5MIRlClSvUpRG9/k/kogn8At7XWnevHRUEa2R8QskP/sTwwl5kz7HenXPkyJEjLSDPSHqTYPjTVxZvXEKy5oQQv5sJuWUiN1OYI0cOh6MXDnY3I67jc+GgijUnhCAXFObIkSPr0LOSaawyHA6ukSF6JWurRgjJVQptx9t0Xu76ExcnxTsj8alebZM9xfrdzx+nH93Wmzwk7l782jFtJYwproM7mISFUYm7BxsaAetniuuRY3nuZw8p3XAyC2RHLsX66eJ62Iwq97XC3hlkF66rssNlxBXWOglH42EDcK4lB2LISgYfDtZuCWa9s6iWtdU1KvPOTwb+1R//G71bDbWPNlYJ/a/n2kcRlpVa3xDwwNaLZEbQIvyNvvi1Y1pVc1V2qOy7JC4Sv3JQWv61p/G8xEX4G70q5aDyy70vXk6Xd4xFou/XqaTC/ewh6FoQ//g4f/OUp65PNbqg3QIB4NXD5j+3dzISibg4yV0+gJQ3uqjatWEf/US9fBiYO68pgke92jKNuWbRs+/9NLtXDyJZi6lRkGhhr6FK4dAkszaX16vQ7mlyVexVeXgCuFb353utb1DCwE+fXnm3mkNpsNbEUlycjL5fK/vFUKzP/cxB74uXlTbc0vJM/GOT1pfpJf7x8UdG11m/t+m8+5mDsqhDV1X+xiNLsrSWsAAAYWE0+n4td7nDiq1rQvgbfcitEVR+ueeF094XR10b25UquMZXTEAMraFPGA6uQRUEa1YIQaK2qFyl0ARaLaTyy7wvjsq74GX4qd7o+3XaRbhWkJZnuSuHVt4p51AeaUz5Lu+rk6pIdDX0WZ4Bj/b9a224+aleR1lsk6DcUA/kpRCouXXuo4OqJR4U6/M0nmefP67cFgIA4G+eWnm3irvcYe/fDjywBUdevrifPZT36nXV3w6uVFxTc5962LVoAiTqn8hi1mhqFPLs722anZvTu9WQ0QzyzmsT5PIB8c6oas8OxLTj18NnXpwUZs8JM+f0IgO9ZKw2AfiV1x85PO1OO0N7FdKLxEWi71bhVyNxlzuU1qlIs1CtJawMXVTtquxgynZZTBpLy7Pxj48hTVz1PLhVlutrmcEJpmuA1f7chJVM7Zbg2f/2ns3HlyGsaSG0sVKIvPPahC6q9r6IaHOQuAh3uR15Sl31KS2qTmhVKnERaekTcXlGWp5JaLWFcfFGlsHy9syq7qnSQhOb9tKCVsKR65ZiQzsJjbMxXuRA4exqyGl21SxUxywbWq8hnxD5t1uz5KqDtrCmhRDkgsLkgFluoNfDokRp/QUx0RyIiTX1zqTK3lfFS6u1MDXNPqZBqKCOI/bKu9VabcMskSDxIoeWdVTR0/JfELZ3KtVOXBiQninsAAAYf0lEQVSBlzLoZ8Bu+MqpoJKeiy5kqjMXDhplrQuhjUEhcqnh2oRi/d4XL+slPDHZNltemtmwz72xQ+/VMWdSPcEgl5a0o13HoSdsEhdZeQf9FuEXKvHTp7XrD20h4R4uiYtE3681txUk+7gfpWqOeiIruXDQBtZuswzEXPto5zaEd194nsZ8ztYUEncP02ZJ5Zd5tp7yNp23dziPLqpm6/q8r06ym4/pqSA/fRqzvlxvcI1ifZ7GD5S9M9xHB2NDO5PaP2kC2HKp1Am6qFpP0sQ7I7rPc/OUqndGiatir/fFUe+Ll10b9qkCd9NQ+eXwb6fsaNWi+u3WOP0jjFYFAQA9ryGmuXLNonjWekQITAWFkRWq5qjnflT9KSwtlMYOR+0/xMwEmWlUId4Z5W+e1tsuSwKcLWPKdiXsuEnYYUGx/rw9uudZbVwIIxgrG27tQlqe5S4fUAk5fl+8qlNGC+GiXfx+YDxw96Rrw16S4mLCA15T6IWDeltxcuEgnpwQApCoUnjyDa4hoJ4gDA27ey4ifHlCrRxy3HANotc1g0SYPS8sjEqLnyRcXwBrTqvLb5+oJ2xZJOwzxIu3xEW4S3tUR2hltbp1kA4AIJEKAp0CoQpjS+cXJ8U7I8LCKFxhiLknU1xHFT3NFNeRl1pzbaIqDO1JzVUHE5ITQgASBYXIOE8vKPTlSeOHYwXe3LsKALZrBgNsDUXeZKgvUYY8mCA59cc/Po40LWM27HNV7DPUUyotz8Zv9AIAjD4QACAuTvI3enWnDra+g1FB8c5o9AKRCCUUVD3ggl/1T/U3EmOApgdGH5XFGF2SilxSKJMLB0FOCGXwQSEyztMLCrsa4sgi4tqEJEGaPJAxHAaK9XtfnUx43sf0+1D55UxxPV1cp7cLXlyclL74tbAwCsc/EA8sfFp3f/3ipLj0ibgwKiyMIOM5fMuljKE0I5Vf7tn6TrqGRnJtolrafswOTarDwQKvNH445stTn8/1dtZDcuEgJCeEq5gICoHOEE8uKFSSxtk7vSl+PORRrHhnNKE3NHgwUWB0AgRmgB++lv68gXx/fMuljLQ8u/KusT8HdEtPffo3p4Ja9IStcxuPtL7KhYMk5ITwISaCQj1bh7Z6/ghqDdjaJPWzdxIXiX98jJ8irW8pIQwKZQjlMHmQSyDEdNcJU76Lre1Lmfl4TgWRIIUtFw5aZK2PTyhJ6D6qrQi2bBKQO5j6R1y5nb0y0Ghb5eycPMQ7o9H3a82pIABA4u7xNww8ln6iztN4Pu/VSVdlB5WqDVOrL11UzT5/POHUgRJpedZ076Uwcy76blVqPFfxgy5rluEwo7NuideqIEjkLPqf/9Nf2nZkGU4uInyEf7L1n06Gw3q3Iot/Q5NM248RQaFe4Xotk7CJwyI2Bmd5r06ac0CFzmHi4nVp8RNkOhROT0LvFaa4jnrsW1R+GSwciovXxcXrcE2g3vPTRdVU/reY4jqSiREtWls1E8CFjkkq/SbVbyHTQdZi9Ao3evkqyGutu/9LTggfkBPCR7h8+RfNL7+id6te8a+lz4PcxJQzXdOSpNk72/OTTHGdp/G8Xc9mDlX3rHXhEWbPx35uW53PdjmEoyD8VG8uEERycsTVfRYR4emNbCFVU+bqL6+Ul5k3u88yckKopvl/eVlveT3QCQrHpundJxCJ+JwTtx5Ufrlrwz7yspYe0vKsMHsuPtWbDMMRc7MfjkVank1GspHKL3fDHRRWVojcGeVvns7Ny2MwaqiWCwcNkRNCNQmDwqHOmLYuqBcUIofxc8iQ752QWR0hWLwuzJxLtuGW54XTTvbXJkdri2M70OWAKa6jC58mEUXxzihMBVvxFVo7vHnW3T+CGNZC+mvrqaZMLhxUkRNCBPigEFn80xtxzZmukaMaGAAAAHYdlV8uLc8A7i4gGCFIxiFlxN6lhKTen0y5QoQqelpafJjjxS/PyqFF7/SiFw7qbaWAfP973z36Z2/ZeXyZT04IEeCDQqDjY6RnetTdHN9fn5uvz1SyQAtzLp2ZDnKCHuiciPDhoM/nu/bLK36/PW7pWUNufAJBbe2W2i1BzB2QJeuuhjhyiD500aWdu8iRKUjcvdjQDnFxMt0HYpKcCmY647dopArur+eRs1tHfubGJEUPfP97ORXUkhNCNPgJm6FJBrmeqa0eUQ6MrFD9I7n1TBkM1EJhNs1NpCbIqWAW0P0zxGV3gVfqQvk44tct+Xy+A9/7rp0Hly3khBBNeVnZa627MXdApuD///buNrat67wD+LmXsV6AmpzyoQsYUgOsb75XQLs6gEha2GstyQKctoDjFFiXDlbSpChaxPAL4K2LsjRetxYyNgxI2jmYs7VLPCNL66GSpWELBoukgbpbCl1qX0YNEGXC+WKBVAdRL+TdhyOf3JDnHJKX916+3P/vQ2Ab1yaBSPrzPOc5z5kZ3+cvCpcO4Xx9V6MzAfZ++d12v5FGmbvFncVppGC3e+vOY5k856f0zHiZe4KeW6xiLl04j+UgF4JQSD5oJp1VlzLVn7yCg+arT/Mnq8m/QKEr7H345zuL0+Zusd1vpA56XUMbp76BI7ZKyhxvrH9kyOSOFRUVUaloJPLi115w8v31EAShUN1F4Su8koVo6Bq3mgpdp/xg2bMxY/bs/fK7pX9J4Cb3HjC39Bh3t29WMMdYPlBN/sne5/CjWebKd14LBoUnvjc2lbd4J3tEk9W4tX7oOuZuYXf5pZ3F6cpHnbXkqny0vP3eaO1didCNNjYV7sHBsSOVCZ3Ti3DzHn8MKZWIx7787Bkn319vQRDKhEIheTGB2xEaG6lwJ6tl8qpkHxu6S/nBcun29G7y6+avcu1+L6Ty0fLO4nTp9jQWgj1DNBdmVrD5Umc5eAHLQRkEYR0vvvC8ZFEo6ggVfbHO3uJcYQHda/9/frz9nr6zON2unlIWgdgR7CWiWyZOHytrYc6H7Kv/KrvuJhGP4dJBucDs7Gy730NHGxgYGOjv//cPPhA9kM4GnnmqHBz8xB9++rC5samu1rR77ewr/YfM2pFI0NXMX62X//e9/ew/mv+XUwafUAY/7f4r5vb/+43d5Ev7q29gFdhjtkrK2et9xZpPzIcHzB/N7PbXLPy2SsrXf9y3sy8Mwls/eR/NonKYLNMQ+Z293KFrxW1l7Eo/d/3HnQcBPUP51HDgiXH1ieOBXx9vZRR1FXO3WPnoTvnBcuXBHVenhkJ7iQakvXJqb4Y3o0o004rCfO1GIAgb8s67N77xzW9JHuDeuDS3dOgqr/sZt1L4SuCJ46Tv16rHqBJCCKk8uEOnbiqf+g36gHVEJ5utWnlwx/s5q9AWzU4tXs2rE1eFd9BjoFqDEISNkk/iFmWb6EowXFUIALWeebOfuzvIvWVC8jx18cL5SxfOO/n+ehSaZRolb7tKZ/kdoaITPy+/K7wqDAD8SdQjc0Irc1NQ9DwVjUQwUK1BCMJG1Z3EPbfE6Qid0Mvcld/GpsKtmgKAP22VFO6MjsMD/HlVoueZSxcvoCjaIARhE+STuDc2mztKce2OrOMZAHzl2p0A9wfCzDh/WJXoeUrXNJygbxyCsAl1h65xs00LV04f499KIT8DCwA+sZpXuT8NRGNFRXNnGFy92xQEYXPkQ9dE2TZ7in9V4c17GEAKAMKh/MJ5jbdklw5OTU7iBH1T8FO4OXWHrnGzLThonjvBv6QeA0gBfE40JlTUIyO/ZYIQcuV1LAebgyBs2osvPB+NRCQPcLNtRnCddCavcid3A4AfbJUU7nJQ1CND6n16vnjh/HDUsTEOPoEgbFooFJJfaCIari2qcnAndwOAH7zyU36RU9QjI7qql8Id9PYgCO348rNndE2TPMAdrh0bqZzQ+F0z8jZoAOhJd9f4H5pFPTKiq3qZK995DUcmbEAQ2iRvyipuK3NLnK/XV59G1wwAHBAN1hBVj0TLRwqXDtqGH742JRLxqclJyQPcoxSRIXNmnLMoJJg1A+AzoruTTh8T9sjILzTFpYO2IQjtq9uaxb1a89yJPW7pH7NmAPxDdHDw8IApmsso75F59swzODJhG4LQvuFo9KJ0oG06qy5lmuiawawZAJ84d4Ofaq8+vRcctNMjgxP0rUAQtkR+fz0h5JWfNtc1IzpXCwA9Q5RqY0f4U6g2Nuv0yFy6cB49Mq1AELYkFArJP4iJBpCKumYWjQB3EQkAvUGSaleftTNHRtc0+ZQPqAtB2Kq6Rynmlg411TXDXUQCQG84d6OPm2ovf54/c6P+HBkURVuGIHRA3S/EZrtmuEcvAKDbvXXnMe40taPhiujgoLyfHD0yjkAQOqDuUQobXTOr4o1xAOhGkqKoqFNUftcSemScgp+2zrjyuuxWCtJk1wwRN5UBQJcSbfWdHd/nHhwUHbFg0CPjFAShM4ajUfl+tahr5uoZftcMhnED9JKlTIC71RcZMs99XnA1jbSHHD0yDkIQOqburRRzS4dqC56SG5rmBIMnAKC7bJWUlwU1nrkzu6KDg9zdRAZFUQchCB0TCoXqzprhfsSbGd8/GuYURorbysz1vpv3AmgiBehSWyXl5r3A6Tf4naKiomjdg4PokXGWYpqcDyNg26kvfDGZSksemDuzW3tmNpNXJ6/2S/5WbKQyoZUndP7NLADQUVbz6mJGXTQCknEwhwfMu5d3uMvBmbf7JEcmgsHgh7/4OXYHHYQgdNh6LvfZzz0leSA4aN69vFO7Lzh761Ajm4KRIXNSL0u6bACgLTY2lXRWXcwE0llVcv6dufbc7oTO+S5eygTOXpcdmfibv/4r3DLhLASh8y7/ybd/8MO/lTxw+li59uBEcVuZuNrf+KZgcNCky8TYSAXLRIB2WcoE0lk1nVUli79aJ7TyW1/lnJ7aKiljV/rldy3d+sn7dt4oiCEInVcoFD7zuaeKxaLkmZsv7Ywdqd4bWDQCM2/buYwJy0QAL91dU9NZNZ0NyPtZRCRF0XM3+uR3Lf3XL34+HI3aeFGQQBC64p13b3zjm9+SPBAZMtOXS7V/fvZ6X4uzRtkykduAAwD2rObVdFZNr6kNVj4lREXRu2vq6TdkvQIXL5y/JL3xBuxBELqlbtfMuRN7L9ecHypuK6ff7HNkrAxqpwAtcjD8qMMD5jNPlblzZLZKyok52eZINBL58D/vtf4eoBaC0C0rhvHbv/v78mfSl0vciCpuK4sZ9bYRSGdVR85ORIbM2EglNoJQBKjD8fAjhBwNVya0yoRe1sR1mldvHbombZe79f4/48iESxCELqrbNRMbqfzTizvyf2TRCKTX1EVDNnKwKQhFgCp0zy+TdzL82DfahFbhbgdWvQF5UfRrLzyPE/TuQRC6qFAo/Nbv/F5uY0PyzOypvbPj/MkyVTY2FbpGdPDCQoQi+NNWSWGtnvYaXkTGjlQm9HJspCJZ/NW+GXlRFAcH3YYgdNf8wsJXnvsjyQOiY4Vyji8TyaM9xdiRivZkpbajFaDb0ZonTT5nhxceDVfo9w63BaauukXRf3j7705OTdl9d1AfgtB1dbtmJvTytef49zHVRc/wOribyNCPtLT7FItF6EYbm4o1/Jz9x2k1hTaj1a18StQtik5NTv7o76/b/vehEQhC19WdNUMIOXdib0Jr9cBDJq8uGoF0Vr275vz3vPbkQS5isQgdixY8M3klnQ1k8opTu32Ms1sJWyXltqHOLR1CUbTtEIRe+Ivvff8vv/f9uo85eC6ejXpy44JfLVw5Gja1MIqo0GZ0zZfJK5m8mrnvcMGTcnwfnTbmyMeQMpim5g0EoUc+85vH5F0zVg4eASxuK6wX3KVb7625qIXNZvc7ARpHY29j0601H0X3/GgJxJHwa3YMKYVpap5BEHokmUyd+uKXbPxFLXzQhNb62ouGIt0vcbx8yrA6qhY2sb8IrdgqKZm8ks6quYfqal5paphns8aOfBx+rez5Mawx9batprZgMPgfH/wbpql5A0HonT/4w68u3L5t+687PimGDUvM5BVXrzykbxhLRpCjsUcrnJm86t6Cj4oMmdaVn1P/bFOVT4nXX/szXEDvGQShdxoZxt0g+q3r4JRtukxcdaGzvFZw0NTCZmTIjD5+UHrCqtGHVvNqsUToao9WDj140bEjFe3JgzNCDn7V0dsHbc/groWiqMcQhJ568wc//ONv/6mz/yabXuHUlG1rBdXtxSKjhSuRx00tXIkMmdHHTSwce8nGprKxqWTu0+QLFLeJq0VOq6OPSvRNnXBvhBuT2CgURb2HIPRa3WOFtrET8c5ePZHJH2zPZO67uLPIRXdrtHAlOEDoR3isHTsZrW0Wt9l/1dxDxe0CQxWWfM4WPCk3JrHVQlHUewhCr60YxqkvfMmRAqmES6FI2pqLFF0y0soqDUhCCE5xeImu8GjI0cDzcpFXxdXkIy1fPdgsHJ9vCwRhe6wYxjvv3pifX2j8TIVt7oUi+WQuelZHFdHCleAgTcoKIYT+WMQ60gYadYQQWs+kaUcI8SYMJA4PmFrYjI1Uoo8fHNpx/CXYPJpmL523LRgMJuLx6ZNTiUQcFdG2QBC22XouNz+/sJxMtdJQ2hR2OtiNVRRdKNAzXvQHiuMvYRuNSUJIbKT86E9M2ijvq7BkIUdrmPQP09kAIaSNCzuRsSMVWgCIjZTZ/y/H3V1TM/fV9Jpbp/K5dE07eXLq5NTkqK5784oggiDsFIVCIZlK/Wx+IZlMebBMpFj3qXsH/uiSMffwoB/V4x0jG6zlNS388ZEyVoZl2h6ftPHSyrpiY8s40pEJx2WNPfoLl17I1TGkEtFIJJGIT5+cSsTjGJzWORCEnWjFMJLJ1PzCgkttNVyeDRTN5NWNhx8fFGvLRqOr2NLTKW6fqGsL+jGC9UO5t9qj2Nl8V0fScNHK5/FE/OTJKVQ+OxOCsKPRZeJyMuXNbqIVHZxG61GO7yxWoQXVTF6lJzeKJdJRNVVo0dFHfb/BARIbKQcHiRt7e1VY8rk3hlQuEY8lEglUPrsCgrBrrOdyyWTqZ/MLyVTK7abTWvQYFuvQ8+AV6Q7WxqZCD1zTHsXOr6z6GV3ksZ5e9mtvXp0N4E5nA94f26Bo+B2PxxOJuPevDrYhCLvSimHML9xOJpNe1k6trFPTPD66wEIx91AtlkjmvkoI6b36aiej/8fpCs/7wGPurqm5hzT8XJ/HJoHw6wEIwq6XTKaWU6k2hiLpmEHb9EAbaxJhPxwRk02hRxQIISzeaJ9tW9KOWc3TGaQK22Nu1zshCL+egyDsKZ0QihRdMnbUNFF2WsDaTsk6Bn1Sd2XLd5ZqtFeFEOLN1l2DOir2iKXhZVTXEX69B0HYszonFClrNAYHiDcbjfbQ6iv9tTU1qdpuey9DlC3XrKrWarRcyX7rxrwVB9GuFlru9ngSqRw96nA8kdB1DQ0vvQ1B6Av0PMZyMmUYhsfdpxL0mEFspIxRoj5hzTzWJNzuN/UJtOY5qms45+crCELfod2nNBqNTKbdb6caS0fyaCmDOaLdiO7L0pzrqHVeFV3TdF3Dss/nEIR+RyuoKytGRy0Wa9FQtAYkrmpqO1oTZpuvdE5bpy3yqkQjEV3XsdsHVghC+Nh6LmcYxoqRSSaTK0bG+9OKNtDefdbxQQeUdPgeZBehxUxSM327i4bd0OQbHdWPx+O6rqHgCbUQhCDUjblYhcYksUwN9eegbS4WcsSyjKOrOtLxCzsJWu0c1fVRXUfyQSMQhNAoloudX0dtlrWvkt1NQSypyX7bycXYquOS1iSjnSmPft01i7lGBIPBUV3TdX1U14ejUVQ7wQYEIdhUKBQMI7NiGCuGkcvlOuSQhsfYipPLmqk2sMVZrXaNEOsEiXgsGo0ODw+j1AlOQRCCY9Zzudx6jrbe5HK5DmxJha5jjb3ocBS3N4AbEITgIhaN6+vrvl01QoOikcjwcFTX9eFodFTXEXvgGQQheIoWVNdzufVcLplMFgpFLBx9iGVeKBQ6Ho8HQ0Gc4YM2QhBC+1nTcWXFKBYLXdqkCrUS8RghJJFIhILBUV1H5kEHQhBC51oxjGKhuGIYhWKRBuT6eq6XulV7Bl3hBYOh0VGdEHI8HieEoIETugWCELoP3XokhCynUoQQmpGosrqNru1o6wp5lHbo24QegCCEXkPXkeRRTBJCksnkwS/QrSNAl3SEELaqo5VMQgiaVqDnIQjBj1hYFoqFFeNgHUlbW9kz3Z6aLNuIJd6IJeEI1nMAhBAEIUCDkslU1Z/QzUvuw7Ra2+IrWtOrFq1MWiHVAOxBEAIAgK9161xdAAAARyAIAQDA1xCEAADgawhCAADwNQQhAAD4GoIQAAB8DUEIAAC+hiAEAABfQxACAICvIQgBAMDXEIQAAOBr/w9J/J/xMgljwgAAAABJRU5ErkJggg==",
    alt = "Logo Universitas Padjadjaran"
  ),
  div(
    class = "brand-text",
    div(class = "brand-project", "Project-Based Learning · S2 Statistika Terapan 2026"),
    div(class = "brand-title", "Simulasi dan Resampling — Ridge regression")
  )
)


# Academic Premium v5
# Direct <style> tag: intentionally independent from tags$head nesting.
finishing_css_v5 <- tags$style(HTML("
  /* ---------- Unpad identity ---------- */
  .project-brand {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
  }

  .brand-logo {
    width: 64px !important;
    height: 64px !important;
    object-fit: contain !important;
    flex: 0 0 64px !important;
    border-radius: 12px !important;
    padding: 3px !important;
    background: #ffffff !important;
    box-shadow: 0 4px 14px rgba(36,52,71,.14) !important;
  }

  /* ---------- Control panels ---------- */
  .control-card {
    border: 1px solid #cbdde6 !important;
    border-top: 4px solid #4f8aa8 !important;
    background: #fafdfe !important;
    box-shadow: 0 5px 16px rgba(36,52,71,.08) !important;
  }

  .control-card .card-header {
    background: #eaf3f7 !important;
    color: #245d7d !important;
    border-bottom: 1px solid #cddfe7 !important;
    font-weight: 750 !important;
  }

  /* ---------- KPI cards: explicit semantic classes ---------- */
  .kpi-row .kpi-card {
    min-height: 112px !important;
    border: 1px solid #d4dee7 !important;
    box-shadow: 0 4px 14px rgba(36,52,71,.07) !important;
  }

  .kpi-row .kpi-observasi {
    background: #f2f7fb !important;
    border-top: 5px solid #2a6f97 !important;
  }

  .kpi-row .kpi-vif {
    background: #fff8e9 !important;
    border-top: 5px solid #e0a458 !important;
  }

  .kpi-row .kpi-lambda-optimal {
    background: #eaf9f8 !important;
    border-top: 5px solid #00a6a6 !important;
  }

  .kpi-row .kpi-lambda-selected {
    background: #f1f7fb !important;
    border-top: 5px solid #2a6f97 !important;
  }

  .kpi-row .kpi-mse-ols {
    background: #fff1ed !important;
    border-top: 5px solid #e76f51 !important;
  }

  .kpi-row .kpi-mse-ridge {
    background: #eaf9f8 !important;
    border-top: 5px solid #00a6a6 !important;
  }

  .kpi-row .kpi-mse-ols .kpi-number {
    color: #c9573d !important;
    font-weight: 750 !important;
  }

  .kpi-row .kpi-mse-ridge .kpi-number,
  .kpi-row .kpi-lambda-optimal .kpi-number {
    color: #008f8f !important;
    font-weight: 750 !important;
  }

  .kpi-row .kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(36,52,71,.12) !important;
  }

  /* ---------- VIF panel ---------- */
  .vif-card {
    background: #f7fbfd !important;
    border: 1px solid #c7dce8 !important;
    border-top: 5px solid #2a6f97 !important;
    box-shadow: 0 5px 16px rgba(42,111,151,.08) !important;
  }

  .vif-card .card-header {
    background: #e7f1f7 !important;
    color: #245d7d !important;
    border-bottom: 1px solid #c7dce8 !important;
    font-weight: 800 !important;
  }

  /* ---------- Beta plot frame only: coral/teal Plotly colors untouched ---------- */
  .beta-card {
    background: #fbffff !important;
    border: 1px solid #c4e2e2 !important;
    border-top: 5px solid #00a6a6 !important;
    box-shadow: 0 5px 16px rgba(0,166,166,.08) !important;
  }

  .beta-card .card-header {
    background: #e7f8f7 !important;
    color: #007f80 !important;
    border-bottom: 1px solid #c4e2e2 !important;
    font-weight: 800 !important;
  }

  /* ---------- Temuan utama ---------- */
  .insight-card {
    background: #eaf9f8 !important;
    border: 1px solid #9ed4d3 !important;
    border-left: 7px solid #00a6a6 !important;
    box-shadow: 0 7px 20px rgba(0,166,166,.11) !important;
  }

  .insight-card .card-header {
    background: #d8f1ef !important;
    color: #006f70 !important;
    border-bottom: 1px solid #b7dfdd !important;
    font-weight: 800 !important;
  }

  /* ---------- Coefficient table ---------- */
  .coef-table-card {
    border: 1px solid #cbd8e2 !important;
    border-top: 5px solid #2a6f97 !important;
    box-shadow: 0 5px 16px rgba(42,111,151,.07) !important;
  }

  .coef-table-card .card-header {
    background: #eaf2f7 !important;
    color: #243447 !important;
    border-bottom: 1px solid #cbd8e2 !important;
    font-weight: 800 !important;
  }

  .coef-table-card table.dataTable thead th {
    background: #dfeef4 !important;
    color: #243447 !important;
    border-bottom: 2px solid #82afbf !important;
  }

  /* ---------- Interpretation / conclusion ---------- */
  .interpretation-card {
    background: #fbfcfd !important;
    border: 1px solid #d0d9e1 !important;
    border-top: 5px solid #7b8b99 !important;
    box-shadow: 0 4px 14px rgba(36,52,71,.06) !important;
  }

  .interpretation-card .card-header {
    background: #eef2f5 !important;
    color: #334155 !important;
    border-bottom: 1px solid #d5dde4 !important;
    font-weight: 800 !important;
  }
"))

ui <- page_fluid(
  dashboard_css,
  project_css,
  finishing_css,
  finishing_css_v3,
  finishing_css_v4,
  finishing_css_v5,
  finishing_css_v6,
  finishing_css_v7,
  finishing_css_v8,
  finishing_css_v9,
  brand_header,
  theme = bs_theme(version = 5, bootswatch = "flatly",
                   base_font = font_google("Noto Sans"),
                   heading_font = font_google("Noto Sans")),

  div(
    class = "dashboard-tab-nav",
    navset_card_tab(
      id = "main_tabs",
    nav_panel("1. Analisis Ridge",
    div(class = "page-wrap",
      layout_columns(
        card(class = "control-card", card_header("Pilihan data"),
             selectInput("wine1", "Jenis wine:", c("Red Wine", "White Wine"), "Red Wine")),
        card(class = "control-card", card_header("Lambda interaktif"),
             sliderInput("lambda1", "log10(lambda):", -4, 3, log10(red$lambda_min), 0.01, width = "100%"),
             div(class = "helper-text", "Geser pada skala log10(lambda). Nilai aktual ditampilkan pada panel Lambda Terpilih.")),
        col_widths = c(4, 8)
      ),
      h2("Data dan Performa Model"),
      layout_columns(
        class = "kpi-row",
        div(class = "kpi-card kpi-observasi",
            div(class = "kpi-label", "Observasi"),
            div(class = "kpi-number", textOutput("n_total"))),
        div(class = "kpi-card kpi-vif",
            div(class = "kpi-label", "Max VIF"),
            div(class = "kpi-number", textOutput("max_vif"))),
        actionButton(
          "use_lambda_optimal",
          uiOutput("lambda_optimal_box"),
          class = "kpi-card kpi-lambda-optimal kpi-optimal-btn"
        ),
        div(class = "kpi-card kpi-lambda-selected",
            div(class = "kpi-label", "Lambda Terpilih"),
            div(class = "kpi-number", textOutput("lambda_card"))),
        div(class = "kpi-card kpi-mse-ols",
            div(class = "kpi-label", "MSE OLS"),
            div(class = "kpi-number", textOutput("mse_ols"))),
        div(class = "kpi-card kpi-mse-ridge",
            div(class = "kpi-label", "MSE Ridge"),
            div(class = "kpi-number", textOutput("mse_lambda"))),
        col_widths = c(2, 2, 2, 2, 2, 2)
      ),
      br(),
      layout_columns(
        card(class = "analysis-card vif-card", card_header("Nilai VIF per prediktor"), DTOutput("vif_table")),
        card(class = "analysis-card beta-card", card_header("Efek lambda terhadap koefisien"), plotlyOutput("lambda_coef_plot", height = "500px")),
        col_widths = c(4, 8)
      ),
      br(),
      card(class = "insight-card", card_header("Temuan utama"), uiOutput("main_finding")),
      br(),
      card(class = "coef-table-card", card_header("Koefisien pada lambda terpilih"),
           DTOutput("lambda_coef_table", width = "100%", height = "auto")),
      br(),
      card(class = "interpretation-card", card_header("Cara membaca hasil"), uiOutput("tab1_interpretation")),
      br(),
      card(class = "interpretation-card", card_header("Kesimpulan"), uiOutput("tab1_conclusion"))
    )
  ),

  nav_panel("2. Simulasi Resampling",
    div(class = "page-wrap",
      layout_columns(
        card(class = "tab2-vif-card", card_header("Pengaturan simulasi"),
             selectInput("wine2", "Jenis wine:", c("Red Wine", "White Wine"), "Red Wine"),
             div(class = "helper-text", "Setiap bootstrap menggunakan 5-fold cross-validation.")),
        card(class = "tab2-vif-card", card_header("Pilihan bootstrap"),
             radioButtons("boot_B", "Pilih jumlah bootstrap:",
                          c("B = 500" = 500, "B = 1000" = 1000),
                          selected = 1000, inline = TRUE),
             div(class = "helper-text",
                 "Pilihan 500 atau 1000 digunakan sekaligus sebagai jumlah resampling dan bootstrap terpilih.")),
        col_widths = c(5, 7)
      ),
      h2("Simulasi dan Resampling Ridge Regression"),
      layout_columns(
        div(class = "tab2-kpi-card",
            value_box(title = "Bootstrap ke-", value = textOutput("boot_kpi"))),
        div(class = "tab2-kpi-card",
            value_box(title = "Rata-rata |Beta Ridge|", value = textOutput("boot_beta_kpi"))),
        div(class = "tab2-kpi-card",
            value_box(title = "RMSE", value = textOutput("boot_rmse_kpi"))),
        div(class = "tab2-kpi-card",
            value_box(title = "Lambda* bootstrap (5-fold CV)", value = textOutput("boot_lambda_kpi"))),
        col_widths = c(3, 3, 3, 3)
      ),
      br(),
      card(class = "tab2-vif-card", card_header("Perbandingan koefisien OLS vs Ridge dengan 95% Bootstrap CI Ridge"),
           plotlyOutput("coef_ci_plot", height = "520px")),
      br(),
      card(class = "tab2-vif-card", card_header("Tabel Ringkasan beta Ridge seluruh prediktor"),
           DTOutput("boot_summary_table", width = "100%", height = "auto"),
           style = "min-height: 520px; width: 100%;"),
      br(),
      card(class = "tab2-vif-card", card_header("Interpretasi simulasi"), uiOutput("tab2_interpretation"))
    )
  )
  )
  )
)

# ============================================================
# 5. SERVER
# ============================================================
# Format lambda sebagai desimal biasa dengan koma desimal Indonesia.
# Contoh: 3.92e-02 -> 0,0392
lambda_decimal <- function(x) {
  x <- as.numeric(x)
  out <- format(x, scientific = FALSE, digits = 5, trim = TRUE)
  out <- sub('\\.$', '', out)
  sub('\\.', ',', out, fixed = TRUE)
}

server <- function(input, output, session) {
  d1 <- reactive(wine_data[[input$wine1]])
  d2 <- reactive(wine_data[[input$wine2]])

  # ---------------- TAB 1 ----------------
  output$n_total <- renderText(comma(d1()$n_total))
  output$max_vif <- renderText(number(max(d1()$vif_table$VIF), accuracy = 0.01))

  output$lambda_optimal_box <- renderUI({
    div(
      span(class = "lambda-optimal-label", "Lambda Optimal (10-fold CV)"),
      span(class = "lambda-optimal-value",
           lambda_decimal(d1()$lambda_min))
    )
  })
  # Lambda yang benar-benar dipakai oleh seluruh komponen Tab 1.
  # Disimpan sebagai reactiveVal agar klik Lambda Optimal langsung
  # mengubah lambda terpilih, MSE, grafik, dan tabel secara bersamaan.
  selected_lambda1 <- reactiveVal(red$lambda_min)

  observeEvent(input$lambda1, {
    req(input$lambda1)
    selected_lambda1(10 ^ as.numeric(input$lambda1))
  }, ignoreInit = TRUE)

  observeEvent(input$wine1, {
    model <- d1()
    selected_lambda1(model$lambda_min)
    lam <- model$ridge_path$lambda
    updateSliderInput(
      session, "lambda1",
      min = floor(log10(min(lam))),
      max = ceiling(log10(max(lam))),
      value = log10(model$lambda_min),
      step = 0.01
    )
  }, ignoreInit = TRUE)

  observeEvent(input$use_lambda_optimal, {
    model <- d1()
    lam_opt <- model$lambda_min
    # Update reactive state lebih dulu, lalu sinkronkan slider.
    selected_lambda1(lam_opt)
    updateSliderInput(
      session, "lambda1",
      value = log10(lam_opt)
    )
  })

  # Sinkronisasi awal dan setiap perubahan dataset.
  observeEvent(d1(), {
    model <- d1()
    lam <- model$ridge_path$lambda
    selected_lambda1(model$lambda_min)
    updateSliderInput(
      session, "lambda1",
      min = floor(log10(min(lam))),
      max = ceiling(log10(max(lam))),
      value = log10(model$lambda_min),
      step = 0.01
    )
  }, ignoreInit = FALSE)

  output$lambda_card <- renderText({
    lambda_decimal(selected_lambda1())
  })

  # MSE OLS pada test set.
  # Diambil langsung dari performance hasil pipeline utama
  # agar nilainya konsisten dengan R/RMarkdown.
  output$mse_ols <- renderText({
    model <- d1()
    mse <- model$performance$MSE_Test[
      model$performance$Model == "OLS"
    ]
    number(mse, accuracy = 0.0001)
  })

  output$mse_lambda <- renderText({
    model <- d1()
    lam <- selected_lambda1()
    req(is.finite(lam), lam > 0)
    ridge_at_lambda <- glmnet::glmnet(
      x = model$X_train, y = model$Y_train, alpha = 0,
      lambda = lam, standardize = FALSE
    )
    pred <- as.numeric(predict(ridge_at_lambda, newx = model$X_test))
    number(mean((model$Y_test - pred)^2), accuracy = 0.0001)
  })

  # Helper: ambil koefisien OLS dan Ridge hanya untuk prediktor.
  # Intercept sengaja dikeluarkan agar jumlah baris selalu sama
  # dengan jumlah prediktor (11 untuk data Wine Quality).
  get_lambda_compare <- function(model, lambda) {
    vars <- model$predictor_names

    ols_all <- coef(model$ols)
    ols_all <- setNames(as.numeric(ols_all), names(coef(model$ols)))

    ridge_mat <- as.matrix(coef(model$ridge_path, s = lambda))
    ridge_all <- setNames(as.numeric(ridge_mat[, 1]), rownames(ridge_mat))

    out <- data.frame(
      Prediktor = vars,
      Beta_OLS = unname(ols_all[vars]),
      Beta_Ridge = unname(ridge_all[vars]),
      stringsAsFactors = FALSE
    )

    # Validasi sederhana agar error tidak tersembunyi.
    if (nrow(out) != length(vars)) {
      stop("Jumlah koefisien prediktor tidak sama dengan jumlah prediktor.")
    }

    out
  }

  output$vif_table <- renderDT({
    d1()$vif_table |>
      arrange(desc(VIF)) |>
      mutate(VIF = round(VIF, 4))
  }, options = list(pageLength = 12, dom = "t", ordering = FALSE), rownames = FALSE)

  output$lambda_coef_plot <- renderPlotly({
    model <- d1()
    lam <- selected_lambda1()
    cmp <- get_lambda_compare(model, lam)

    df <- bind_rows(
      data.frame(Variable = cmp$Prediktor, Model = "OLS", Coefficient = cmp$Beta_OLS),
      data.frame(Variable = cmp$Prediktor, Model = "Ridge", Coefficient = cmp$Beta_Ridge)
    )

    p <- ggplot(df, aes(x = Variable, y = Coefficient, fill = Model,
                        text = paste0("Prediktor: ", Variable,
                                      "<br>Model: ", Model,
                                      "<br>Beta: ", round(Coefficient, 4)))) +
      geom_col(position = position_dodge(width = .78), width = .68) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#98A2B3") +
      scale_fill_manual(values = c("OLS" = "#E76F51", "Ridge" = "#00A6A6")) +
      labs(title = paste0("Beta OLS vs Ridge pada lambda = ",
                          lambda_decimal(lam)),
           x = NULL, y = "Nilai beta", fill = "Model") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank())

    ggplotly(p, tooltip = "text") |>
      layout(margin = list(b = 110, l = 55, r = 25, t = 75))
  })

  output$lambda_coef_table <- renderDT({
    model <- d1()
    lam <- selected_lambda1()
    cmp <- get_lambda_compare(model, lam)

    data.frame(
      Prediktor = cmp$Prediktor,
      Beta_OLS = round(cmp$Beta_OLS, 4),
      Beta_Ridge = round(cmp$Beta_Ridge, 4),
      Selisih = round(cmp$Beta_Ridge - cmp$Beta_OLS, 4),
      Rasio_Ridge_OLS = round(ifelse(abs(cmp$Beta_OLS) > 1e-12,
                                     cmp$Beta_Ridge / cmp$Beta_OLS,
                                     NA_real_), 4),
      stringsAsFactors = FALSE
    )
  }, options = list(pageLength = 12, dom = "t", ordering = FALSE), rownames = FALSE)

  # Ringkasan dinamis OLS vs Ridge pada lambda yang sedang dipilih.
  output$main_finding <- renderUI({
    model <- d1()
    lam <- selected_lambda1()

    mse_ols <- model$performance$MSE_Test[
      model$performance$Model == "OLS"
    ]

    ridge_at_lambda <- glmnet::glmnet(
      x = model$X_train, y = model$Y_train, alpha = 0,
      lambda = lam, standardize = FALSE
    )
    pred_ridge <- as.numeric(predict(ridge_at_lambda, newx = model$X_test))
    mse_ridge <- mean((model$Y_test - pred_ridge)^2)

    delta_mse <- mse_ridge - mse_ols
    pct_change <- if (abs(mse_ols) > 1e-12) 100 * delta_mse / mse_ols else NA_real_
    direction <- if (delta_mse < 0) "lebih rendah" else if (delta_mse > 0) "lebih tinggi" else "sama"

    div(class = "interpretation-text",
        strong("Pada lambda terpilih = "), lambda_decimal(lam),
        ". MSE Ridge = ", number(mse_ridge, accuracy = 0.0001),
        ", sedangkan MSE OLS = ", number(mse_ols, accuracy = 0.0001),
        ". Dengan demikian, MSE Ridge ", direction, " daripada OLS sebesar ",
        number(abs(delta_mse), accuracy = 0.0001),
        if (is.finite(pct_change)) paste0(" (", ifelse(delta_mse <= 0, "", "+"), round(pct_change, 2), "%).") else ".",
        tags$br(), tags$br(),
        strong("Interpretasi: "),
        "ringkasan ini menunjukkan perubahan kesalahan prediksi pada data uji ketika OLS dibandingkan dengan Ridge pada lambda yang sedang dipilih. Nilai akan berubah ketika lambda atau jenis wine diubah.")
  })

  output$tab1_interpretation <- renderUI({
    model <- d1()
    lam <- selected_lambda1()
    cmp <- get_lambda_compare(model, lam)

    shrink <- mean(abs(cmp$Beta_Ridge), na.rm = TRUE) /
      mean(abs(cmp$Beta_OLS), na.rm = TRUE)

    div(class = "interpretation-text",
        strong("VIF: "),
        paste0("nilai VIF maksimum = ", round(max(model$vif_table$VIF), 2),
               ". Tabel menampilkan VIF tiap prediktor."),
        tags$br(), tags$br(),
        strong("Lambda: "),
        paste0("pada lambda = ", format(lam, scientific = TRUE, digits = 3),
               ", rata-rata |beta Ridge| sekitar ", round(shrink * 100, 1),
               "% dari rata-rata |beta OLS|."),
        tags$br(), tags$br(),
        "Geser lambda untuk melihat perubahan beta Ridge, sementara beta OLS tetap sebagai pembanding.",
        tags$br(), tags$br(),
        strong("Intuisi: "),
        "Semakin besar λ, semakin kuat penyusutan (shrinkage) koefisien Ridge menuju 0, sehingga model menjadi lebih sederhana; λ dipilih untuk menyeimbangkan kompleksitas model dan kesalahan prediksi.")
  })

  output$tab1_conclusion <- renderUI({
    model <- d1()
    lam <- selected_lambda1()
    mse_ols <- model$performance$MSE_Test[model$performance$Model == "OLS"]
    ridge_at_lambda <- glmnet::glmnet(
      x = model$X_train, y = model$Y_train, alpha = 0,
      lambda = lam, standardize = FALSE
    )
    pred_ridge <- as.numeric(predict(ridge_at_lambda, newx = model$X_test))
    mse_ridge <- mean((model$Y_test - pred_ridge)^2)
    max_vif <- max(model$vif_table$VIF)

    div(class = "interpretation-text",
        tags$ol(
          tags$li(paste0("Nilai VIF maksimum sebesar ", round(max_vif, 2), ".")),
          tags$li(paste0("Lambda Ridge yang sedang dipilih adalah ", lambda_decimal(lam), ".")),
          tags$li(paste0("Pada data uji, MSE OLS = ", number(mse_ols, accuracy = 0.0001),
                         " dan MSE Ridge = ", number(mse_ridge, accuracy = 0.0001), ".")),
          tags$li("Ridge melakukan shrinkage terhadap koefisien, sehingga perubahan koefisien dapat diamati melalui grafik dan tabel."),
          tags$li("Stabilitas estimasi koefisien selanjutnya dievaluasi melalui bootstrap pada Tab 2.")
        )
    )
  })

  # ---------------- TAB 2 ----------------
  # Pilihan B sekaligus menentukan jumlah bootstrap yang diringkas
  # dan bootstrap terpilih yang ditampilkan. Hanya tersedia 500 dan 1000.
  boot_B <- reactive({
    val <- input$boot_B
    if (is.null(val) || !nzchar(val)) 1000L else as.integer(val)
  })

  # Bootstrap terpilih selalu sama dengan jumlah B yang dipilih:
  # B=500 -> bootstrap ke-500
  # B=1000 -> bootstrap ke-1000
  boot_i <- reactive({
    boot_B()
  })

  output$boot_kpi <- renderText(boot_i())

  output$boot_beta_kpi <- renderText({
    m <- d2()
    i <- boot_i()
    vars <- intersect(m$predictor_names, colnames(m$boot_beta_ridge))
    beta <- as.numeric(m$boot_beta_ridge[i, vars, drop = TRUE])
    format(mean(abs(beta), na.rm = TRUE), nsmall = 4)
  })

  output$boot_rmse_kpi <- renderText({
    m <- d2()
    i <- boot_i()
    format(m$boot_rmse[i], nsmall = 4, digits = 4)
  })

  output$boot_lambda_kpi <- renderText({
    m <- d2()
    i <- boot_i()
    req(!is.null(m$boot_lambda), length(m$boot_lambda) >= i)
    lam <- as.numeric(m$boot_lambda[i])
    req(length(lam) == 1, is.finite(lam), lam > 0)
    lambda_decimal(lam)
  })

  output$boot_summary_table <- renderDT({
    m <- d2()
    Bsel <- boot_B()
    vars <- intersect(m$predictor_names, colnames(m$boot_beta_ridge))
    req(length(vars) > 0, nrow(m$boot_beta_ridge) >= Bsel)

    # Ringkasan benar-benar mengikuti jumlah bootstrap yang dipilih.
    mat <- m$boot_beta_ridge[seq_len(Bsel), vars, drop = FALSE]
    mean_beta <- colMeans(mat, na.rm = TRUE)
    se_beta <- apply(mat, 2, sd, na.rm = TRUE)
    ci_low <- apply(mat, 2, quantile, probs = 0.025, na.rm = TRUE, names = FALSE)
    ci_high <- apply(mat, 2, quantile, probs = 0.975, na.rm = TRUE, names = FALSE)

    out <- data.frame(
      Prediktor = vars,
      Mean = round(as.numeric(mean_beta), 4),
      SE = round(as.numeric(se_beta), 4),
      `CI 95% Lower` = round(as.numeric(ci_low), 4),
      `CI 95% Upper` = round(as.numeric(ci_high), 4),
      `Status Informatif` = ifelse(ci_low > 0 | ci_high < 0, "Ya", "Tidak"),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )

    datatable(
      out,
      rownames = FALSE,
      width = "100%",
      class = "compact stripe hover",
      options = list(
        pageLength = 11,
        lengthChange = FALSE,
        searching = FALSE,
        ordering = FALSE,
        info = FALSE,
        dom = "t",
        autoWidth = FALSE,
        columnDefs = list(
          list(width = "24%", targets = 0),
          list(width = "13%", targets = 1:4),
          list(width = "24%", targets = 5)
        )
      )
    )
  })

  output$coef_ci_plot <- renderPlotly({
    m <- d2()
    Bsel <- boot_B()
    vars <- intersect(m$predictor_names, colnames(m$boot_beta_ridge))
    req(length(vars) > 0, nrow(m$boot_beta_ridge) >= Bsel)

    mat <- m$boot_beta_ridge[seq_len(Bsel), vars, drop = FALSE]
    ci_low <- apply(mat, 2, quantile, probs = 0.025, na.rm = TRUE, names = FALSE)
    ci_high <- apply(mat, 2, quantile, probs = 0.975, na.rm = TRUE, names = FALSE)

    bo_all <- coef(m$ols)
    bo_all <- setNames(as.numeric(bo_all), names(coef(m$ols)))
    bo <- unname(bo_all[vars])

    # Ridge yang dibandingkan adalah model final pada lambda optimal 10-fold CV.
    br_all <- as.matrix(coef(m$ridge_final))
    br_all <- setNames(as.numeric(br_all[, 1]), rownames(br_all))
    br <- unname(br_all[vars])

    df <- bind_rows(
      data.frame(Variable = vars, Model = "OLS", Coefficient = bo),
      data.frame(Variable = vars, Model = "Ridge", Coefficient = br)
    )
    ci_df <- data.frame(Variable = vars, Lower = ci_low, Upper = ci_high)

    p <- ggplot(df, aes(x = Variable, y = Coefficient, fill = Model,
                        text = paste0(Variable, "<br>", Model,
                                      ": ", format(Coefficient, digits = 5)))) +
      geom_col(position = position_dodge(width = .78), width = .68) +
      geom_errorbar(data = ci_df,
                    aes(x = Variable, ymin = Lower, ymax = Upper),
                    inherit.aes = FALSE, width = .16, color = "#667085", linewidth = .8) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#98A2B3") +
      scale_fill_manual(values = c("OLS" = "#E76F51", "Ridge" = "#00A6A6")) +
      labs(title = paste0("OLS vs Ridge + 95% Bootstrap CI Ridge (B = ", Bsel, ")"),
           subtitle = "CI dihitung dari bootstrap yang dipilih",
           x = NULL, y = "Koefisien", fill = "Model") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank())

    ggplotly(p, tooltip = "text") |>
      layout(margin = list(b = 110, l = 55, r = 25, t = 85))
  })

  output$tab2_interpretation <- renderUI({
    m <- d2()
    vars <- intersect(m$predictor_names, colnames(m$boot_beta_ridge))
    i <- boot_i()
    Bsel <- boot_B()
    current <- as.numeric(m$boot_beta_ridge[i, vars, drop = TRUE])
    mean_abs <- mean(abs(current), na.rm = TRUE)
    lam_i <- as.numeric(m$boot_lambda[i])

    div(class = "interpretation-text",
        paste0("Dengan B = ", Bsel, " dan 5-fold CV, bootstrap ke-", i,
               " memiliki lambda* = ", format(lam_i, scientific = TRUE, digits = 3),
               ", rata-rata |beta Ridge| = ", round(mean_abs, 4),
               " dan RMSE = ", format(m$boot_rmse[i], nsmall = 4, digits = 4), "."),
        tags$br(), tags$br(),
        paste0("Grafik menunjukkan koefisien OLS dan Ridge final beserta 95% CI Ridge berdasarkan B = ", Bsel, "."),
        tags$br(), tags$br(),
        paste0("Ketika pilihan diubah dari B = 500 menjadi B = 1000, seluruh ringkasan bootstrap (Mean, SE, CI, dan Status Informatif) dihitung ulang berdasarkan jumlah replikasi tersebut."),
        tags$br(), tags$br(),
        strong("Intuisi: "),
        "Bootstrap digunakan untuk melihat seberapa stabil koefisien Ridge terhadap variasi sampel; semakin banyak replikasi bootstrap, semakin banyak variasi sampel yang digunakan untuk merangkum Mean, SE, dan CI.")
  })

}

shinyApp(ui,server)
