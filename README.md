# NoctTwin: a personalised digital twin for nocturnal hypoglycemia in type 2 diabetes

## Title and team
- **Project:** NoctTwin: a personalised digital twin for nocturnal hypoglycemia in type 2 diabetes
- **Team:** AlooDUM
- **College:** Arka Jain University, Jamshedpur
- **Members:** Nivesh Kumar · Kriti Kumari · Bhoomi Singh · Abhinav Tiwary
- **Team Leader:** Abhinav Tiwary
- **Competition:** Happiest Health Digital Twin Challenge 2026

---

## Project Title and Problem Statement
**NoctTwin: A Personalised Digital Twin for Nocturnal Hypoglycemia in Type 2 Diabetes**

Nocturnal hypoglycemia is an acute, dangerous complication in the management of Type 2 Diabetes Mellitus (T2DM). As patients sleep, counter-regulatory symptom perception is blunted, leaving severe drops in blood glucose undetected. Clinicians face two difficult bedside decisions each evening:
1. **Risk Stratification at Bedtime (22:00):** Accurately predicting which patient will suffer a persistent, clinically significant nocturnal glucose low (<70 mg/dL between 23:00 and 06:00) using continuous glucose monitoring (CGM) sensor history.
2. **Personalised Trajectory & Countermeasure Evaluation:** Forecasting the expected nocturnal trajectory and simulating what-if preventative adjustments—such as bedside carbohydrate snacks (15g or 30g) or secretagogue dose down-titration—prior to sleep.

Sulfonylurea-treated patients represent a high-risk group due to glucose-independent insulin secretagogue activity; in this study, sulfonylurea results are analyzed as an exploratory subgroup.

---

## Healthcare Use Case
NoctTwin operates as a clinical decision support tool designed for inpatient and bedside settings:
- **Triage Board:** Ranks patients at the 22:00 bedtime anchor by calibrated probability of experiencing a persistent nocturnal low, presenting positive predictive value (PPV) and number needed to warn (NNW) alongside alert thresholds.
- **Personalised Digital Twin Trajectory:** Displays the patient's individual overnight expected trajectory synchronized to bedtime glucose, bounded by calibrated 80% and 90% conformal uncertainty bands.
- **What-If Countermeasure Simulation:** Allows clinicians to simulate the physiological response to a 15g or 30g carbohydrate bedtime snack or a 50% secretagogue dose reduction.
- **Interpretability & Clinical Factors:** Presents patient-specific feature contributions (SHAP values) and verified data points (prior-night events, 2-hour bedtime glucose velocity, nadir trend) to support clinical judgment without opaque recommendations.

---

## AI/ML Model and Framework Details
- **Classification Engine (Group G):** Regularized Logistic Regression ($L_2$ penalty, $C=0.1$) trained on patient-level grouped folds, combining rolling bedside CGM features (anchor glucose, 2h/4h velocity, glucose variance, nadir) with strictly historical prior-night hypoglycemia records. Probabilities are calibrated using inner patient-grouped Platt sigmoid scaling.
- **Physiological Digital Twin (Synchronised Blended Forecaster):**
  $$\text{Forecast}(t) = \text{Patient\_Profile}(t) + \left(G_{\text{bedtime}} - \text{Profile}(\text{22:00})\right) \cdot \exp(-t / \tau)$$
  where $\tau$ is a physiological relaxation constant fitted via patient-grouped cross-validation on training folds ($\tau_{\text{mean}} = 3.305$ hours).
- **Two-Compartment Mechanistic ODE:** Minimal Bergman-derived glucose-insulin dynamics model with meal carbohydrate absorption kinematics, initialized at bedtime glucose and utilized exclusively to compute differential what-if perturbation deltas ($\Delta G$).
- **Uncertainty Quantification:** Split-conformal prediction intervals computed on forecaster residuals across held-out patient splits, providing verified empirical 80% and 90% prediction coverage.

---

## Technical Stack
- **Backend & Data Pipeline:** Python 3.14.2, FastAPI 0.141.1, Uvicorn 0.52.4, NumPy 2.5.2, Pandas 3.0.5, SciPy 1.18.1, Scikit-learn 1.9.0, LightGBM 4.7.0, SHAP 0.52.0, PyArrow 25.0.1, Pytest 9.1.1, Ruff 0.16.5.
- **Frontend Application:** React 18.3.1, TypeScript 5.6.3 (strict), Vite 6.0.1, Tailwind CSS 3.4.15, Apache ECharts 5.5.1 (`echarts-for-react`), Lucide React 0.468.0, Vitest 2.1.8.
- **Architectural Contract:** Shared OpenAPI and TypeScript interfaces (`shared/types.ts`) decoupling frontend and backend with dual-mode operation: active FastAPI connection or full client-side static JSON fallback.

---

## Results

### 6.1 Headline Model Performance (20 Repeated GroupKFold Runs)
Evaluated across 20 repeated 5-fold patient-grouped cross-validation splits with different random seeds on nights with historical data available ($N = 1,081$ nights, 101 strict events):

| Metric | Group B: CGM Only (Mean [Range]) | Group G: CGM + History (Mean [Range]) | Paired Difference ($\Delta$ G - B) | Paired Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **AUROC** | 0.736 [0.707 – 0.753] | **0.832 [0.824 – 0.840]** | **+0.097** [+0.082 – +0.121] | **[+0.051, +0.149]** |
| **AUPRC** | 0.354 [0.252 – 0.392] | **0.476 [0.417 – 0.496]** | **+0.123** [+0.087 – +0.165] | **[+0.047, +0.191]** |
| **Brier Score** | 0.073 [0.071 – 0.078] | **0.065 [0.064 – 0.068]** | **-0.008** [-0.010 – -0.006] | **[-0.014, -0.004]** |

### 6.2 Digital Twin Forecaster Benchmark
Evaluated on nights with historical prior data (1,072 nights at 1-hour horizon; 1,071 nights at 3-hour and 6-hour horizons due to 1 missing sensor reading):

| Forecast Horizon | Model / Baseline | MAE [95% CI] (mg/dL) | RMSE [95% CI] (mg/dL) |
| :--- | :--- | :---: | :---: |
| **1-Hour (23:00)** | **Blended Synchronised Forecaster** | **14.83 [13.54, 16.14]** | **20.33 [18.59, 22.01]** |
| | Persistence Baseline | 18.48 [16.81, 20.06] | 24.28 [22.25, 26.02] |
| | Patient Hist Overnight Profile | 24.62 [22.18, 27.07] | 33.39 [30.16, 36.66] |
| **3-Hour (01:00)** | **Blended Synchronised Forecaster** | **19.35 [17.32, 21.39]** | **27.32 [24.31, 30.12]** |
| | Patient Hist Overnight Profile | 21.69 [19.38, 24.22] | 30.74 [27.60, 33.68] |
| | Persistence Baseline | 31.97 [28.50, 35.54] | 43.07 [38.17, 47.75] |
| **6-Hour (04:00)** | **Blended Synchronised Forecaster** | **16.05 [14.21, 17.91]** | **22.78 [20.03, 25.43]** |
| | Patient Hist Overnight Profile | 17.18 [15.24, 19.13] | 24.03 [21.36, 26.54] |
| | Persistence Baseline | 36.98 [32.38, 42.02] | 50.02 [43.30, 56.71] |

### 6.3 Split-Conformal Empirical Coverage (200 Patient-Level Splits)
Evaluated across 200 random patient splits (half calibration, half test on held-out patients):

| Target Coverage | Mean Empirical Coverage (%) | Range Across Splits (%) | 95% Range Across Splits (%) |
| :--- | :---: | :---: | :---: |
| **80% Nominal Band** | **79.93%** | [70.79% – 86.66%] | [74.17% – 85.82%] |
| **90% Nominal Band** | **89.84%** | [83.39% – 94.13%] | [85.03% – 93.72%] |
| **Conditional (<110 mg/dL) 80%** | **89.13%** | [83.11% – 94.81%] | [84.99% – 93.14%] |
| **Conditional (<110 mg/dL) 90%** | **95.18%** | [91.83% – 98.07%] | [92.45% – 97.21%] |

<details>
<summary>Click to view Cohort Regimen Distribution & Subgroup Breakdown</summary>

Canonical classification counts across the 100 hospitalized patients (1,181 valid nights):
- **SU without Insulin:** 10 patients, 124 nights, 24 strict events (19.4% base rate).
- **SU with Insulin:** 8 patients, 123 nights, 8 strict events (6.5% base rate).
- **Insulin without SU:** 51 patients, 520 nights, 38 strict events (7.3% base rate).
- **Neither SU nor Insulin:** 31 patients, 414 nights, 54 strict events (13.0% base rate).
- **All SU Combined:** 18 patients, 247 nights, 32 strict events across 11 patients (13.0% base rate).
- **All Patients Combined:** 100 patients, 1,181 nights, 124 strict events across 45 patients (10.5% base rate).

Event concentration: the top 10% of patients account for 48.0% of all strict nocturnal events (60 of 124 events). When single outlier patients are excluded, the SU-without-insulin rate falls from 19.4% to 13.5% (excluding Pt 2066). All sulfonylurea findings are exploratory.
</details>

<details>
<summary>Click to view Prior-Night History Ablation & Cold-Start Analysis</summary>

- **History Availability Stratification:**
  - 1 Prior Night ($N=99$ nights, 9 events): Group G AUROC 0.778 vs Group B 0.730 ($\Delta = +0.048$).
  - 2–3 Prior Nights ($N=190$ nights, 15 events): Group G AUROC 0.812 vs Group B 0.741 ($\Delta = +0.071$).
  - 4+ Prior Nights ($N=792$ nights, 77 events): Group G AUROC 0.840 vs Group B 0.760 ($\Delta = +0.080$).
- **Cold-Start Analysis (Night 0, No Prior History, $N=100$ nights, 23 events):**
  - CGM Only (Group B): AUROC 0.741 [0.612, 0.852], AUPRC 0.442 [0.248, 0.651].
  - CGM + Meds (Group C): AUROC 0.748 [0.620, 0.859], AUPRC 0.450 [0.254, 0.659].
  - CGM + Meds + Demographics (Group D): AUROC 0.752 [0.625, 0.864], AUPRC 0.458 [0.261, 0.665].
  - Differences on Night 0 show no statistically measurable benefit from medication or static indicators over CGM alone.
</details>

---

## Limitations
1. **Repeated-Run Variation vs Statistical Uncertainty:** The repeated GroupKFold range (e.g. AUROC 0.824–0.840) reflects sensitivity to how patients are partitioned across cross-validation folds, not true statistical sampling uncertainty; the paired bootstrap confidence interval (+0.097, 95% CI [+0.051, +0.149]) should be used for inference.
2. **Model Selection & Lack of External Validation:** The regularized logistic regression model and regularization hyperparameters were selected on this single dataset without external cohort validation; all performance estimates are exploratory.
3. **Exploratory Sulfonylurea Subgroup:** The cohort includes only 18 SU-treated patients (247 nights, 32 strict events across 11 patients). Subgroup comparisons are underpowered and cannot establish causal risk ratios.
4. **Hospitalized Inpatient Setting:** All records originate from hospitalized inpatients where dietary timing, intravenous infusions, and clinical nursing interventions differ substantially from free-living outpatients.
5. **Simulation Hypothesis for Bedtime Countermeasures:** Carbohydrate absorption kinetics and secretagogue scaling parameters in the ODE what-if simulation are based on literature parameters rather than prospective intervention trial data; outputs are labeled simulation hypotheses.
6. **Night 0 Sensor Warm-Up Discrepancy:** Night 0 event prevalence is 23.0% compared to 9.3% on subsequent nights (1+), reflecting flash sensor calibration and stabilization during the initial 24 hours of wear.
7. **Cardiac & ECG Context:** Hypoglycemia has been associated with QTc prolongation and arrhythmia in the literature; NoctTwin does not predict ECG quantities or Holter waveforms.

---

## Demo Video
Unlisted YouTube video: https://youtu.be/55ZW_UGlgbQ
Updated video walkthrough: https://youtu.be/043O1jUg5GI

---

## Architecture Diagram
The system architecture and end-to-end data pipeline is documented in [docs/architecture.pdf](docs/architecture.pdf).

---

## Presentation
- Presentation Deck (PDF): [docs/AlooDUM_slides.pdf](docs/AlooDUM_slides.pdf)
- Presentation Deck (PowerPoint): [docs/AlooDUM_slides.pptx](docs/AlooDUM_slides.pptx)

---

## Repository Structure
```text
.
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── api/
│   ├── main.py
│   ├── models.py
│   └── routes.py
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── AlooDUM_slides.pdf
│   ├── AlooDUM_slides.pptx
│   └── architecture.pdf
├── reports/
│   └── figures/
├── scripts/
│   ├── evaluate_conformal_splits.py
│   ├── evaluate_group_g_robustness.py
│   ├── evaluate_repeated_groupkfold.py
│   ├── evaluate_twin_forecast.py
│   ├── export_home_traces.py
│   ├── export_ui_data.py
│   ├── generate_displayed_model_figures.py
│   ├── run_demo.py
│   └── run_phase0.py
├── shared/
│   └── types.ts
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   └── twin/
├── tests/
│   └── test_no_leakage.py
└── web/
    ├── eslint.config.js
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── postcss.config.js
    ├── tailwind.config.js
    ├── tsconfig.json
    ├── vite.config.ts
    ├── public/
    └── src/
```

---

## Quick Start and Retraining

### Quick Start
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run unit and zero-leakage test suite
python -m pytest tests/

# 3. Check code style and formatting
python -m ruff check .

# 4. Install web dependencies, run tests, and build frontend
cd web && npm install && npm test && npm run build
cd ..

# 5. Launch the demo application (starts backend and serves UI)
python scripts/run_demo.py
```
The launcher starts the FastAPI service, serves the production web interface, and automatically opens `http://localhost:8000`. Press `Ctrl+C` to terminate.

### Retraining from Raw Data
The repository includes all extracted datasets in `data/processed/` and precomputed assets in `web/public/data/` for immediate execution. If you download the raw Shanghai_T2DM dataset files into `data/raw/`, execute:
```bash
python scripts/run_phase0.py
python scripts/export_ui_data.py
```

### Tested Environment
- **Operating System:** Windows 11 (64-bit)
- **Python:** 3.14.2
- **Node.js:** v24.14.1
- **npm:** 11.11.0

---

## Data Provenance
- Dataset: Shanghai_T2DM continuous glucose monitoring dataset from Zhao, X. et al. *Scientific Data* 10, 836 (2023), published under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
- Figshare Download Link: [https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397/3](https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397/3)
- Retraining requirement: Downloading the raw Shanghai_T2DM files into `data/raw/` is only required if running data ingestion from scratch.

---

## Open-Source License Details
- Software License: MIT License (see [LICENSE](LICENSE)). Copyright (c) 2026 Team AlooDUM.
- Dataset License: Creative Commons Attribution 4.0 International (CC BY 4.0), Zhao et al., *Scientific Data*, 2023.

---

### Regulatory Disclaimer
*NoctTwin is an academic decision support prototype, not an FDA- or CE-marked medical device. It does not provide automated diagnostic or therapeutic decisions.*
