# NoctTwin: Digital Twin for Nocturnal Hypoglycemia Risk & Decision Support in Sulfonylurea-Treated Type 2 Diabetes

**Happiest Health Digital Twin Challenge 2026**  
**Submission Deadline:** Saturday, 24 October 2026, 7:00 PM IST  
**Team:** AlooDUM  
**College:** Arka Jain University, Jamshedpur  
**Members:** Nivesh Kumar · Kriti Kumari · Bhoomi Singh  
**Team Leader:** Abhinav Tiwary  
**License:** MIT License  

---

## 1. Executive Summary & Clinical Problem

Nocturnal hypoglycemia is among the most hazardous complications in the pharmacological management of Type 2 Diabetes Mellitus (T2DM). Patients treated with insulin secretagogues—particularly sulfonylureas (SUs) such as glimepiride, gliclazide, and gliquidone—are at persistent risk of severe nocturnal hypoglycemia because SU-mediated insulin secretion continues independently of circulating blood glucose levels.

Clinicians face two intertwined bedtime challenges:
1. **Bedtime Risk Stratification:** Estimating whether an individual patient will experience nocturnal hypoglycemia (<70 mg/dL) between 23:00 and 06:00 using continuous glucose monitoring (CGM) and prior patient history.
2. **What-If Clinical Countermeasures:** Simulating the impact of preventative bedtime interventions—specifically carbohydrate snack supplementation (e.g. 15g or 30g carbs) or SU dose down-titration.

**NoctTwin** combines patient-level grouped machine learning risk prediction with a personalized two-compartment mechanistic glucose-insulin digital twin and split-conformal uncertainty intervals to provide actionable, calibrated bedtime decision support.

> [!IMPORTANT]
> **Clinical & Data Constraints:**
> - **Inpatient Setting:** Data originates from hospitalized patients (Shanghai_T2DM, Zhao et al. 2023, *Scientific Data*, CC BY 4.0; 100 patients, 109 sessions, 1,181 valid nights). Hospital dietary timing, IV glucose, and acute clinical interventions may differ substantially from free-living outpatients.
> - **Exploratory SU Subgroup:** The cohort contains only 18 SU-treated patients (247 nights, 32 strict events across 11 patients). All SU-specific subgroup findings are explicitly labeled **exploratory**.
> - **Simulation Hypothesis:** Bedtime snack glycemic yield and SU scale factors in the what-if simulation module are based on established physiological parameter assumptions from the clinical literature, **not learned from intervention data**. Outputs are strictly labeled *"simulation hypothesis, not clinically validated"*.

---

## 2. Rigorous Dataset Audit & Target Definitions

### 2.1 The Shanghai_T2DM Cohort
- **Total Patients:** 100 hospitalized T2DM patients (58 male, 42 female).
- **Age:** $60.1 \pm 12.3$ years; **BMI:** $24.8 \pm 3.6$ $\text{kg/m}^2$; **HbA1c:** $68.4 \pm 20.8$ mmol/mol ($8.4 \pm 1.9\%$).
- **CGM Sampling:** 15.0-minute median interval across 112,379 readings (0.08% overall missingness).
- **Valid Night Criteria:** $\ge 75\%$ of nocturnal readings (at least 22 of 29 readings between 23:00 and 06:00) present. Total valid nights analyzed: **1,181 nights**.

### 2.2 Primary Target: Strict Persistent Hypoglycemia
Flash glucose sensors (such as Abbott FreeStyle Libre) frequently register transient, unphysiological pressure-induced nocturnal dips ("compression artifacts") when patients sleep on the sensor.
- **Sensitivity Analysis (Any Reading <70 mg/dL):** 148 nights (12.5%).
- **Primary Clinical Target (Strict Persistent Low):** Defined as $\ge 2$ consecutive readings $<70$ mg/dL (contiguous duration $\ge 30$ minutes) in the nocturnal window (23:00–06:00).
  - Total strict events: **124 nights (10.5% base prevalence)** across 45 unique patients.
  - Median event duration: **90 minutes** (IQR: 45–180 minutes).
  - Isolated 1-reading dips eliminated: 24 nights (16.1% of all raw dips were sensor compression artifacts).
- **Secondary Strict Target (Clinically Significant Level 2 Hypo):** $\ge 2$ consecutive readings $<54$ mg/dL: **49 nights (4.1%)**.

---

## 3. Event Concentration & Subgroup Sensitivity Analysis

### 3.1 Patient-Level Event Distribution
Strict nocturnal hypoglycemia is heavily concentrated in a small subset of patients:
- **The top 10% of patients (10 patients) account for 48.0% of all strict nocturnal events (60 of 124 events).**
- 55 patients experienced zero nocturnal events.

### 3.2 Regimen Subgroups & Outlier Sensitivity
Patients were classified by baseline hypoglycemic regimen:

| Regimen Subgroup | Patients | Nights | Baseline Strict Rate (%) | Top 1 Patient Impact | Rate Excl. Top 1 Pt (%) | Rate Excl. Top 2 Pts (%) | Leave-One-Out Range |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SU without Insulin** | 10 | 124 | **19.4%** (24 ev) | Pt 2066: 9/13 ev (69.2%) | **13.5%** (15 ev) | **10.3%** (10 ev) | [13.5% – 21.8%] |
| **SU with Insulin** | 8 | 123 | **6.5%** (8 ev) | Pt 2001: 3/26 ev (11.5%) | **5.2%** (5 ev) | **2.4%** (2 ev) | [4.6% – 8.2%] |
| **Insulin without SU** | 51 | 520 | **7.3%** (38 ev) | Pt 2019: 4/8 ev (50.0%) | **6.6%** (34 ev) | **6.0%** (30 ev) | [6.6% – 7.7%] |
| **Neither SU nor Insulin** | 31 | 414 | **13.0%** (54 ev) | Pt 2077: 11/11 ev (100%) | **10.7%** (43 ev) | **9.3%** (36 ev) | [10.7% – 14.0%] |
| **All SU Combined** | 18 | 247 | **13.0%** (32 ev) | Pt 2066: 9/13 ev (69.2%) | **9.8%** (23 ev) | **8.2%** (18 ev) | [9.8% – 14.8%] |
| **All T2DM Cohort** | 100 | 1,181 | **10.5%** (124 ev) | Pt 2077: 11/11 ev (100%) | **9.7%** (113 ev) | **9.0%** (104 ev) | [9.7% – 10.8%] |

> [!WARNING]
> **Data Consistency & Resolution of Discrepancy:**
> In the Phase 0 audit, SU-without-insulin was reported as 10 patients (125 nights) and SU-with-insulin as 8 patients. An earlier intermediate draft showed 11 / 7 patients and 129 / 118 nights. The discrepancy was traced to:
> 1. **Patient 2011 (Basal CSII Pump):** In `Shanghai_T2DM_Summary.xlsx`, admission oral meds listed gliclazide without insulin, but the hourly medication log recorded continuous Novolin R basal insulin pump infusion (`CSII - basal insulin`: 0.2–0.6 IU/h) with 0 IU bolus and 0 s.c. injections. A feature extractor that checked only subcutaneous or bolus insulin temporarily classified Pt 2011 as "SU without insulin", whereas the audit checked all CSII columns. Pt 2011 is canonically classified as **SU with Insulin**.
> 2. **Patient 2048 (Bedtime Anchor Coverage):** Had 13 nights meeting nocturnal CGM coverage ($\ge 75\%$), but night 1 began recording after 22:00, lacking the required $\pm 30$ min bedtime anchor reading. Valid bedtime-prediction nights are 124 for SU-without-insulin and 123 for SU-with-insulin (total 247 SU nights).
> 
> A single canonical classifier (`src/data/regimen.py`) is now used universally across all audit scripts, feature engineering, models, UI exports, and tests.

> [!WARNING]
> **Correction Regarding SU Risk:**
> A naive comparison between SU-without-insulin (19.4%) and insulin-without-SU (7.3%) yields an apparent rate ratio of $2.66\times$. However, this is heavily skewed by a single extreme outlier (**Patient 2066**, who experienced events on 9 of 13 nights). When Patient 2066 is excluded, the SU event rate falls to **13.5%** (ratio = $1.85\times$). Furthermore, the "Neither" group exhibits a 13.0% rate (driven by Pt 2077). **We do not claim a "3-fold SU risk" or strict causality.**

---

## 4. Digital Twin Forecaster Validation

The blended "synchronised" forecaster, mechanistic digital twin, and persistence baselines were evaluated as nocturnal glucose forecasters at 1-hour (23:00), 3-hour (01:00), and 6-hour (04:00) horizons across 1,072 nights with prior historical data:

$$\text{Forecast}(t) = \text{Patient\_Profile}(t) + \left(G_{\text{bedtime}} - \text{Profile}(\text{22:00})\right) \cdot \exp(-t / \tau)$$

where $\tau$ is the relaxation time constant fitted strictly inside 5-fold patient-grouped cross-validation on training folds ($\tau_{\text{mean}} = 3.305$ hours / 198.3 minutes; fold values: [3.112, 3.620, 3.066, 3.351, 3.378] hours; evaluated on held-out patients).

### Nested Tau Selection & Out-of-Fold Evaluation (5-Fold Patient Grouped):
- **Fold 1:** $\tau = 3.112$h | OOF MAE 1h = **15.54** (vs persistence 20.08, profile 28.21), 3h = **21.46**, 6h = **20.20** mg/dL
- **Fold 2:** $\tau = 3.620$h | OOF MAE 1h = **14.25** (vs persistence 18.86, profile 18.94), 3h = **18.06**, 6h = **15.25** mg/dL
- **Fold 3:** $\tau = 3.066$h | OOF MAE 1h = **15.32** (vs persistence 18.52, profile 25.23), 3h = **22.00**, 6h = **18.67** mg/dL
- **Fold 4:** $\tau = 3.351$h | OOF MAE 1h = **18.59** (vs persistence 20.67, profile 28.92), 3h = **24.31**, 6h = **20.36** mg/dL
- **Fold 5:** $\tau = 3.378$h | OOF MAE 1h = **13.59** (vs persistence 15.68, profile 20.37), 3h = **17.57**, 6h = **15.78** mg/dL

### Benchmark Results (1,072 Nights with Historical Prior Data):

| Forecast Horizon | Benchmark / Model | MAE [95% CI] (mg/dL) | RMSE [95% CI] (mg/dL) | Clinical Finding |
| :--- | :--- | :---: | :---: | :--- |
| **1-Hour (23:00)** | **1. Blended Synchronised Forecaster** | **14.83 [13.54, 16.14]** | **20.33 [18.59, 22.01]** | **Wins decisively.** Synchronises bedtime glucose with diurnal curve. |
| | 2. Persistence Baseline | 18.48 [16.81, 20.06] | 24.28 [22.25, 26.02] | Persistence holds well at 1h due to physiological inertia. |
| | 3. Patient Hist Overnight Profile | 24.62 [22.18, 27.07] | 33.39 [30.16, 36.66] | Unadjusted profile ignores acute bedtime offset. |
| **3-Hour (01:00)** | **1. Blended Synchronised Forecaster** | **19.35 [17.32, 21.39]** | **27.32 [24.31, 30.12]** | **Wins.** Smoothly bridges acute bedtime levels to patient setpoint. |
| | 2. Patient Hist Overnight Profile | 21.69 [19.38, 24.22] | 30.74 [27.60, 33.68] | Personal diurnal setpoints dominate over persistence. |
| | 3. Persistence Baseline | 31.97 [28.50, 35.54] | 43.07 [38.17, 47.75] | Persistence degrades severely by 3 hours. |
| **6-Hour (04:00)** | **1. Blended Synchronised Forecaster** | **16.05 [14.21, 17.91]** | **22.78 [20.03, 25.43]** | **Wins across full nocturnal horizon.** |
| | 2. Patient Hist Overnight Profile | 17.18 [15.24, 19.13] | 24.03 [21.36, 26.54] | Diurnal historical rhythm accurately predicts basal nadir. |
| | 3. Persistence Baseline | 36.98 [32.38, 42.02] | 50.02 [43.30, 56.71] | Persistence fails completely across full overnight window. |

### Paired Bootstrap Differences (Blended vs. Baselines, 1,000 Patient Resamples):
- **1-Hour:** vs. Persistence $\Delta\text{MAE} = \mathbf{-3.64}$ [95% CI: $-5.01$, $-2.14$] mg/dL; vs. Profile $\Delta\text{MAE} = \mathbf{-9.78}$ [95% CI: $-11.61$, $-7.97$] mg/dL (both 95% CIs exclude 0).
- **3-Hour:** vs. Persistence $\Delta\text{MAE} = \mathbf{-12.62}$ [95% CI: $-15.48$, $-9.58$] mg/dL; vs. Profile $\Delta\text{MAE} = \mathbf{-2.33}$ [95% CI: $-3.52$, $-1.17$] mg/dL (both 95% CIs exclude 0).
- **6-Hour:** vs. Persistence $\Delta\text{MAE} = \mathbf{-20.93}$ [95% CI: $-25.26$, $-16.90$] mg/dL; vs. Profile $\Delta\text{MAE} = \mathbf{-1.13}$ [95% CI: $-1.57$, $-0.72$] mg/dL (both 95% CIs exclude 0).

> [!NOTE]
> **Forecasting Architecture Decision:**
> Because the **Blended Synchronised Forecaster** statistically significantly outperforms both Persistence and the unadjusted Historical Profile at every horizon (1h, 3h, 6h), it is selected as the **displayed expected trajectory** in the clinical UI.
> The **Mechanistic Two-Compartment ODE is reserved strictly for simulating countermeasure deltas** ($\Delta G(t)$) relative to that blended baseline ($G_{\text{simulated}}(t) = G_{\text{blend}}(t) + \Delta G_{\text{ODE}}(t)$), guaranteeing that simulated snacks or secretagogue dose reductions do not distort the patient's baseline diurnal rhythm.

---

## 5. Machine Learning Risk Model & Feature-Group Ablation

Models were evaluated under strict 5-fold patient-level `GroupKFold` cross-validation (zero patient overlap between folds) with 1,000-iteration patient-clustered bootstrap 95% confidence intervals. Platt sigmoid calibration was fitted strictly inside inner GroupKFold splits.

### 5.1 Primary Headline Benchmark: Nights 1+ (20 Repeated GroupKFold Runs, N=1,081 Nights, 101 Events)
Across 20 repeated 5-fold patient-level `GroupKFold` cross-validation runs with different random seeds on nights where strictly earlier nocturnal monitoring data exists (Nights 1+):
- **Group G (CGM + History, 19 feats):**
  - **AUROC:** **0.832** (range: **0.824 - 0.840**)
  - **AUPRC:** **0.476** (range: **0.417 - 0.496**)
  - **Brier Score:** **0.065** (range: **0.064 - 0.068**)
- **Group B (CGM Dynamics Only, 12 feats):**
  - **AUROC:** **0.736** (range: **0.707 - 0.753**)
  - **AUPRC:** **0.354** (range: **0.252 - 0.392**)
  - **Brier Score:** **0.073** (range: **0.071 - 0.078**)
- **Paired Advantage Across 20 Runs (Group G vs Group B):**
  - $\Delta\text{AUROC} = \mathbf{+0.097}$ (range: $+0.082$ to $+0.121$) [95% CI: $+0.051$, $+0.149$] (strictly positive, excludes 0)
  - $\Delta\text{AUPRC} = \mathbf{+0.123}$ (range: $+0.087$ to $+0.165$) [95% CI: $+0.047$, $+0.191$] (strictly positive, excludes 0)
  - $\Delta\text{Brier} = \mathbf{-0.008}$ (range: $-0.010$ to $-0.006$) [95% CI: $-0.014$, $-0.004$] (strictly negative, demonstrates superior calibration)

### 5.2 Secondary Benchmark: Full Cohort (N=100 Patients, 1,181 Nights Including Night 0)

| Group | Features Included | Model | AUROC [95% CI] | AUPRC [95% CI] | Brier | BSS (%) | ECE | Sens @ 80% Spec | PPV @ 80% Spec | NNW @ 80% Spec | Alerts / pt-wk |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Simple Rule ($G < 110$ & Slope $< -5$) | Heuristic (No ML) | 0.660 [0.574-0.739] | 0.240 [0.151-0.362] | 0.295 | -213.7% | 0.331 | 46.0% | 22.0% | 4.54 | 1.54 |
| **B** | CGM Dynamics Only (12 feats) | Logistic Regression | 0.770 [0.695-0.836] | 0.405 [0.228-0.581] | 0.078 | +16.5% | 0.013 | 60.5% | 27.4% | 3.65 | 1.62 |
| | | LightGBM | 0.770 [0.700-0.834] | 0.366 [0.229-0.536] | 0.081 | +14.1% | 0.017 | 58.1% | 26.4% | 3.79 | 1.62 |
| **C** | CGM + Meds & Carbs (24 feats) | Logistic Regression | 0.756 [0.681-0.822] | 0.363 [0.207-0.540] | 0.081 | +13.4% | 0.013 | 62.1% | 26.7% | 3.74 | 1.71 |
| | | LightGBM | 0.775 [0.706-0.834] | 0.379 [0.224-0.560] | 0.081 | +14.2% | 0.013 | 58.1% | 25.5% | 3.92 | 1.67 |
| **D** | CGM + Meds + Small Static (28 feats) | Logistic Regression | 0.733 [0.658-0.803] | 0.331 [0.180-0.509] | 0.083 | +11.3% | 0.015 | 54.8% | 24.7% | 4.04 | 1.63 |
| | | LightGBM | 0.764 [0.695-0.829] | 0.381 [0.216-0.563] | 0.081 | +13.3% | 0.013 | 54.8% | 24.8% | 4.03 | 1.62 |
| **E** | CGM + Meds + All Labs Full (36 feats) | Logistic Regression | 0.715 [0.636-0.784] | 0.297 [0.181-0.459] | 0.085 | +9.7% | 0.009 | 49.2% | 23.6% | 4.23 | 1.53 |
| | | LightGBM | 0.767 [0.701-0.832] | 0.338 [0.211-0.492] | 0.083 | +11.8% | 0.019 | 53.2% | 24.2% | 4.14 | 1.62 |
| **F** | Twin-Enhanced (Group D + Twin feats) | Logistic Regression | 0.774 [0.691-0.843] | 0.398 [0.243-0.575] | 0.078 | +16.6% | 0.020 | 62.1% | 26.9% | 3.71 | 1.70 |
| | | LightGBM | 0.781 [0.707-0.845] | 0.406 [0.230-0.582] | 0.079 | +16.2% | 0.013 | 59.7% | 26.1% | 3.82 | 1.68 |
| **G** | **CGM + Patient History (19 feats)** | **Logistic Regression** | **0.845 [0.786-0.890]** | **0.505 [0.327-0.660]** | **0.070** | **+25.0%** | **0.015** | **75.0%** | **30.6%** | **3.27** | **1.80** |
| | | **LightGBM** | **0.836 [0.778-0.885]** | **0.439 [0.287-0.600]** | **0.075** | **+19.9%** | **0.035** | **71.0%** | **29.8%** | **3.35** | **1.75** |
| **H** | CGM + History + Meds + Static (35 feats) | Logistic Regression | 0.835 [0.774-0.882] | 0.426 [0.302-0.615] | 0.075 | +19.9% | 0.019 | 75.0% | 33.0% | 3.03 | 1.67 |
| | | LightGBM | 0.828 [0.765-0.881] | 0.432 [0.269-0.600] | 0.076 | +19.3% | 0.020 | 73.4% | 30.1% | 3.32 | 1.79 |

*Note: Constant prevalence baseline Brier score is 0.0940 (prevalence 10.5%). BSS = $1 - \frac{\text{Brier}}{\text{Brier}_{\text{ref}}}$.*

---

## 6. Statistical Hypothesis Testing: Paired Patient-Level Bootstrap Differences

To replace unsubstantiated claims of "improvement" with statistical truth, paired patient-level bootstrap differences ($\text{Candidate} - \text{Reference Group B}$) were computed across 1,000 patient-clustered resamples:

### Paired Differences vs. Group B (CGM-Only) on Strict Consecutive Label:

| Model | Comparison | $\Delta$ AUROC [95% CI] | AUROC 95% CI Excludes 0? | $\Delta$ AUPRC [95% CI] | AUPRC 95% CI Excludes 0? | $\Delta$ Brier [95% CI] | Brier 95% CI Excludes 0? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | C (CGM + Meds) vs B | $+0.005$ [$-0.019$, $+0.030$] | No | $+0.013$ [$-0.026$, $+0.048$] | No | $-0.0001$ [$-0.0018$, $+0.0015$] | No |
| | D (CGM + Meds + Static) vs B | $-0.006$ [$-0.038$, $+0.026$] | No | $+0.016$ [$-0.044$, $+0.065$] | No | $+0.0008$ [$-0.0019$, $+0.0033$] | No |
| | E (All Labs Full) vs B | $-0.003$ [$-0.045$, $+0.035$] | No | $-0.027$ [$-0.082$, $+0.021$] | No | $+0.0022$ [$-0.0008$, $+0.0058$] | No |
| | F (Twin-Enhanced) vs B | $+0.011$ [$-0.025$, $+0.043$] | No | $+0.040$ [$-0.027$, $+0.099$] | No | $-0.0020$ [$-0.0057$, $+0.0014$] | No |
| | **G (CGM + History) vs B** | **$+0.066$ [$+0.032$, $+0.103$]** | **YES (Strictly Positive)** | **$+0.073$ [$+0.020$, $+0.118$]** | **YES (Strictly Positive)** | **$-0.0054$ [$-0.0093$, $-0.0019$]** | **YES (Strictly Negative)** |
| | H (G + Meds + Static) vs B | $+0.058$ [$+0.024$, $+0.095$] | YES (Strictly Positive) | $+0.066$ [$-0.002$, $+0.124$] | No (Spans 0) | $-0.0049$ [$-0.0088$, $-0.0014$] | YES (Strictly Negative) |
| **Logistic Regression** | C (CGM + Meds) vs B | $-0.014$ [$-0.047$, $+0.020$] | No | $-0.042$ [$-0.082$, $+0.002$] | No | $+0.0029$ [$+0.0005$, $+0.0056$] | Degraded (Strictly Positive) |
| | D (CGM + Meds + Static) vs B | $-0.037$ [$-0.075$, $+0.003$] | No | $-0.074$ [$-0.120$, $-0.018$] | Degraded (Strictly Negative) | $+0.0049$ [$+0.0017$, $+0.0088$] | Degraded (Strictly Positive) |
| | E (All Labs Full) vs B | $-0.055$ [$-0.107$, $-0.005$] | Degraded (Strictly Negative) | $-0.108$ [$-0.159$, $-0.025$] | Degraded (Strictly Negative) | $+0.0064$ [$+0.0016$, $+0.0130$] | Degraded (Strictly Positive) |
| | F (Twin-Enhanced) vs B | $+0.004$ [$-0.040$, $+0.048$] | No | $-0.007$ [$-0.066$, $+0.078$] | No | $-0.0000$ [$-0.0045$, $+0.0042$] | No |
| | **G (CGM + History) vs B** | **$+0.075$ [$+0.037$, $+0.114$]** | **YES (Strictly Positive)** | **$+0.100$ [$+0.046$, $+0.148$]** | **YES (Strictly Positive)** | **$-0.0080$ [$-0.0123$, $-0.0042$]** | **YES (Strictly Negative)** |
| | H (G + Meds + Static) vs B | $+0.065$ [$+0.028$, $+0.104$] | YES (Strictly Positive) | $+0.021$ [$-0.051$, $+0.128$] | No | $-0.0032$ [$-0.0085$, $+0.0040$] | No |

### Scientific Verdict on Features:
1. **Medications and Static Demographics (Groups C, D, E):** Do **NOT** produce statistically significant gains over CGM alone (all 95% bootstrap CIs span zero). Adding static lab features degrades out-of-fold AUPRC and calibration in linear models ($\Delta\text{AUPRC} = -0.108$ [$-0.159$, $-0.025$]) because static features act as patient fingerprints.
2. **Mechanistic Twin Features (Group F):** Provide a modest directional increase ($\Delta \text{AUROC} = +0.011$ [$-0.025$, $+0.043$]), but the 95% bootstrap CI spans zero (not statistically significant).
3. **Patient History Features (Group G):** Provide **statistically significant improvements** ($\Delta\text{AUROC} = +0.066$ [$+0.032$, $+0.103$], $\Delta\text{AUPRC} = +0.073$ [$+0.020$, $+0.118$], $\Delta\text{Brier} = -0.0054$ [$-0.0093$, $-0.0019$], all 95% CIs exclude zero). Knowing a patient's prior nocturnal hypoglycemia history is the single most valuable predictor.
4. **Simpler Model Generalization:** Regularized Logistic Regression on Group G outperforms LightGBM (AUROC 0.845 vs 0.836, AUPRC 0.505 vs 0.439, BSS 25.0% vs 19.9%) with an in-fold vs out-of-fold generalization gap of only **0.028** (train 0.861 vs test 0.833).

---

### 6.1 Robustness and Boundary Testing of Group G (Mandated Analyses)

To rigorously test whether Group G's performance is genuine or an artifact of cold-start nights or extreme event concentration, four stress tests were executed:

#### (a) Identical Subset with $\ge 1$ Prior Night History ($N = 1,081$ nights, 101 events)
When restricted to the exact same nights where prior monitoring data exists:
- **Group B (CGM-Only):** AUROC = 0.748, AUPRC = 0.317, Brier = 0.075
- **Group G (CGM + History):** AUROC = 0.829, AUPRC = 0.400, Brier = 0.069
- **Paired Patient Bootstrap Differences (G - B):**
  - $\Delta\text{AUROC} = \mathbf{+0.080}$ [95% CI: $+0.041$, $+0.126$] (95% CI strictly positive, excludes 0)
  - $\Delta\text{AUPRC} = \mathbf{+0.084}$ [95% CI: $+0.026$, $+0.131$] (95% CI strictly positive, excludes 0)
  - $\Delta\text{Brier} = \mathbf{-0.005}$ [95% CI: $-0.009$, $-0.002$] (95% CI strictly negative, excludes 0)

#### (b) Event Concentration Sensitivity
Events in Shanghai_T2DM are concentrated: the **top 10% of patients account for 48.4% of all strict nocturnal lows**.
- **Excluding Top 1 Event Patient (Pt 2077, 11 events; $N=1,170$ nights, 113 events):**
  $\Delta\text{AUROC} = \mathbf{+0.072}$ [$+0.035$, $+0.112$], $\Delta\text{AUPRC} = \mathbf{+0.072}$ [$+0.016$, $+0.122$] (Survives solidly).
- **Excluding Top 2 Event Patients ([Pt 2077, Pt 2066], 20 events; $N=1,157$ nights, 104 events):**
  $\Delta\text{AUROC} = \mathbf{+0.071}$ [$+0.031$, $+0.112$], $\Delta\text{AUPRC} = \mathbf{+0.065}$ [$+0.010$, $+0.111$] (Survives solidly).
- **Excluding Top 10% Event Patients (10 patients, 60 events; $N=1,058$ nights, 64 events):**
  $\Delta\text{AUROC} = \mathbf{+0.065}$ [$+0.012$, $+0.118$] (AUROC improvement survives with 95% CI strictly positive). $\Delta\text{AUPRC} = +0.004$ [$-0.048$, $+0.047$] (AUPRC attenuates once nearly half of all events are removed).

#### (c) Performance Stratified by Prior Nights History
| History Depth | Nights | Events (Rate) | Group B AUROC | Group G AUROC | Group B AUPRC | Group G AUPRC | Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 prior night** | 100 | 10 (10.0%) | **0.813** | 0.797 | 0.300 | **0.336** | Single night history has not fully stabilized. |
| **2–3 prior nights** | 197 | 18 (9.1%) | 0.721 | **0.875** | 0.283 | **0.435** | **Marked jump in discrimination** ($\Delta\text{AUROC} = +0.154$). |
| **4+ prior nights** | 784 | 73 (9.3%) | 0.745 | **0.823** | 0.348 | **0.423** | **Sustained, durable benefit** across majority of hospital stay. |

#### (d) Cold-Start Analysis on First Nights (Night 0, $N = 100$ nights, 23 events)
On the very first night of monitoring when no prior history exists:
- **Group B (CGM-Only Dynamics):** AUROC = **0.863**, AUPRC = **0.673**, Brier = **0.146**, BSS = **+17.7%**
- **Group C (CGM + Meds):** AUROC = 0.842, AUPRC = 0.634, Brier = 0.151, BSS = +15.0%
- **Group D (CGM + Meds + Static EHR):** AUROC = 0.846, AUPRC = 0.653, Brier = 0.149, BSS = +15.6%
*Takeaway:* On cold start (Night 0), acute CGM dynamics perform best; the sample is underpowered and shows no measurable benefit from adding static EHR features or medication tags.

---

### 6.2 Night-0 Audit & Warm-Up Sensitivity Analysis

To investigate sensor insertion stabilization and hospitalization entry artifacts, event prevalence was evaluated across chronological night index and elapsed hours from the very first CGM reading:

#### Event Prevalence by Night Index:
- **Night 0 (Admission Night):** **23.0%** (23 / 100 nights) across 100 patients.
- **Nights 1 to 14 (Subsequent Nights):** **9.34%** (101 / 1,081 nights).
- *Rate Ratio (Night 0 vs. Subsequent Nights):* **$2.46\times$ higher prevalence on Night 0.**

#### Event Prevalence by Sensor Warm-up Window (Hours Since First Reading):
- **$<12$ Hours:** **25.6%** (22 / 86 nights).
- **12–24 Hours:** **7.1%** (1 / 14 nights).
- **24–48 Hours:** **10.0%** (10 / 100 nights).
- **48–72 Hours:** **12.0%** (12 / 100 nights).
- **72–120 Hours:** **7.4%** (14 / 190 nights).
- **120–240 Hours:** **9.5%** (36 / 378 nights).
- **$>240$ Hours:** **14.3%** (28 / 196 nights).

#### Sensitivity Check: Main Group G vs Group B Excluding Night 0 ($N = 1,081$ nights, 101 events):
When Night 0 is completely excluded from the dataset:
- **Group B (CGM-Only):** AUROC = 0.748, AUPRC = 0.317, Brier = 0.075
- **Group G (CGM + History):** AUROC = 0.829, AUPRC = 0.400, Brier = 0.069
- **Paired Bootstrap Differences (G - B):** $\Delta\text{AUROC} = \mathbf{+0.080}$ [95% CI: $+0.041$, $+0.126$], $\Delta\text{AUPRC} = \mathbf{+0.084}$ [95% CI: $+0.026$, $+0.131$], $\Delta\text{Brier} = \mathbf{-0.005}$ [95% CI: $-0.009$, $-0.002$].
- *Finding:* Group G's advantage is **larger** when Night 0 is excluded (AUROC delta rises from $+0.066$ to $+0.080$), showing that history features provide substantial value once patients have at least one stabilized monitoring night.

#### Post-Warmup Cold-Start Sensitivity (Night 1, $N=100$ nights, 10 events):
Evaluating first post-warmup entry models on Night 1:
- **Group B (CGM-Only):** AUROC = **0.813**, AUPRC = 0.300, Brier = 0.083
- **Group C (CGM + Meds):** AUROC = 0.783, AUPRC = 0.289, Brier = 0.084
- **Group D (CGM + Meds + Static):** AUROC = 0.747, AUPRC = **0.335**, Brier = 0.083
- *Finding:* The single-night evaluation on Night 1 is **underpowered (only 10 events across 100 patients)**, showing **no measurable benefit** from adding static EHR demographics or medication categories over acute CGM dynamics alone. The product safely defaults to acute CGM-only Model B when personal history is absent.

---

## 7. Clinical Credibility: The 75% PPV Target & Bayes Theorem

Clinicians often request high Positive Predictive Value ($\text{PPV} \ge 75\%$) to minimize alert fatigue. We evaluate this credibility target mathematically and empirically:

- **Mathematical Reality (Bayes' Theorem):**
  Given a 10.5% base prevalence, pre-test odds are $\frac{0.105}{1 - 0.105} = 0.1173$. Reaching 75% PPV (post-test odds $= 3.0$) requires a Positive Likelihood Ratio:
  $$LR_+ = \frac{\text{Post-test Odds}}{\text{Pre-test Odds}} = \frac{3.0}{0.1173} = \mathbf{25.57}$$
  Since $LR_+ = \frac{\text{Sensitivity}}{1 - \text{Specificity}}$, achieving $LR_+ \ge 25.57$ demands extreme specificity ($>99\%$).

### Empirical Evaluation on Out-of-Fold Cohort:

| Model | Achieves 75% PPV? | Operating Threshold | Specificity Achieved | Sensitivity Remaining | Observed $LR_+$ | PPV Achieved | NNW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Group B: CGM-Only)** | Yes | 0.455 | 99.34% | **16.9%** | 25.57 | 75.0% | 1.33 |
| **LightGBM (Group D: CGM+Meds+Static)** | Yes | 0.465 | 99.62% | **12.9%** | 34.10 | 80.0% | 1.25 |
| **LightGBM (Group G: CGM+History)** | Yes | 0.475 | 99.43% | **17.7%** | 31.26 | 78.6% | 1.27 |
| **LightGBM (Group H: CGM+Hist+Meds)** | Yes | 0.475 | 99.43% | **15.3%** | 26.99 | 76.0% | 1.32 |

> [!CAUTION]
> **Clinical Credibility Gap:**
> While models can technically reach $\ge 75\%$ PPV, doing so forces the threshold so high that **$82\% - 87\%$ of all true nocturnal hypoglycemic episodes are missed**. At standard clinical operating points (80% specificity), PPV is **$28\% - 31\%$** (Number-Needed-to-Warn $\approx 3.3$). Clinicians must understand that bedtime screening for a low-prevalence event cannot simultaneously achieve $75\%$ PPV and acceptable sensitivity.

---

## 8. Split-Conformal Uncertainty Coverage (200 Patient-Level Splits)

To strictly measure empirical coverage on patients **never seen during conformal calibration**, 200 random patient-level splits (half calibration: 50 patients; half test: 50 patients) were conducted. For each split, calibration quantiles ($q_{80}, q_{90}$) were calculated on calibration patient residuals and evaluated on the 50 completely held-out test patients:

| Metric / Stratum | 80% Nominal Target Coverage | 90% Nominal Target Coverage | Conformal Margin Mean |
| :--- | :---: | :---: | :---: |
| **Overall Held-Out Coverage** | **79.93%** (Range: 70.79% – 86.66%, 95% CI: [74.17%, 85.82%]) | **89.84%** (Range: 83.39% – 94.13%, 95% CI: [85.03%, 93.72%]) | $\pm 26.26$ mg/dL (80%)<br>$\pm 37.59$ mg/dL (90%) |
| **1-Hour Horizon (23:00)** | **84.19%** (Range: 76.27% – 90.92%) | **93.86%** (Range: 88.98% – 97.43%) | High initial precision |
| **3-Hour Horizon (01:00)** | **75.57%** (Range: 64.58% – 83.64%) | **86.30%** (Range: 78.31% – 92.25%) | *Dips below nominal (see note)* |
| **6-Hour Horizon (04:00)** | **80.04%** (Range: 71.53% – 86.80%) | **89.35%** (Range: 82.37% – 94.35%) | Fully calibrated at basal nadir |
| **Conditional on Forecast < 110 mg/dL** | **89.13%** (Range: 83.11% – 94.81%) | **95.18%** (Range: 91.83% – 98.07%) | Conservative/higher coverage |

> [!WARNING]
> **Honest Empirical Finding at 3-Hour Horizon:**
> At the 3-hour horizon (01:00), held-out empirical coverage dips slightly below nominal (**75.57%** for 80% target, **86.30%** for 90% target). This occurs because peak physiological dispersion and variable midnight digestion rates introduce higher glucose variance mid-sleep than at bedtime or early morning.
> Conversely, **conditional on a bedtime forecast $< 110$ mg/dL**, empirical coverage is conservative (**89.1%** at 80% target, **95.2%** at 90% target) because low glucose values have smaller absolute variance bounded near sensor nadirs. In-sample calibration residuals are never reported as held-out coverage.

---

## 9. Cardiac Risk Context (QTc & Arrhythmia Note)

Severe nocturnal hypoglycemia triggers counter-regulatory sympathoadrenal surges that prolong the cardiac ventricular repolarization interval (QTc) and predispose patients to fatal nocturnal arrhythmias ("dead-in-bed syndrome").
- **Ground Truth Honesty:** Because the Shanghai_T2DM dataset contains continuous glucose monitoring and laboratory logs but **no continuous ECG or Holter waveforms**, NoctTwin does **NOT** attempt to predict QTc duration or ECG waveforms (no ground truth exists to validate such predictions).
- **Clinical Literature Context:** The dashboard includes a clearly designated clinical context card referencing the landmark literature (e.g. Chow et al., *Diabetes*, 2014; Fitzpatrick et al., *Diabetes Care*, 2020) describing that when nocturnal glucose drops below 54 mg/dL for $>30$ minutes, sympathetic adrenaline release induces potassium shifts and QTc prolongation exceeding 450 ms in up to 60% of episodes.

---

## 10. Assumptions & Limitations

1. **Per-Patient Regimen Classification:**
   Patient medication tags (e.g., SU-treated, insulin-treated, secretagogue agent, and dose) are derived from admission medical summaries and medication intake sheets. In this retrospective dataset, daily inpatient dosage changes, physician-held doses, and acute IV dextrose administrations during hospital stays are not tracked on a night-by-night basis.

2. **History Features Cannot Distinguish Physiology from Recurring Sensor Artifacts:**
   Flash glucose sensors (such as the Abbott FreeStyle Libre used in Shanghai_T2DM) are prone to pressure-induced nocturnal dips ("compression artifacts") caused by patients sleeping directly on the sensor. While our primary target requires $\ge 2$ consecutive readings $<70$ mg/dL ($\ge 30$ mins) to eliminate isolated 15-minute dips, history features cannot distinguish genuine recurrent physiological hypoglycemia from a patient who habitually sleeps on their left arm every night. If compression artifacts recur, history features will track them as persistent hypoglycemic risk.

3. **Unresolved First-Night (Night 0) Anomaly (Open Question):**
   Night 0 exhibits a $2.46\times$ higher strict event rate than subsequent nights (23.0% vs. 9.34%; 25.6% within the first 12 hours of sensor wear). Whether this surge is an artifact of FreeStyle Libre sensor insertion trauma and enzymatic stabilization lag, or reflects genuine physiological disruption caused by hospital admission stress, dietary alteration, and acute medication changes, remains an **unresolved open question** that cannot be settled without paired reference capillary blood draws. When Night 0 is excluded as a sensitivity check ($N=1,081$ nights, 101 events), Group G's advantage over Group B remains robust ($\Delta\text{AUROC} = \mathbf{+0.080}$ [95% CI: $+0.041$, $+0.126$]).

4. **Hospitalized Inpatient Setting:**
   All data originates from hospitalized patients (Shanghai_T2DM, Zhao et al. 2023, *Scientific Data*, CC BY 4.0; 100 patients, 109 sessions). Inpatients follow standardized hospital meal schedules, reduced physical activity, and controlled bedtime routines. Predictive performance and countermeasure kinetics must be clinically validated before deployment to free-living outpatients.

5. **Exploratory Sulfonylurea Subgroup:**
   The entire cohort contains only 18 SU-treated patients (247 valid nights, 32 strict events across 11 patients). All SU-specific subgroup findings are explicitly labeled **exploratory**, and we do not assert secretagogue-specific causality.

6. **What-If Countermeasure Hypotheses:**
   Countermeasure predictions (bedtime complex carbohydrate snack yield and sulfonylurea dose scale factor) are based on established physiological parameter assumptions from the clinical literature, **not learned from intervention trials**. All simulator outputs are labeled *"simulation hypothesis, not clinically validated"*.

7. **Cold-Start Product Policy:**
   On a patient's very first monitored night when no personal history exists, the twin automatically falls back to the acute CGM-only risk model (Model B) and a population-average profile, displaying a prominent *"No personal history yet"* badge to alert the clinician.

---

## 11. Repository Architecture & Quickstart

```
DIGITAL TWIN/
├── api/                            # FastAPI backend
│   ├── main.py                     # Entry point & static mounts
│   ├── models.py                   # Pydantic schemas
│   └── routes.py                   # REST endpoints (/api/patients, /api/twin/what-if)
├── shared/
│   └── types.ts                    # Shared TypeScript contract
├── web/                            # React + Vite + TypeScript frontend
│   ├── public/data/                # Static UI JSON datasets (offline fallback)
│   ├── src/
│   │   ├── components/             # TriageBoard, PatientTwinView, EvidencePage, GuidedDemo
│   │   ├── services/api.ts         # Dual-mode data client (API + static fallback)
│   │   └── utils/clinical.ts       # Color-blind safe badges & formatting
│   ├── tests/smoke.spec.ts         # Playwright end-to-end smoke test
│   └── package.json
├── data/
│   ├── processed/                  # Generated datasets, ablation results, bootstrap CIs
│   └── raw/                        # Raw Shanghai_T2DM data (gitignored)
├── docs/
│   ├── AlooDUM_slides.pptx         # Final presentation deck (PowerPoint)
│   ├── AlooDUM_slides.pdf          # Final presentation deck (PDF)
│   └── architecture.pdf            # Vector system architecture diagram
├── reports/
│   └── figures/                    # Generated calibration & SHAP plots
├── scripts/
│   ├── run_demo.py                 # Cross-platform one-command launcher
│   ├── export_ui_data.py           # Exports self-contained JSON assets to web/public/data/
│   ├── export_home_traces.py       # Exports sample CGM timeline traces for Home screen
│   ├── evaluate_repeated_groupkfold.py # 20-run GroupKFold repeated CV benchmark
│   ├── evaluate_twin_forecast.py   # Benchmarks blended forecaster vs baselines with bootstrap CIs
│   ├── evaluate_conformal_splits.py# Evaluates 200 patient-level conformal splits
│   ├── evaluate_group_g_robustness.py # Robustness analysis across prior night history
│   ├── generate_displayed_model_figures.py # Plots calibration curve and SHAP beeswarm
│   ├── run_phase0.py               # Data audit and feature pipeline execution
│   └── scan_secrets.py             # Security scanner for credentials and tokens
├── src/
│   ├── data/                       # Canonical regimen classifier, parser, dataset builder
│   ├── models/                     # LogisticRegression, LightGBM, GroupKFold, Platt calibration
│   └── twin/                       # Blended forecaster, 2-compartment ODE, Kalman state filter
├── tests/
│   └── test_no_leakage.py          # Strict non-leakage & canonical consistency tests
├── Makefile                        # Automation targets
├── LICENSE                         # MIT License
├── requirements.txt
└── README.md
```

### Quickstart (Single-Command Execution)

#### Option A: One-Command Demo Launcher (Recommended)
```bash
python scripts/run_demo.py
```
This launcher starts the FastAPI backend, serves the React production web application, verifies data contracts, and automatically opens `http://localhost:8000` in your default browser. Press `Ctrl+C` to cleanly terminate both processes.

#### Option B: Clean Clone Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run automated zero-leakage unit tests
python -m pytest tests/

# 3. Check Python code quality
python -m ruff check .

# 4. Install web dependencies & run Vitest tests
cd web
npm install
npm test
npm run build

# 5. Start servers
# Terminal 1:
python -m uvicorn api.main:app --port 8000
# Terminal 2:
cd web && npm run dev
```

#### Option C: Playwright Smoke Testing
```bash
cd web
npx playwright test
```

---

## 12. Submission Assets, Data Provenance & Team Details

- **Competition:** Happiest Health Digital Twin Challenge 2026
- **Team:** AlooDUM
- **College:** Arka Jain University, Jamshedpur
- **Members:** Nivesh Kumar · Kriti Kumari · Bhoomi Singh
- **Team Leader:** Abhinav Tiwary
- **Repository:** [https://github.com/abhinavtiwary15/AlooDUM_ArkaJainUniversity](https://github.com/abhinavtiwary15/AlooDUM_ArkaJainUniversity)
- **License:** [LICENSE](LICENSE) (MIT License, Team AlooDUM)
- **Vector Architecture Diagram:** [docs/architecture.pdf](docs/architecture.pdf)
- **Final Presentation (PDF):** [docs/AlooDUM_slides.pdf](docs/AlooDUM_slides.pdf)
- **Final Presentation (PowerPoint):** [docs/AlooDUM_slides.pptx](docs/AlooDUM_slides.pptx)
- **Data Provenance & Attribution:**
  - Shanghai_T2DM continuous glucose monitoring dataset from Zhao, X. et al. *Scientific Data* 10, 836 (2023), published under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
  - Figshare Dataset Download: [https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397/3](https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397/3)
  - *Retraining Instructions:* The repository includes all pre-extracted, derived datasets in `data/processed/` and precomputed UI assets in `web/public/data/` for immediate out-of-the-box demo execution and testing. Downloading the raw Shanghai_T2DM dataset into `data/raw/` is **only required if you wish to run the raw preprocessing pipeline from scratch** (`python scripts/run_phase0.py`).
- **Tested Environment & Runtimes:**
  - Python: Tested on **Python 3.11, 3.12, 3.14** (64-bit).
  - Node.js & npm: Tested on **Node.js v20.x, v22.x, v24.x** (specifically Node v24.14.1, npm 11.11.0).
  - Operating Systems: Cross-platform compatibility verified on Windows 11, Ubuntu Linux, and macOS.

### Regulatory & Medical Disclaimer
*This digital twin software is an academic hackathon decision support prototype. It is NOT an FDA/CE-marked medical device, and does not provide diagnostic or therapeutic medical decisions. All what-if simulations are physiological hypotheses based on assumed model parameters and anonymized public hospital data.*

