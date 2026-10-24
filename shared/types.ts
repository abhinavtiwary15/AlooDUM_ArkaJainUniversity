/**
 * Shared Type Contract between FastAPI Backend and React Frontend.
 * Ensures compile-time strictness and zero mismatch across boundaries.
 */

export type RiskTier = 'HIGH' | 'MODERATE' | 'LOW';

export type RegimenCohort =
  | 'SU_WITHOUT_INSULIN'
  | 'SU_WITH_INSULIN'
  | 'INSULIN_WITHOUT_SU'
  | 'NEITHER';

export interface PatientSummary {
  patientId: string;
  age: number;
  gender: string;
  bmi: number;
  hba1c: number;
  egfr: number;
  regimenCohort: RegimenCohort;
  isSu: boolean;
  isInsulin: boolean;
  suAgent: string;
  suDoseMg: number;
  totalDayInsulin: number;
  tonightDate: string;
  bedtimeGlucose: number;
  calibratedRisk: number;
  riskTier: RiskTier;
  totalRecordedNights: number;
  historicalStrictLows: number;
  nightsSinceLastLow: number;
  histNPriorNights?: number;
  tonightStrictLow?: boolean;
  isColdStartTonight?: boolean;
}

export interface CGMPoint {
  timestamp: string; // ISO 8601 string
  glucose: number;   // mg/dL
  isPostAnchor: boolean;
  isNocturnalWindow: boolean;
}

export interface TwinTrajectoryPoint {
  timestamp: string;
  minutes: number;
  predictedGlucose: number;
  lowerConformal: number;
  upperConformal: number;
  persistenceBaseline: number;
}

export interface SHAPFactor {
  feature: string;
  label: string;
  value: number | string;
  impact: number; // positive = increases risk, negative = protective
  direction: 'INCREASES_RISK' | 'DECREASES_RISK';
}

export interface DeterministicClinicalSummary {
  headline: string;
  riskLevel: string;
  calibratedProbabilityPct: number;
  operatingPointContext: string;
  primaryDrivers: string[];
  recommendedAction: string;
  disclaimer: string;
}

export interface NightDetail {
  patientId: string;
  nightDate: string;
  bedtimeGlucose: number;
  bedtimeSlope1h: number;
  bedtimeSlope3h: number;
  calibratedRisk: number;
  riskTier: RiskTier;
  actualGroundTruthStrictLow: boolean;
  actualGroundTruthNadir: number;
  cgmSeries: CGMPoint[];
  twinForecast: TwinTrajectoryPoint[];
  shapFactors: SHAPFactor[];
  clinicalSummary: DeterministicClinicalSummary;
  isColdStart: boolean;
  riskModelUsed: string;
  fallbackForecastType?: string;
  priorNightLow?: boolean;
  priorLowCount?: number;
  nightsSinceLastLow?: number | null;
  personalMedianNadir?: number | null;
}

export interface WhatIfRequest {
  patientId: string;
  bedtimeDate?: string;
  snackCarbsG: number;
  suDoseScale: number;
}

export interface WhatIfResponse {
  patientId: string;
  snackCarbsG: number;
  suDoseScale: number;
  baselineNadir: number;
  simulatedNadir: number;
  deltaNadirFromBaseline: number;
  simulatedMinutesBelow70: number;
  simulatedTrajectory: TwinTrajectoryPoint[];
  disclaimer: string;
}

export interface ModelCardData {
  cohortSummary: {
    totalPatients: number;
    totalNights: number;
    baseStrictPrevalencePct: number;
    strictEventsCount: number;
    suPatientsCount: number;
    suStrictEventsCount: number;
    conformalMarginMgDl: number;
    conformalCoveragePct: number;
    conformalMargin80MgDl?: number;
    conformalMargin90MgDl?: number;
    conformalCoverage80Pct?: number;
    conformalCoverage90Pct?: number;
    alertThreshold80Spec?: number;
    alertThresholdModerate?: number;
    ppvAt80SpecPct?: number;
    alertsPerPtWkAt80Spec?: number;
    nnwAt80Spec?: number;
  };
  ablationTable: Array<{
    group: string;
    model: string;
    cohort: string;
    aurocCi: string;
    auprcCi: string;
    brier: number;
    bssPct: number;
    ece: number;
    sens80SpecPct: number;
    ppv80SpecPct: number;
    nnw80Spec: number;
    alertsPerPtWk: number;
  }>;
  pairedDifferences: Array<{
    model: string;
    comparison: string;
    deltaAurocCi: string;
    aurocSignificant: boolean;
    deltaAuprcCi: string;
    auprcSignificant: boolean;
    deltaBrierCi: string;
    brierSignificant: boolean;
  }>;
  forecasterBenchmark: Array<{
    horizon: string;
    model: string;
    maeMgDl: number;
    rmseMgDl: number;
    nNights: number;
  }>;
  credibility75Ppv: Array<{
    model: string;
    threshold: number | null;
    sensAt75PpvPct: number;
    specAt75PpvPct: number;
    ppvAchievedPct: number;
    nnwAchieved: number;
    requiredLrPlus: number;
    observedLrPlus: number;
  }>;
  outlierSensitivity: {
    top10PctEventSharePct: number;
    suBaselineRatePct: number;
    suExclTop1RatePct: number;
    suTop1PatientId: string;
    suTop1EventFraction: string;
    softenedRiskRatio: number;
  };
  conformal200Splits?: {
    fold_results?: Array<{
      fold: number;
      tau_hours: number;
      n_train_pts: number;
      n_val_pts: number;
      mae_blend_1h: number;
      mae_pers_1h: number;
      mae_prof_1h: number;
      mae_blend_3h: number;
      mae_pers_3h: number;
      mae_prof_3h: number;
      mae_blend_6h: number;
      mae_pers_6h: number;
      mae_prof_6h: number;
    }>;
    conformal_splits_summary?: {
      overall_80: { mean: number; range: [number, number]; ci95: [number, number] };
      overall_90: { mean: number; range: [number, number]; ci95: [number, number] };
      h1_80: { mean: number; range: [number, number]; ci95: [number, number] };
      h1_90: { mean: number; range: [number, number]; ci95: [number, number] };
      h3_80: { mean: number; range: [number, number]; ci95: [number, number] };
      h3_90: { mean: number; range: [number, number]; ci95: [number, number] };
      h6_80: { mean: number; range: [number, number]; ci95: [number, number] };
      h6_90: { mean: number; range: [number, number]; ci95: [number, number] };
      cond_low_80: { mean: number; range: [number, number]; ci95: [number, number] };
      cond_low_90: { mean: number; range: [number, number]; ci95: [number, number] };
      q80_margin_mean: number;
      q90_margin_mean: number;
    };
  };
  groupGRobustness?: {
    identical_subset_ge1_prior_night: {
      auc_a: number;
      auc_b: number;
      diff_auc: number;
      diff_auc_ci: [number, number];
      prc_a: number;
      prc_b: number;
      diff_prc: number;
      diff_prc_ci: [number, number];
      brier_a: number;
      brier_b: number;
      diff_brier: number;
      diff_brier_ci: [number, number];
    };
    event_concentration_sensitivity: {
      top1_excluded: { auc_a: number; auc_b: number; diff_auc: number; diff_auc_ci: [number, number] };
      top2_excluded: { auc_a: number; auc_b: number; diff_auc: number; diff_auc_ci: [number, number] };
      top10pct_excluded: { auc_a: number; auc_b: number; diff_auc: number; diff_auc_ci: [number, number] };
    };
    stratified_by_prior_nights: Array<{
      bin: string;
      nights: number;
      events: number;
      event_rate: number;
      b_auroc: number;
      g_auroc: number;
      b_auprc: number;
      g_auprc: number;
    }>;
    cold_start_night_0: Array<{
      model: string;
      auroc: number;
      auprc: number;
      brier: number;
      bss: number;
    }>;
  };
  repeatedGroupKFoldHeadline?: {
    n_repeats: number;
    n_nights_history: number;
    n_events_history: number;
    model: string;
    group_b_cgm_only: {
      auroc: { mean: number; min: number; max: number; std: number };
      auprc: { mean: number; min: number; max: number; std: number };
      brier: { mean: number; min: number; max: number; std: number };
    };
    group_g_cgm_history: {
      auroc: { mean: number; min: number; max: number; std: number };
      auprc: { mean: number; min: number; max: number; std: number };
      brier: { mean: number; min: number; max: number; std: number };
    };
    differences_g_minus_b: {
      delta_auroc: { mean: number; min: number; max: number; std: number };
      delta_auprc: { mean: number; min: number; max: number; std: number };
      delta_brier: { mean: number; min: number; max: number; std: number };
    };
    paired_bootstrap_ci: {
      diff_auc: number;
      diff_auc_ci: [number, number];
      diff_prc: number;
      diff_prc_ci: [number, number];
      diff_brier: number;
      diff_brier_ci: [number, number];
    };
  };
}
