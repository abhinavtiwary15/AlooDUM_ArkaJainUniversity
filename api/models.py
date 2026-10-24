"""Pydantic schemas for the FastAPI backend matching shared/types.ts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskTier = Literal["HIGH", "MODERATE", "LOW"]
RegimenCohort = Literal[
    "SU_WITHOUT_INSULIN",
    "SU_WITH_INSULIN",
    "INSULIN_WITHOUT_SU",
    "NEITHER",
]


class PatientSummaryModel(BaseModel):
    patientId: str
    age: int
    gender: str
    bmi: float
    hba1c: float
    egfr: float
    regimenCohort: RegimenCohort
    isSu: bool
    isInsulin: bool
    suAgent: str
    suDoseMg: float
    totalDayInsulin: float
    tonightDate: str
    bedtimeGlucose: float
    calibratedRisk: float
    riskTier: RiskTier
    totalRecordedNights: int
    historicalStrictLows: int
    nightsSinceLastLow: float
    histNPriorNights: int = 0
    tonightStrictLow: bool = False
    isColdStartTonight: bool = False


class CGMPointModel(BaseModel):
    timestamp: str
    glucose: float
    isPostAnchor: bool
    isNocturnalWindow: bool


class TwinTrajectoryPointModel(BaseModel):
    timestamp: str
    minutes: int
    predictedGlucose: float
    lowerConformal: float
    upperConformal: float
    persistenceBaseline: float


class SHAPFactorModel(BaseModel):
    feature: str
    label: str
    value: Any
    impact: float
    direction: Literal["INCREASES_RISK", "DECREASES_RISK"]


class DeterministicClinicalSummaryModel(BaseModel):
    headline: str
    riskLevel: str
    calibratedProbabilityPct: float
    operatingPointContext: str
    primaryDrivers: list[str]
    recommendedAction: str
    disclaimer: str


class NightDetailModel(BaseModel):
    patientId: str
    nightDate: str
    bedtimeGlucose: float
    bedtimeSlope1h: float
    bedtimeSlope3h: float
    calibratedRisk: float
    riskTier: RiskTier
    actualGroundTruthStrictLow: bool
    actualGroundTruthNadir: float
    cgmSeries: list[CGMPointModel]
    twinForecast: list[TwinTrajectoryPointModel]
    shapFactors: list[SHAPFactorModel]
    clinicalSummary: DeterministicClinicalSummaryModel
    isColdStart: bool = False
    riskModelUsed: str = "CGM_AND_PATIENT_HISTORY"
    fallbackForecastType: str | None = None


class WhatIfRequestModel(BaseModel):
    patientId: str
    bedtimeDate: str | None = None
    snackCarbsG: float = Field(default=0.0, ge=0.0, le=100.0)
    suDoseScale: float = Field(default=1.0, ge=0.0, le=2.0)


class WhatIfResponseModel(BaseModel):
    patientId: str
    snackCarbsG: float
    suDoseScale: float
    baselineNadir: float
    simulatedNadir: float
    deltaNadirFromBaseline: float
    simulatedMinutesBelow70: int
    simulatedTrajectory: list[TwinTrajectoryPointModel]
    disclaimer: str
