import React, { useState, useMemo } from 'react';
import { ModelCardData, PatientSummary, RegimenCohort, RiskTier } from '../../../shared/types';
import { formatRegimenLabel, getRiskBadgeProps } from '../utils/clinical';
import { Footer } from './Footer';

interface TriageBoardProps {
  patients: PatientSummary[];
  selectedPatientId: string;
  onSelectPatient: (patientId: string) => void;
  modelCard: ModelCardData | null;
  showRecordedOutcome: boolean;
  onToggleRecordedOutcome: (show: boolean) => void;
}

export const TriageBoard: React.FC<TriageBoardProps> = ({
  patients,
  onSelectPatient,
  modelCard,
  showRecordedOutcome,
  onToggleRecordedOutcome,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCohort, setSelectedCohort] = useState<RegimenCohort | 'ALL'>('ALL');
  const [selectedTier, setSelectedTier] = useState<RiskTier | 'ALL'>('ALL');

  // Filter and sort patients strictly descending by calibrated risk
  const filteredPatients = useMemo(() => {
    return patients
      .filter((p) => {
        const matchesSearch =
          searchQuery.trim() === '' ||
          p.patientId.toLowerCase().includes(searchQuery.trim().toLowerCase());

        const matchesCohort =
          selectedCohort === 'ALL' ||
          (selectedCohort === 'SU_WITHOUT_INSULIN' && p.regimenCohort === 'SU_WITHOUT_INSULIN') ||
          (selectedCohort === 'SU_WITH_INSULIN' && p.regimenCohort === 'SU_WITH_INSULIN') ||
          (selectedCohort === 'INSULIN_WITHOUT_SU' && p.regimenCohort === 'INSULIN_WITHOUT_SU') ||
          (selectedCohort === 'NEITHER' && p.regimenCohort === 'NEITHER');

        const matchesTier = selectedTier === 'ALL' || p.riskTier === selectedTier;

        return matchesSearch && matchesCohort && matchesTier;
      })
      .sort((a, b) => b.calibratedRisk - a.calibratedRisk);
  }, [patients, searchQuery, selectedCohort, selectedTier]);

  // Dynamic numbers from data
  const totalPatients = modelCard?.cohortSummary.totalPatients || patients.length || 100;
  const totalNights = modelCard?.cohortSummary.totalNights
    ? modelCard.cohortSummary.totalNights.toLocaleString()
    : '1,181';
  const alertsPerPtWk = modelCard?.cohortSummary.alertsPerPtWkAt80Spec
    ? modelCard.cohortSummary.alertsPerPtWkAt80Spec.toFixed(1)
    : '1.8';

  return (
    <div className="min-h-screen bg-white text-[#0E1330] font-sans flex flex-col justify-between">
      <div className="max-w-[1160px] w-full mx-auto px-4 sm:px-6 pt-10 pb-16">
        {/* Overline & Title */}
        <div className="font-mono text-xs text-[#4A4F73] mb-2">Bedtime triage</div>
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0E1330] mb-4">
          Tonight's list, ranked by risk
        </h1>
        <p className="text-sm sm:text-base text-[#3A3F66] leading-relaxed max-w-3xl mb-10">
          Each patient's latest recorded night, replayed as if the forecast were made at 22:00. High means
          above the alert threshold at 80% specificity. Moderate means above half of it.
        </p>

        {/* 4 Metric Columns */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
          <div className="pt-3 border-t-2 border-[#1B1E4A]">
            <div className="font-mono text-xs text-[#4A4F73] mb-1">Patients</div>
            <div className="text-3xl sm:text-4xl font-bold text-[#0E1330] font-sans">
              {totalPatients}
            </div>
          </div>

          <div className="pt-3 border-t-2 border-[#1B1E4A]">
            <div className="font-mono text-xs text-[#4A4F73] mb-1">Nights replayed</div>
            <div className="text-3xl sm:text-4xl font-bold text-[#0E1330] font-sans">
              {totalNights}
            </div>
          </div>

          <div className="pt-3 border-t-2 border-[#1B1E4A]">
            <div className="font-mono text-xs text-[#4A4F73] mb-1">True lows among alerts</div>
            <div className="text-3xl sm:text-4xl font-bold text-[#0E1330] font-sans">
              about 1 in 3
            </div>
          </div>

          <div className="pt-3 border-t-2 border-[#1B1E4A]">
            <div className="font-mono text-xs text-[#4A4F73] mb-1">Alerts per patient-week</div>
            <div className="text-3xl sm:text-4xl font-bold text-[#0E1330] font-sans">
              {alertsPerPtWk}
            </div>
          </div>
        </div>

        {/* Filters and Controls Row */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex flex-wrap items-center gap-3">
            {/* Monotonic Tier Filter Buttons */}
            <div className="inline-flex rounded-[2px] border border-[#CBD0E2] overflow-hidden bg-white">
              {(['ALL', 'HIGH', 'MODERATE', 'LOW'] as const).map((tier) => {
                const isSelected = selectedTier === tier;
                const label = tier === 'ALL' ? 'All' : tier === 'HIGH' ? 'High' : tier === 'MODERATE' ? 'Moderate' : 'Low';
                return (
                  <button
                    key={tier}
                    onClick={() => setSelectedTier(tier)}
                    className={`px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-colors border-r last:border-r-0 border-[#CBD0E2] focus-visible:outline-[#1B1E4A] ${
                      isSelected
                        ? 'bg-[#1B1E4A] text-white font-semibold'
                        : 'bg-white text-[#0E1330] hover:bg-[#F1F3F9]'
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>

            {/* Regimen Dropdown */}
            <div className="relative">
              <select
                value={selectedCohort}
                onChange={(e) => setSelectedCohort(e.target.value as RegimenCohort | 'ALL')}
                className="bg-white border border-[#CBD0E2] text-[#0E1330] text-xs sm:text-sm rounded-[2px] px-3 py-1.5 pr-8 focus:outline-none focus:border-[#1B1E4A] cursor-pointer appearance-none"
              >
                <option value="ALL">All regimens ▾</option>
                <option value="SU_WITHOUT_INSULIN">Sulfonylurea</option>
                <option value="SU_WITH_INSULIN">SU + Insulin</option>
                <option value="INSULIN_WITHOUT_SU">Insulin</option>
                <option value="NEITHER">Neither</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-[#4A4F73] text-xs">
                ▼
              </div>
            </div>

            {/* Search Input */}
            <div>
              <input
                type="text"
                placeholder="Search patient ID"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-white border border-[#CBD0E2] text-[#0E1330] text-xs sm:text-sm rounded-[2px] px-3 py-1.5 w-44 sm:w-56 placeholder-[#4A4F73] focus:outline-none focus:border-[#1B1E4A]"
              />
            </div>
          </div>

          {/* Toggle: Show recorded outcome */}
          <div className="flex items-center space-x-2.5 select-none">
            <button
              role="switch"
              aria-checked={showRecordedOutcome}
              onClick={() => onToggleRecordedOutcome(!showRecordedOutcome)}
              className={`w-11 h-6 flex items-center rounded-full p-0.5 transition-colors focus-visible:outline-[#1B1E4A] ${
                showRecordedOutcome ? 'bg-[#1B1E4A]' : 'bg-[#CBD0E2]'
              }`}
            >
              <div
                className={`bg-white w-5 h-5 rounded-full transition-transform ${
                  showRecordedOutcome ? 'transform translate-x-5' : ''
                }`}
              />
            </button>
            <span className="text-xs sm:text-sm font-sans text-[#0E1330]">
              Show recorded outcome
            </span>
          </div>
        </div>

        {/* Triage Table */}
        <div className="border border-[#CBD0E2] rounded-[4px] overflow-hidden bg-white">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#F1F3F9] border-b border-[#CBD0E2] font-mono text-xs text-[#4A4F73]">
                  <th className="py-2.5 px-4 font-normal">Patient</th>
                  <th className="py-2.5 px-4 font-normal">Risk tonight</th>
                  <th className="py-2.5 px-4 font-normal">Tier</th>
                  <th className="py-2.5 px-4 font-normal">Bedtime</th>
                  <th className="py-2.5 px-4 font-normal">Prior lows</th>
                  <th className="py-2.5 px-4 font-normal">Regimen</th>
                  {showRecordedOutcome && (
                    <th className="py-2.5 px-4 font-normal text-right">Recorded outcome</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E3E6F1] text-sm">
                {filteredPatients.length === 0 ? (
                  <tr>
                    <td colSpan={showRecordedOutcome ? 7 : 6} className="py-8 text-center text-[#4A4F73] font-mono text-sm">
                      No matching patients found.
                    </td>
                  </tr>
                ) : (
                  filteredPatients.map((p) => {
                    const pct = Math.round(p.calibratedRisk * 100);
                    const tierProps = getRiskBadgeProps(p.riskTier);

                    return (
                      <tr
                        key={p.patientId}
                        onClick={() => onSelectPatient(p.patientId)}
                        className="hover:bg-[#F1F3F9] cursor-pointer transition-colors focus-within:bg-[#F1F3F9]"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            onSelectPatient(p.patientId);
                          }
                        }}
                      >
                        {/* Patient */}
                        <td className="py-3 px-4 font-medium text-[#0E1330] font-sans">
                          Patient {p.patientId}
                        </td>

                        {/* Risk tonight: Progress bar + % */}
                        <td className="py-3 px-4">
                          <div className="flex items-center space-x-3 max-w-[200px]">
                            <div className="flex-1 h-1.5 bg-[#E3E6F1] rounded-none overflow-hidden">
                              <div
                                className="h-full bg-[#1B1E4A]"
                                style={{ width: `${Math.min(100, pct)}%` }}
                              />
                            </div>
                            <span className="font-mono text-xs text-[#0E1330] w-9 text-right">
                              {pct}%
                            </span>
                          </div>
                        </td>

                        {/* Tier Chip */}
                        <td className="py-3 px-4">
                          <span
                            className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-[2px] text-xs font-medium border ${tierProps.badgeClass}`}
                          >
                            <span>{tierProps.symbol}</span>
                            <span>{tierProps.label}</span>
                          </span>
                        </td>

                        {/* Bedtime */}
                        <td className="py-3 px-4 font-mono text-xs text-[#0E1330]">
                          {Math.round(p.bedtimeGlucose)} mg/dL
                        </td>

                        {/* Prior lows */}
                        <td className="py-3 px-4 font-mono text-xs text-[#0E1330]">
                          {p.historicalStrictLows} of {p.totalRecordedNights} nights
                        </td>

                        {/* Regimen */}
                        <td className="py-3 px-4 font-sans text-xs text-[#0E1330]">
                          {formatRegimenLabel(p.regimenCohort)}
                        </td>

                        {/* Recorded Outcome (if toggled) */}
                        {showRecordedOutcome && (
                          <td className="py-3 px-4 text-right font-mono text-xs">
                            {p.tonightStrictLow ? (
                              <span className="text-[#A30F14] font-medium">Persistent low</span>
                            ) : (
                              <span className="text-[#046A38]">No persistent low</span>
                            )}
                          </td>
                        )}
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Table Caption */}
        <div className="mt-3 font-mono text-xs text-[#4A4F73]">
          Sample rows from replayed recorded nights. Click a row to open that patient's twin. Recorded outcomes stay hidden unless the switch is on.
        </div>
      </div>

      {/* Light Footer */}
      <Footer variant="light" />
    </div>
  );
};
