import React, { useEffect, useState } from 'react';
import { ModelCardData, PatientSummary } from '../../../shared/types';
import { PROJECT_NAME, TEAM_NAME, INSTITUTION, CHALLENGE_NAME } from '../config';
import { Footer } from './Footer';

interface HomeViewProps {
  onOpenTriage: () => void;
  onOpenEvidence: () => void;
  onSelectPatient: (patientId: string) => void;
  modelCard: ModelCardData | null;
  patients: PatientSummary[];
}

interface HomeTrace {
  patientId: string;
  hasLow: boolean;
  values: number[]; // 11 points: 20:00 to 06:00
}

interface HomeTracesData {
  metadata: {
    caption: string;
    totalTraces: number;
    lowTraces: number;
  };
  traces: HomeTrace[];
}

export const HomeView: React.FC<HomeViewProps> = ({
  onOpenTriage,
  onOpenEvidence,
  onSelectPatient,
  modelCard,
  patients,
}) => {
  const [tracesData, setTracesData] = useState<HomeTracesData | null>(null);

  useEffect(() => {
    fetch('/data/home_traces.json')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.traces) {
          setTracesData(data);
        }
      })
      .catch(() => {
        // Fallback gracefully if not reachable
      });
  }, []);

  // Compute metrics dynamically from model_card
  const rep = modelCard?.repeatedGroupKFoldHeadline;
  const groupG = modelCard?.ablationTable.find(
    (r) => r.group === 'G_cgm_history' && r.model === 'LogisticRegression' && r.cohort.includes('All')
  );

  const aurocG = rep ? rep.group_g_cgm_history.auroc.mean.toFixed(3) : '0.832';
  const aurocGRange = rep ? `${rep.group_g_cgm_history.auroc.min.toFixed(3)}–${rep.group_g_cgm_history.auroc.max.toFixed(3)}` : '0.824–0.840';
  const aurocB = rep ? rep.group_b_cgm_only.auroc.mean.toFixed(3) : '0.736';
  const aurocBRange = rep ? `${rep.group_b_cgm_only.auroc.min.toFixed(3)}–${rep.group_b_cgm_only.auroc.max.toFixed(3)}` : '0.707–0.753';
  const aurocDelta = rep ? `+${rep.differences_g_minus_b.delta_auroc.mean.toFixed(3)} [95% CI: +${rep.paired_bootstrap_ci.diff_auc_ci[0].toFixed(3)} to +${rep.paired_bootstrap_ci.diff_auc_ci[1].toFixed(3)}]` : '+0.097 [95% CI: +0.051 to +0.149]';

  // Forecaster 1h, 3h error
  const f1hBlend = modelCard?.forecasterBenchmark.find((b) => b.horizon.includes('1-Hour') && b.model.includes('Blended'));
  const f1hPers = modelCard?.forecasterBenchmark.find((b) => b.horizon.includes('1-Hour') && b.model.includes('Persistence'));
  const f3hBlend = modelCard?.forecasterBenchmark.find((b) => b.horizon.includes('3-Hour') && b.model.includes('Blended'));
  const f3hPers = modelCard?.forecasterBenchmark.find((b) => b.horizon.includes('3-Hour') && b.model.includes('Persistence'));

  const mae1h = f1hBlend ? f1hBlend.maeMgDl.toFixed(1) : '14.8';
  const mae1hPers = f1hPers ? f1hPers.maeMgDl.toFixed(1) : '18.5';
  const mae3h = f3hBlend ? f3hBlend.maeMgDl.toFixed(1) : '19.4';
  const mae3hPers = f3hPers ? f3hPers.maeMgDl.toFixed(1) : '32.0';

  // Uncertainty 90%
  const margin90 = modelCard?.cohortSummary.conformalMargin90MgDl
    ? Math.round(modelCard.cohortSummary.conformalMargin90MgDl)
    : 38;
  const coverageOverall90 = modelCard?.conformal200Splits?.conformal_splits_summary?.overall_90.mean
    ? modelCard.conformal200Splits.conformal_splits_summary.overall_90.mean.toFixed(1)
    : '89.8';
  const coverage3h90 = modelCard?.conformal200Splits?.conformal_splits_summary?.h3_90.mean
    ? modelCard.conformal200Splits.conformal_splits_summary.h3_90.mean.toFixed(1)
    : '86.3';

  // Alert burden
  const alertsWk = groupG?.alertsPerPtWk ? groupG.alertsPerPtWk.toFixed(1) : '1.8';

  // Sort patients descending by calibrated risk for the preview
  const previewPatients = [...patients]
    .sort((a, b) => b.calibratedRisk - a.calibratedRisk)
    .slice(0, 4);

  return (
    <div className="min-h-screen bg-white text-[#0E1330] font-sans">
      {/* 1. HERO SECTION (NAVY) */}
      <section className="relative bg-[#1B1E4A] text-white pt-12 pb-16 px-4 sm:px-6 overflow-hidden border-b border-[#2A2E63]">
        <div className="max-w-[1160px] mx-auto relative z-10">
          {/* Metadata overline */}
          <div className="font-mono text-xs sm:text-sm text-[#AEB4E0] mb-6 tracking-wide">
            {CHALLENGE_NAME} &nbsp;/&nbsp; {TEAM_NAME} &nbsp;/&nbsp; {INSTITUTION}
          </div>

          {/* Main Headline */}
          <h1 className="text-4xl sm:text-6xl md:text-7xl font-bold tracking-tight text-white mb-6 leading-[1.08] max-w-4xl">
            See the night low before it happens.
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-[#C9CEF0] leading-relaxed max-w-2xl mb-8">
            {PROJECT_NAME} learns each patient's overnight glucose pattern from their earlier nights. At bedtime it
            estimates the risk of a persistent low, with an uncertainty range a clinician can inspect.
          </p>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-4 mb-14">
            <button
              onClick={onOpenTriage}
              className="bg-white text-[#1B1E4A] hover:bg-[#F1F3F9] font-semibold text-sm sm:text-base px-6 py-3 rounded-[2px] transition-colors focus-visible:outline-white"
            >
              Open the live demo
            </button>
            <button
              onClick={onOpenEvidence}
              className="bg-[#1B1E4A] hover:bg-[#252A63] text-white border border-[#CBD0E2]/40 font-medium text-sm sm:text-base px-6 py-3 rounded-[2px] transition-colors focus-visible:outline-white"
            >
              Read the evidence
            </button>
          </div>

          {/* Hero Chart with Real ShanghaiT2DM Traces Background */}
          <div className="relative w-full rounded-[4px] bg-[#15173A] border border-[#2B306B] p-4 sm:p-6 overflow-hidden">
            {/* Real Recorded Traces Background (20:00 to 06:00) */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none opacity-40"
              viewBox="0 0 1000 360"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              {tracesData?.traces.map((tr, i) => {
                // Map 11 points (20:00 to 06:00) across x: 100 to 950, y: glucose 40 to 250 -> y: 320 to 40
                const pts = tr.values.map((v, idx) => {
                  const x = 100 + idx * (850 / 10);
                  const y = 320 - ((v - 40) / 210) * 280;
                  return `${x.toFixed(1)},${y.toFixed(1)}`;
                });
                const d = 'M ' + pts.join(' L ');
                const strokeColor = tr.hasLow ? '#FF7A7F' : '#FFFFFF';
                const strokeOpacity = tr.hasLow ? '0.35' : '0.07';
                return (
                  <path
                    key={`${tr.patientId}-${i}`}
                    d={d}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={tr.hasLow ? '1.4' : '0.8'}
                    strokeOpacity={strokeOpacity}
                  />
                );
              })}
            </svg>

            {/* Foreground Labelled Illustration Chart */}
            <div className="relative z-10 w-full">
              <div className="w-full h-64 sm:h-72">
                <svg className="w-full h-full" viewBox="0 0 1000 300" preserveAspectRatio="none">
                  {/* Grid Lines & Labels */}
                  {/* 200 mg/dL */}
                  <line x1="80" y1="50" x2="950" y2="50" stroke="#2B306B" strokeWidth="1" />
                  <text x="35" y="54" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">200</text>

                  {/* 150 mg/dL */}
                  <line x1="80" y1="110" x2="950" y2="110" stroke="#2B306B" strokeWidth="1" />
                  <text x="35" y="114" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">150</text>

                  {/* 100 mg/dL */}
                  <line x1="80" y1="170" x2="950" y2="170" stroke="#2B306B" strokeWidth="1" />
                  <text x="35" y="174" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">100</text>

                  {/* 70 mg/dL (Ochre Threshold) */}
                  <line x1="80" y1="210" x2="950" y2="210" stroke="#F2B84B" strokeWidth="1.2" strokeDasharray="4 4" />
                  <text x="35" y="214" fill="#F2B84B" fontSize="11" fontFamily="Spline Sans Mono" fontWeight="600">70</text>

                  {/* 54 mg/dL (Severe Low Threshold) */}
                  <line x1="80" y1="235" x2="950" y2="235" stroke="#FF7A7F" strokeWidth="1.2" strokeDasharray="4 4" />
                  <text x="35" y="239" fill="#FF7A7F" fontSize="11" fontFamily="Spline Sans Mono" fontWeight="600">54</text>

                  <text x="35" y="25" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">mg/dL</text>

                  {/* 22:00 Bedtime Anchor Vertical Line */}
                  <line x1="370" y1="30" x2="370" y2="260" stroke="#CBD0E2" strokeWidth="1.2" />
                  <text x="375" y="42" fill="#FFFFFF" fontSize="11" fontFamily="Spline Sans Mono">
                    22:00 bedtime anchor
                  </text>

                  {/* Conformal Uncertainty Shaded Band (22:00 to 06:00) */}
                  <path
                    d="M 370 170 C 450 180, 550 195, 710 205 C 800 200, 880 185, 950 175 L 950 250 C 880 260, 800 265, 710 265 C 550 245, 450 210, 370 170 Z"
                    fill="#3F4685"
                    fillOpacity="0.45"
                  />

                  {/* Recorded Prior Curve (18:00 to 22:00, solid white) */}
                  <path
                    d="M 80 115 C 130 95, 180 80, 220 95 C 270 115, 320 150, 370 170"
                    fill="none"
                    stroke="#FFFFFF"
                    strokeWidth="2.5"
                  />

                  {/* Blended Expected Forecast Curve (22:00 to 06:00, dashed white) */}
                  <path
                    d="M 370 170 C 450 190, 550 215, 710 230 C 800 228, 880 220, 950 205"
                    fill="none"
                    stroke="#FFFFFF"
                    strokeWidth="2"
                    strokeDasharray="6 4"
                  />

                  {/* Simulated What-if Curve (Mint Dotted) */}
                  <path
                    d="M 370 170 C 450 173, 550 183, 710 192 C 800 187, 880 175, 950 160"
                    fill="none"
                    stroke="#4ADE9C"
                    strokeWidth="2.5"
                    strokeDasharray="2 4"
                  />

                  {/* Drop Line at 03:00 pointing to where lower band dips below 70 */}
                  <line x1="710" y1="100" x2="710" y2="265" stroke="#CBD0E2" strokeWidth="1" strokeDasharray="2 3" />
                  <text x="715" y="95" fill="#FFFFFF" fontSize="11" fontFamily="Spline Sans Mono">
                    Forecast range falls below 70 mg/dL
                  </text>

                  {/* Legend inside illustration */}
                  <g transform="translate(760, 120)">
                    <line x1="0" y1="0" x2="20" y2="0" stroke="#FFFFFF" strokeWidth="2.5" />
                    <text x="28" y="4" fill="#FFFFFF" fontSize="11" fontFamily="Archivo">Recorded</text>

                    <line x1="0" y1="20" x2="20" y2="20" stroke="#FFFFFF" strokeWidth="2" strokeDasharray="5 3" />
                    <text x="28" y="24" fill="#FFFFFF" fontSize="11" fontFamily="Archivo">Forecast and 90% range</text>

                    <line x1="0" y1="40" x2="20" y2="40" stroke="#4ADE9C" strokeWidth="2.5" strokeDasharray="2 3" />
                    <text x="28" y="44" fill="#4ADE9C" fontSize="11" fontFamily="Archivo">What-if: countermeasure</text>
                  </g>

                  {/* Time Axis Labels */}
                  <text x="80" y="285" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">18:00</text>
                  <text x="350" y="285" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">22:00</text>
                  <text x="500" y="285" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">00:00</text>
                  <text x="640" y="285" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">02:00</text>
                  <text x="780" y="285" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">04:00</text>
                  <text x="920" y="285" fill="#8E95C7" fontSize="11" fontFamily="Spline Sans Mono">06:00</text>
                </svg>
              </div>

              {/* Caption */}
              <div className="mt-4 pt-3 border-t border-[#2B306B] font-mono text-xs text-[#AEB4E0]">
                Faint lines show 275 recorded nights from the public ShanghaiT2DM dataset, enriched for persistent lows (60 of 275 versus a true rate of about 10.5%). The foreground chart is a labelled illustration, not patient data.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. THE PROBLEM SECTION */}
      <section className="py-20 px-4 sm:px-6 max-w-[1160px] mx-auto">
        <div className="font-mono text-xs uppercase tracking-wider text-[#4A4F73] mb-3">
          The problem
        </div>
        <h2 className="text-2xl sm:text-4xl md:text-5xl font-bold tracking-tight text-[#0E1330] leading-tight mb-8 max-w-4xl">
          Nocturnal hypoglycemia often goes unnoticed. A sleeping patient cannot react to falling glucose, and a clinic visit sees fasting glucose and HbA1c, not the night in between.
        </h2>
        <p className="text-base sm:text-lg text-[#3A3F66] leading-relaxed max-w-3xl">
          Continuous glucose monitors record the night, but thousands of readings are hard to review before the next treatment decision. A clinician needs a short list of who is at risk tonight, and a reason for each name.
        </p>
      </section>

      {/* 3. HOW IT WORKS / ONE NIGHT IN THREE MOMENTS */}
      <section className="bg-[#F1F3F9] py-20 px-4 sm:px-6 border-y border-[#E3E6F1]">
        <div className="max-w-[1160px] mx-auto">
          <div className="font-mono text-xs uppercase tracking-wider text-[#4A4F73] mb-3">
            How it works
          </div>
          <h2 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0E1330] mb-12">
            One night, in three moments
          </h2>

          <div className="space-y-12">
            {/* Moment 1 */}
            <div className="pt-8 border-t border-[#CBD0E2] grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="font-sans text-4xl sm:text-5xl font-bold text-[#0E1330]">
                18:00
              </div>
              <div className="md:col-span-3">
                <h3 className="text-xl font-bold text-[#0E1330] mb-2 font-sans">Learn</h3>
                <p className="text-base text-[#3A3F66] leading-relaxed">
                  Earlier nights, 15-minute CGM readings, medications and labs build a personal overnight profile. Nothing recorded after 22:00 is ever used.
                </p>
              </div>
            </div>

            {/* Moment 2 */}
            <div className="pt-8 border-t border-[#CBD0E2] grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="font-sans text-4xl sm:text-5xl font-bold text-[#0E1330]">
                22:00
              </div>
              <div className="md:col-span-3">
                <h3 className="text-xl font-bold text-[#0E1330] mb-2 font-sans">Synchronise</h3>
                <p className="text-base text-[#3A3F66] leading-relaxed">
                  At bedtime the forecast starts from the actual reading, then relaxes toward the patient's own profile as the night goes on.
                </p>
              </div>
            </div>

            {/* Moment 3 */}
            <div className="pt-8 border-t border-[#CBD0E2] grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="font-sans text-4xl sm:text-5xl font-bold text-[#0E1330]">
                23:00 <span className="text-lg font-normal block sm:inline text-[#4A4F73]">to 06:00</span>
              </div>
              <div className="md:col-span-3">
                <h3 className="text-xl font-bold text-[#0E1330] mb-2 font-sans">Estimate and explore</h3>
                <p className="text-base text-[#3A3F66] leading-relaxed">
                  A calibrated probability of a persistent low, an uncertainty range, and what-if scenarios that are labelled as simulations.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-12 pt-6 border-t border-[#CBD0E2] font-mono text-xs text-[#4A4F73]">
            A persistent low means two or more consecutive readings below 70 mg/dL between 23:00 and 06:00. Every result is validated by patient, so no patient appears in both training and testing.
          </div>
        </div>
      </section>

      {/* 4. EVIDENCE / WHAT THE DATA SUPPORTS */}
      <section className="py-20 px-4 sm:px-6 max-w-[1160px] mx-auto">
        <div className="font-mono text-xs uppercase tracking-wider text-[#4A4F73] mb-3">
          Evidence
        </div>
        <h2 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0E1330] mb-14">
          What the data supports
        </h2>

        {/* 4 Metric Columns with Top Hairlines */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {/* Card 1 */}
          <div className="pt-4 border-t-2 border-[#0E1330]">
            <div className="font-mono text-xs text-[#4A4F73] mb-2">Personal history</div>
            <div className="text-5xl sm:text-6xl font-bold text-[#0E1330] font-sans mb-3 tracking-tight">
              {aurocG}
            </div>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              AUROC with earlier nights: {aurocG} ({aurocGRange}), against {aurocB} ({aurocBRange}) for CGM alone. Paired gain {aurocDelta}.
            </p>
          </div>

          {/* Card 2 */}
          <div className="pt-4 border-t-2 border-[#0E1330]">
            <div className="font-mono text-xs text-[#4A4F73] mb-2">Forecast error</div>
            <div className="text-5xl sm:text-6xl font-bold text-[#0E1330] font-sans mb-3 tracking-tight">
              {mae1h}<span className="text-2xl font-medium text-[#4A4F73]">mg/dL</span>
            </div>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              Mean error at one hour, against {mae1hPers} for holding the bedtime value. At three hours: {mae3h} against {mae3hPers}.
            </p>
          </div>

          {/* Card 3 */}
          <div className="pt-4 border-t-2 border-[#0E1330]">
            <div className="font-mono text-xs text-[#4A4F73] mb-2">Uncertainty</div>
            <div className="text-5xl sm:text-6xl font-bold text-[#0E1330] font-sans mb-3 tracking-tight">
              ±{margin90}<span className="text-2xl font-medium text-[#4A4F73]">mg/dL</span>
            </div>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              The 90% range. Held-out coverage is {coverageOverall90}% overall, and dips to {coverage3h90}% around the three-hour mark.
            </p>
          </div>

          {/* Card 4 */}
          <div className="pt-4 border-t-2 border-[#0E1330]">
            <div className="font-mono text-xs text-[#4A4F73] mb-2">Alert burden</div>
            <div className="text-5xl sm:text-6xl font-bold text-[#0E1330] font-sans mb-3 tracking-tight">
              1 in 3
            </div>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              alerts is a true persistent low at 80% specificity, about {alertsWk} alerts per patient-week.
            </p>
          </div>
        </div>

        {/* What Did Not Help Strip */}
        <div className="bg-[#F1F3F9] rounded-[4px] p-6 border border-[#E3E6F1]">
          <div className="font-mono text-xs text-[#4A4F73] mb-4">
            What did not help, reported as found
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div>
              <div className="font-bold text-[#0E1330] text-base mb-1">Static labs</div>
              <div className="font-mono text-xs text-[#4A4F73]">No measurable gain</div>
            </div>
            <div>
              <div className="font-bold text-[#0E1330] text-base mb-1">Demographics</div>
              <div className="font-mono text-xs text-[#4A4F73]">No measurable gain</div>
            </div>
            <div>
              <div className="font-bold text-[#0E1330] text-base mb-1">Medication tags</div>
              <div className="font-mono text-xs text-[#4A4F73]">No measurable gain</div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. THE CLINICIAN'S VIEW / PREVIEW OF TRIAGE */}
      <section className="py-20 px-4 sm:px-6 max-w-[1160px] mx-auto border-t border-[#E3E6F1]">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
          {/* Left Text Column */}
          <div className="lg:col-span-5">
            <div className="font-mono text-xs uppercase tracking-wider text-[#4A4F73] mb-3">
              The clinician's view
            </div>
            <h2 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0E1330] mb-6 leading-tight">
              Tonight's list, ranked by risk
            </h2>
            <p className="text-base text-[#3A3F66] leading-relaxed mb-8">
              Open a patient to see the glucose history, the twin's forecast and range, the factors behind the flag, and what-if options for a snack or a dose change.
            </p>
            <button
              onClick={onOpenTriage}
              className="bg-[#1B1E4A] hover:bg-[#252A63] text-white font-semibold text-sm px-6 py-3 rounded-[2px] transition-colors focus-visible:outline-[#1B1E4A]"
            >
              Open the full list
            </button>
          </div>

          {/* Right Preview Card */}
          <div className="lg:col-span-7 bg-[#F1F3F9] rounded-[4px] border border-[#CBD0E2] overflow-hidden">
            <div className="px-5 py-3 border-b border-[#CBD0E2] font-mono text-xs font-semibold text-[#0E1330] bg-white">
              Bedtime triage · retrospective replay
            </div>
            <div className="divide-y divide-[#E3E6F1] bg-white">
              <div className="grid grid-cols-12 px-5 py-2.5 font-mono text-xs text-[#4A4F73] bg-[#F8FAFC]">
                <div className="col-span-5">Patient</div>
                <div className="col-span-4">Risk tonight</div>
                <div className="col-span-3 text-right">Tier</div>
              </div>

              {previewPatients.map((p) => {
                const pct = Math.round(p.calibratedRisk * 100);
                const tierSymbol = p.riskTier === 'HIGH' ? '▲' : p.riskTier === 'MODERATE' ? '◆' : '●';
                const tierBadge =
                  p.riskTier === 'HIGH'
                    ? 'text-[#A30F14] bg-[#FCE4E4] border-[#F8B4B4]'
                    : p.riskTier === 'MODERATE'
                    ? 'text-[#7A4F00] bg-[#FFF3D6] border-[#FFE194]'
                    : 'text-[#046A38] bg-[#E2F1E8] border-[#BDE3CC]';
                const tierLabel = p.riskTier === 'HIGH' ? 'High' : p.riskTier === 'MODERATE' ? 'Moderate' : 'Low';

                return (
                  <button
                    key={p.patientId}
                    onClick={() => onSelectPatient(p.patientId)}
                    className="w-full text-left grid grid-cols-12 px-5 py-3 items-center hover:bg-[#F1F3F9] transition-colors focus-visible:outline-[#1B1E4A]"
                  >
                    <div className="col-span-5 text-sm font-semibold text-[#0E1330] font-sans">
                      Patient {p.patientId}
                    </div>
                    <div className="col-span-4 flex items-center space-x-2">
                      <div className="w-16 h-1.5 bg-[#E3E6F1] rounded-none overflow-hidden">
                        <div
                          className="h-full bg-[#1B1E4A]"
                          style={{ width: `${Math.min(100, pct)}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs text-[#0E1330]">{pct}%</span>
                    </div>
                    <div className="col-span-3 text-right">
                      <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-[2px] text-xs font-medium border ${tierBadge}`}>
                        <span>{tierSymbol}</span>
                        <span>{tierLabel}</span>
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="px-5 py-2.5 font-mono text-xs text-[#4A4F73] bg-[#F1F3F9] border-t border-[#CBD0E2]">
              Sample rows from replayed recorded nights. Not live patient data.
            </div>
          </div>
        </div>
      </section>

      {/* 6. LIMITS / WHAT WE STATE UP FRONT */}
      <section className="py-20 px-4 sm:px-6 max-w-[1160px] mx-auto border-t border-[#E3E6F1]">
        <div className="font-mono text-xs uppercase tracking-wider text-[#4A4F73] mb-3">
          Limits
        </div>
        <h2 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0E1330] mb-14">
          What we state up front
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="pt-4 border-t-2 border-[#0E1330]">
            <h3 className="text-lg font-bold text-[#0E1330] mb-3 font-sans">
              Hospitalised cohort
            </h3>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              100 patients from the public ShanghaiT2DM dataset, monitored for 3 to 14 days. Free-living outpatients may behave differently.
            </p>
          </div>

          <div className="pt-4 border-t-2 border-[#0E1330]">
            <h3 className="text-lg font-bold text-[#0E1330] mb-3 font-sans">
              Sulfonylurea results are exploratory
            </h3>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              Only 18 patients take them, and 11 have events. We report the subgroup, but do not claim a causal effect.
            </p>
          </div>

          <div className="pt-4 border-t-2 border-[#0E1330]">
            <h3 className="text-lg font-bold text-[#0E1330] mb-3 font-sans">
              What-if effects are assumptions
            </h3>
            <p className="text-sm text-[#3A3F66] leading-relaxed">
              Snack and dose scenarios use assumed physiology, not effects learned from the data. They generate hypotheses to discuss.
            </p>
          </div>
        </div>
      </section>

      {/* 7. NAVY FOOTER */}
      <Footer variant="navy" />
    </div>
  );
};
