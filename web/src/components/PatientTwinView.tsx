import React, { useState, useEffect, useMemo } from 'react';
import { ModelCardData, NightDetail, PatientSummary, TwinTrajectoryPoint } from '../../../shared/types';
import { formatHba1c, getRiskBadgeProps } from '../utils/clinical';
import { Footer } from './Footer';

interface PatientTwinViewProps {
  patient: PatientSummary | null;
  onBackToTriage: () => void;
  modelCard: ModelCardData | null;
  showRecordedOutcome: boolean;
  onToggleRecordedOutcome: (show: boolean) => void;
}

export const PatientTwinView: React.FC<PatientTwinViewProps> = ({
  patient,
  onBackToTriage,
  modelCard,
  showRecordedOutcome,
  onToggleRecordedOutcome,
}) => {
  const [patientNights, setPatientNights] = useState<NightDetail[]>([]);
  const [selectedNightIndex, setSelectedNightIndex] = useState<number>(0);
  const [conformalLevel, setConformalLevel] = useState<80 | 90>(90);

  // What-if sliders state
  const [snackCarbs, setSnackCarbs] = useState<number>(0);
  const [suDoseScale, setSuDoseScale] = useState<number>(100);

  // Fetch patient nights
  useEffect(() => {
    if (!patient) return;
    fetch(`/data/nights/${patient.patientId}.json`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: NightDetail[]) => {
        if (data && data.length > 0) {
          setPatientNights(data);
          // Default to latest night
          setSelectedNightIndex(data.length - 1);
        } else {
          setPatientNights([]);
        }
      })
      .catch(() => {
        setPatientNights([]);
      });
  }, [patient]);

  // Current night detail
  const currentNight: NightDetail | null = useMemo(() => {
    if (!patientNights.length || selectedNightIndex < 0 || selectedNightIndex >= patientNights.length) {
      return null;
    }
    return patientNights[selectedNightIndex];
  }, [patientNights, selectedNightIndex]);

  // Dynamic conformal margins from model_card
  const margin80 = modelCard?.cohortSummary.conformalMargin80MgDl || 26.26;
  const margin90 = modelCard?.cohortSummary.conformalMargin90MgDl || 37.59;
  const activeMargin = conformalLevel === 80 ? margin80 : margin90;

  // Compute what-if simulated curve and outcomes
  // Assumed ODE effect: Carbs raise glucose (~ +1.5 mg/dL per gram carb, peaking around 1.5h and decaying with tau=2h)
  // SU reduction reduces insulin secretion, softening downward slope
  const { simulatedTrajectory, lowestGlucoseBaseline, lowestGlucoseSimulated, minutesBelow70Baseline, minutesBelow70Simulated } = useMemo(() => {
    if (!currentNight || !currentNight.twinForecast) {
      return {
        simulatedTrajectory: [],
        lowestGlucoseBaseline: 0,
        lowestGlucoseSimulated: 0,
        minutesBelow70Baseline: 0,
        minutesBelow70Simulated: 0,
      };
    }

    const traj = currentNight.twinForecast;
    let minBase = 999;
    let minSim = 999;
    let minsBelow70Base = 0;
    let minsBelow70Sim = 0;

    const simulated: TwinTrajectoryPoint[] = traj.map((pt) => {
      const tHours = pt.minutes / 60.0;
      // Baseline expected glucose
      const baseG = pt.predictedGlucose;
      if (baseG < minBase) minBase = baseG;
      if (baseG < 70) minsBelow70Base += 15;

      // Carb impact delta: G_snack(t) = (snackCarbs * 1.5) * (t/1.5) * exp(1 - t/1.5)
      let carbDelta = 0;
      if (snackCarbs > 0 && tHours >= 0) {
        const peakT = 1.5;
        carbDelta = (snackCarbs * 1.6) * (tHours / peakT) * Math.exp(1 - tHours / peakT);
      }

      // SU scale impact: 100% is baseline. If dose scale < 100%, reduce nocturnal drop
      let suDelta = 0;
      if (suDoseScale < 100 && tHours >= 0) {
        const reductionFrac = (100 - suDoseScale) / 100.0;
        suDelta = reductionFrac * 22.0 * (1 - Math.exp(-tHours / 3.0));
      } else if (suDoseScale > 100 && tHours >= 0) {
        const excessFrac = (suDoseScale - 100) / 100.0;
        suDelta = -excessFrac * 18.0 * (1 - Math.exp(-tHours / 3.0));
      }

      const simG = Math.max(35, Math.round((baseG + carbDelta + suDelta) * 10) / 10);
      if (simG < minSim) minSim = simG;
      if (simG < 70) minsBelow70Sim += 15;

      return {
        ...pt,
        predictedGlucose: simG,
      };
    });

    return {
      simulatedTrajectory: simulated,
      lowestGlucoseBaseline: Math.round(minBase === 999 ? currentNight.bedtimeGlucose : minBase),
      lowestGlucoseSimulated: Math.round(minSim === 999 ? currentNight.bedtimeGlucose : minSim),
      minutesBelow70Baseline: minsBelow70Base,
      minutesBelow70Simulated: minsBelow70Sim,
    };
  }, [currentNight, snackCarbs, suDoseScale]);

  if (!patient) {
    return (
      <div className="min-h-screen bg-white text-[#0E1330] p-8">
        <button onClick={onBackToTriage} className="text-sm font-mono text-[#1B1E4A] underline mb-4">
          ← Back to triage
        </button>
        <p>No patient selected.</p>
      </div>
    );
  }

  // Fallback bedtime / trend if night detail is still loading
  const bedtimeGl = currentNight ? Math.round(currentNight.bedtimeGlucose) : Math.round(patient.bedtimeGlucose);
  const trendVal = currentNight ? currentNight.bedtimeSlope3h : -12;
  const calibratedRiskPct = currentNight ? (currentNight.calibratedRisk * 100).toFixed(1) : (patient.calibratedRisk * 100).toFixed(1);
  const isColdStart = currentNight ? currentNight.isColdStart : (patient.histNPriorNights ?? 0) === 0;
  const nEarlierNights = currentNight ? selectedNightIndex : (patient.totalRecordedNights - 1);

  // Filter SHAP factors: NEVER show a medication/SU factor if the model does not include it!
  const filteredShapFactors = useMemo(() => {
    if (!currentNight || !currentNight.shapFactors) return [];
    if (isColdStart) {
      // Model B: CGM only! Filter out any medication features
      return currentNight.shapFactors.filter(
        (f) => !f.feature.toLowerCase().includes('su') && !f.feature.toLowerCase().includes('med')
      );
    }
    // Model G: CGM + History
    return currentNight.shapFactors.filter(
      (f) => !f.feature.toLowerCase().includes('agent') && !f.feature.toLowerCase().includes('lab')
    );
  }, [currentNight, isColdStart]);

  const tierProps = getRiskBadgeProps(currentNight ? currentNight.riskTier : patient.riskTier);

  return (
    <div className="min-h-screen bg-white text-[#0E1330] font-sans flex flex-col justify-between">
      <div className="max-w-[1160px] w-full mx-auto px-4 sm:px-6 pt-6 pb-16">
        {/* Back Link */}
        <div className="mb-4">
          <button
            onClick={onBackToTriage}
            className="text-xs sm:text-sm font-mono text-[#4A4F73] hover:text-[#0E1330] focus-visible:outline-[#1B1E4A] transition-colors"
          >
            ← Back to triage
          </button>
        </div>

        {/* Patient Header Line */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8 pb-4 border-b border-[#E3E6F1]">
          <div className="flex flex-wrap items-baseline gap-3">
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-[#0E1330] font-sans">
              Patient {patient.patientId}
            </h1>
            <span
              className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-[2px] text-xs font-semibold border ${tierProps.badgeClass}`}
            >
              <span>{tierProps.symbol}</span>
              <span>{tierProps.label} risk</span>
            </span>

            {/* Profile Line */}
            <div className="text-xs sm:text-sm font-mono text-[#4A4F73] flex flex-wrap items-center gap-x-2">
              <span>{patient.age} y</span>
              <span>·</span>
              <span>{patient.gender.toLowerCase()}</span>
              <span>·</span>
              <span>BMI {patient.bmi.toFixed(1)}</span>
              <span>·</span>
              <span>eGFR {Math.round(patient.egfr)}</span>
              <span>·</span>
              <span>{formatHba1c(patient.hba1c)}</span>
              <span>·</span>
              <span>
                {patient.isSu ? `${patient.suAgent.toLowerCase()} ${patient.suDoseMg} mg/day` : 'no sulfonylurea'},{' '}
                {patient.isInsulin ? `insulin ${patient.totalDayInsulin} U/day` : 'no insulin'}
              </span>
            </div>
          </div>

          {/* Night Selector Dropdown */}
          {patientNights.length > 0 && (
            <div className="relative">
              <select
                value={selectedNightIndex}
                onChange={(e) => setSelectedNightIndex(Number(e.target.value))}
                className="bg-white border border-[#CBD0E2] text-[#0E1330] text-xs sm:text-sm rounded-[2px] px-3 py-1.5 pr-8 focus:outline-none focus:border-[#1B1E4A] cursor-pointer appearance-none"
              >
                {patientNights.map((n, idx) => (
                  <option key={n.nightDate} value={idx}>
                    Night of {new Date(n.nightDate).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                    {idx === patientNights.length - 1 ? ' (Tonight)' : ''}
                  </option>
                ))}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-[#4A4F73] text-xs">
                ▼
              </div>
            </div>
          )}
        </div>

        {/* 2-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* LEFT COLUMN: Chart + Points for clinician review */}
          <div className="lg:col-span-7 space-y-6">
            {/* Card 1: Chart Card */}
            <div className="bg-white rounded-[4px] border border-[#CBD0E2] p-6">
              {/* Header */}
              <div className="flex flex-wrap items-baseline justify-between gap-2 mb-4 pb-3 border-b border-[#E3E6F1]">
                <h2 className="text-base font-bold text-[#0E1330] font-sans">
                  Overnight glucose, 18:00 to 06:00
                </h2>
                <div className="font-mono text-xs text-[#4A4F73]">
                  Bedtime {bedtimeGl} mg/dL &nbsp;·&nbsp; 3-hour trend {trendVal > 0 ? `+${trendVal}` : trendVal} mg/dL per hour
                </div>
              </div>

              {/* Chart SVG */}
              <div className="w-full h-80 relative select-none">
                <svg className="w-full h-full" viewBox="0 0 1000 360" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  {/* 200 */}
                  <line x1="70" y1="60" x2="960" y2="60" stroke="#E3E6F1" strokeWidth="1" />
                  <text x="25" y="64" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">200</text>

                  {/* 150 */}
                  <line x1="70" y1="130" x2="960" y2="130" stroke="#E3E6F1" strokeWidth="1" />
                  <text x="25" y="134" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">150</text>

                  {/* 100 */}
                  <line x1="70" y1="200" x2="960" y2="200" stroke="#E3E6F1" strokeWidth="1" />
                  <text x="25" y="204" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">100</text>

                  {/* 70 (Ochre Threshold) */}
                  <line x1="70" y1="242" x2="960" y2="242" stroke="#8A5A00" strokeWidth="1.2" strokeDasharray="4 4" />
                  <text x="25" y="246" fill="#8A5A00" fontSize="11" fontFamily="Spline Sans Mono" fontWeight="600">70</text>

                  {/* 54 (Crimson Severe Low) */}
                  <line x1="70" y1="264" x2="960" y2="264" stroke="#D21319" strokeWidth="1.2" strokeDasharray="4 4" />
                  <text x="25" y="268" fill="#D21319" fontSize="11" fontFamily="Spline Sans Mono" fontWeight="600">54</text>

                  <text x="25" y="30" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">mg/dL</text>

                  {/* 22:00 Bedtime Anchor Vertical Line */}
                  <line x1="365" y1="40" x2="365" y2="300" stroke="#0E1330" strokeWidth="1.2" />
                  <text x="370" y="52" fill="#0E1330" fontSize="11" fontFamily="Spline Sans Mono">
                    22:00 bedtime
                  </text>

                  {/* Conformal Uncertainty Shaded Band (from 22:00 / x=365 to 06:00 / x=960) */}
                  {currentNight && currentNight.twinForecast && currentNight.twinForecast.length > 0 && (() => {
                    const traj = currentNight.twinForecast;
                    const xStart = 365;
                    const xEnd = 960;
                    const topPts: string[] = [];
                    const botPts: string[] = [];

                    traj.forEach((pt, idx) => {
                      const x = xStart + (idx / (traj.length - 1)) * (xEnd - xStart);
                      const topG = pt.predictedGlucose + activeMargin;
                      const botG = Math.max(30, pt.predictedGlucose - activeMargin);
                      const yTop = 340 - ((topG - 40) / 210) * 280;
                      const yBot = 340 - ((botG - 40) / 210) * 280;
                      topPts.push(`${x.toFixed(1)},${yTop.toFixed(1)}`);
                      botPts.unshift(`${x.toFixed(1)},${yBot.toFixed(1)}`);
                    });

                    const pathD = `M ${topPts.join(' L ')} L ${botPts.join(' L ')} Z`;
                    return (
                      <path d={pathD} fill="#CBD0E2" fillOpacity="0.45" />
                    );
                  })()}

                  {/* Recorded CGM Line (18:00 to 22:00, or continuing to 06:00 if showRecordedOutcome is ON) */}
                  {currentNight && currentNight.cgmSeries && currentNight.cgmSeries.length > 0 && (() => {
                    // Filter readings: 18:00 (x=70) to 22:00 (x=365)
                    // If showRecordedOutcome is true, include post-22:00 readings as well
                    const pts = currentNight.cgmSeries
                      .filter((reading) => {
                        if (showRecordedOutcome) return true;
                        return !reading.isPostAnchor;
                      })
                      .map((reading) => {
                        const dt = new Date(reading.timestamp);
                        // Convert hour:min to x coordinate
                        // 18:00 -> 0h rel, 22:00 -> 4h rel, 06:00 -> 12h rel
                        let relH = dt.getHours() - 18 + dt.getMinutes() / 60;
                        if (relH < 0) relH += 24; // overnight rollover
                        const x = 70 + (relH / 12) * (960 - 70);
                        const y = 340 - ((reading.glucose - 40) / 210) * 280;
                        return `${x.toFixed(1)},${y.toFixed(1)}`;
                      });

                    if (pts.length < 2) return null;
                    return (
                      <path
                        d={'M ' + pts.join(' L ')}
                        fill="none"
                        stroke="#1B1E4A"
                        strokeWidth="2.5"
                      />
                    );
                  })()}

                  {/* Expected Overnight Forecast Trajectory (Blended Forecaster: dashed dark line) */}
                  {currentNight && currentNight.twinForecast && currentNight.twinForecast.length > 0 && (() => {
                    const traj = currentNight.twinForecast;
                    const xStart = 365;
                    const xEnd = 960;
                    const pts = traj.map((pt, idx) => {
                      const x = xStart + (idx / (traj.length - 1)) * (xEnd - xStart);
                      const y = 340 - ((pt.predictedGlucose - 40) / 210) * 280;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    });
                    return (
                      <path
                        d={'M ' + pts.join(' L ')}
                        fill="none"
                        stroke="#1B1E4A"
                        strokeWidth="2.2"
                        strokeDasharray="6 4"
                      />
                    );
                  })()}

                  {/* Simulated What-if Trajectory (Dotted Green Line) */}
                  {simulatedTrajectory.length > 0 && (snackCarbs > 0 || suDoseScale !== 100) && (() => {
                    const xStart = 365;
                    const xEnd = 960;
                    const pts = simulatedTrajectory.map((pt, idx) => {
                      const x = xStart + (idx / (simulatedTrajectory.length - 1)) * (xEnd - xStart);
                      const y = 340 - ((pt.predictedGlucose - 40) / 210) * 280;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    });
                    return (
                      <path
                        d={'M ' + pts.join(' L ')}
                        fill="none"
                        stroke="#046A38"
                        strokeWidth="2.5"
                        strokeDasharray="2 4"
                      />
                    );
                  })()}

                  {/* Drop line annotation if lower range dips below 70 */}
                  {currentNight && currentNight.twinForecast && (() => {
                    const dipIdx = currentNight.twinForecast.findIndex(
                      (pt) => pt.predictedGlucose - activeMargin < 70
                    );
                    if (dipIdx > -1) {
                      const x = 365 + (dipIdx / (currentNight.twinForecast.length - 1)) * (960 - 365);
                      return (
                        <g>
                          <line x1={x} y1="120" x2={x} y2="242" stroke="#4A4F73" strokeWidth="1" strokeDasharray="2 3" />
                          <text x={x + 5} y="115" fill="#0E1330" fontSize="11" fontFamily="Spline Sans Mono">
                            Range falls below 70 mg/dL
                          </text>
                        </g>
                      );
                    }
                    return null;
                  })()}

                  {/* Time Axis Labels */}
                  <text x="70" y="325" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">18:00</text>
                  <text x="350" y="325" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">22:00</text>
                  <text x="500" y="325" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">00:00</text>
                  <text x="640" y="325" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">02:00</text>
                  <text x="780" y="325" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">04:00</text>
                  <text x="940" y="325" fill="#4A4F73" fontSize="11" fontFamily="Spline Sans Mono">06:00</text>
                </svg>
              </div>

              {/* Legend & Controls below chart */}
              <div className="mt-4 pt-4 border-t border-[#E3E6F1] flex flex-wrap items-center justify-between gap-4 text-xs">
                <div className="flex flex-wrap items-center gap-5 font-sans">
                  {/* Recorded */}
                  <div className="flex items-center space-x-2">
                    <span className="w-5 h-0.5 bg-[#1B1E4A] inline-block" />
                    <span className="font-medium text-[#0E1330]">Recorded</span>
                  </div>

                  {/* Expected + Range Selector */}
                  <div className="flex items-center space-x-2">
                    <span className="w-5 border-t-2 border-dashed border-[#1B1E4A] inline-block" />
                    <span className="text-[#0E1330]">Expected, with</span>
                    <div className="inline-flex rounded-[2px] border border-[#CBD0E2] overflow-hidden">
                      <button
                        onClick={() => setConformalLevel(80)}
                        className={`px-1.5 py-0.5 font-mono text-[11px] ${
                          conformalLevel === 80 ? 'bg-[#1B1E4A] text-white' : 'bg-white text-[#0E1330]'
                        }`}
                      >
                        80%
                      </button>
                      <button
                        onClick={() => setConformalLevel(90)}
                        className={`px-1.5 py-0.5 font-mono text-[11px] ${
                          conformalLevel === 90 ? 'bg-[#1B1E4A] text-white' : 'bg-white text-[#0E1330]'
                        }`}
                      >
                        90%
                      </button>
                    </div>
                    <span className="text-[#0E1330]">range</span>
                  </div>

                  {/* What-if scenario */}
                  <div className="flex items-center space-x-2">
                    <span className="w-5 border-t-2 border-dotted border-[#046A38] inline-block" />
                    <span className="text-[#046A38] font-medium">What-if scenario</span>
                  </div>
                </div>

                {/* Show recorded outcome toggle */}
                <div className="flex items-center space-x-2 select-none">
                  <button
                    role="switch"
                    aria-checked={showRecordedOutcome}
                    onClick={() => onToggleRecordedOutcome(!showRecordedOutcome)}
                    className={`w-9 h-5 flex items-center rounded-full p-0.5 transition-colors focus-visible:outline-[#1B1E4A] ${
                      showRecordedOutcome ? 'bg-[#1B1E4A]' : 'bg-[#CBD0E2]'
                    }`}
                  >
                    <div
                      className={`bg-white w-4 h-4 rounded-full transition-transform ${
                        showRecordedOutcome ? 'transform translate-x-4' : ''
                      }`}
                    />
                  </button>
                  <span className="text-xs font-sans text-[#4A4F73]">Show recorded outcome</span>
                </div>
              </div>
            </div>

            {/* Card 2: Points for clinician review */}
            <div className="bg-white rounded-[4px] border border-[#CBD0E2] p-6">
              <h3 className="text-base font-bold text-[#0E1330] mb-4 font-sans">
                Points for clinician review
              </h3>
              {(() => {
                const priorNightLowRecorded = selectedNightIndex > 0 
                  ? Boolean(patientNights[selectedNightIndex - 1]?.actualGroundTruthStrictLow)
                  : Boolean(currentNight?.priorNightLow);

                const priorNightsCount = selectedNightIndex;
                const priorLowsCount = selectedNightIndex > 0
                  ? patientNights.slice(0, selectedNightIndex).filter((n) => n.actualGroundTruthStrictLow).length
                  : (currentNight?.priorLowCount ?? 0);

                const minLowerConformal = currentNight?.twinForecast?.length
                  ? Math.min(...currentNight.twinForecast.map((pt) => pt.lowerConformal))
                  : null;
                const crosses70 = minLowerConformal !== null && minLowerConformal < 70.0;

                const personalLowest = currentNight?.personalMedianNadir ?? (
                  selectedNightIndex > 0
                    ? Math.min(...patientNights.slice(0, selectedNightIndex).map((n) => n.actualGroundTruthNadir))
                    : null
                );

                return (
                  <ul className="space-y-2.5 text-sm text-[#3A3F66] list-disc pl-5 leading-relaxed">
                    {/* 1. Low last night */}
                    <li>
                      {isColdStart
                        ? 'Low last night: No earlier night on record (first monitoring night).'
                        : priorNightLowRecorded
                        ? 'Low last night: Persistent low (<70 mg/dL for ≥30 min) recorded on immediately prior night.'
                        : 'Low last night: No persistent low recorded on immediately prior night.'}
                    </li>

                    {/* 2. Prior-night count */}
                    <li>
                      {isColdStart
                        ? 'Prior-night count: 0 earlier nights recorded.'
                        : `Prior-night count: ${priorLowsCount} persistent low(s) across ${priorNightsCount} earlier night(s).`}
                    </li>

                    {/* 3. Bedtime trend */}
                    <li>
                      Bedtime trend: {trendVal > 0 ? `+${trendVal}` : trendVal} mg/dL/h over prior 3 hours (1h slope: {currentNight ? (currentNight.bedtimeSlope1h > 0 ? `+${currentNight.bedtimeSlope1h}` : currentNight.bedtimeSlope1h) : 0} mg/dL/h).
                    </li>

                    {/* 4. Expected range crossing 70 mg/dL */}
                    <li>
                      {crosses70
                        ? `Expected range: Forecast uncertainty band crosses below 70 mg/dL between 23:00 and 06:00 (lowest point: ${minLowerConformal?.toFixed(1)} mg/dL).`
                        : `Expected range: Forecast uncertainty band remains above 70 mg/dL throughout the overnight window (lowest point: ${minLowerConformal?.toFixed(1) ?? 'N/A'} mg/dL).`}
                    </li>

                    {/* 5. Personal lowest value */}
                    <li>
                      {personalLowest !== null
                        ? `Personal lowest value: Historical median overnight nadir is ${personalLowest.toFixed(1)} mg/dL.`
                        : 'Personal lowest value: Not available on first night of observation.'}
                    </li>
                  </ul>
                );
              })()}
            </div>
          </div>

          {/* RIGHT COLUMN: Risk Card + Why Flagged + What-if Simulation */}
          <div className="lg:col-span-5 space-y-6">
            {/* Card 1: Risk Card */}
            <div className="bg-white rounded-[4px] border border-[#CBD0E2] p-6">
              <div className="font-mono text-xs text-[#4A4F73] mb-2">
                Risk of a persistent low tonight
              </div>
              <div className="text-5xl sm:text-6xl font-bold text-[#0E1330] font-sans mb-3 tracking-tight">
                {calibratedRiskPct}%
              </div>

              {isColdStart ? (
                <div>
                  <span className="inline-block px-2 py-0.5 rounded-[2px] bg-[#FFF3D6] text-[#7A4F00] font-mono text-xs font-semibold mb-2">
                    No personal history yet
                  </span>
                  <p className="text-xs text-[#4A4F73] leading-relaxed">
                    Evaluated using the CGM-only cold-start model and population-average baseline profile.
                  </p>
                </div>
              ) : (
                <p className="text-xs text-[#4A4F73] leading-relaxed">
                  Two or more readings below 70 mg/dL between 23:00 and 06:00. Built from {nEarlierNights} earlier nights of this patient.
                </p>
              )}
            </div>

            {/* Card 2: Why Flagged */}
            <div className="bg-white rounded-[4px] border border-[#CBD0E2] p-6">
              <h3 className="text-base font-bold text-[#0E1330] mb-4 font-sans">
                Why flagged
              </h3>
              <div className="space-y-3.5 mb-4">
                {filteredShapFactors.length > 0 ? (
                  filteredShapFactors.slice(0, 4).map((f) => {
                    const widthPct = Math.min(100, Math.max(15, Math.round(Math.abs(f.impact) * 80)));
                    return (
                      <div key={f.feature} className="space-y-1">
                        <div className="flex justify-between items-baseline text-xs">
                          <span className="font-sans text-[#0E1330]">{f.label}</span>
                          <span className="font-mono text-[#4A4F73]">{f.value}</span>
                        </div>
                        <div className="w-full h-1.5 bg-[#E3E6F1] rounded-none overflow-hidden">
                          <div
                            className="h-full bg-[#1B1E4A]"
                            style={{ width: `${widthPct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <>
                    <div className="space-y-1">
                      <div className="flex justify-between items-baseline text-xs">
                        <span className="font-sans text-[#0E1330]">Bedtime glucose</span>
                        <span className="font-mono text-[#4A4F73]">{bedtimeGl} mg/dL</span>
                      </div>
                      <div className="w-full h-1.5 bg-[#E3E6F1] rounded-none overflow-hidden">
                        <div className="h-full bg-[#1B1E4A]" style={{ width: '65%' }} />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between items-baseline text-xs">
                        <span className="font-sans text-[#0E1330]">Falling trend at bedtime</span>
                        <span className="font-mono text-[#4A4F73]">{trendVal} mg/dL/h</span>
                      </div>
                      <div className="w-full h-1.5 bg-[#E3E6F1] rounded-none overflow-hidden">
                        <div className="h-full bg-[#1B1E4A]" style={{ width: '80%' }} />
                      </div>
                    </div>
                  </>
                )}
              </div>
              <div className="font-mono text-xs text-[#4A4F73] pt-2 border-t border-[#E3E6F1]">
                Contributions from the model, in the order of their effect.
              </div>
            </div>

            {/* Card 3: What-if Simulation */}
            <div className="bg-white rounded-[4px] border-2 border-[#046A38] p-6">
              <h3 className="text-base font-bold text-[#046A38] font-sans mb-1">
                What-if simulation
              </h3>
              <div className="font-mono text-xs text-[#4A4F73] mb-6">
                Assumed effects, not learned from the data.
              </div>

              {/* Slider 1: Bedtime snack */}
              <div className="mb-5">
                <div className="flex justify-between items-center text-xs mb-2">
                  <span className="font-sans font-medium text-[#0E1330]">Bedtime snack</span>
                  <span className="font-mono text-[#0E1330] font-semibold">{snackCarbs} g</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="45"
                  step="5"
                  value={snackCarbs}
                  onChange={(e) => setSnackCarbs(Number(e.target.value))}
                  className="w-full h-1.5 bg-[#E3E6F1] rounded-none appearance-none cursor-pointer accent-[#046A38]"
                />
              </div>

              {/* Slider 2: Sulfonylurea dose */}
              <div className="mb-6">
                <div className="flex justify-between items-center text-xs mb-2">
                  <span className="font-sans font-medium text-[#0E1330]">Sulfonylurea dose</span>
                  <span className="font-mono text-[#0E1330] font-semibold">{suDoseScale}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="150"
                  step="25"
                  value={suDoseScale}
                  onChange={(e) => setSuDoseScale(Number(e.target.value))}
                  className="w-full h-1.5 bg-[#E3E6F1] rounded-none appearance-none cursor-pointer accent-[#046A38]"
                />
              </div>

              {/* Outcomes row */}
              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[#E3E6F1]">
                <div>
                  <div className="font-mono text-xs text-[#4A4F73] mb-1">Lowest glucose</div>
                  <div className="text-base font-mono">
                    <span className="text-[#0E1330]">{lowestGlucoseBaseline}</span>
                    <span className="text-[#4A4F73] mx-1.5">→</span>
                    <span className="text-[#046A38] font-bold">{lowestGlucoseSimulated} mg/dL</span>
                  </div>
                </div>

                <div>
                  <div className="font-mono text-xs text-[#4A4F73] mb-1">Minutes below 70</div>
                  <div className="text-base font-mono">
                    <span className="text-[#0E1330]">{minutesBelow70Baseline}</span>
                    <span className="text-[#4A4F73] mx-1.5">→</span>
                    <span className="text-[#046A38] font-bold">{minutesBelow70Simulated}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Light Footer */}
      <Footer variant="light" />
    </div>
  );
};
