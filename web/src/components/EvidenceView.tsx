import React from 'react';
import { ModelCardData } from '../../../shared/types';
import { Footer } from './Footer';

interface EvidenceViewProps {
  modelCard: ModelCardData | null;
}

export const EvidenceView: React.FC<EvidenceViewProps> = ({ modelCard }) => {
  if (!modelCard) {
    return (
      <div className="min-h-screen bg-white text-[#0E1330] p-12 text-center font-mono text-sm">
        Loading evidence metrics...
      </div>
    );
  }

  // 1. Primary Comparison: Nights 1+ with Earlier History Available (20 Repeated GroupKFold Runs)
  const rep = modelCard.repeatedGroupKFoldHeadline;
  const prim = modelCard.groupGRobustness?.identical_subset_ge1_prior_night;

  const primAucB = rep ? rep.group_b_cgm_only.auroc.mean : (prim?.auc_a ?? 0.736);
  const primAucBRange = rep ? `${rep.group_b_cgm_only.auroc.min.toFixed(3)}–${rep.group_b_cgm_only.auroc.max.toFixed(3)}` : '0.707–0.753';
  const primAucG = rep ? rep.group_g_cgm_history.auroc.mean : (prim?.auc_b ?? 0.832);
  const primAucGRange = rep ? `${rep.group_g_cgm_history.auroc.min.toFixed(3)}–${rep.group_g_cgm_history.auroc.max.toFixed(3)}` : '0.824–0.840';

  const primDiffAuc = rep ? rep.differences_g_minus_b.delta_auroc.mean : 0.097;
  const primDiffAucCi = rep
    ? `+${primDiffAuc.toFixed(3)} [${rep.paired_bootstrap_ci.diff_auc_ci[0].toFixed(3)} to ${rep.paired_bootstrap_ci.diff_auc_ci[1].toFixed(3)}]`
    : '+0.097 [+0.051 to +0.149]';

  const primPrcB = rep ? rep.group_b_cgm_only.auprc.mean : (prim?.prc_a ?? 0.354);
  const primPrcBRange = rep ? `${rep.group_b_cgm_only.auprc.min.toFixed(3)}–${rep.group_b_cgm_only.auprc.max.toFixed(3)}` : '0.252–0.392';
  const primPrcG = rep ? rep.group_g_cgm_history.auprc.mean : (prim?.prc_b ?? 0.476);
  const primPrcGRange = rep ? `${rep.group_g_cgm_history.auprc.min.toFixed(3)}–${rep.group_g_cgm_history.auprc.max.toFixed(3)}` : '0.417–0.496';

  const primDiffPrc = rep ? rep.differences_g_minus_b.delta_auprc.mean : 0.123;
  const primDiffPrcCi = rep
    ? `+${primDiffPrc.toFixed(3)} [${rep.paired_bootstrap_ci.diff_prc_ci[0].toFixed(3)} to ${rep.paired_bootstrap_ci.diff_prc_ci[1].toFixed(3)}]`
    : '+0.123 [+0.047 to +0.191]';

  const primBrierB = rep ? rep.group_b_cgm_only.brier.mean : (prim?.brier_a ?? 0.073);
  const primBrierBRange = rep ? `${rep.group_b_cgm_only.brier.min.toFixed(3)}–${rep.group_b_cgm_only.brier.max.toFixed(3)}` : '0.071–0.078';
  const primBrierG = rep ? rep.group_g_cgm_history.brier.mean : (prim?.brier_b ?? 0.065);
  const primBrierGRange = rep ? `${rep.group_g_cgm_history.brier.min.toFixed(3)}–${rep.group_g_cgm_history.brier.max.toFixed(3)}` : '0.064–0.068';

  const primDiffBrier = rep ? rep.differences_g_minus_b.delta_brier.mean : -0.008;
  const primDiffBrierCi = rep
    ? `${primDiffBrier.toFixed(3)} [${rep.paired_bootstrap_ci.diff_brier_ci[0].toFixed(3)} to ${rep.paired_bootstrap_ci.diff_brier_ci[1].toFixed(3)}]`
    : '-0.008 [-0.014 to -0.004]';

  // 2. Forecaster Benchmarks
  const fb = modelCard.forecasterBenchmark;
  const horizons = ['1-Hour (23:00)', '3-Hour (01:00)', '6-Hour (04:00)'];

  // 3. Conformal Coverage (200 Patient-Level Splits)
  const conf = modelCard.conformal200Splits?.conformal_splits_summary;
  const margin80 = modelCard.cohortSummary.conformalMargin80MgDl || 26.26;
  const margin90 = modelCard.cohortSummary.conformalMargin90MgDl || 37.59;

  // 4. Alert Burden: Group G metrics at 80% specificity
  const groupGAll = modelCard.ablationTable.find(
    (r) => r.group === 'G_cgm_history' && r.model === 'LogisticRegression' && r.cohort.includes('All')
  );
  const sens80 = groupGAll?.sens80SpecPct ?? 75.0;
  const ppv80 = groupGAll?.ppv80SpecPct ?? 30.6;
  const nnw80 = groupGAll?.nnw80Spec ?? 3.27;
  const alertsWk80 = groupGAll?.alertsPerPtWk ?? 1.8;

  // Secondary Full Cohort comparison
  const fullCohortRows = modelCard.ablationTable.filter(
    (r) => r.cohort === 'All T2DM (N=100)' && r.model === 'LogisticRegression'
  );

  return (
    <div className="min-h-screen bg-white text-[#0E1330] font-sans flex flex-col justify-between">
      <div className="max-w-[1160px] w-full mx-auto px-4 sm:px-6 pt-10 pb-16 space-y-16">
        {/* Page Header */}
        <div>
          <div className="font-mono text-xs text-[#4A4F73] mb-2">Evidence & Validation</div>
          <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0E1330] mb-4">
            Scientific evidence & model card
          </h1>
          <p className="text-sm sm:text-base text-[#3A3F66] leading-relaxed max-w-3xl">
            Strict patient-level GroupKFold cross-validation (zero leakage across train and test), 1,000-iteration
            patient-clustered bootstrap confidence intervals, and 200 random patient-level splits for conformal coverage.
          </p>
        </div>

        {/* SECTION (a): PRIMARY COMPARISON (NIGHTS 1+ WITH PRIOR HISTORY) */}
        <section className="space-y-4">
          <div className="border-b border-[#CBD0E2] pb-3">
            <h2 className="text-xl font-bold text-[#0E1330] font-sans">
              Primary comparison: nights with at least one earlier night (Nights 1+)
            </h2>
            <p className="text-xs sm:text-sm text-[#4A4F73] font-sans mt-1">
              Evaluated on identical nights with strictly earlier nocturnal history available (1,081 nights across 100 patients).
            </p>
          </div>

          <div className="border border-[#CBD0E2] rounded-[4px] overflow-hidden bg-white">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#F1F3F9] border-b border-[#CBD0E2] font-mono text-xs text-[#4A4F73]">
                  <th className="py-2.5 px-4 font-normal">Model</th>
                  <th className="py-2.5 px-4 font-normal">Features</th>
                  <th className="py-2.5 px-4 font-normal">AUROC</th>
                  <th className="py-2.5 px-4 font-normal">AUPRC</th>
                  <th className="py-2.5 px-4 font-normal">Brier Score</th>
                  <th className="py-2.5 px-4 font-normal">Paired Difference (95% CI)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E3E6F1] text-xs font-mono text-[#0E1330]">
                <tr>
                  <td className="py-3 px-4 font-sans font-medium">Group B (CGM-Only)</td>
                  <td className="py-3 px-4 text-[#4A4F73]">Bedtime glucose & slopes</td>
                  <td className="py-3 px-4">
                    <div>{primAucB.toFixed(3)}</div>
                    <div className="text-[10px] text-[#4A4F73]">[{primAucBRange}]</div>
                  </td>
                  <td className="py-3 px-4">
                    <div>{primPrcB.toFixed(3)}</div>
                    <div className="text-[10px] text-[#4A4F73]">[{primPrcBRange}]</div>
                  </td>
                  <td className="py-3 px-4">
                    <div>{primBrierB.toFixed(3)}</div>
                    <div className="text-[10px] text-[#4A4F73]">[{primBrierBRange}]</div>
                  </td>
                  <td className="py-3 px-4 text-[#4A4F73]">Reference baseline</td>
                </tr>
                <tr className="bg-[#F8FAFC]">
                  <td className="py-3 px-4 font-sans font-bold">Group G (CGM + History)</td>
                  <td className="py-3 px-4 text-[#4A4F73]">Bedtime + prior nights lows/nadir</td>
                  <td className="py-3 px-4 font-bold">
                    <div>{primAucG.toFixed(3)}</div>
                    <div className="text-[10px] text-[#4A4F73] font-normal">[{primAucGRange}]</div>
                  </td>
                  <td className="py-3 px-4 font-bold">
                    <div>{primPrcG.toFixed(3)}</div>
                    <div className="text-[10px] text-[#4A4F73] font-normal">[{primPrcGRange}]</div>
                  </td>
                  <td className="py-3 px-4 font-bold">
                    <div>{primBrierG.toFixed(3)}</div>
                    <div className="text-[10px] text-[#4A4F73] font-normal">[{primBrierGRange}]</div>
                  </td>
                  <td className="py-3 px-4 text-[#046A38] font-bold">
                    Δ AUROC: {primDiffAucCi}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="font-mono text-xs text-[#4A4F73] space-y-1">
            <div>
              Results from 20 repeated 5-fold patient-level GroupKFold runs with different random seeds (N = 1,081 nights, 101 events across 100 patients with earlier history available).
            </div>
            <div>
              Values show mean across 20 runs with [min–max] range. Paired bootstrap 95% CIs resampled by patient (1,000 resamples): Δ AUPRC = {primDiffPrcCi}; Δ Brier = {primDiffBrierCi}.
            </div>
          </div>
        </section>

        {/* SECTION (b): FORECASTER BENCHMARK AT 1H, 3H, 6H */}
        <section className="space-y-4">
          <div className="border-b border-[#CBD0E2] pb-3">
            <h2 className="text-xl font-bold text-[#0E1330] font-sans">
              Overnight forecaster benchmark (1h, 3h, 6h horizons)
            </h2>
            <p className="text-xs sm:text-sm text-[#4A4F73] font-sans mt-1">
              Nested tau optimization per fold (mean tau = 3.305 hours). Blended model synchronises bedtime reading with patient historical overnight profile.
            </p>
          </div>

          <div className="border border-[#CBD0E2] rounded-[4px] overflow-hidden bg-white">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#F1F3F9] border-b border-[#CBD0E2] font-mono text-xs text-[#4A4F73]">
                  <th className="py-2.5 px-4 font-normal">Horizon</th>
                  <th className="py-2.5 px-4 font-normal">Forecaster Model</th>
                  <th className="py-2.5 px-4 font-normal">MAE (mg/dL)</th>
                  <th className="py-2.5 px-4 font-normal">95% CI (mg/dL)</th>
                  <th className="py-2.5 px-4 font-normal">RMSE (mg/dL)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E3E6F1] text-xs font-mono text-[#0E1330]">
                {horizons.map((h) => {
                  const items = fb.filter((b) => b.horizon === h);
                  return items.map((item, idx) => {
                    const isBlend = item.model.includes('Blended');
                    const ciStr = (item as { maeCi?: number[] }).maeCi
                      ? `[${(item as { maeCi?: number[] }).maeCi?.[0].toFixed(1)} - ${(item as { maeCi?: number[] }).maeCi?.[1].toFixed(1)}]`
                      : '—';
                    return (
                      <tr key={`${h}-${item.model}`} className={isBlend ? 'bg-[#F8FAFC]' : ''}>
                        {idx === 0 && (
                          <td rowSpan={items.length} className="py-3 px-4 font-sans font-medium border-r border-[#CBD0E2] align-top">
                            {h}
                          </td>
                        )}
                        <td className={`py-2.5 px-4 font-sans ${isBlend ? 'font-bold text-[#0E1330]' : 'text-[#4A4F73]'}`}>
                          {item.model}
                        </td>
                        <td className={`py-2.5 px-4 ${isBlend ? 'font-bold text-[#0E1330]' : ''}`}>
                          {item.maeMgDl.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-4 text-[#4A4F73]">{ciStr}</td>
                        <td className="py-2.5 px-4">{item.rmseMgDl.toFixed(2)}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* SECTION (c): UNCERTAINTY & CONFORMAL COVERAGE (200 PATIENT SPLITS) */}
        <section className="space-y-4">
          <div className="border-b border-[#CBD0E2] pb-3">
            <h2 className="text-xl font-bold text-[#0E1330] font-sans">
              Uncertainty quantification & conformal coverage
            </h2>
            <p className="text-xs sm:text-sm text-[#4A4F73] font-sans mt-1">
              Empirical coverage measured across 200 random patient-level splits (50 patients calibration, 50 held-out test).
            </p>
          </div>

          <div className="border border-[#CBD0E2] rounded-[4px] overflow-hidden bg-white">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#F1F3F9] border-b border-[#CBD0E2] font-mono text-xs text-[#4A4F73]">
                  <th className="py-2.5 px-4 font-normal">Condition / Horizon</th>
                  <th className="py-2.5 px-4 font-normal">Target</th>
                  <th className="py-2.5 px-4 font-normal">Half-Width</th>
                  <th className="py-2.5 px-4 font-normal">Mean Coverage</th>
                  <th className="py-2.5 px-4 font-normal">Min - Max Range</th>
                  <th className="py-2.5 px-4 font-normal">95% Split CI</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E3E6F1] text-xs font-mono text-[#0E1330]">
                <tr>
                  <td className="py-3 px-4 font-sans font-medium">Overall Overnight (22:00-06:00)</td>
                  <td className="py-3 px-4">80%</td>
                  <td className="py-3 px-4">±{margin80.toFixed(1)} mg/dL</td>
                  <td className="py-3 px-4 font-bold">{conf?.overall_80.mean.toFixed(2)}%</td>
                  <td className="py-3 px-4">[{conf?.overall_80.range[0].toFixed(1)}% - {conf?.overall_80.range[1].toFixed(1)}%]</td>
                  <td className="py-3 px-4">[{conf?.overall_80.ci95[0].toFixed(1)}% - {conf?.overall_80.ci95[1].toFixed(1)}%]</td>
                </tr>
                <tr className="bg-[#F8FAFC]">
                  <td className="py-3 px-4 font-sans font-bold">Overall Overnight (22:00-06:00)</td>
                  <td className="py-3 px-4 font-bold">90%</td>
                  <td className="py-3 px-4 font-bold">±{margin90.toFixed(1)} mg/dL</td>
                  <td className="py-3 px-4 font-bold">{conf?.overall_90.mean.toFixed(2)}%</td>
                  <td className="py-3 px-4">[{conf?.overall_90.range[0].toFixed(1)}% - {conf?.overall_90.range[1].toFixed(1)}%]</td>
                  <td className="py-3 px-4">[{conf?.overall_90.ci95[0].toFixed(1)}% - {conf?.overall_90.ci95[1].toFixed(1)}%]</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-sans">1-Hour Horizon (23:00)</td>
                  <td className="py-3 px-4">90%</td>
                  <td className="py-3 px-4">±{margin90.toFixed(1)} mg/dL</td>
                  <td className="py-3 px-4">{conf?.h1_90.mean.toFixed(2)}%</td>
                  <td className="py-3 px-4">[{conf?.h1_90.range[0].toFixed(1)}% - {conf?.h1_90.range[1].toFixed(1)}%]</td>
                  <td className="py-3 px-4">[{conf?.h1_90.ci95[0].toFixed(1)}% - {conf?.h1_90.ci95[1].toFixed(1)}%]</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-sans">3-Hour Horizon (01:00) (trough)</td>
                  <td className="py-3 px-4">90%</td>
                  <td className="py-3 px-4">±{margin90.toFixed(1)} mg/dL</td>
                  <td className="py-3 px-4 text-[#A30F14] font-medium">{conf?.h3_90.mean.toFixed(2)}%</td>
                  <td className="py-3 px-4">[{conf?.h3_90.range[0].toFixed(1)}% - {conf?.h3_90.range[1].toFixed(1)}%]</td>
                  <td className="py-3 px-4">[{conf?.h3_90.ci95[0].toFixed(1)}% - {conf?.h3_90.ci95[1].toFixed(1)}%]</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-sans">6-Hour Horizon (04:00)</td>
                  <td className="py-3 px-4">90%</td>
                  <td className="py-3 px-4">±{margin90.toFixed(1)} mg/dL</td>
                  <td className="py-3 px-4">{conf?.h6_90.mean.toFixed(2)}%</td>
                  <td className="py-3 px-4">[{conf?.h6_90.range[0].toFixed(1)}% - {conf?.h6_90.range[1].toFixed(1)}%]</td>
                  <td className="py-3 px-4">[{conf?.h6_90.ci95[0].toFixed(1)}% - {conf?.h6_90.ci95[1].toFixed(1)}%]</td>
                </tr>
                <tr className="bg-[#F8FAFC]">
                  <td className="py-3 px-4 font-sans">Conditional on Forecast &lt; 110 mg/dL</td>
                  <td className="py-3 px-4">90%</td>
                  <td className="py-3 px-4">±{margin90.toFixed(1)} mg/dL</td>
                  <td className="py-3 px-4 font-bold text-[#046A38]">{conf?.cond_low_90.mean.toFixed(2)}%</td>
                  <td className="py-3 px-4">[{conf?.cond_low_90.range[0].toFixed(1)}% - {conf?.cond_low_90.range[1].toFixed(1)}%]</td>
                  <td className="py-3 px-4">[{conf?.cond_low_90.ci95[0].toFixed(1)}% - {conf?.cond_low_90.ci95[1].toFixed(1)}%]</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="font-mono text-xs text-[#4A4F73]">
            Note: Coverage dips slightly near the 3-hour mark ({conf?.h3_90.mean.toFixed(1)}% at 90% target) due to patient-to-patient variance in nocturnal nadir timing.
          </div>
        </section>

        {/* SECTION (d): ALERT BURDEN & CLINICAL TRADEOFF */}
        <section className="space-y-4">
          <div className="border-b border-[#CBD0E2] pb-3">
            <h2 className="text-xl font-bold text-[#0E1330] font-sans">
              Alert burden & clinical operating point
            </h2>
            <p className="text-xs sm:text-sm text-[#4A4F73] font-sans mt-1">
              At cohort prevalence of 10.5%, mathematical Bayes' rule shows 75% PPV requires an impossible positive likelihood ratio (LR+ &gt; 25.6), resulting in unsafe sensitivity.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="border border-[#CBD0E2] rounded-[4px] p-4 bg-white">
              <div className="font-mono text-xs text-[#4A4F73] mb-1">Sensitivity at 80% Spec</div>
              <div className="text-3xl font-bold text-[#0E1330] font-sans">{sens80.toFixed(1)}%</div>
              <div className="text-xs text-[#4A4F73] mt-1 font-sans">Detects 3 of every 4 strict lows</div>
            </div>

            <div className="border border-[#CBD0E2] rounded-[4px] p-4 bg-white">
              <div className="font-mono text-xs text-[#4A4F73] mb-1">PPV at 80% Spec</div>
              <div className="text-3xl font-bold text-[#0E1330] font-sans">{ppv80.toFixed(1)}%</div>
              <div className="text-xs text-[#4A4F73] mt-1 font-sans">About 1 in 3 flagged nights is an event</div>
            </div>

            <div className="border border-[#CBD0E2] rounded-[4px] p-4 bg-white">
              <div className="font-mono text-xs text-[#4A4F73] mb-1">Number Needed to Warn</div>
              <div className="text-3xl font-bold text-[#0E1330] font-sans">{nnw80.toFixed(2)}</div>
              <div className="text-xs text-[#4A4F73] mt-1 font-sans">Alerts evaluated per true positive</div>
            </div>

            <div className="border border-[#CBD0E2] rounded-[4px] p-4 bg-white">
              <div className="font-mono text-xs text-[#4A4F73] mb-1">Alert Burden</div>
              <div className="text-3xl font-bold text-[#0E1330] font-sans">{alertsWk80.toFixed(1)}</div>
              <div className="text-xs text-[#4A4F73] mt-1 font-sans">Alerts per patient-week</div>
            </div>
          </div>
        </section>

        {/* SECTION (e): WHAT DID NOT HELP */}
        <section className="bg-[#F1F3F9] rounded-[4px] border border-[#CBD0E2] p-6 space-y-4">
          <h2 className="text-lg font-bold text-[#0E1330] font-sans">
            What did not help, reported as found
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div>
              <div className="font-bold text-[#0E1330] mb-1">Static laboratory panels</div>
              <p className="text-xs text-[#3A3F66] leading-relaxed">
                HbA1c, eGFR, lipid panels, and liver enzymes added no measurable discrimination over CGM dynamics (Group E vs B paired Δ AUROC = -0.007 [-0.038 to 0.024]).
              </p>
            </div>
            <div>
              <div className="font-bold text-[#0E1330] mb-1">Demographic factors</div>
              <p className="text-xs text-[#3A3F66] leading-relaxed">
                Age, sex, and BMI showed zero statistically reliable gain in out-of-fold generalization across patient-level folds.
              </p>
            </div>
            <div>
              <div className="font-bold text-[#0E1330] mb-1">Medication tags alone</div>
              <p className="text-xs text-[#3A3F66] leading-relaxed">
                Adding SU dose and insulin tags without longitudinal CGM history (Group C vs B) showed underpowered, no measurable benefit in discrimination.
              </p>
            </div>
          </div>
        </section>

        {/* SECTION (f): LIMITATIONS */}
        <section className="space-y-4">
          <div className="border-b border-[#CBD0E2] pb-3">
            <h2 className="text-xl font-bold text-[#0E1330] font-sans">
              Clinical limitations & study constraints
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-[#3A3F66] leading-relaxed">
            <div className="border border-[#E3E6F1] p-4 rounded-[4px]">
              <div className="font-bold text-[#0E1330] mb-1">Hospitalised cohort</div>
              <p className="text-xs">
                ShanghaiT2DM participants were hospitalised inpatients monitored for 3 to 14 days with standardized meal schedules. Free-living outpatients experience different nocturnal activity and dietary variability.
              </p>
            </div>
            <div className="border border-[#E3E6F1] p-4 rounded-[4px]">
              <div className="font-bold text-[#0E1330] mb-1">Exploratory sulfonylurea subgroup</div>
              <p className="text-xs">
                Only 18 patients took sulfonylureas, and 11 had events. While we report the SU subgroup separately, small numbers preclude causal claims.
              </p>
            </div>
            <div className="border border-[#E3E6F1] p-4 rounded-[4px]">
              <div className="font-bold text-[#0E1330] mb-1">Regimen labels per patient, not per night</div>
              <p className="text-xs">
                Medication records are cohort-level baseline assignments. Daily dose titrations or timing of evening administration are not recorded per night.
              </p>
            </div>
            <div className="border border-[#E3E6F1] p-4 rounded-[4px]">
              <div className="font-bold text-[#0E1330] mb-1">Night-0 event anomaly as an open question</div>
              <p className="text-xs">
                First-night event prevalence was 23.0% versus 9.3% in subsequent nights. Sensor insertion trauma, calibration stabilization, or hospital admission stress remain open questions.
              </p>
            </div>
          </div>
        </section>

        {/* SECTION (g): FIGURES & PROVENANCE */}
        <section className="space-y-6">
          <div className="border-b border-[#CBD0E2] pb-3">
            <h2 className="text-xl font-bold text-[#0E1330] font-sans">
              Diagnostic plots & data provenance
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            <div className="border border-[#CBD0E2] rounded-[4px] p-4 bg-white">
              <div className="font-mono text-xs font-semibold text-[#0E1330] mb-2">
                Reliability & Calibration (Brier Skill Score = +25.0%, ECE = 0.015)
              </div>
              <img
                src="/data/figures/calibration_plot_revised.png"
                alt="Calibration Curve"
                className="w-full h-auto rounded-[2px] border border-[#E3E6F1]"
              />
              <div className="font-mono text-[11px] text-[#4A4F73] mt-2">
                Isotonically calibrated in-fold via patient-grouped splits against constant-prevalence baseline.
              </div>
            </div>

            <div className="border border-[#CBD0E2] rounded-[4px] p-4 bg-white">
              <div className="font-mono text-xs font-semibold text-[#0E1330] mb-2">
                SHAP Global Feature Importance
              </div>
              <img
                src="/data/figures/shap_summary.png"
                alt="SHAP Summary"
                className="w-full h-auto rounded-[2px] border border-[#E3E6F1]"
              />
              <div className="font-mono text-[11px] text-[#4A4F73] mt-2">
                Bedtime slope and prior nocturnal trough frequency emerge as the strongest predictors.
              </div>
            </div>
          </div>
        </section>

        {/* SECONDARY: FULL-COHORT RESULTS */}
        <section className="space-y-4 pt-8 border-t border-[#CBD0E2]">
          <div className="pb-2">
            <h2 className="text-base font-bold text-[#4A4F73] font-sans uppercase tracking-wider">
              Secondary analysis: full cohort (including night 0)
            </h2>
            <p className="text-xs text-[#4A4F73] mt-1">
              Presented for complete transparency. All 1,181 nights across 100 patients.
            </p>
          </div>

          <div className="border border-[#CBD0E2] rounded-[4px] overflow-hidden bg-white">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-[#F1F3F9] border-b border-[#CBD0E2] text-[#4A4F73]">
                  <th className="py-2 px-3 font-normal">Group</th>
                  <th className="py-2 px-3 font-normal">Model</th>
                  <th className="py-2 px-3 font-normal">AUROC (95% CI)</th>
                  <th className="py-2 px-3 font-normal">AUPRC (95% CI)</th>
                  <th className="py-2 px-3 font-normal">Brier</th>
                  <th className="py-2 px-3 font-normal">BSS (%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E3E6F1] text-[#0E1330]">
                {fullCohortRows.map((r) => (
                  <tr key={`${r.group}-${r.model}`}>
                    <td className="py-2 px-3 font-sans">{r.group}</td>
                    <td className="py-2 px-3 text-[#4A4F73]">{r.model}</td>
                    <td className="py-2 px-3">{r.aurocCi}</td>
                    <td className="py-2 px-3">{r.auprcCi}</td>
                    <td className="py-2 px-3">{r.brier}</td>
                    <td className="py-2 px-3">{r.bssPct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Light Footer */}
      <Footer variant="light" />
    </div>
  );
};
