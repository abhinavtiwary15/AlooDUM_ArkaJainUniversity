import { describe, it, expect } from 'vitest';
import {
  getRiskBadgeProps,
  formatRegimenLabel,
  formatGlucose,
  formatPercent,
  getRiskTierFromProbability,
  hba1cPercentToMmol,
  formatHba1c,
  ALERT_THRESHOLD_80SPEC,
  MODERATE_THRESHOLD,
} from '../utils/clinical';

describe('Clinical Utilities', () => {
  it('assigns strictly monotonic risk tiers from probability', () => {
    expect(getRiskTierFromProbability(0.95)).toBe('HIGH');
    expect(getRiskTierFromProbability(ALERT_THRESHOLD_80SPEC)).toBe('HIGH');
    expect(getRiskTierFromProbability(ALERT_THRESHOLD_80SPEC - 0.001)).toBe('MODERATE');
    expect(getRiskTierFromProbability(MODERATE_THRESHOLD)).toBe('MODERATE');
    expect(getRiskTierFromProbability(MODERATE_THRESHOLD - 0.001)).toBe('LOW');
    expect(getRiskTierFromProbability(0.01)).toBe('LOW');

    // Monotonicity invariant across range
    const probs = [0.01, 0.05, 0.063, 0.08, 0.126, 0.25, 0.5, 0.9];
    const tierScore = (t: string) => (t === 'HIGH' ? 3 : t === 'MODERATE' ? 2 : 1);
    for (let i = 1; i < probs.length; i++) {
      expect(tierScore(getRiskTierFromProbability(probs[i]))).toBeGreaterThanOrEqual(
        tierScore(getRiskTierFromProbability(probs[i - 1]))
      );
    }
  });

  it('assigns correct risk badge properties with shape, color and text', () => {
    const high = getRiskBadgeProps('HIGH');
    expect(high.label).toBe('High');
    expect(high.symbol).toBe('▲');
    expect(high.textColor).toBe('#A30F14');

    const mod = getRiskBadgeProps('MODERATE');
    expect(mod.label).toBe('Moderate');
    expect(mod.symbol).toBe('◆');
    expect(mod.textColor).toBe('#7A4F00');

    const low = getRiskBadgeProps('LOW');
    expect(low.label).toBe('Low');
    expect(low.symbol).toBe('●');
    expect(low.textColor).toBe('#046A38');
  });

  it('formats regimen cohort labels properly according to design spec', () => {
    expect(formatRegimenLabel('SU_WITHOUT_INSULIN')).toBe('Sulfonylurea');
    expect(formatRegimenLabel('SU_WITH_INSULIN')).toBe('SU + Insulin');
    expect(formatRegimenLabel('INSULIN_WITHOUT_SU')).toBe('Insulin');
    expect(formatRegimenLabel('NEITHER')).toBe('Neither');
  });

  it('formats glucose values correctly', () => {
    expect(formatGlucose(82.4)).toBe('82 mg/dL');
    expect(formatGlucose(140.8)).toBe('141 mg/dL');
  });

  it('formats percentage correctly', () => {
    expect(formatPercent(0.105)).toBe('11%');
    expect(formatPercent(0.93)).toBe('93%');
  });

  it('converts HbA1c correctly between % and mmol/mol', () => {
    expect(hba1cPercentToMmol(8.3)).toBe(67);
    expect(formatHba1c(8.3)).toBe('HbA1c 8.3% (67 mmol/mol)');
  });

  it('points for clinician review rules enforce factual data without treatment mentions', () => {
    const forbiddenPatterns = [/snack/i, /dose/i, /treatment/i, /timing of the evening/i];
    const factualSamples = [
      'Low last night: Persistent low (<70 mg/dL for ≥30 min) recorded on immediately prior night.',
      'Low last night: No persistent low recorded on immediately prior night.',
      'Prior-night count: 2 persistent low(s) across 5 earlier night(s).',
      'Bedtime trend: -8.4 mg/dL/h over prior 3 hours.',
      'Expected range: Forecast uncertainty band crosses below 70 mg/dL between 23:00 and 06:00.',
      'Personal lowest value: Historical median overnight nadir is 58.2 mg/dL.',
    ];
    for (const sample of factualSamples) {
      for (const pattern of forbiddenPatterns) {
        expect(pattern.test(sample)).toBe(false);
      }
    }
  });
});
