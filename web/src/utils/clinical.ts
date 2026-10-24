/**
 * Clinical utilities, formatting, and design tokens for NoctTwin.
 * Follows design specification: Never uses color alone to indicate state.
 */

import { RegimenCohort, RiskTier } from '../../../shared/types';
import { TIERS } from '../config';

export const ALERT_THRESHOLD_80SPEC = 0.126;
export const MODERATE_THRESHOLD = 0.063;

export function getRiskTierFromProbability(p: number): RiskTier {
  if (p >= ALERT_THRESHOLD_80SPEC) return 'HIGH';
  if (p >= MODERATE_THRESHOLD) return 'MODERATE';
  return 'LOW';
}

export function getRiskBadgeProps(tier: RiskTier): {
  label: string;
  symbol: string;
  textColor: string;
  bgColor: string;
  borderColor: string;
  badgeClass: string;
} {
  switch (tier) {
    case 'HIGH':
      return {
        label: TIERS.HIGH.label,
        symbol: TIERS.HIGH.symbol,
        textColor: TIERS.HIGH.color,
        bgColor: TIERS.HIGH.bg,
        borderColor: TIERS.HIGH.border,
        badgeClass: 'text-[#A30F14] bg-[#FCE4E4] border-[#F8B4B4]',
      };
    case 'MODERATE':
      return {
        label: TIERS.MODERATE.label,
        symbol: TIERS.MODERATE.symbol,
        textColor: TIERS.MODERATE.color,
        bgColor: TIERS.MODERATE.bg,
        borderColor: TIERS.MODERATE.border,
        badgeClass: 'text-[#7A4F00] bg-[#FFF3D6] border-[#FFE194]',
      };
    case 'LOW':
    default:
      return {
        label: TIERS.LOW.label,
        symbol: TIERS.LOW.symbol,
        textColor: TIERS.LOW.color,
        bgColor: TIERS.LOW.bg,
        borderColor: TIERS.LOW.border,
        badgeClass: 'text-[#046A38] bg-[#E2F1E8] border-[#BDE3CC]',
      };
  }
}

export function formatRegimenLabel(cohort: RegimenCohort): string {
  switch (cohort) {
    case 'SU_WITHOUT_INSULIN':
      return 'Sulfonylurea';
    case 'SU_WITH_INSULIN':
      return 'SU + Insulin';
    case 'INSULIN_WITHOUT_SU':
      return 'Insulin';
    case 'NEITHER':
      return 'Neither';
    default:
      return cohort;
  }
}

export function formatGlucose(val: number): string {
  return `${Math.round(val)} mg/dL`;
}

export function formatPercent(val: number): string {
  return `${Math.round(val * 100)}%`;
}

export function formatPercentDecimal(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

/**
 * Convert HbA1c from % to mmol/mol (IFCC equation)
 * mmol/mol = (HbA1c% - 2.15) * 10.929
 */
export function hba1cPercentToMmol(pct: number): number {
  return Math.round((pct - 2.15) * 10.929);
}

export function formatHba1c(pct: number): string {
  const mmol = hba1cPercentToMmol(pct);
  return `HbA1c ${pct.toFixed(1)}% (${mmol} mmol/mol)`;
}
