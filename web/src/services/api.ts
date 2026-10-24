/**
 * Dual-Mode Data Client:
 * Tries FastAPI endpoints first; seamlessly falls back to static JSON files
 * in web/public/data/ if the API is offline or unreachable.
 */

import {
  ModelCardData,
  NightDetail,
  PatientSummary,
  WhatIfRequest,
  WhatIfResponse,
} from '../../../shared/types';

let apiAvailable: boolean | null = null;

export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch('/api/health', { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      apiAvailable = true;
      return true;
    }
  } catch {
    apiAvailable = false;
  }
  apiAvailable = false;
  return false;
}

export function isApiAvailable(): boolean {
  return apiAvailable === true;
}

export async function fetchPatients(): Promise<PatientSummary[]> {
  try {
    const res = await fetch('/api/patients', { signal: AbortSignal.timeout(2500) });
    if (res.ok) {
      apiAvailable = true;
      return await res.json();
    }
  } catch {
    apiAvailable = false;
  }

  // Fallback to static JSON
  const res = await fetch('/data/patients.json');
  if (!res.ok) throw new Error('Failed to load static patients data');
  return await res.json();
}

export async function fetchPatientNights(patientId: string): Promise<NightDetail[]> {
  try {
    const res = await fetch(`/api/patients/${patientId}/nights`, {
      signal: AbortSignal.timeout(2500),
    });
    if (res.ok) {
      apiAvailable = true;
      return await res.json();
    }
  } catch {
    apiAvailable = false;
  }

  // Fallback to static JSON
  const res = await fetch(`/data/nights/${patientId}.json`);
  if (!res.ok) throw new Error(`Failed to load static nights for patient ${patientId}`);
  return await res.json();
}

export async function fetchModelCard(): Promise<ModelCardData> {
  try {
    const res = await fetch('/api/model-card', { signal: AbortSignal.timeout(2500) });
    if (res.ok) {
      apiAvailable = true;
      return await res.json();
    }
  } catch {
    apiAvailable = false;
  }

  // Fallback to static JSON
  const res = await fetch('/data/model_card.json');
  if (!res.ok) throw new Error('Failed to load static model card');
  return await res.json();
}

export async function runWhatIfSimulation(req: WhatIfRequest): Promise<WhatIfResponse> {
  // Try live API first
  try {
    const res = await fetch('/api/twin/what-if', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      apiAvailable = true;
      return await res.json();
    }
  } catch {
    apiAvailable = false;
  }

  // Offline Fallback: Use precomputed whatif_grid.json
  const gridRes = await fetch('/data/whatif_grid.json');
  if (!gridRes.ok) throw new Error('Failed to load precomputed what-if grid');
  const gridMap: Record<string, WhatIfResponse[]> = await gridRes.json();
  const ptGrid = gridMap[req.patientId] || [];

  if (ptGrid.length === 0) {
    throw new Error(`No precomputed what-if grid found for patient ${req.patientId}`);
  }

  // Find nearest discrete grid point (carbs in [0, 15, 30, 45], su_scale in [0.0, 0.5, 0.75, 1.0])
  const targetCarbs = req.snackCarbsG;
  const targetScale = req.suDoseScale;

  let bestMatch = ptGrid[0];
  let minDistance = Infinity;

  for (const item of ptGrid) {
    const dCarbs = Math.abs(item.snackCarbsG - targetCarbs);
    const dScale = Math.abs(item.suDoseScale - targetScale) * 30.0;
    const dist = dCarbs + dScale;
    if (dist < minDistance) {
      minDistance = dist;
      bestMatch = item;
    }
  }

  return {
    ...bestMatch,
    snackCarbsG: req.snackCarbsG,
    suDoseScale: req.suDoseScale,
    disclaimer:
      'Simulation hypothesis (loaded from precomputed offline grid). Not clinically validated.',
  };
}
