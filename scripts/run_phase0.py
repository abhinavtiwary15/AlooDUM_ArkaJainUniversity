"""Phase 0 Execution Script: Data Audit & Pipeline Ingestion.

Executes the audit pipeline on Shanghai T2DM dataset and prints exact counts.
"""

from __future__ import annotations

import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.audit import run_full_data_audit


def main() -> None:
    print("Running Phase 0 Data Audit and Cohort Evaluation...")
    stats = run_full_data_audit(raw_dir="data/raw")
    print(f"Total patients: {stats['total_patients']}")
    print(f"Total nights: {stats['total_nights_all_22']}")
    print("Data audit completed successfully.")


if __name__ == "__main__":
    main()
