"""Medication and dietary intake parsers for Shanghai T2DM dataset."""

from __future__ import annotations

import math
import re
from typing import Any

# Target Sulfonylurea keywords (English & Chinese)
SU_KEYWORDS: dict[str, str] = {
    "glimepiride": "glimepiride",
    "gliclazide": "gliclazide",
    "glipizide": "glipizide",
    "glibenclamide": "glibenclamide",
    "glyburide": "glibenclamide",
    "gliquidone": "gliquidone",
    "格列美脲": "glimepiride",
    "格列齐特": "gliclazide",
    "格列吡嗪": "glipizide",
    "格列本脲": "glibenclamide",
    "格列喹酮": "gliquidone",
}

# Target Insulin keywords (brand names, types, typos, and Chinese)
INSULIN_KEYWORDS: dict[str, str] = {
    "insulin": "insulin",
    "novolin": "novolin",
    "humulin": "humulin",
    "gansulin": "gansulin",
    "degludec": "degludec",
    "detemir": "detemir",
    "glargine": "glargine",
    "glarigine": "glargine",  # observed dataset typo in summary sheet
    "aspart": "aspart",
    "lispro": "lispro",
    "glulisine": "glulisine",
    "csii": "csii",
    "胰岛素": "insulin",
    "诺和灵": "novolin",
    "优泌林": "humulin",
    "甘舒霖": "gansulin",
    "德谷": "degludec",
    "地特": "detemir",
    "甘精": "glargine",
    "门冬": "aspart",
    "赖脯": "lispro",
    "谷赖": "glulisine",
}


def parse_medications(text: Any) -> dict[str, Any]:
    """Parse medication string for Sulfonylureas, Insulin, and specific agent names.

    Args:
        text: Free-text medication string from summary or daily log.

    Returns:
        dict with boolean flags and lists of identified agents.
    """
    if not isinstance(text, str) or not text.strip() or text.strip().lower() in ("none", "nan", "null"):
        return {
            "has_su": False,
            "has_insulin": False,
            "su_agents": [],
            "insulin_agents": [],
            "raw_text": "" if text is None else str(text).strip(),
        }

    raw = text.strip()
    raw_lower = raw.lower()

    detected_su: list[str] = []
    for kw, standard_name in SU_KEYWORDS.items():
        if kw in raw_lower and standard_name not in detected_su:
            detected_su.append(standard_name)

    detected_ins: list[str] = []
    for kw, standard_name in INSULIN_KEYWORDS.items():
        if kw in raw_lower and standard_name not in detected_ins:
            detected_ins.append(standard_name)

    return {
        "has_su": len(detected_su) > 0,
        "has_insulin": len(detected_ins) > 0,
        "su_agents": sorted(detected_su),
        "insulin_agents": sorted(detected_ins),
        "raw_text": raw,
    }


# Approximate carbohydrate fraction per gram of common food items (literature / nutritional tables)
FOOD_CARB_FRACTION: dict[str, float] = {
    "rice": 0.28,        # cooked white/brown rice ~28% carbs
    "fried rice": 0.30,
    "coarse grain": 0.25,
    "congee": 0.12,
    "porridge": 0.12,
    "noodle": 0.25,
    "bread": 0.50,
    "buckwheat bread": 0.45,
    "steamed bun": 0.45,
    "steamed dumpling": 0.30,
    "dumpling": 0.30,
    "wonton": 0.25,
    "pancake": 0.40,
    "potato": 0.17,
    "sweet potato": 0.20,
    "corn": 0.19,
    "lotus root": 0.16,
    "oats": 0.60,
    "oatmeal": 0.15,
    "biscuit": 0.70,
    "cracker": 0.70,
    "apple": 0.14,
    "banana": 0.23,
    "orange": 0.12,
    "pear": 0.15,
    "peach": 0.10,
    "watermelon": 0.08,
    "milk": 0.05,        # 5g carbs per 100ml
    "yogurt": 0.08,
    "vegetable": 0.04,   # leafy greens ~3-5% carbs
    "cucumber": 0.03,
    "tomato": 0.04,
    "egg": 0.01,         # negligible carbs
    "meat": 0.00,
    "pork": 0.00,
    "beef": 0.00,
    "chicken": 0.00,
    "duck": 0.00,
    "fish": 0.00,
    "shrimp": 0.00,
    "tofu": 0.02,
    "bean curd": 0.02,
}


def parse_diet_carbs(text: Any) -> float:
    """Extract estimated total carbohydrate grams from dietary intake logs.

    Lines typically formatted as:
      'Rice 150 g'
      'Vegetable 100 g'
      'Sliced lotus root 50 g'

    Returns:
        float: Estimated total carbohydrate grams, or 0.0 if empty / unavailable.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    raw = text.strip()
    if "not available" in raw.lower() or raw.lower() in ("none", "nan"):
        return 0.0

    total_carbs = 0.0
    lines = raw.split("\n")

    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue

        # Match number followed by g or ml
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|ml|gram|grams)", line_clean)
        amount_g = float(match.group(1)) if match else 0.0

        if amount_g <= 0.0:
            continue

        # Identify food item
        matched_fraction = 0.05  # default conservative carbohydrate fraction
        for food_key, frac in FOOD_CARB_FRACTION.items():
            if food_key in line_clean:
                matched_fraction = frac
                break

        total_carbs += amount_g * matched_fraction

    return round(total_carbs, 2)


def parse_insulin_dose(val: Any) -> float:
    """Parse insulin dose in IU from numeric or text log entries."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val) if not math.isnan(val) else 0.0

    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return 0.0

    # Match IU or U notation e.g. 'Novolin R, 14 IU' or '12 IU'
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:iu|u|units?)", s, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Fallback to any positive number in the string
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m2.group(1)) if m2 else 0.0


def parse_su_dose(text: Any) -> tuple[float, str]:
    """Parse Sulfonylurea agent name and dosage in mg from medication log entries.

    Returns:
        tuple[float, str]: (dose_mg, agent_name)
    """
    if not isinstance(text, str):
        return 0.0, "none"

    su_kws = ["glimepiride", "gliclazide", "glipizide", "glibenclamide", "glyburide", "gliquidone"]
    tl = text.lower()

    for kw in su_kws:
        if kw in tl:
            std_name = "glibenclamide" if kw == "glyburide" else kw
            # Look for number near the keyword e.g. 'gliclazide 120 mg' or '120 mg gliclazide'
            pattern = rf"{kw}\s*(\d+(?:\.\d+)?)\s*mg|(\d+(?:\.\d+)?)\s*mg\s*{kw}"
            m = re.search(pattern, tl)
            if m:
                val = m.group(1) or m.group(2)
                return float(val), std_name
            # Fallback: look for any number after kw
            m2 = re.search(rf"{kw}\s*(\d+(?:\.\d+)?)", tl)
            if m2:
                return float(m2.group(1)), std_name
            return 0.0, std_name

    return 0.0, "none"
