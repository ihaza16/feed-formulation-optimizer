#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feed Formulation Optimizer
Author: Imuetinyanosa Benjamin Ihaza
Purpose: Minimize ration cost while satisfying nutrient requirements.

Features
- Reads ingredients from Excel/CSV if provided (columns described below)
- Falls back to built-in demo ingredients if no file is passed
- Supports multiple diet types (edit or add your own easily)
- Prints optimal inclusion rates, cost, and nutrient satisfaction
- Saves results to CSV if --out is provided

USAGE EXAMPLES
--------------
# 1) Use built-in demo ingredients (no file), solve for broiler starter
python feed_formulation_optimizer.py --diet broiler_starter

# 2) Use your spreadsheet (Excel/CSV), solve for layers
python feed_formulation_optimizer.py --file Livestock_Feed_Composition.xlsx --diet layers --out result_layers.csv

INPUT FILE FORMAT (Excel or CSV)
--------------------------------
Required columns (case-insensitive; spaces/underscores allowed):
- ingredient
- cost_per_kg
- cp      (crude protein, %)
- me      (metabolizable energy, kcal/kg)
- cf      (crude fibre, %)
- lys     (lysine, %)
- met     (methionine, %)
- ca      (calcium, %)
- avp     (available phosphorus, %)
Optional:
- min_inclusion (% of diet), max_inclusion (% of diet)

All % values are on as-fed basis. cost_per_kg in same currency unit.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys
import os

try:
    import pandas as pd  # for reading Excel/CSV (optional)
except Exception:  # pragma: no cover
    pd = None

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD


# --------------------------- Data structures ---------------------------

@dataclass
class Ingredient:
    cost: float         # cost per kg
    cp: float           # crude protein (%)
    me: float           # metabolizable energy (kcal/kg)
    cf: float           # crude fibre (%)
    lys: float          # lysine (%)
    met: float          # methionine (%)
    ca: float           # calcium (%)
    avp: float          # available phosphorus (%)
    min_incl: float = 0.0   # minimum inclusion (% of diet)
    max_incl: float = 100.0 # maximum inclusion (% of diet)

# Requirements per diet (min or max); Percent values are on as-fed basis
# These are typical targets; adjust to your formulation standard.
DIET_REQUIREMENTS = {
    "broiler_starter": {
        "cp_min": 24.0,
        "me_min": 3200.0,     # kcal/kg
        "cf_max": 4.0,
        "lys_min": 1.40,
        "met_min": 0.55,
        "ca_min": 1.00,
        "avp_min": 0.55,
    },
    "layers": {
        "cp_min": 17.0,
        "me_min": 2700.0,
        "cf_max": 7.0,
        "lys_min": 0.75,
        "met_min": 0.35,
        "ca_min": 3.50,
        "avp_min": 0.45,
    },
    "goat_growth": {
        "cp_min": 14.0,
        "me_min": 2400.0,
        "cf_max": 15.0,
        "lys_min": 0.60,
        "met_min": 0.25,
        "ca_min": 0.60,
        "avp_min": 0.30,
    }
}

# Built-in demo ingredient table (edit freely or override with --file)
DEMO_INGREDIENTS: Dict[str, Ingredient] = {
    "maize":            Ingredient(cost=0.25, cp=8.5,  me=3300, cf=2.4, lys=0.25, met=0.18, ca=0.03, avp=0.08, max_incl=65),
    "soybean_meal":     Ingredient(cost=0.48, cp=46.0, me=2450, cf=6.0, lys=2.90, met=0.65, ca=0.30, avp=0.20, max_incl=35),
    "groundnut_cake":   Ingredient(cost=0.42, cp=42.0, me=2500, cf=7.0, lys=1.40, met=0.55, ca=0.25, avp=0.20, max_incl=20),
    "fishmeal":         Ingredient(cost=0.90, cp=60.0, me=2800, cf=1.0, lys=4.50, met=1.60, ca=5.50, avp=3.00, max_incl=10),
    "rice_bran":        Ingredient(cost=0.15, cp=12.0, me=1800, cf=12.0, lys=0.55, met=0.25, ca=0.10, avp=0.30, max_incl=20),
    "wheat_offal":      Ingredient(cost=0.17, cp=16.0, me=1700, cf=10.0, lys=0.55, met=0.25, ca=0.13, avp=0.28, max_incl=25),
    "limestone":        Ingredient(cost=0.08, cp=0.0,  me=0,    cf=0.0, lys=0.00, met=0.00, ca=38.0, avp=0.00, max_incl=8),
    "dcp":              Ingredient(cost=0.75, cp=0.0,  me=0,    cf=0.0, lys=0.00, met=0.00, ca=23.0, avp=18.0, max_incl=3),
    "premix":           Ingredient(cost=1.20, cp=0.0,  me=0,    cf=0.0, lys=0.00, met=0.00, ca=0.00, avp=0.00, min_incl=0.5, max_incl=1.0),
    "salt":             Ingredient(cost=0.10, cp=0.0,  me=0,    cf=0.0, lys=0.00, met=0.00, ca=0.00, avp=0.00, min_incl=0.2, max_incl=0.5),
}

# --------------------------- Utility functions ---------------------------

def _normalize(col: str) -> str:
    return col.strip().lower().replace(" ", "_")

def load_ingredients_from_file(path: str) -> Dict[str, Ingredient]:
    if pd is None:
        raise RuntimeError("pandas is required to read Excel/CSV files. Install with: pip install pandas openpyxl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df.columns = [_normalize(c) for c in df.columns]

    required = ["ingredient", "cost_per_kg", "cp", "me", "cf", "lys", "met", "ca", "avp"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}")

    ingredients: Dict[str, Ingredient] = {}
    for _, row in df.iterrows():
        name = str(row["ingredient"]).strip().lower()
        ingredients[name] = Ingredient(
            cost=float(row["cost_per_kg"]),
            cp=float(row["cp"]),
            me=float(row["me"]),
            cf=float(row["cf"]),
            lys=float(row["lys"]),
            met=float(row["met"]),
            ca=float(row["ca"]),
            avp=float(row["avp"]),
            min_incl=float(row.get("min_inclusion", 0) or 0),
            max_incl=float(row.get("max_inclusion", 100) or 100),
        )
    return ingredients

# --------------------------- LP model builder ---------------------------

def build_and_solve(ingredients: Dict[str, Ingredient],
                    requirements: Dict[str, float],
                    solver: Optional[PULP_CBC_CMD] = None
                   ) -> Tuple[str, Dict[str, float], Dict[str, float], float]:
    """
    Returns: (status, inclusion_percent_by_ingredient, nutrient_values, total_cost_per_ton)
    """
    names = list(ingredients.keys())
    model = LpProblem("Feed_Formulation", LpMinimize)

    # Variables: inclusion fraction (0..1)
    x = {n: LpVariable(f"x_{n}", lowBound=0.0) for n in names}

    # Objective: minimize cost per kg
    model += lpSum(ingredients[n].cost * x[n] for n in names), "Total_Cost_per_kg"

    # Sum to 1 (100%)
    model += lpSum(x[n] for n in names) == 1.0, "Sum_to_100pct"

    # Inclusion bounds
    for n in names:
        ing = ingredients[n]
        if ing.min_incl > 0:
            model += x[n] >= ing.min_incl / 100.0, f"MinIncl_{n}"
        if ing.max_incl < 100:
            model += x[n] <= ing.max_incl / 100.0, f"MaxIncl_{n}"

    # Nutrient constraints
    def nutrient(expr, attr):
        return lpSum(getattr(ingredients[n], attr) * x[n] for n in names)

    if "cp_min"  in requirements: model += nutrient(x, "cp")  >= requirements["cp_min"],  "CP_min"
    if "me_min"  in requirements: model += nutrient(x, "me")  >= requirements["me_min"],  "ME_min"
    if "cf_max"  in requirements: model += nutrient(x, "cf")  <= requirements["cf_max"],  "CF_max"
    if "lys_min" in requirements: model += nutrient(x, "lys") >= requirements["lys_min"], "LYS_min"
    if "met_min" in requirements: model += nutrient(x, "met") >= requirements["met_min"], "MET_min"
    if "ca_min"  in requirements: model += nutrient(x, "ca")  >= requirements["ca_min"],  "CA_min"
    if "avp_min" in requirements: model += nutrient(x, "avp") >= requirements["avp_min"], "AVP_min"

    # Solve
    if solver is None:
        solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = LpStatus[model.status]
    inclusion = {n: round(100.0 * x[n].value(), 3) for n in names if x[n].value() is not None and x[n].value() > 1e-8}
    cost_per_kg = value(model.objective)
    cost_per_ton = round(1000.0 * cost_per_kg, 2) if cost_per_kg is not None else float("nan")

    # Compute achieved nutrients
    def val(attr): return round(sum(getattr(ingredients[n], attr) * (x[n].value() or 0.0) for n in names), 4)
    achieved = {
        "cp":  val("cp"),
        "me":  val("me"),
        "cf":  val("cf"),
        "lys": val("lys"),
        "met": val("met"),
        "ca":  val("ca"),
        "avp": val("avp"),
    }

    return status, inclusion, achieved, cost_per_ton


# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Feed Formulation Optimizer")
    parser.add_argument("--file", help="Excel/CSV file with ingredient table", default=None)
    parser.add_argument("--diet", help=f"Diet type ({', '.join(DIET_REQUIREMENTS.keys())})", default="broiler_starter")
    parser.add_argument("--out", help="Optional CSV path to save inclusion and nutrient summary", default=None)
    args = parser.parse_args()

    # Load ingredients
    if args.file:
        try:
            ings = load_ingredients_from_file(args.file)
            print(f"Loaded {len(ings)} ingredients from {args.file}")
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    else:
        ings = DEMO_INGREDIENTS
        print(f"Using built-in demo ingredients ({len(ings)} items). Pass --file to use your spreadsheet.")

    diet = args.diet.lower().strip()
    if diet not in DIET_REQUIREMENTS:
        print(f"[ERROR] Unknown diet '{args.diet}'. Choose from: {', '.join(DIET_REQUIREMENTS.keys())}")
        sys.exit(1)

    req = DIET_REQUIREMENTS[diet]
    print(f"Solving for diet: {diet} with requirements: {req}")

    status, inclusion, achieved, cost_per_ton = build_and_solve(ings, req)

    print("\n=== OPTIMIZATION STATUS ===")
    print(status)
    if status != "Optimal":
        print("Solution is not optimal. Check requirements and inclusion bounds.")
    print("\n=== OPTIMAL INCLUSION RATES (% of diet) ===")
    for k, v in sorted(inclusion.items(), key=lambda kv: -kv[1]):
        print(f"{k:20s} : {v:6.3f} %")

    print("\n=== NUTRIENT SUMMARY (achieved vs requirement) ===")
    def show(name, unit, key_min=None, key_max=None):
        if key_min and key_min in req:
            print(f"{name:24s}: {achieved[unit]:8.3f}  >=  {req[key_min]:8.3f}  ({unit})")
        elif key_max and key_max in req:
            print(f"{name:24s}: {achieved[unit]:8.3f}  <=  {req[key_max]:8.3f}  ({unit})")
        else:
            print(f"{name:24s}: {achieved[unit]:8.3f}  ({unit})")

    show("Crude Protein", "cp", key_min="cp_min")
    show("Metabolizable Energy", "me", key_min="me_min")
    show("Crude Fibre", "cf", key_max="cf_max")
    show("Lysine", "lys", key_min="lys_min")
    show("Methionine", "met", key_min="met_min")
    show("Calcium", "ca", key_min="ca_min")
    show("Available Phosphorus", "avp", key_min="avp_min")

    print(f"\n=== TOTAL COST ===")
    print(f"Cost per ton: {cost_per_ton:.2f}\n")

    # Optional CSV export
    if args.out:
        try:
            if pd is None:
                print("[WARN] pandas not available; cannot save CSV.")
            else:
                inc_df = pd.DataFrame(
                    [{"ingredient": k, "inclusion_percent": v} for k, v in inclusion.items()]
                ).sort_values("inclusion_percent", ascending=False)
                nut_df = pd.DataFrame(
                    [{"nutrient": k, "achieved": v, "requirement": req.get(f"{k}_min", req.get(f"{k}_max", None))}]
                )
                with pd.ExcelWriter(args.out.replace(".csv", ".xlsx"), engine="openpyxl") as xlw:
                    inc_df.to_excel(xlw, index=False, sheet_name="inclusion")
                    nut_df.to_excel(xlw, index=False, sheet_name="nutrients")
                print(f"Saved summary to {args.out.replace('.csv', '.xlsx')}")
        except Exception as e:
            print(f"[WARN] Could not save output: {if __name__ == "__main__":main()
