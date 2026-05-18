"""
Real Estate Price Normalizer — Avito Morocco v3
================================================
Handles two real problems found in Avito-scraped data:

  1. PLACEHOLDER PRICES — round numbers repeated across many listings
     (e.g. 500000 appearing on 80 listings). These are fake/default prices
     sellers enter. They get flagged and excluded from analysis.

  2. PER-M² vs TOTAL classification — done conservatively:
     - Default assumption is TOTAL (most Avito listings are total price)
     - Only flips to per_m² when the price is impossibly small to be a total
       AND fits perfectly as a per-m² value for that city/operation
     - When uncertain → flag for review, never blindly multiply

Usage:
    python price_normalizer.py --input combined_dataset.csv --output cleaned.csv
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PLACEHOLDER DETECTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROUND_PRICE_BASE     = 50_000   # sales: multiples of 50k are suspicious
PLACEHOLDER_MIN_COUNT = 10      # flag if same round price appears >= N times

ABSOLUTE_FLOOR = {
    "à vendre": 100_000,
    "à louer":  800,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CITY-AWARE THRESHOLDS (MAD)
# ─────────────────────────────────────────────────────────────────────────────
CITY_THRESHOLDS = {
    "casablanca": {
        "à vendre": dict(total_min=300_000,  total_max=15_000_000, per_m2_min=5_000,  per_m2_max=45_000, per_m2_price_max=45_000),
        "à louer":  dict(total_min=2_000,    total_max=60_000,     per_m2_min=50,     per_m2_max=400,    per_m2_price_max=400),
    },
    "rabat": {
        "à vendre": dict(total_min=250_000,  total_max=12_000_000, per_m2_min=5_000,  per_m2_max=40_000, per_m2_price_max=40_000),
        "à louer":  dict(total_min=2_000,    total_max=50_000,     per_m2_min=45,     per_m2_max=350,    per_m2_price_max=350),
    },
    "marrakech": {
        "à vendre": dict(total_min=250_000,  total_max=20_000_000, per_m2_min=5_000,  per_m2_max=50_000, per_m2_price_max=50_000),
        "à louer":  dict(total_min=2_000,    total_max=70_000,     per_m2_min=50,     per_m2_max=450,    per_m2_price_max=450),
    },
    "tanger": {
        "à vendre": dict(total_min=200_000,  total_max=8_000_000,  per_m2_min=4_500,  per_m2_max=35_000, per_m2_price_max=35_000),
        "à louer":  dict(total_min=1_800,    total_max=45_000,     per_m2_min=40,     per_m2_max=300,    per_m2_price_max=300),
    },
    "agadir": {
        "à vendre": dict(total_min=200_000,  total_max=6_000_000,  per_m2_min=4_500,  per_m2_max=30_000, per_m2_price_max=30_000),
        "à louer":  dict(total_min=1_500,    total_max=40_000,     per_m2_min=40,     per_m2_max=280,    per_m2_price_max=280),
    },
    "fes": {
        "à vendre": dict(total_min=150_000,  total_max=4_000_000,  per_m2_min=3_500,  per_m2_max=20_000, per_m2_price_max=20_000),
        "à louer":  dict(total_min=1_500,    total_max=25_000,     per_m2_min=30,     per_m2_max=200,    per_m2_price_max=200),
    },
    "meknes": {
        "à vendre": dict(total_min=120_000,  total_max=3_500_000,  per_m2_min=3_000,  per_m2_max=18_000, per_m2_price_max=18_000),
        "à louer":  dict(total_min=1_200,    total_max=20_000,     per_m2_min=25,     per_m2_max=180,    per_m2_price_max=180),
    },
    "oujda": {
        "à vendre": dict(total_min=100_000,  total_max=3_000_000,  per_m2_min=2_500,  per_m2_max=15_000, per_m2_price_max=15_000),
        "à louer":  dict(total_min=1_000,    total_max=18_000,     per_m2_min=20,     per_m2_max=150,    per_m2_price_max=150),
    },
    "kenitra": {
        "à vendre": dict(total_min=150_000,  total_max=3_500_000,  per_m2_min=3_500,  per_m2_max=20_000, per_m2_price_max=20_000),
        "à louer":  dict(total_min=1_500,    total_max=22_000,     per_m2_min=30,     per_m2_max=200,    per_m2_price_max=200),
    },
    "tetouan": {
        "à vendre": dict(total_min=150_000,  total_max=4_000_000,  per_m2_min=3_500,  per_m2_max=22_000, per_m2_price_max=22_000),
        "à louer":  dict(total_min=1_200,    total_max=22_000,     per_m2_min=30,     per_m2_max=200,    per_m2_price_max=200),
    },
    "el jadida": {
        "à vendre": dict(total_min=120_000,  total_max=3_500_000,  per_m2_min=3_500,  per_m2_max=20_000, per_m2_price_max=20_000),
        "à louer":  dict(total_min=1_200,    total_max=20_000,     per_m2_min=25,     per_m2_max=180,    per_m2_price_max=180),
    },
    "safi": {
        "à vendre": dict(total_min=100_000,  total_max=2_500_000,  per_m2_min=2_500,  per_m2_max=14_000, per_m2_price_max=14_000),
        "à louer":  dict(total_min=1_000,    total_max=15_000,     per_m2_min=20,     per_m2_max=130,    per_m2_price_max=130),
    },
    "default": {
        "à vendre": dict(total_min=100_000,  total_max=15_000_000, per_m2_min=2_500,  per_m2_max=45_000, per_m2_price_max=45_000),
        "à louer":  dict(total_min=800,      total_max=60_000,     per_m2_min=20,     per_m2_max=400,    per_m2_price_max=400),
    },
}


def get_thresholds(ville, operation: str) -> dict:
    city_key = str(ville).strip().lower() if pd.notna(ville) else "default"
    op_key   = "à louer" if "louer" in operation.strip().lower() else "à vendre"
    return CITY_THRESHOLDS.get(city_key, CITY_THRESHOLDS["default"])[op_key]


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLACEHOLDER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def flag_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_placeholder"] = False

    # A) Below absolute floor
    sale_mask = ~df["operation"].str.contains("louer")
    rent_mask =  df["operation"].str.contains("louer")
    df.loc[sale_mask & (df["prix"] < ABSOLUTE_FLOOR["à vendre"]), "is_placeholder"] = True
    df.loc[rent_mask & (df["prix"] < ABSOLUTE_FLOOR["à louer"]),  "is_placeholder"] = True

    # B) Round number repeated >= PLACEHOLDER_MIN_COUNT times
    price_counts = df["prix"].value_counts()
    repeated     = df["prix"].map(price_counts) >= PLACEHOLDER_MIN_COUNT

    sale_round = df["prix"] % ROUND_PRICE_BASE == 0
    rent_round = df["prix"] % 500 == 0

    df.loc[sale_mask & sale_round & repeated, "is_placeholder"] = True
    df.loc[rent_mask & rent_round & repeated, "is_placeholder"] = True

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSERVATIVE CLASSIFICATION — total-first, never guess
# ─────────────────────────────────────────────────────────────────────────────

def classify_price(row) -> tuple:
    t      = get_thresholds(row.get("ville", "default"), row["operation"])
    prix   = row["prix"]
    surf   = row["surface"]
    ratio  = prix / surf       # implied per-m² if prix is total
    total  = prix * surf       # implied total  if prix is per-m²
    op_key = "à louer" if "louer" in row["operation"] else "à vendre"
    floor  = ABSOLUTE_FLOOR[op_key]

    # 1. Below floor → only valid if it works as per-m²
    if prix < floor:
        if (t["per_m2_min"] <= prix <= t["per_m2_price_max"]
                and t["total_min"] <= total <= t["total_max"]):
            return "per_m2", "prix below total floor, valid as per_m2"
        return "flag", "prix below floor and doesn't fit per_m2"

    # 2. Fits total range with a realistic implied per-m² ratio → total
    if (t["total_min"] <= prix <= t["total_max"]
            and t["per_m2_min"] <= ratio <= t["per_m2_max"]):
        return "total", "fits total range with realistic per-m² ratio"

    # 3. Prix is too small to be a total but fits per-m² range → per_m2
    if (prix < t["total_min"]
            and t["per_m2_min"] <= prix <= t["per_m2_price_max"]
            and t["total_min"]  <= total <= t["total_max"]):
        return "per_m2", "prix too small for total, valid as per_m2"

    # 4. Total range OK but ratio is off → flag (don't touch)
    if t["total_min"] <= prix <= t["total_max"]:
        return "flag", f"total range OK but implied {ratio:,.0f} MAD/m² is unrealistic"

    # 5. Nothing fits
    return "flag", f"prix {prix:,.0f} fits neither total nor per_m2 for {row.get('ville','?')}"


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    results = df.apply(classify_price, axis=1)
    df["price_label"]  = results.apply(lambda x: x[0])
    df["label_reason"] = results.apply(lambda x: x[1])

    df["prix_normalise"] = np.where(
        df["price_label"] == "per_m2",
        df["prix"] * df["surface"],
        df["prix"],   # total or flag → keep as-is, NEVER blindly multiply
    )
    df["prix_m2"] = df["prix_normalise"] / df["surface"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame):
    total        = len(df)
    clean        = (df["price_label"] == "total").sum()
    per_m2       = (df["price_label"] == "per_m2").sum()
    flagged      = (df["price_label"] == "flag").sum()
    placeholders = df["is_placeholder"].sum()

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total rows              : {total:,}")
    print(f"  ├─ Clean (total price)  : {clean:,}  ({100*clean/total:.1f}%)")
    print(f"  ├─ Converted (per m²)   : {per_m2:,}  ({100*per_m2/total:.1f}%)")
    print(f"  ├─ Placeholder prices   : {placeholders:,}  ({100*placeholders/total:.1f}%)")
    print(f"  └─ Flagged for review   : {flagged:,}  ({100*flagged/total:.1f}%)")
    print("=" * 60)

    usable = df[~df["is_placeholder"] & (df["price_label"] != "flag")]
    if "ville" in df.columns and len(usable) > 0:
        print(f"\n  {'Ville':<15} {'Operation':<12} {'Median total':>14} {'Median /m²':>12} {'N':>6}")
        print("  " + "-" * 62)
        for (ville, op), grp in usable.groupby(["ville", "operation"]):
            print(f"  {ville:<15} {op:<12} {grp['prix_normalise'].median():>14,.0f} "
                  f"{grp['prix_m2'].median():>12,.0f} {len(grp):>6,}")

    top_ph = df[df["is_placeholder"]]["prix"].value_counts().head(10)
    if len(top_ph) > 0:
        print("\n  TOP PLACEHOLDER PRICES")
        for prix, count in top_ph.items():
            print(f"    {prix:>12,.0f} MAD  ×  {count} listings")

    flagged_rows = df[df["price_label"] == "flag"].head(5)
    if len(flagged_rows) > 0:
        print("\n  SAMPLE FLAGGED ROWS")
        cols = ["ville", "operation", "prix", "surface", "label_reason"]
        print(flagged_rows[[c for c in cols if c in flagged_rows.columns]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",             default="combined_dataset.csv")
    parser.add_argument("--output",            default="prix_normalise.csv")
    parser.add_argument("--sep",               default=",")
    parser.add_argument("--drop-placeholders", action="store_true",
                        help="Remove placeholder rows from output")
    parser.add_argument("--review-only",       action="store_true",
                        help="Export only flagged rows")
    args = parser.parse_args()

    print(f"📂 Loading {args.input} …")
    df = pd.read_csv(args.input, sep=args.sep)
    df.columns = df.columns.str.strip().str.lower()

    required = {"prix", "surface", "operation"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    if "ville" not in df.columns:
        print("⚠️  No 'ville' column — using default thresholds.")
        df["ville"] = "default"

    df["prix"]      = pd.to_numeric(df["prix"],    errors="coerce")
    df["surface"]   = pd.to_numeric(df["surface"], errors="coerce")
    df["ville"]     = df["ville"].fillna("default").str.strip().str.lower()
    df["operation"] = df["operation"].str.strip().str.lower()
    df.dropna(subset=["prix", "surface", "operation"], inplace=True)
    df = df[df["surface"] > 0].reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} valid rows")

    df = flag_placeholders(df)
    df = normalize_prices(df)
    summarize(df)

    # Enriched output (all rows + metadata)
    out = df.copy()
    if args.drop_placeholders:
        n = out["is_placeholder"].sum()
        out = out[~out["is_placeholder"]]
        print(f"\n🗑️  Dropped {n:,} placeholder rows")
    if args.review_only:
        out = out[out["price_label"] == "flag"]
        print(f"\n⚠️  Exporting {len(out):,} flagged rows")

    out.to_csv(args.output, index=False)
    print(f"\n✅ Saved → {args.output}")

    # Clean output: only good rows, prix replaced by normalised value
    meta = {"price_label", "label_reason", "is_placeholder", "prix_normalise", "prix_m2"}
    good = df[~df["is_placeholder"] & (df["price_label"] != "flag")].copy()
    orig_cols = [c for c in df.columns if c not in meta]
    clean = good[orig_cols].copy()
    clean["prix"] = good["prix_normalise"]
    clean.to_csv("prix.csv", index=False)
    print(f"✅ Saved → prix.csv  ({len(clean):,} clean rows)\n")


if __name__ == "__main__":
    main()