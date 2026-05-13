"""
Predict Missing Surface for clean_avito_lastPages.csv
======================================================
Trains on Masaken_dataset_completed.csv (complete dataset),
then fills missing surface values in clean_avito_lastPages.csv.

Usage:
    python predict_avito_surface.py

Or with custom paths:
    python predict_avito_surface.py \
        --train  Masaken_dataset_completed.csv \
        --target clean_avito_lastPages.csv \
        --output clean_avito_filled.csv
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import re
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def parse_rooms(desc):
    if pd.isna(desc): return np.nan
    m = re.search(r'(\d+)\s*chambre', str(desc), re.IGNORECASE)
    return int(m.group(1)) if m else np.nan

def parse_salles(desc):
    if pd.isna(desc): return np.nan
    m = re.search(r'(\d+)\s*salle', str(desc), re.IGNORECASE)
    return int(m.group(1)) if m else np.nan

def parse_salons(desc):
    if pd.isna(desc): return np.nan
    m = re.search(r'(\d+)\s*salon', str(desc), re.IGNORECASE)
    return int(m.group(1)) if m else np.nan

def safe_surface(val):
    """Convert surface column: 'None' / None / NaN → np.nan, else float."""
    if pd.isna(val): return np.nan
    if str(val).strip().lower() in ("none", "nan", "", "n/a"): return np.nan
    try: return float(val)
    except: return np.nan


# ─────────────────────────────────────────────────────────────
# NORMALISE COLUMN NAMES
# ─────────────────────────────────────────────────────────────

# Map from possible raw names → canonical internal names
_COL_MAP = {
    # price
    "prix": "Prix", "price": "Prix",
    # surface
    "surface": "Surface",
    # address
    "adresse": "Adresse", "address": "Adresse",
    # type
    "type": "Type",
    # operation
    "operation": "Operation",
    # description
    "description": "Description",
    # city
    "ville": "Ville", "city": "Ville",
    # date
    "date": "Date",
    # pre-computed amenities (Masaken complete dataset)
    "rooms": "rooms", "bathrooms": "bathrooms", "salons": "salons",
    "garage": "garage", "terrasse": "terrasse", "ascenseur": "ascenseur",
    "piscine": "piscine", "meuble": "meuble", "balcon": "balcon",
    "city_median_price": "city_median_price",
    "price_vs_city": "price_vs_city",
    "surface_regex": "surface_regex",
}

def normalise_columns(df):
    """Lowercase all column names, then rename to canonical form."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns=_COL_MAP, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (works on both datasets)
# ─────────────────────────────────────────────────────────────

def engineer_features(df, label_encoders=None, fit_encoders=True,
                      global_prix_per_m2_median=None):
    """
    df               : normalised dataframe
    label_encoders   : dict of fitted LabelEncoders (pass when transforming target)
    fit_encoders     : True = fit new encoders (training); False = use supplied ones
    global_prix_per_m2_median : median from training set (used when Surface unknown)
    """
    df = df.copy()

    # ── Surface: clean up "None" strings ──────────────────────
    df["Surface"] = df["Surface"].apply(safe_surface)

    # ── Prix: numeric ─────────────────────────────────────────
    df["Prix"] = pd.to_numeric(df["Prix"], errors="coerce")
    df["log_prix"] = np.log1p(df["Prix"].clip(lower=0))

    # ── Date ──────────────────────────────────────────────────
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["year"]  = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
    else:
        df["year"]  = np.nan
        df["month"] = np.nan

    # ── Room features ─────────────────────────────────────────
    if "rooms" in df.columns:
        df["nb_chambres"] = pd.to_numeric(df["rooms"], errors="coerce")
    else:
        df["nb_chambres"] = df["Description"].apply(parse_rooms) if "Description" in df.columns else np.nan

    if "bathrooms" in df.columns:
        df["nb_salles"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    else:
        df["nb_salles"] = df["Description"].apply(parse_salles) if "Description" in df.columns else np.nan

    if "salons" in df.columns:
        df["nb_salons"] = pd.to_numeric(df["salons"], errors="coerce")
    else:
        df["nb_salons"] = df["Description"].apply(parse_salons) if "Description" in df.columns else np.nan

    # ── Binary amenities ──────────────────────────────────────
    for col in ["garage", "terrasse", "ascenseur", "piscine", "meuble", "balcon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # ── Categorical encoding ──────────────────────────────────
    if label_encoders is None:
        label_encoders = {}

    for col in ["Adresse", "Type", "Operation", "Ville"]:
        enc_col = col + "_enc"
        if col not in df.columns:
            df[enc_col] = -1
            continue
        if fit_encoders:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                df[enc_col] = -1
            else:
                # Handle unseen labels safely
                known = set(le.classes_)
                df[enc_col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )

    # ── Price / m² (only computable where surface is known) ───
    mask = df["Surface"].notna() & (df["Surface"] > 0) & df["Prix"].notna()
    df["prix_per_m2"] = np.nan
    df.loc[mask, "prix_per_m2"] = df.loc[mask, "Prix"] / df.loc[mask, "Surface"]

    # Fill unknown prix_per_m2 with the median (from training if supplied)
    if global_prix_per_m2_median is None:
        global_prix_per_m2_median = df["prix_per_m2"].median()
    df["prix_per_m2_filled"] = df["prix_per_m2"].fillna(global_prix_per_m2_median)

    # ── Pre-computed city stats (only in complete dataset) ────
    for col in ["city_median_price", "price_vs_city"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df, label_encoders, global_prix_per_m2_median


# ─────────────────────────────────────────────────────────────
# CANDIDATE FEATURES  (dynamic — only used if present & non-empty)
# ─────────────────────────────────────────────────────────────

_CANDIDATES = [
    "log_prix", "prix_per_m2_filled",
    "Adresse_enc", "Type_enc", "Operation_enc", "Ville_enc",
    "nb_chambres", "nb_salles", "nb_salons",
    "garage", "terrasse", "ascenseur", "piscine", "meuble", "balcon",
    "city_median_price", "price_vs_city",
    "year", "month",
]

def get_feature_cols(df):
    return [c for c in _CANDIDATES if c in df.columns]


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, min_samples_leaf=5,
            subsample=0.8, random_state=42,
        )),
    ])


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def train(train_path):
    print(f"\n{'='*55}")
    print(f"  STEP 1 — Train on: {train_path}")
    print(f"{'='*55}")

    df_raw = pd.read_csv(train_path)
    df_raw = normalise_columns(df_raw)
    print(f"  Shape: {df_raw.shape}  |  Columns: {list(df_raw.columns)}")

    df_feat, encoders, ppm2_median = engineer_features(df_raw, fit_encoders=True)
    feature_cols = get_feature_cols(df_feat)

    known = df_feat[df_feat["Surface"].notna()].copy()
    X_train = known[feature_cols]
    y_train = known["Surface"]

    print(f"\n  Rows with known Surface: {len(known)}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    model = build_model()

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
    print(f"\n   CV R²: {cv_r2.round(3)}  →  mean={cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    if cv_r2.mean() >= 0.75:
        print("   Target R² ≥ 0.75 achieved!")
    else:
        print("    R² below 0.75 — check data quality.")

    # Final fit
    model.fit(X_train, y_train)
    y_hat = model.predict(X_train)
    print(f"  In-sample R²: {r2_score(y_train, y_hat):.3f}  |  MAE: {mean_absolute_error(y_train, y_hat):.1f} m²")

    # Feature importances
    gbr = model.named_steps["gbr"]
    surviving = [c for c in feature_cols if X_train[c].notna().any()]
    n_used = len(gbr.feature_importances_)
    if len(surviving) != n_used:
        surviving = feature_cols[:n_used]
    fi = pd.Series(gbr.feature_importances_, index=surviving).sort_values(ascending=False)
    print("\n   Feature importances:")
    print(fi.round(3).to_string(index=True))

    return model, feature_cols, encoders, ppm2_median


# ─────────────────────────────────────────────────────────────
# PREDICT ON TARGET CSV
# ─────────────────────────────────────────────────────────────

def predict_and_fill(target_path, output_path, model, feature_cols,
                     encoders, ppm2_median):
    print(f"\n{'='*55}")
    print(f"  STEP 2 — Predict surface in: {target_path}")
    print(f"{'='*55}")

    df_raw = pd.read_csv(target_path)
    df_raw = normalise_columns(df_raw)
    print(f"  Shape: {df_raw.shape}  |  Columns: {list(df_raw.columns)}")

    # Count "None" strings before cleaning
    if "Surface" in df_raw.columns:
        none_count = df_raw["Surface"].astype(str).str.strip().str.lower().isin(
            ["none", "nan", "", "n/a"]
        ).sum()
        print(f"  Rows with missing/None surface: {none_count}")
    else:
        print("   No 'surface' column found — will predict for all rows.")
        df_raw["Surface"] = np.nan

    # Engineer features — use fitted encoders from training, don't refit
    df_feat, _, _ = engineer_features(
        df_raw,
        label_encoders=encoders,
        fit_encoders=False,
        global_prix_per_m2_median=ppm2_median,
    )

    # Add any feature cols that exist in training but not in target (fill with NaN)
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan

    missing_mask = df_feat["Surface"].isna()
    n_missing = missing_mask.sum()

    if n_missing == 0:
        print("   No missing Surface values — nothing to predict.")
    else:
        X_pred = df_feat.loc[missing_mask, feature_cols]
        preds  = model.predict(X_pred).round(1)
        preds  = np.clip(preds, 10, 2000)

        df_raw.loc[missing_mask, "Surface"] = preds
        df_raw.loc[missing_mask, "surface_predicted"] = True
        df_raw["surface_predicted"] = df_raw["surface_predicted"].fillna(False)

        print(f"\n  Predicted surface for {n_missing} rows.")
        print(f"  Predicted Surface stats:")
        desc = pd.Series(preds).describe().round(1)
        for k, v in desc.items():
            print(f"    {k:>8}: {v}")

    df_raw.to_csv(output_path, index=False)
    print(f"\n   Saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # ── Train once ────────────────────────────────────────────
    TRAIN_CSV = "Masaken_dataset_completed.csv"

    # ── List every (target_csv, output_csv) pair you want to fill ──
    TARGETS = [
        (
            "clean_avito_lastPages.csv",
            "clean_avito_filled.csv",
        ),
        (
            r"C:\Users\HP\Downloads\PFA_web_scraping\clean_avito_selected_pages2.csv",
            r"C:\Users\HP\Downloads\PFA_web_scraping\clean_avito_selected_pages2_filled.csv",
        ),
    ]

    model, feature_cols, encoders, ppm2_median = train(TRAIN_CSV)

    for target_path, output_path in TARGETS:
        predict_and_fill(target_path, output_path, model, feature_cols, encoders, ppm2_median)

    print("\n All done!\n")

if __name__ == "__main__":
    main()