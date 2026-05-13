"""
Surface Area Predictor for Masaken Dataset
==========================================
Predicts missing 'Surface' values using regression.
Target: R² ≥ 0.75

Usage:
    python surface_predictor.py --input Masaken_dataset_completed.csv --output filled_dataset.csv
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import re
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def extract_rooms_from_description(desc):
    """Parse number of bedrooms (chambres) from description string."""
    if pd.isna(desc):
        return np.nan
    match = re.search(r'(\d+)\s*chambre', str(desc), re.IGNORECASE)
    return int(match.group(1)) if match else np.nan

def extract_bathrooms_from_description(desc):
    """Parse number of bathrooms from description string."""
    if pd.isna(desc):
        return np.nan
    match = re.search(r'(\d+)\s*salle', str(desc), re.IGNORECASE)
    return int(match.group(1)) if match else np.nan

def extract_price_per_m2_proxy(row):
    """If surface is known, compute price/m². Later used as a feature."""
    if pd.notna(row["Surface"]) and row["Surface"] > 0:
        return row["Prix"] / row["Surface"]
    return np.nan

def engineer_features(df):
    """Build all features from raw columns."""
    df = df.copy()

    # --- Date features ---
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    # --- Use existing parsed columns if available, else parse from Description ---
    if "rooms" in df.columns:
        df["nb_chambres"] = pd.to_numeric(df["rooms"], errors="coerce")
    else:
        df["nb_chambres"] = df["Description"].apply(extract_rooms_from_description)

    if "bathrooms" in df.columns:
        df["nb_salles"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    else:
        df["nb_salles"] = df["Description"].apply(extract_bathrooms_from_description)

    if "salons" in df.columns:
        df["nb_salons"] = pd.to_numeric(df["salons"], errors="coerce")
    else:
        df["nb_salons"] = np.nan

    # --- Binary amenity features already in dataset ---
    for col in ["garage", "terrasse", "ascenseur", "piscine", "meuble", "balcon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Encode categoricals ---
    for col in ["Adresse", "Type", "Operation", "Ville"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # --- Log-transform Prix (heavy right skew) ---
    df["log_prix"] = np.log1p(df["Prix"].clip(lower=0))

    # --- Price / m² from rows that DO have Surface (used only for training) ---
    df["prix_per_m2"] = df.apply(extract_price_per_m2_proxy, axis=1)
    median_ppm2 = df["prix_per_m2"].median()
    df["prix_per_m2_filled"] = df["prix_per_m2"].fillna(median_ppm2)

    # --- Use pre-computed city median price if available ---
    if "city_median_price" in df.columns:
        df["city_median_price"] = pd.to_numeric(df["city_median_price"], errors="coerce")
    if "price_vs_city" in df.columns:
        df["price_vs_city"] = pd.to_numeric(df["price_vs_city"], errors="coerce")

    return df


# ─────────────────────────────────────────────
# 2. FEATURE SELECTION
# ─────────────────────────────────────────────

# Build feature list dynamically — only include cols that exist after engineering
_CANDIDATE_FEATURES = [
    "log_prix",
    "prix_per_m2_filled",
    "Adresse_enc",
    "Type_enc",
    "Operation_enc",
    "Ville_enc",
    "nb_chambres",
    "nb_salles",
    "nb_salons",
    "garage",
    "terrasse",
    "ascenseur",
    "piscine",
    "meuble",
    "balcon",
    "city_median_price",
    "price_vs_city",
    "year",
    "month",
]

def get_feature_cols(df_feat):
    """Return only candidates that actually exist in the dataframe."""
    return [c for c in _CANDIDATE_FEATURES if c in df_feat.columns]

def build_model():
    """Gradient Boosting Regressor inside a simple pipeline."""
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )),
    ])
    return model


# ─────────────────────────────────────────────
# 3. TRAIN + EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(df_feat):
    """Train on rows with known Surface; report cross-validated R²."""
    feature_cols = get_feature_cols(df_feat)

    known = df_feat[df_feat["Surface"].notna()].copy()
    X = known[feature_cols]
    y = known["Surface"]

    print(f"\n📊 Training set: {len(known)} rows with known Surface")
    print(f"   Missing Surface rows to fill: {df_feat['Surface'].isna().sum()}")
    print(f"   Features used ({len(feature_cols)}): {feature_cols}")

    model = build_model()

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    print(f"\n✅ Cross-validated R² scores: {cv_scores.round(3)}")
    print(f"   Mean R²: {cv_scores.mean():.3f}  ±  {cv_scores.std():.3f}")

    if cv_scores.mean() < 0.65:
        print("\n⚠️  R² below 0.65 — consider adding more features or checking data quality.")
    elif cv_scores.mean() >= 0.75:
        print("\n🎯 Target R² ≥ 0.75 achieved!")

    # Final fit on all known data
    model.fit(X, y)

    # In-sample stats (for reference)
    y_pred_train = model.predict(X)
    print(f"\n   In-sample R²:  {r2_score(y, y_pred_train):.3f}")
    print(f"   MAE (m²):      {mean_absolute_error(y, y_pred_train):.1f}")

    # Feature importance — SimpleImputer drops all-NaN cols, so derive
    # surviving column names from the training data directly.
    gbr = model.named_steps["gbr"]
    surviving_cols = [c for c in feature_cols if X[c].notna().any()]
    n_used = len(gbr.feature_importances_)
    if len(surviving_cols) != n_used:          # safety fallback
        surviving_cols = feature_cols[:n_used]
    fi = pd.Series(gbr.feature_importances_, index=surviving_cols).sort_values(ascending=False)
    print("\n📌 Feature importances:")
    print(fi.round(3).to_string())

    return model, feature_cols


# ─────────────────────────────────────────────
# 4. PREDICT MISSING VALUES
# ─────────────────────────────────────────────

def fill_missing_surface(df_orig, df_feat, model, feature_cols):
    """Predict Surface for rows where it is NaN."""
    missing_mask = df_feat["Surface"].isna()
    if missing_mask.sum() == 0:
        print("\nℹ️  No missing Surface values found — dataset is already complete.")
        return df_orig

    X_missing = df_feat.loc[missing_mask, feature_cols]
    predictions = model.predict(X_missing).round(1)
    predictions = np.clip(predictions, 10, 2000)

    df_out = df_orig.copy()
    df_out.loc[missing_mask, "Surface"] = predictions
    df_out.loc[missing_mask, "Surface_predicted"] = True
    df_out["Surface_predicted"] = df_out["Surface_predicted"].fillna(False)

    print(f"\n🔮 Filled {missing_mask.sum()} missing Surface values.")
    print(f"   Predicted Surface stats:")
    print(pd.Series(predictions).describe().round(1).to_string())

    return df_out


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main(input_path, output_path):
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Missing values per column:\n{df.isnull().sum().to_string()}")

    # Feature engineering
    df_feat = engineer_features(df)

    # Train & evaluate
    model, feature_cols = train_and_evaluate(df_feat)

    # Fill missing
    df_filled = fill_missing_surface(df, df_feat, model, feature_cols)

    # Save
    df_filled.to_csv(output_path, index=False)
    print(f"\n💾 Saved filled dataset → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict missing Surface values")
    parser.add_argument("--input",  default="Masaken_dataset_completed.csv", help="Path to input CSV")
    parser.add_argument("--output", default="Masaken_filled.csv",            help="Path to output CSV")
    args = parser.parse_args()
    main(args.input, args.output)