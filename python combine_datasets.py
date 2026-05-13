"""
Combine 3 CSV files into one
"""
import pandas as pd

FILES = [
    r"C:\Users\HP\Downloads\MasakenPFA\Masaken_dataset.csv",
    r"C:\Users\HP\Downloads\MasakenPFA\clean_avito_filled.csv",
    r"C:\Users\HP\Downloads\PFA_web_scraping\clean_avito_selected_pages2_filled.csv",
]

OUTPUT = r"C:\Users\HP\Downloads\MasakenPFA\combined_dataset.csv"

dfs = []
for path in FILES:
    df = pd.read_csv(path)
    # Normalise column names to lowercase for consistent merging
    df.columns = [c.strip().lower() for c in df.columns]
    print(f" Loaded: {path.split(chr(92))[-1]}  →  {df.shape[0]} rows, {df.shape[1]} cols")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

print(f"\nCombined shape: {combined.shape}")
print(f"   Columns: {list(combined.columns)}")
print(f"\n   Missing values:\n{combined.isnull().sum().to_string()}")

combined.to_csv(OUTPUT, index=False)
print(f"\n Saved → {OUTPUT}")