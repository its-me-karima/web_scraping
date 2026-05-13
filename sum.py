import pandas as pd
import glob

# Get all CSV files
files = glob.glob("*.csv")

# Remove the file you don't want
files = [f for f in files if f != "clean_avito_lastPages.csv"]

# Read all remaining files
df_list = [pd.read_csv(f) for f in files]

# Merge everything
combined_df = pd.concat(df_list, ignore_index=True)

# Save final dataset
combined_df.to_csv("Masaken_dataset.csv", index=False)

print("Merged successfully!")
print(combined_df.shape)