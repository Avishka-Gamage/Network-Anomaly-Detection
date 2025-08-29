import pandas as pd
import os

# Folder with your dataset CSVs
data_dir = r"data\MachineLearningCVE"

# Get all .csv files
csv_files = [file for file in os.listdir(data_dir) if file.endswith(".csv")]

# Read and combine
df_list = []
for file in csv_files:
    path = os.path.join(data_dir, file)
    print(f"Reading {file}")
    df = pd.read_csv(path, low_memory=False)
    df_list.append(df)

# Merge all DataFrames
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged dataset
output_path = os.path.join(data_dir, "CICIDS2017_Merged.csv")
merged_df.to_csv(output_path, index=False)
print(f"[✔] Merged dataset saved to: {output_path}")
