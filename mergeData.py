import pandas as pd
import os

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = "data"   # your data folder
OUTPUT_FILE = "merged_data.csv"

# -----------------------------
# MERGE FUNCTION
# -----------------------------
def merge_all_data(base_path):
    all_data = []

    # Loop through year folders
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Skip if not a folder
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder}")

        # Loop through files inside folder
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)

                try:
                    df = pd.read_csv(file_path)

                    # Add year column (from folder name)
                    df['year'] = folder

                    # Optional: store file name
                    df['source_file'] = file

                    all_data.append(df)

                except Exception as e:
                    print(f"Error reading {file}: {e}")

    # Merge all data
    if len(all_data) == 0:
        print("No data found!")
        return None

    merged_df = pd.concat(all_data, ignore_index=True)

    return merged_df


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    merged_df = merge_all_data(BASE_PATH)

    if merged_df is not None:
        print("Merged shape:", merged_df.shape)

        # Save file
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Merged data saved as {OUTPUT_FILE}")