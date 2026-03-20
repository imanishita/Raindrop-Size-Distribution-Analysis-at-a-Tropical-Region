import pandas as pd
import os

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = "data"   # main folder

# -----------------------------
# FUNCTION TO CONVERT
# -----------------------------
def convert_txt_to_csv(base_path):

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder}")

        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                txt_path = os.path.join(folder_path, file)

                try:
                    # Try reading (adjust delimiter if needed)
                    df = pd.read_csv(txt_path, sep=None, engine='python')

                    # Create csv filename
                    csv_file = file.replace(".txt", ".csv")
                    csv_path = os.path.join(folder_path, csv_file)

                    # Save as CSV
                    df.to_csv(csv_path, index=False)

                    print(f"Converted: {file} → {csv_file}")

                except Exception as e:
                    print(f"Error converting {file}: {e}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    convert_txt_to_csv(BASE_PATH)