import pandas as pd
import glob
import os


path = "data/"   

# Loop through all RD-*.txt files
for file in glob.glob(os.path.join(path, "RD-*.txt")):
    try:
        # Read tab-separated TXT file
        df = pd.read_csv(file, sep='\t', engine='python')
        
        # Save to CSV with same base name
        csv_name = os.path.splitext(file)[0] + '.csv'
        df.to_csv(csv_name, index=False)
        
        print(f"✅ Converted {os.path.basename(file)} → {os.path.basename(csv_name)}")
    
    except Exception as e:
        print(f"⚠️ Error in {file}: {e}")

print("\nAll conversions completed!")
