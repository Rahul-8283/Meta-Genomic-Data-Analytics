import GEOparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import requests
import gzip
import shutil

# Define the file details
GSE_ID = "GSE55235"
FILE_NAME_GZ = f"{GSE_ID}_family.soft.gz"
FILE_NAME_SOFT = f"{GSE_ID}_family.soft"
DOWNLOAD_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/{GSE_ID}/soft/{FILE_NAME_GZ}"
DEST_DIR = "./"

# ------------------------------
# Step 1: Manual Download and Decompression (To bypass FTP/Firewall issues)
# ------------------------------
print(f"Bypassing FTP issues by manually downloading {GSE_ID} via HTTPS...")

# 1. Check if the final SOFT file already exists
if os.path.exists(os.path.join(DEST_DIR, FILE_NAME_SOFT)):
    print(f"Found local file {FILE_NAME_SOFT}. Skipping download.")
else:
    # 2. Download the compressed file
    download_path_gz = os.path.join(DEST_DIR, FILE_NAME_GZ)
    print(f"Downloading {DOWNLOAD_URL} to {download_path_gz}...")
    try:
        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        with open(download_path_gz, 'wb') as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        print("Download successful.")
    except requests.exceptions.RequestException as e:
        print(f"FATAL ERROR: Manual download failed. Check your internet connection or the URL.")
        print(f"Error details: {e}")
        exit(1)

    # 3. Decompress the file
    download_path_soft = os.path.join(DEST_DIR, FILE_NAME_SOFT)
    print(f"Decompressing {FILE_NAME_GZ} to {FILE_NAME_SOFT}...")
    try:
        with gzip.open(download_path_gz, 'rb') as f_in:
            with open(download_path_soft, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Decompression successful.")
        os.remove(download_path_gz) # Clean up the .gz file
    except Exception as e:
        print(f"FATAL ERROR: Decompression failed. Error details: {e}")
        exit(1)


# ------------------------------
# Step 2: Load dataset from local file
# ------------------------------
print(f"Loading dataset {GSE_ID} from local file...")
# Load the data using the file path instead of the GEO ID
gse = GEOparse.get_GEO(filepath=os.path.join(DEST_DIR, FILE_NAME_SOFT))

# Extract expression data
expr_data = gse.pivot_samples('VALUE')
print("Expression data shape:", expr_data.shape)

expr_data.to_csv("expression_data.csv")
print("Raw expression data saved to expression_data.csv")

# ------------------------------
# Step 3: Get sample annotations and define groups
# ------------------------------
metadata = []
for gsm_name, gsm in gse.gsms.items():
    meta = gsm.metadata
    metadata.append({
        "Sample": gsm_name,
        "Title": meta.get("title", [""])[0],
        "Source": meta.get("source_name_ch1", [""])[0],
    })

metadata_df = pd.DataFrame(metadata)

# Define groups based on the 'Source' column (robust matching)
ra_samples = metadata_df[metadata_df["Source"].str.lower().str.contains("ra|rheumatoid")]["Sample"].values
healthy_samples = metadata_df[metadata_df["Source"].str.lower().str.contains("control|healthy")]["Sample"].values

print(f"Found {len(ra_samples)} RA samples and {len(healthy_samples)} Healthy control samples.")

if len(ra_samples) == 0 or len(healthy_samples) == 0:
    print("ERROR: Could not find distinct RA and Healthy Control groups in metadata.")
    exit(1)

metadata_df.to_csv("metadata.csv", index=False)
print("\nSample Metadata (First 5 Rows):")
print(metadata_df.head())

# ------------------------------
# Step 4: Simple Differential Expression Analysis
# ------------------------------
expr_ra = expr_data[expr_data.columns.intersection(ra_samples)]
expr_healthy = expr_data[expr_data.columns.intersection(healthy_samples)]

mean_ra = expr_ra.mean(axis=1)
mean_healthy = expr_healthy.mean(axis=1)

# Log Fold Change (Log-difference, as data is likely already log2-transformed)
logFC = (mean_ra - mean_healthy) 

deg_results = pd.DataFrame({
    "Gene": expr_data.index,
    "RA_mean": mean_ra,
    "Healthy_mean": mean_healthy,
    "logFC": logFC
})

deg_results = deg_results.sort_values(by="logFC", ascending=False)
deg_results.to_csv("DEG_results.csv", index=False)

print("\nTop 5 Differentially Expressed Genes (Most Upregulated in RA):")
print(deg_results.head())

# ------------------------------
# Step 5: Visualization
# ------------------------------
top_genes_up = deg_results.head(20)
top_genes_down = deg_results.tail(20)
top_genes_combined = pd.concat([top_genes_up, top_genes_down]).set_index("Gene")

plt.figure(figsize=(10, 8))
colors = ['firebrick' if fc > 0 else 'royalblue' for fc in top_genes_combined["logFC"]]

sns.barplot(x=top_genes_combined["logFC"], y=top_genes_combined.index, palette=colors)
plt.title("Top 40 Differentially Expressed Genes (RA vs Healthy)")
plt.xlabel(r"Log Fold Change ($\log_{2}$ expression RA / $\log_{2}$ expression Healthy)") 
plt.ylabel("Gene ID")
plt.axvline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()

print("\nScript execution complete. Results saved to expression_data.csv, metadata.csv, and DEG_results.csv.")