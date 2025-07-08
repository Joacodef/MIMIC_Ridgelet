import argparse
import os
import pandas as pd
from dotenv import load_dotenv

# --- Custom Imports ---
# Make sure the script can find the 'src' directory
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.config import load_config, AppConfig


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get paths from environment variables
MIMIC_CXR_METADATA_PATH = os.getenv("MIMIC_CXR_METADATA_PATH")
PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")

# Define file names
METADATA_FILE = "mimic-cxr-2.0.0-metadata.csv.gz"
CHEXPERT_FILE = "mimic-cxr-2.0.0-chexpert.csv.gz"
SPLIT_FILE = "mimic-cxr-2.0.0-split.csv.gz"
AVAILABLE_STUDIES_FILE = "raw/available_studies_complete.csv"


def create_splits(config: AppConfig):
    """
    Generates train, validation, and test splits for the MIMIC-CXR dataset
    based on the provided configuration.
    """
    print("--- Starting Split Creation ---")
    pathology = config.pathology
    print(f"Target pathology: {pathology}")

    # --- 1. Load Data ---
    print("Loading data sources...")
    try:
        df_available = pd.read_csv(os.path.join(PROJECT_DATA_FOLDER_PATH, AVAILABLE_STUDIES_FILE))
        df_meta = pd.read_csv(os.path.join(MIMIC_CXR_METADATA_PATH, METADATA_FILE))
        df_chexpert = pd.read_csv(os.path.join(MIMIC_CXR_METADATA_PATH, CHEXPERT_FILE))
        df_split = pd.read_csv(os.path.join(MIMIC_CXR_METADATA_PATH, SPLIT_FILE))
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        print("Please ensure your .env variables and file paths are correct.")
        return

    # --- 2. Clean and Merge Data ---
    print("Cleaning and merging data...")

    # Clean the ID columns from the custom CSV
    if 'study_id' in df_available.columns:
        df_available['study_id'] = df_available['study_id'].astype(str).str.lstrip('s').astype(int)
    if 'subject_id' in df_available.columns:
        df_available['subject_id'] = df_available['subject_id'].astype(str).str.lstrip('p').astype(int)
    
    available_study_ids = df_available['study_id'].unique()

    # Merge official MIMIC-CXR files
    df_merged = df_meta.merge(df_split, on=['subject_id', 'study_id', 'dicom_id'])
    df_full = df_merged.merge(df_chexpert, on=['subject_id', 'study_id'])

    # Filter to only include studies that are actually available
    df_full = df_full[df_full['study_id'].isin(available_study_ids)]

    # Filter for frontal views only (AP or PA)
    df_frontal = df_full[df_full['ViewPosition'].isin(['AP', 'PA'])].copy()

    if df_frontal.empty:
        print("\nERROR: No frontal view images found in the filtered dataset.")
        return
        
    # Create binary label based on the configured pathology
    if pathology not in df_frontal.columns:
        raise ValueError(f"The pathology '{pathology}' is not a valid column in the CheXpert labels file.")
    
    # Create a generic 'label' column. 1.0 indicates presence of the pathology.
    df_frontal['label'] = (df_frontal[pathology] == 1.0).astype(int)
    
    print(f"Total available frontal images after filtering: {len(df_frontal)}")
    print(f"Total positive cases for '{pathology}': {df_frontal['label'].sum()}")

    # --- 3. Separate by Official Split ---
    df_train_full = df_frontal[df_frontal['split'] == 'train']
    df_val_full = df_frontal[df_frontal['split'] == 'validate']
    df_test_full = df_frontal[df_frontal['split'] == 'test']

    # --- 4. Build Balanced Training Set ---
    train_size = config.data.train_size
    print(f"Building balanced training set of size {train_size}...")

    samples_per_class = train_size // 2
    
    pos_cases = df_train_full[df_train_full['label'] == 1]
    neg_cases = df_train_full[df_train_full['label'] == 0]

    if len(pos_cases) < samples_per_class:
        raise ValueError(f"Not enough positive cases for training. Requested {samples_per_class}, found {len(pos_cases)}.")
    if len(neg_cases) < samples_per_class:
        raise ValueError(f"Not enough negative cases for training. Requested {samples_per_class}, found {len(neg_cases)}.")

    train_pos = pos_cases.sample(n=samples_per_class, random_state=42)
    train_neg = neg_cases.sample(n=samples_per_class, random_state=42)
    df_train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 5. Build Validation and Test Sets ---
    print("Building validation and test sets...")
    # These sets are not balanced and use all available data from their respective splits
    df_val = df_val_full
    df_test = df_test_full

    # --- 6. Save Outputs ---
    # Output directory is now named using the pathology and split size
    output_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", f"split_{pathology}_{train_size}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving splits to: {output_dir}")

    # Save 'label' column
    final_columns = ['dicom_id', 'study_id', 'subject_id', 'ViewPosition', 'split', 'label']
    df_train[final_columns].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val[final_columns].to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    df_test[final_columns].to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("--- Split Creation Complete ---")
    print(f"Train set: {len(df_train)} images ({df_train['label'].sum()} positive cases)")
    print(f"Validation set: {len(df_val)} images ({df_val['label'].sum()} positive cases)")
    print(f"Test set: {len(df_test)} images ({df_test['label'].sum()} positive cases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train, validation, and test splits for MIMIC-CXR based on a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    # Load configuration from the specified file
    app_config = load_config(args.config)
    
    # Run the split creation process
    create_splits(app_config)