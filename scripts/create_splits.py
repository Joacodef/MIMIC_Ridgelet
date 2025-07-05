import argparse
import os
import pandas as pd
from dotenv import load_dotenv

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

def create_splits(train_size, val_size, test_size):
    """
    Generates train, validation, and test splits for the MIMIC-CXR dataset.
    """
    print("--- Starting Split Creation ---")

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

    # *** FIX: Clean the ID columns from the custom CSV ***
    # Remove leading 's' from study_id and 'p' from subject_id, then cast to integer
    if 'study_id' in df_available.columns:
        df_available['study_id'] = df_available['study_id'].astype(str).str.lstrip('s').astype(int)
    if 'subject_id' in df_available.columns:
        df_available['subject_id'] = df_available['subject_id'].astype(str).str.lstrip('p').astype(int)
    
    # Keep only the unique, cleaned study_ids to act as a filter
    available_study_ids = df_available['study_id'].unique()

    # Merge official MIMIC-CXR files
    df_merged = df_meta.merge(df_split, on=['subject_id', 'study_id', 'dicom_id'])
    df_full = df_merged.merge(df_chexpert, on=['subject_id', 'study_id'])

    # Filter to only include studies that are actually available
    df_full = df_full[df_full['study_id'].isin(available_study_ids)]

    # Filter for frontal views only (AP or PA)
    df_frontal = df_full[df_full['ViewPosition'].isin(['AP', 'PA'])].copy()

    if df_frontal.empty:
        print("\nERROR: No frontal view images found in the filtered dataset. Please check the contents of your CSVs.")
        return
        
    # Create binary fracture label
    df_frontal['fracture'] = (df_frontal['Fracture'] == 1.0).astype(int)
    
    print(f"Total available frontal images after filtering: {len(df_frontal)}")
    print(f"Total fracture cases found: {df_frontal['fracture'].sum()}")

    # --- 3. Separate by Official Split ---
    df_train_full = df_frontal[df_frontal['split'] == 'train']
    df_val_full = df_frontal[df_frontal['split'] == 'validate']
    df_test_full = df_frontal[df_frontal['split'] == 'test']

    # --- 4. Build Balanced Training Set ---
    print(f"Building balanced training set of size {train_size}...")
    samples_per_class = train_size // 2
    
    pos_cases = df_train_full[df_train_full['fracture'] == 1]
    neg_cases = df_train_full[df_train_full['fracture'] == 0]

    if len(pos_cases) < samples_per_class:
        raise ValueError(f"Not enough fracture cases for training. Requested {samples_per_class}, found {len(pos_cases)}.")
    if len(neg_cases) < samples_per_class:
        raise ValueError(f"Not enough non-fracture cases for training. Requested {samples_per_class}, found {len(neg_cases)}.")

    train_pos = pos_cases.sample(n=samples_per_class, random_state=42)
    train_neg = neg_cases.sample(n=samples_per_class, random_state=42)
    df_train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 5. Build Validation and Test Sets ---
    print("Building validation and test sets...")
    val_size_final = val_size if val_size is not None else len(df_val_full)
    test_size_final = test_size if test_size is not None else len(df_test_full)

    df_val = df_val_full.sample(n=min(val_size_final, len(df_val_full)), random_state=42)
    df_test = df_test_full.sample(n=min(test_size_final, len(df_test_full)), random_state=42)

    # --- 6. Save Outputs ---
    output_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", f"split_{train_size}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving splits to: {output_dir}")

    final_columns = ['dicom_id', 'study_id', 'subject_id', 'ViewPosition', 'split', 'fracture']
    df_train[final_columns].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val[final_columns].to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    df_test[final_columns].to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("--- Split Creation Complete ---")
    print(f"Train set: {len(df_train)} images ({df_train['fracture'].sum()} fractures)")
    print(f"Validation set: {len(df_val)} images ({df_val['fracture'].sum()} fractures)")
    print(f"Test set: {len(df_test)} images ({df_test['fracture'].sum()} fractures)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create balanced train, validation, and test splits for MIMIC-CXR fracture detection.")
    parser.add_argument("--train_size", type=int, required=True, help="The total size of the balanced training set.")
    parser.add_argument("--val_size", type=int, default=None, help="The size of the validation set. Defaults to using all available validation images.")
    parser.add_argument("--test_size", type=int, default=None, help="The size of the test set. Defaults to using all available test images.")
    args = parser.parse_args()
    
    create_splits(args.train_size, args.val_size, args.test_size)