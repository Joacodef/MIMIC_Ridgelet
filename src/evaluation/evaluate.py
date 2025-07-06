import argparse
import os
import yaml
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# MONAI imports to match the training script
from monai.data import DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized

# Absolute imports from the 'src' directory
from src.data.dataset import CXRFractureDataset
from src.models.model import FractureDetector

def load_config(run_dir):
    """Loads the configuration from a specific run directory."""
    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        # Using a loader that can handle custom tags if any, safe otherwise
        config = yaml.safe_load(f)
    return config

def evaluate(run_dir, data_split='test', checkpoint_name='best_model.pth'):
    """
    Evaluates a trained model from a checkpoint.

    Args:
        run_dir (str): The directory of the training run.
        data_split (str): The data split to evaluate on ('test', 'validation').
        checkpoint_name (str): The name of the checkpoint file.
    """
    print(f"--- Starting Evaluation ---")
    print(f"Run Directory: {run_dir}")
    print(f"Data Split: {data_split}")
    print(f"Checkpoint: {checkpoint_name}")

    # 1. Load Configuration and setup device
    config = load_config(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment variables for paths ---
    # This assumes .env file is in the root directory. Adjust if needed.
    from dotenv import load_dotenv
    load_dotenv()
    IMAGE_ROOT_DIR = os.getenv("MIMIC_CXR_P_FOLDERS_PATH")
    PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")

    # 2. Recreate the Validation Transforms from the training script
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(config['data']['image_size'], config['data']['image_size']))
    ])
    
    # 3. Load Data
    split_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", config['data']['split_folder_name'])
    csv_file_name = "test.csv" if data_split == 'test' else 'validation.csv'
    csv_path = os.path.join(split_dir, csv_file_name)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file for data split '{data_split}' not found at {csv_path}")

    dataset = CXRFractureDataset(
        csv_path=csv_path,
        image_root_dir=IMAGE_ROOT_DIR,
        transform=val_transforms
    )
    
    # Use MONAI's DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers']
    )
    print(f"Loaded {len(dataset)} images for evaluation from {csv_path}")

    # 4. Load Model
    model = FractureDetector(
        base_model_name=config['model']['base_model']
    ).to(device)

    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
    # Load the state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # 5. Run Evaluation Loop
    all_labels = []
    all_preds = []
    all_probs = []

    progress_bar = tqdm(loader, desc=f"Evaluating on {data_split}", unit="batch")
    with torch.no_grad():
        for batch_data in progress_bar:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).float().unsqueeze(1)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()

    # 6. Calculate and Report Metrics
    roc_auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    results_str = f"""
    --- Evaluation Results for {data_split} set ---
    Checkpoint: {checkpoint_name}
    
    Metrics:
    - ROC AUC Score: {roc_auc:.4f}
    - Accuracy:      {accuracy:.4f}
    - Precision:     {precision:.4f}
    - Recall:        {recall:.4f}
    - F1-Score:      {f1:.4f}

    Confusion Matrix:
    [TN, FP]  [{tn}, {fp}]
    [FN, TP]  [{fn}, {tp}]
    
    ------------------------------------
    """
    
    print(results_str)

    # 7. Save results to file
    results_filename = f"{data_split}_results.txt"
    results_path = os.path.join(run_dir, results_filename)
    with open(results_path, 'w') as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the directory of a training run (e.g., output/models/split_name/run_name).')
    parser.add_argument('--split', type=str, default='test', choices=['validation', 'test'],
                        help='The data split to evaluate on.')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='The name of the checkpoint file to use (e.g., best_model.pth, last_model.pth).')

    args = parser.parse_args()
    
    evaluate(args.run_dir, args.split, args.checkpoint)

if __name__ == '__main__':
    main()