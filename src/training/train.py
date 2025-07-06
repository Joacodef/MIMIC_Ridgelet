import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
import shutil
from datetime import datetime
from dataclasses import asdict

import wandb

from monai.data import DataLoader
from monai.losses import FocalLoss
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized, RandFlipd, RandAffined

from ..models.model import FractureDetector
from ..data.dataset import CXRFractureDataset
from ..config.config import load_config

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Performs one full training epoch."""
    model.train()
    running_loss = 0.0
    
    # Using tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for batch_data in progress_bar:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Performs validation on the validation set."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for batch_data in progress_bar:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Store predictions and labels for AUC calculation
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": loss.item()})

    val_loss = running_loss / len(val_loader)
    val_auc = roc_auc_score(all_labels, all_preds)
    
    return val_loss, val_auc

def main(config_path, resume_dir=None, resume_from="last"):
    """Main function to run the training pipeline."""
    # --- 1. Determine Configuration Path ---
    if resume_dir:
        run_config_path = os.path.join(resume_dir, 'config.yaml')
        if not os.path.isfile(run_config_path):
            raise FileNotFoundError(
                f"Configuration file not found in resume directory: {run_config_path}"
            )
        config_path = run_config_path
        print(f"Resuming run. Using configuration from: {config_path}")
    elif not config_path:
        raise ValueError("A configuration file must be provided with --config for a new run.")

    # --- 1.2. Load Configurations ---
    config = load_config(config_path)

    load_dotenv()
    IMAGE_ROOT_DIR = os.getenv("MIMIC_CXR_P_FOLDERS_PATH")
    PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")
    PROJECT_OUTPUT_FOLDER_PATH = os.getenv("PROJECT_OUTPUT_FOLDER_PATH")
    
    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 3. Data Preparation ---
    print("Setting up data pipelines...")
    augs = config.data.augmentations  # A shorthand for convenience

    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(config.data.image_size, config.data.image_size)),
        RandFlipd(keys=["image"], prob=augs.rand_flip_prob, spatial_axis=0),
        RandAffined(
            keys=["image"], 
            prob=augs.rand_affine_prob, 
            rotate_range=(augs.rotate_range), 
            scale_range=(augs.scale_range)
        )
    ])

    # The val_transforms block remains the same as it has no random augmentations
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(config.data.image_size, config.data.image_size))
    ])

    split_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", config.data.split_folder_name)
    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "validation.csv")

    train_dataset = CXRFractureDataset(csv_path=train_csv, image_root_dir=IMAGE_ROOT_DIR, transform=train_transforms)
    val_dataset = CXRFractureDataset(csv_path=val_csv, image_root_dir=IMAGE_ROOT_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size, shuffle=True, num_workers=config.dataloader.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers)

    # --- 4. Model, Loss, Optimizer ---
    print("Initializing model, criterion, and optimizer...")
    model = FractureDetector(base_model_name=config.model.base_model).to(device)
    
    # Initialize loss function based on the configuration
    loss_config = config.training.loss
    if loss_config.name.lower() == 'focalloss':
        print(f"Using Focal Loss with gamma={loss_config.gamma}")
        criterion = FocalLoss(gamma=loss_config.gamma)
    elif loss_config.name.lower() == 'bcewithlogitsloss':
        print("Using BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss function '{loss_config.name}' is not supported.")

    # Select optimizer from config
    optimizer_name = config.training.optimizer.lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

    print(f"Using optimizer: {optimizer_name.capitalize()}")
    
    # --- 5. Training Setup & Checkpoint Loading ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0
    patience = config.training.early_stopping_patience
    output_model_name = config.training.output_model_name

    # Handle output directory and resume logic
    if resume_dir:

        # Set the output directory to the resume directory
        output_run_dir = resume_dir 

        if resume_from == "best":
            checkpoint_filename = config.training.output_model_name # e.g., 'best_model.pth'
        else: # Default to 'last'
            checkpoint_filename = "last_model.pth"
        
        checkpoint_path = os.path.join(output_run_dir, checkpoint_filename)
        print(f"Attempting to resume training from '{resume_from}' checkpoint: {checkpoint_path}")

        if os.path.isfile(checkpoint_path):
            print(f"Checkpoint found. Loading state from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            epochs_no_improve = 0  # Reset patience on resume
            
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
            print(f"Previous best validation loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: No checkpoint file found at '{checkpoint_path}'.")
            print("Starting a new training run in the specified directory.")
            os.makedirs(output_run_dir, exist_ok=True)
            shutil.copy(config_path, os.path.join(output_run_dir, 'config.yaml'))

    else:
        # Create a unique directory for a new training run
        model_name = config.model.base_model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        
        output_run_dir = os.path.join(PROJECT_OUTPUT_FOLDER_PATH, "models", config.data.split_folder_name, run_name)
        os.makedirs(output_run_dir, exist_ok=True)
        print(f"Starting new training run. Saving artifacts to: {output_run_dir}")
        shutil.copy(config_path, os.path.join(output_run_dir, 'config.yaml'))

    if config.wandb.enabled:
        run_name = os.path.basename(output_run_dir)
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            config=asdict(config)
        )
        print("Weights & Biases logging enabled.")

    best_model_path = os.path.join(output_run_dir, output_model_name)

    print("--- Starting Training ---")
    for epoch in range(start_epoch, config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if config.wandb.enabled:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_auc": val_auc,
            })
        
        # --- 6. Model Checkpoint Creation ---
        # Create checkpoint dictionary on every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }

        # --- 7. Model Saving & Early Stopping ---
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Update the best_val_loss in the checkpoint before saving
            checkpoint['best_val_loss'] = best_val_loss 
            torch.save(checkpoint, best_model_path)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
        
        # Save the last checkpoint on every epoch
        last_model_path = os.path.join(output_run_dir, "last_model.pth")
        torch.save(checkpoint, last_model_path)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break
            
    print("\n--- Training Complete ---")
    print(f"Finished training. Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    if config.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fracture detection model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to the configuration YAML file. Required for new runs."
    )
    parser.add_argument(
        "--resume_dir", 
        type=str, 
        default=None, 
        help="Path to a previous run's directory to resume training from."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="last",
        choices=['best', 'last'],
        help="Checkpoint to resume from: 'best' or 'last' (default: 'last')."
    )
    args = parser.parse_args()
    main(args.config, args.resume_dir, args.resume_from)