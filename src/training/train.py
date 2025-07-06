import os
import argparse
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
import shutil
from datetime import datetime
from dataclasses import asdict
from typing import Optional

import wandb

from monai.data import DataLoader
from monai.losses import FocalLoss
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized, RandFlipd, RandAffined

from ..config.config import load_config, AppConfig
from ..data.dataset import CXRFractureDataset
from ..models.model import FractureDetector


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Performs one full training epoch."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for batch_data in progress_bar:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device).float().unsqueeze(1)

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
            labels = batch_data["label"].to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": loss.item()})

    val_loss = running_loss / len(val_loader)
    val_auc = roc_auc_score(all_labels, all_preds)
    
    return val_loss, val_auc

def run_training(
    config: AppConfig, 
    config_path: str, 
    resume_dir: Optional[str] = None, 
    resume_from: str = "last",
    train_csv_override: Optional[str] = None,
    output_dir_override: Optional[str] = None
) -> tuple[float, str]:
    """
    Main function to run the training pipeline.
    This function is designed to be called by both the CLI and other scripts.
    """
    # --- 1. Load Environment Variables and Setup ---
    load_dotenv()
    IMAGE_ROOT_DIR = os.getenv("MIMIC_CXR_P_FOLDERS_PATH")
    PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")
    PROJECT_OUTPUT_FOLDER_PATH = os.getenv("PROJECT_OUTPUT_FOLDER_PATH")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Data Preparation ---
    print("Setting up data pipelines...")
    augs = config.data.augmentations
    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(config.data.image_size, config.data.image_size)),
        RandFlipd(keys=["image"], prob=augs.rand_flip_prob, spatial_axis=0),
        RandAffined(
            keys=["image"], prob=augs.rand_affine_prob, 
            rotate_range=(augs.rotate_range), scale_range=(augs.scale_range)
        )
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(config.data.image_size, config.data.image_size))
    ])

    split_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", config.data.split_folder_name)
    train_csv = train_csv_override if train_csv_override else os.path.join(split_dir, "train.csv")
    print(f"Using training data from: {train_csv}")
    val_csv = os.path.join(split_dir, "validation.csv")

    train_dataset = CXRFractureDataset(csv_path=train_csv, image_root_dir=IMAGE_ROOT_DIR, transform=train_transforms)
    val_dataset = CXRFractureDataset(csv_path=val_csv, image_root_dir=IMAGE_ROOT_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size, shuffle=True, num_workers=config.dataloader.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers)

    # --- 3. Model, Loss, Optimizer ---
    print("Initializing model, criterion, and optimizer...")
    model = FractureDetector(base_model_name=config.model.base_model).to(device)
    
    loss_config = config.training.loss
    if loss_config.name.lower() == 'focalloss':
        print(f"Using Focal Loss with gamma={loss_config.gamma} and alpha={loss_config.alpha}")
        criterion = FocalLoss(gamma=loss_config.gamma, alpha=loss_config.alpha)
    elif loss_config.name.lower() == 'bcewithlogitsloss':
        print("Using BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss function '{loss_config.name}' is not supported.")

    optimizer_name = config.training.optimizer.lower()
    lr = config.training.learning_rate
    weight_decay = config.training.weight_decay
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")
    print(f"Using optimizer: {optimizer_name.capitalize()} with learning rate {lr} and weight decay {weight_decay}")
    
    scheduler_config = config.training.scheduler
    if scheduler_config.name.lower() == 'reducelronplateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_config.factor, patience=scheduler_config.patience
        )
        print(f"Using ReduceLROnPlateau scheduler with factor={scheduler_config.factor} and patience={scheduler_config.patience}")
    else:
        scheduler = None
        print("No learning rate scheduler will be used.")

    # --- 4. Training Setup & Checkpoint Loading ---
    best_val_loss = float('inf')
    best_val_auc = 0.0  
    epochs_no_improve = 0
    start_epoch = 0
    patience = config.training.early_stopping_patience
    output_model_name = config.training.output_model_name

    if output_dir_override:
        output_run_dir = output_dir_override
        os.makedirs(output_run_dir, exist_ok=True)
        # We don't resume when an override is given, it's a fresh run in a specific dir
        print(f"Using provided output directory: {output_run_dir}")
        shutil.copy(config_path, os.path.join(output_run_dir, 'config.yaml'))
    elif resume_dir:
        output_run_dir = resume_dir
        checkpoint_filename = "last_model.pth" if resume_from == "last" else config.training.output_model_name
        checkpoint_path = os.path.join(output_run_dir, checkpoint_filename)
        print(f"Attempting to resume training from '{resume_from}' checkpoint: {checkpoint_path}")

        if os.path.isfile(checkpoint_path):
            print(f"Checkpoint found. Loading state...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_auc = checkpoint.get('best_val_auc', 0.0) 
            epochs_no_improve = 0
            print(f"Resuming from epoch {start_epoch}. Best validation loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint not found at '{checkpoint_path}'. Starting new run in directory.")
            os.makedirs(output_run_dir, exist_ok=True)
            shutil.copy(config_path, os.path.join(output_run_dir, 'config.yaml'))
    else:
        model_name = config.model.base_model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        output_run_dir = os.path.join(PROJECT_OUTPUT_FOLDER_PATH, "models", config.data.split_folder_name, run_name)
        os.makedirs(output_run_dir, exist_ok=True)
        print(f"Starting new run. Saving artifacts to: {output_run_dir}")
        shutil.copy(config_path, os.path.join(output_run_dir, 'config.yaml'))

    if config.wandb.enabled:
        run_name = os.path.basename(output_run_dir)
        wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=run_name, config=asdict(config))
        print("Weights & Biases logging enabled.")

    best_model_path = os.path.join(output_run_dir, output_model_name)

    print("--- Starting Training ---")
    for epoch in range(start_epoch, config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}")

        best_val_auc = max(val_auc, best_val_auc)

        if config.wandb.enabled:
            wandb.log({
                "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                "validation_auc": val_auc, "learning_rate": current_lr
            })
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc
        }

        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, best_model_path)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
        
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

    # Return a tuple of the best metric and the output directory
    return (best_val_auc, output_run_dir)

#  Main function dedicated to parsing CLI arguments
def main():
    """Command-line interface for the main training function."""
    parser = argparse.ArgumentParser(description="Train a fracture detection model.")
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration YAML file.")
    parser.add_argument("--resume_dir", type=str, default=None, help="Path to a run's directory to resume training.")
    parser.add_argument("--resume_from", type=str, default="last", choices=['best', 'last'], help="Checkpoint to resume from.")
    args = parser.parse_args()

    config_target_path = args.config
    if args.resume_dir:
        run_config_path = os.path.join(args.resume_dir, 'config.yaml')
        if not os.path.isfile(run_config_path):
            raise FileNotFoundError(f"Config file not found in resume directory: {run_config_path}")
        config_target_path = run_config_path
        print(f"Resuming run. Using configuration from: {config_target_path}")
    elif not config_target_path:
        raise ValueError("A config file must be provided with --config for a new run.")
    
    config = load_config(config_target_path)
    run_training(config, config_target_path, args.resume_dir, args.resume_from)


if __name__ == "__main__":
    main()