import os
import argparse
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
import yaml
from datetime import datetime
from dataclasses import asdict
from typing import Optional
import sys

import wandb

from monai.data import DataLoader, CacheDataset
from monai.losses import FocalLoss
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    CopyItemsd,     
    DeleteItemsd, 
    Resized, 
    RandAffine
)

# Add project root to path to allow absolute imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import load_config, AppConfig
from data.dataset import CXRFractureDataset
from models.model import FractureDetector
from data.transforms import RidgeletTransformd, HaarTransformd, ConcatenateChannelsd


def train_one_epoch(model, train_loader, optimizer, criterion, device, augment_fn=None):
    """Performs one full training epoch."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for batch_data in progress_bar:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device).float().unsqueeze(1)

        # Apply GPU augmentations if provided
        if augment_fn:
            # Apply the transform to each image in the batch individually and stack them back together
            images = torch.stack([augment_fn(image) for image in images])

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
    resume_dir: Optional[str] = None,
    resume_from: str = "last",
    train_csv_override: Optional[str] = None,
    output_dir_override: Optional[str] = None
) -> tuple[float, str]:
    """
    Main function to run the training pipeline.
    This function is self-contained and responsible for its own artifacts.
    """
    # --- 1. Load Environment Variables and Setup ---
    load_dotenv()
    IMAGE_ROOT_DIR = os.getenv("MIMIC_CXR_P_FOLDERS_PATH")
    PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")
    PROJECT_OUTPUT_FOLDER_PATH = os.getenv("PROJECT_OUTPUT_FOLDER_PATH")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Setup Directories and Save Config ---
    if output_dir_override:
        output_run_dir = output_dir_override
    elif resume_dir:
        output_run_dir = resume_dir
    else:
        # Use run_name from config if provided, otherwise generate one
        run_name = getattr(config, 'run_name', None) or f"{config.model.base_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_run_dir = os.path.join(PROJECT_OUTPUT_FOLDER_PATH, "models", config.data.split_folder_name, run_name)

    os.makedirs(output_run_dir, exist_ok=True)

    config_save_path = os.path.join(output_run_dir, "config.yaml")
    try:
        config_dict = asdict(config)
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"FATAL: Could not save configuration to {config_save_path}. Error: {e}")
        raise

    # --- 3. Data Preparation ---
    print("Setting up data pipelines with in-memory caching...")
    augs = config.data.augmentations

    # Define transforms that are ALWAYS applied before caching (heavy, deterministic)
    pre_cache_transforms_list = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(config.data.image_size, config.data.image_size)),
    ]

    # Define the key for the transformed image output
    transform_output_key = "image_transformed"

    # Add the special transform (Haar/Ridgelet) to the pre-cache list
    if config.data.transform_name == 'ridgelet':
        ridgelet_params = asdict(config.data.transform_params.ridgelet)
        pre_cache_transforms_list.append(RidgeletTransformd(keys=["image"], output_key=transform_output_key, threshold_ratio=config.data.transform_threshold_ratio, **ridgelet_params))
    elif config.data.transform_name == 'haar':
        haar_params = asdict(config.data.transform_params.haar)
        pre_cache_transforms_list.append(HaarTransformd(keys=["image"], output_key=transform_output_key, threshold_ratio=config.data.transform_threshold_ratio, **haar_params))

    # Add multichannel logic to the pre-cache list and determine input channels
    if config.data.transform_name is not None:
        if config.data.multichannel:
            pre_cache_transforms_list.append(ConcatenateChannelsd(keys=["image", transform_output_key], output_key="image"))
            input_channels = 2
        else:
            pre_cache_transforms_list.extend([
                DeleteItemsd(keys=["image"]),
                CopyItemsd(keys=[transform_output_key], names=["image"]),
            ])
            input_channels = 1
        pre_cache_transforms_list.append(DeleteItemsd(keys=[transform_output_key]))
    else:
        input_channels = 1

    pre_cache_transforms = Compose(pre_cache_transforms_list)

    # --- Create Base Datasets ---
    # The base dataset is now responsible for loading and applying the heavy transforms once
    split_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", config.data.split_folder_name)
    train_csv = train_csv_override if train_csv_override else os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "validation.csv")
    print(f"Using training data from: {train_csv}")

    base_train_dataset = CXRFractureDataset(csv_path=train_csv, image_root_dir=IMAGE_ROOT_DIR, transform=pre_cache_transforms)
    base_val_dataset = CXRFractureDataset(csv_path=val_csv, image_root_dir=IMAGE_ROOT_DIR, transform=pre_cache_transforms)

    # --- Wrap with CacheDataset for In-Memory Caching ---
    # TUNE THIS VALUE based on your available system RAM.
    # Start with a lower value like 0.25 and increase if you have enough memory.
    RAM_CACHE_RATE = 1.0
    print(f"Initializing in-memory CacheDataset with cache_rate={RAM_CACHE_RATE}. This may take a while on the first run...")

    train_dataset = CacheDataset(
        data=base_train_dataset,
        cache_rate=RAM_CACHE_RATE,
        num_workers=config.dataloader.num_workers
    )
    val_dataset = CacheDataset(
        data=base_val_dataset,
        cache_rate=RAM_CACHE_RATE,
        num_workers=config.dataloader.num_workers
    )

    # --- Create DataLoaders ---
    dl_config = config.dataloader
    train_loader = DataLoader(train_dataset, batch_size=dl_config.batch_size, shuffle=True, num_workers=dl_config.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=dl_config.batch_size, shuffle=False, num_workers=dl_config.num_workers, pin_memory=True, persistent_workers=True)

    # --- 4. Model, Loss, Optimizer ---
    print("Initializing model, criterion, and optimizer...")

    print("Initializing GPU-based augmentations...")
    gpu_augmentations = RandAffine(
        prob=config.data.augmentations.rand_affine_prob,
        rotate_range=config.data.augmentations.rotate_range,
        scale_range=config.data.augmentations.scale_range,
        device=device  # This assigns the transform to the GPU
    )
    
    model = FractureDetector(base_model_name=config.model.base_model, in_channels=input_channels).to(device)
    
    loss_config = config.training.loss
    if loss_config.name.lower() == 'focalloss':
        criterion = FocalLoss(gamma=loss_config.gamma, alpha=loss_config.alpha)
    elif loss_config.name.lower() == 'bcewithlogitsloss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss function '{loss_config.name}' is not supported.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    
    scheduler_config = config.training.scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_config.factor, patience=scheduler_config.patience
    ) if scheduler_config.name.lower() == 'reducelronplateau' else None

    # --- 5. Checkpoint Loading & Training State ---
    start_epoch = 0
    best_val_auc = 0.0
    epochs_no_improve = 0
    
    if resume_dir:
        checkpoint_filename = "last_model.pth" if resume_from == "last" else config.training.output_model_name
        checkpoint_path = os.path.join(output_run_dir, checkpoint_filename)
        print(f"Attempting to resume training from '{resume_from}' checkpoint: {checkpoint_path}")
        if os.path.isfile(checkpoint_path):
            print("Checkpoint found. Loading state...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint.get('best_val_auc', 0.0)
            print(f"Resuming from epoch {start_epoch}. Best validation AUC: {best_val_auc:.4f}")
        else:
            print(f"Warning: Checkpoint not found at '{checkpoint_path}'. Starting a new run in the directory.")

    # --- 6. W&B Integration ---
    if config.wandb.enabled:
        # Use the final run name for W&B
        run_name = os.path.basename(output_run_dir)
        wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=run_name, config=asdict(config))
        print("Weights & Biases logging enabled.")

    # --- 7. Training Loop ---
    best_model_path = ""
    print("--- Starting Training ---")
    try:
        for epoch in range(start_epoch, config.training.epochs):
            print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
            
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, augment_fn=gpu_augmentations)
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            if scheduler:
                scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}")

            if val_auc > best_val_auc:
                print(f"Validation AUC improved from {best_val_auc:.4f} to {val_auc:.4f}. Saving best model...")
                best_val_auc = val_auc
                epochs_no_improve = 0
                best_model_path = os.path.join(output_run_dir, config.training.output_model_name)
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_auc': best_val_auc
                }, best_model_path)
            else:
                epochs_no_improve += 1
                print(f"Validation AUC did not improve for {epochs_no_improve} epoch(s).")
            
            # Always save the last model checkpoint
            last_model_path = os.path.join(output_run_dir, "last_model.pth")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': val_auc
            }, last_model_path)
            
            if config.wandb.enabled:
                wandb.log({
                    "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                    "validation_auc": val_auc, "learning_rate": current_lr
                })
                
            if epochs_no_improve >= config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {config.training.early_stopping_patience} epochs without improvement.")
                break

    except KeyboardInterrupt:
        print("\n\n--- Training interrupted by user. Exiting gracefully. ---")
        # The 'last_model.pth' from the last completed epoch is already saved.
    
    print("\n--- Training Complete ---")
    print(f"Finished training. Best validation AUC achieved: {best_val_auc:.4f}")

    if config.wandb.enabled:
        wandb.finish()

    return best_val_auc, output_run_dir

def main():
    """Command-line interface for the main training function."""
    parser = argparse.ArgumentParser(description="Train a fracture detection model.")
    parser.add_argument("--config", type=str, help="Path to the configuration YAML file for a new run.")
    parser.add_argument("--resume_dir", type=str, help="Path to a run's directory to resume training.")
    parser.add_argument("--resume_from", type=str, default="last", choices=['best', 'last'], help="Checkpoint to resume from.")
    args = parser.parse_args()

    if args.resume_dir:
        config_path = os.path.join(args.resume_dir, 'config.yaml')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found in resume directory: {config_path}")
        print(f"Resuming run. Using configuration from: {config_path}")
        config = load_config(config_path)
        run_training(config, resume_dir=args.resume_dir, resume_from=args.resume_from)
    elif args.config:
        config = load_config(args.config)
        run_training(config)
    else:
        raise ValueError("Either --config for a new run or --resume_dir to resume a run must be provided.")

if __name__ == "__main__":
    main()