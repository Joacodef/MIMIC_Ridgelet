import argparse
import copy
import os
import sys
from typing import Any, Dict

import pandas as pd
import optuna
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Add project root to path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config.config import load_config, AppConfig
from src.training.train import run_training

def create_stratified_subset(data_path: str, output_dir: str, percentage: float) -> str:
    """
    Creates a stratified subset of a CSV file and saves it.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Full training data not found at {data_path}")
    
    full_df = pd.read_csv(data_path)
    
    min_samples = 2 if len(full_df['fracture'].unique()) > 1 else 1
    train_size = int(len(full_df) * percentage)
    if train_size < min_samples:
        train_size = min_samples

    if train_size >= len(full_df):
        subset_df = full_df
    else:
        subset_df, _ = train_test_split(
            full_df,
            train_size=train_size,
            stratify=full_df['fracture'],
            random_state=42
        )
    
    subset_filename = f"train_subset_{int(percentage*100)}.csv"
    subset_path = os.path.join(output_dir, subset_filename)
    subset_df.to_csv(subset_path, index=False)
    
    print(f"Created stratified subset with {len(subset_df)} samples at {subset_path}")
    return subset_path

def get_suggested_params(trial: optuna.Trial, config: AppConfig) -> Dict[str, Any]:
    """
    Suggests hyperparameters for a given Optuna trial.
    """
    params = {}
    
    # Existing Parameters 
    params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    params['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Optimizer Choice
    params['optimizer'] = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    # Batch Size
    params['batch_size'] = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Scheduler Patience for ReduceLROnPlateau
    params['scheduler_patience'] = trial.suggest_int("scheduler_patience", 3, 10)

    # Augmentation Probability for RandAffined
    params['rand_affine_prob'] = trial.suggest_float("rand_affine_prob", 0.3, 0.7)

    # -- Conditional Parameters for FocalLoss --
    if config.training.loss.name.lower() == 'focalloss':
        params['loss_gamma'] = trial.suggest_float("loss_gamma", 1.0, 3.0)
        params['loss_alpha'] = trial.suggest_float("loss_alpha", 0.25, 0.75)
        
    return params

def main():
    """Main function to start the staged hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Run staged HPO for fracture detection.")
    parser.add_argument('--config', type=str, required=True, help='Path to the base configuration YAML file.')
    parser.add_argument('--percentages', type=float, nargs='+', default=[0.05, 0.20, 1.00], help='Data percentages for each stage.')
    parser.add_argument('--trials_per_stage', type=int, nargs='+', default=[25, 15, 10], help='Trial counts for each stage.')
    parser.add_argument('--study-name', type=str, default="fracture-detection-staged-hpo", help='Unique name for the Optuna study and output folder.')
    parser.add_argument('--study-db', type=str, default="hpo_study.db", help='Name of the SQLite database file.')
    args = parser.parse_args()

    if len(args.percentages) != len(args.trials_per_stage):
        raise ValueError("The number of items in --percentages and --trials_per_stage must be the same.")
    
    STAGES = [{'percentage': p, 'n_trials': t} for p, t in zip(args.percentages, args.trials_per_stage)]

    load_dotenv()
    PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")
    PROJECT_OUTPUT_FOLDER_PATH = os.getenv("PROJECT_OUTPUT_FOLDER_PATH")

    base_config = load_config(args.config)
    split_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", base_config.data.split_folder_name)
    full_train_csv_path = os.path.join(split_dir, "train.csv")
    subsets_dir = os.path.join(split_dir, "hpo_subsets")
    os.makedirs(subsets_dir, exist_ok=True)
    
    study_name = args.study_name
    db_path = os.path.join(PROJECT_OUTPUT_FOLDER_PATH, "optuna_db", args.study_db)
    storage_url = f"sqlite:///{db_path}"
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    print(f"Using Optuna study '{study_name}' with database at: {storage_url}")

    study = optuna.create_study(
        study_name=study_name, storage=storage_url, direction="maximize", load_if_exists=True
    )

    for i, stage_info in enumerate(STAGES):
        stage_num = i + 1
        n_trials_for_stage = stage_info['n_trials']
        percentage = stage_info['percentage']
        
        print("\n" + "="*60)
        print(f"--- HPO STAGE {stage_num}/{len(STAGES)}: {n_trials_for_stage} trials on {percentage*100:.0f}% data ---")
        print("="*60)

        subset_csv_path = create_stratified_subset(full_train_csv_path, subsets_dir, percentage)

        def objective(trial: optuna.Trial) -> float:
            """Defines the objective for a single Optuna trial."""
            trial_config = copy.deepcopy(base_config)
            
            suggested_params = get_suggested_params(trial, trial_config)
           
            trial_config.training.learning_rate = suggested_params['learning_rate']
            trial_config.training.weight_decay = suggested_params['weight_decay']
            trial_config.training.optimizer = suggested_params['optimizer']
            trial_config.dataloader.batch_size = suggested_params['batch_size']
            trial_config.training.scheduler.patience = suggested_params['scheduler_patience']
            trial_config.data.augmentations.rand_affine_prob = suggested_params['rand_affine_prob']

            # Conditional parameters
            if 'loss_gamma' in suggested_params:
                trial_config.training.loss.gamma = suggested_params['loss_gamma']
            if 'loss_alpha' in suggested_params:
                trial_config.training.loss.alpha = suggested_params['loss_alpha']
                
            # Set a descriptive run name for this trial
            trial_config.run_name = f"{trial.study.study_name}_trial_{trial.number}"
            trial_config.wandb.enabled = False
            
            # Define the output directory for the training run
            trial_output_dir = os.path.join(
                PROJECT_OUTPUT_FOLDER_PATH, trial.study.study_name, f"trial_{trial.number}"
            )
            
            try:
                # Pass the config object directly. train.py is now responsible for saving it.
                best_val_auc, _ = run_training(
                    config=trial_config,
                    train_csv_override=subset_csv_path,
                    output_dir_override=trial_output_dir
                )
                return best_val_auc
            except Exception as e:
                print(f"Trial {trial.number} failed with exception: {e}")
                raise optuna.exceptions.TrialPruned()

        study.optimize(objective, n_trials=n_trials_for_stage)

    print("\n" + "="*60)
    print(f"--- Staged Optimization Finished for study: '{study_name}' ---")
    
    try:
        best_trial = study.best_trial
        print("\nBest trial found across all stages:")
        print(f"  Value (Best Val AUC): {best_trial.value:.4f}")
        print("  Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("\nNo completed trials found in the study. Cannot determine the best trial.")

if __name__ == '__main__':
    main()