import argparse
import copy
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Add project root to path to allow absolute imports
import sys
import os
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
    
    subset_df, _ = train_test_split(
        full_df,
        train_size=percentage,
        stratify=full_df['fracture'],
        random_state=42
    )
    
    subset_filename = f"train_subset_{int(percentage*100)}.csv"
    subset_path = os.path.join(output_dir, subset_filename)
    subset_df.to_csv(subset_path, index=False)
    
    print(f"Created stratified subset with {len(subset_df)} samples at {subset_path}")
    return subset_path


def main():
    """Main function to start the staged hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Run staged HPO for fracture detection.")
    parser.add_argument('--config', type=str, required=True, help='Path to the base configuration YAML file.')
    parser.add_argument('--percentages', type=float, nargs='+', default=[0.05, 0.20, 1.00],
                        help='A list of data percentages for each stage.')
    parser.add_argument('--trials_per_stage', type=int, nargs='+', default=[25, 15, 10],
                        help='A list of trial counts corresponding to each percentage stage.')
    parser.add_argument('--study-db', type=str, default="hpo_study.db",
                        help='Name of the SQLite database file for the study, will be saved in the project output folder.')
    args = parser.parse_args()

    # --- 1. Validate and construct HPO Stages ---
    if len(args.percentages) != len(args.trials_per_stage):
        raise ValueError("The number of items in --percentages and --trials_per_stage must be the same.")
    
    STAGES = [{'percentage': p, 'n_trials': t} for p, t in zip(args.percentages, args.trials_per_stage)]

    # --- 2. Setup ---
    load_dotenv()
    PROJECT_DATA_FOLDER_PATH = os.getenv("PROJECT_DATA_FOLDER_PATH")
    PROJECT_OUTPUT_FOLDER_PATH = os.getenv("PROJECT_OUTPUT_FOLDER_PATH")

    base_config = load_config(args.config)
    split_dir = os.path.join(PROJECT_DATA_FOLDER_PATH, "splits", base_config.data.split_folder_name)
    full_train_csv_path = os.path.join(split_dir, "train.csv")
    subsets_dir = os.path.join(split_dir, "hpo_subsets")
    os.makedirs(subsets_dir, exist_ok=True)
    
    # --- 3. Create a single, persistent Optuna study ---
    study_name = "fracture-detection-staged-hpo"
    db_path = os.path.join(PROJECT_OUTPUT_FOLDER_PATH, args.study_db)
    storage_url = f"sqlite:///{db_path}"
    
    # Create the directory for the database if it doesn't exist
    print(f"Ensuring database directory exists at: {os.path.dirname(db_path)}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True  # This is key for resuming
    )

    # --- 4. Run the HPO process stage by stage ---
    for i, stage_info in enumerate(STAGES):
        stage_num = i + 1
        percentage = stage_info['percentage']
        n_trials = stage_info['n_trials']
        
        print("\n" + "="*60)
        print(f"--- HPO STAGE {stage_num}/{len(STAGES)}: {n_trials} trials on {percentage*100:.0f}% data ---")
        print("="*60)

        subset_csv_path = create_stratified_subset(full_train_csv_path, subsets_dir, percentage)

        def objective(trial: optuna.Trial) -> float:
            trial_config = copy.deepcopy(base_config)

            trial_config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            trial_config.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
            if trial_config.training.loss.name.lower() == 'focalloss':
                trial_config.training.loss.gamma = trial.suggest_float("loss_gamma", 1.0, 3.0)
                trial_config.training.loss.alpha = trial.suggest_float("loss_alpha", 0.25, 0.75)
            
            trial_config.wandb.enabled = False
            
            try:
                trial_output_dir = os.path.join(
                    PROJECT_OUTPUT_FOLDER_PATH, 
                    "optuna", 
                    trial.study.study_name, 
                    f"trial_{trial.number}"
                )

                best_val_auc, _ = run_training(
                    config=trial_config, 
                    config_path=args.config,
                    train_csv_override=subset_csv_path,
                    output_dir_override=trial_output_dir
                )
                return best_val_auc
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return 0.0

        study.optimize(objective, n_trials=n_trials)

    # --- 5. Print the final results ---
    print("\n" + "="*60)
    print("--- Staged Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("\nBest trial found across all stages:")
    best_trial = study.best_trial
    print(f"  Value (Best Val AUC): {best_trial.value:.4f}")
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()