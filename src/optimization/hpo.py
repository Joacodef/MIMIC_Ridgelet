import argparse
import copy
import optuna

# Add project root to path to allow absolute imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config.config import load_config, AppConfig
from src.training.train import run_training

def objective(trial: optuna.Trial, base_config: AppConfig, base_config_path: str) -> float:
    """
    The objective function for Optuna to optimize.
    A single "trial" consists of an entire training run.
    """
    # Create a deep copy of the base config for this trial
    trial_config = copy.deepcopy(base_config)

    # --- 1. Define the hyperparameter search space ---
    # Suggest values for the hyperparameters we want to tune.
    trial_config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    trial_config.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Example for Focal Loss parameters
    if trial_config.training.loss.name.lower() == 'focalloss':
        trial_config.training.loss.gamma = trial.suggest_float("loss_gamma", 1.0, 3.0)
        trial_config.training.loss.alpha = trial.suggest_float("loss_alpha", 0.25, 0.75)

    trial_config.data.augmentations.rand_affine_prob = trial.suggest_float("rand_affine_prob", 0.3, 0.7)

    # Note: The config expects a tuple for ranges, but we can suggest one value
    # and construct the tuple from it.
    rotate_magnitude = trial.suggest_float("rotate_magnitude", 0.05, 0.2)
    trial_config.data.augmentations.rotate_range = (-rotate_magnitude, rotate_magnitude)

    scale_magnitude = trial.suggest_float("scale_magnitude", 0.05, 0.2)
    trial_config.data.augmentations.scale_range = (-scale_magnitude, scale_magnitude)

    trial_config.model.base_model = trial.suggest_categorical("base_model", ["resnet18", "resnet34"])

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    trial_config.training.optimizer = optimizer_name

    if optimizer_name == "sgd":
        momentum = trial.suggest_float("momentum", 0.85, 0.99)
        # This would require adding 'momentum' to your TrainingConfig
        trial_config.training.momentum = momentum

    trial_config.training.scheduler.factor = trial.suggest_float("scheduler_factor", 0.1, 0.5)
    trial_config.training.scheduler.patience = trial.suggest_int("scheduler_patience", 3, 7)

    # --- 2. Configure the trial ---
    # Disable detailed logging for HPO runs to avoid clutter.
    # The output directory will still be created, which is useful for inspection.
    trial_config.wandb.enabled = False
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Parameters: {trial.params}")

    try:
        # --- 3. Run the training and return the metric to optimize ---
        # We want to maximize the validation AUC.
        best_val_auc = run_training(config=trial_config, config_path=base_config_path)
        return best_val_auc

    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        # Report failure to Optuna so it can prune the trial.
        return 0.0 # Return a poor value

def main():
    """Main function to start the hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for fracture detection.")
    parser.add_argument('--n_trials', type=int, default=50, help='The number of optimization trials to run.')
    parser.add_argument('--config', type=str, required=True, help='Path to the base configuration YAML file.')
    args = parser.parse_args()

    # Load the base configuration from the provided file
    base_config = load_config(args.config)

    # Create an Optuna study
    # We specify a name to allow for resuming studies if needed
    # Direction is "maximize" because we want the highest validation AUC
    study = optuna.create_study(direction="maximize", study_name="fracture_detection_hpo")

    # Start the optimization
    study.optimize(lambda trial: objective(trial, base_config, args.config), n_trials=args.n_trials)

    # --- Print the results ---
    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Val AUC): {best_trial.value:.4f}")

    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()