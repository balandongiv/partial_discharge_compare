import argparse
import json
import logging
import os
import subprocess
from datetime import datetime


def setup_logging(config_path: str) -> None:
    """Configure root logger to write to timestamped and main log files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file_main = os.path.join(log_dir, "main.log")
    log_file_timestamp = os.path.join(log_dir, f"main_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs when running multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler_main = logging.FileHandler(log_file_main)
    file_handler_main.setFormatter(formatter)
    file_handler_timestamp = logging.FileHandler(log_file_timestamp)
    file_handler_timestamp.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler_main)
    root_logger.addHandler(file_handler_timestamp)
    root_logger.addHandler(stream_handler)

    # Add experiment metadata to the log header
    commit_hash = "unknown"
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])\
            .decode().strip()
        )
    except Exception:
        pass

    root_logger.info("Starting experiment")
    root_logger.info("Config path: %s", config_path)
    root_logger.info("Git commit hash: %s", commit_hash)

TRACKER_FILE = 'executed_combinations_tracker.json' # File to save executed combinations

def load_executed_combinations():
    """Loads executed combinations from a JSON file."""
    executed_combinations = set()
    tracker_filepath = os.path.join("ml_flow", TRACKER_FILE)
    if os.path.exists(tracker_filepath):
        try:
            with open(tracker_filepath, 'r') as f:
                loaded_combinations_list = json.load(f)
                executed_combinations = set(tuple(combo) for combo in loaded_combinations_list) # Convert lists back to tuples
            logging.info(f"Loaded executed combinations from {tracker_filepath}")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"Could not load executed combinations from {tracker_filepath}. Starting fresh.")
    return executed_combinations

def save_executed_combinations(executed_combinations):
    """Saves executed combinations to a JSON file."""
    tracker_filepath = os.path.join("../ml_flow", TRACKER_FILE)
    combinations_list = [list(combo) for combo in executed_combinations] # Convert tuples to lists for JSON serialization
    with open(tracker_filepath, 'w') as f:
        json.dump(combinations_list, f, indent=4)
    logging.info(f"Saved executed combinations to {tracker_filepath}")

def main():
    parser = argparse.ArgumentParser(description="ML Workflow Script")
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration JSON file')
    args = parser.parse_args()

    setup_logging(args.config)

    # Import modules after logging is configured
    from data_loader import load_data
    from data_processor import preprocess_data
    from model_runner import train_model, evaluate_model
    from parameter_tuner import tune_hyperparameters
    from feature_selector import select_features_featurewiz
    from helper import load_config

    # Load configuration using helper function
    config = load_config(args.config)
    logging.info("Starting ML Workflow...")

    target_column = config.get('target_column')
    results = [] # List to store results for each configuration
    executed_combinations = load_executed_combinations() # Load executed combinations from file

    for model_type in config.get('model_types'):
        for scaling_method in config.get('scaling_methods'):
            for feature_selection in config.get('feature_selections'):
                logging.info(f"Current Configuration: Model Type: {model_type}, Scaling Method: {scaling_method}, Feature Selection: {feature_selection}")

                combination_key = (model_type, scaling_method, feature_selection)
                if combination_key in executed_combinations:
                    logging.info(f"Combination {combination_key} already executed. Skipping.")
                    continue # Skip to the next combination


                # 1. Load Data
                df = load_data(dataset_name=config.get('dataset_name'))

                # 2. Feature Selection (optional)
                if feature_selection == 'featurewiz':
                    selected_features = select_features_featurewiz(df.copy(), target_column) # Use .copy()
                elif feature_selection is None: # Handle None explicitly
                    selected_features = df.columns.drop(target_column).tolist() # Use all features if no feature selection
                else:
                    raise ValueError(f"Feature selection method '{feature_selection}' not supported.")

                # 3. Data Preprocessing
                X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
                    df,
                    features=selected_features,
                    target=target_column,
                    scaling_method=scaling_method,
                    test_size=config.get('test_size'),
                    val_size=config.get('val_size'),
                    random_state=config.get('random_state')
                )

                # 4. Hyperparameter Tuning (optional)
                if config.get('hyperparameter_tuning'):
                    best_params = tune_hyperparameters(X_train, y_train, model_type=model_type, n_trials=config.get('n_trials_optuna'), cv=config.get('cv_folds'))
                else:
                    best_params = None # Use default parameters in train_model

                # 5. Model Training
                model, cv_score = train_model(X_train, y_train, model_type=model_type, params=best_params, cv=config.get('cv_folds'))

                # 6. Model Evaluation
                accuracy, roc_auc, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)

                results.append({
                    "model_type": model_type,
                    "scaling_method": scaling_method,
                    "feature_selection": feature_selection,
                    "cv_score": cv_score,
                    "test_accuracy": accuracy,
                    "test_roc_auc": roc_auc,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1_score": f1,
                    "test_confusion_matrix": conf_matrix.tolist() # Convert to list for JSON serialization
                })
                executed_combinations.add(combination_key) # Add combination to executed set after successful run
                save_executed_combinations(executed_combinations) # Save after each combination

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.json"
    results_filepath = os.path.join("../ml_flow", results_filename) # Save in ml_flow directory
    with open(results_filepath, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Evaluation results saved to {results_filepath}")
    logging.info("ML Workflow completed for all configurations.")

if __name__ == "__main__":
    main()
