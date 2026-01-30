"""
MLJAR Feature Selection Only Script
Performs feature selection using MLJAR without full model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from supervised.automl import AutoML
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_mljar_feature_selection(csv_path: str, output_dir: str = "mljar_feature_selection"):
    """
    Run MLJAR feature selection only (no full training).
    
    Args:
        csv_path: Path to the CSV file with features
        output_dir: Directory to save MLJAR results
    """
    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    log.info(f"Data shape: {df.shape}")
    
    # Find label column
    label_cols = [c for c in df.columns if any(x in c.lower() for x in ['label', 'target', 'class', 'station'])]
    log.info(f"Potential label columns: {label_cols}")
    
    if 'label' in df.columns:
        label_col = 'label'
    elif 'target' in df.columns:
        label_col = 'target'
    elif 'station_id' in df.columns:
        label_col = 'station_id'
    else:
        # Use the first column as label if it looks categorical
        label_col = df.columns[0]
        log.warning(f"No obvious label column, using: {label_col}")
    
    log.info(f"Using label column: {label_col}")
    
    # Separate features and target
    y = df[label_col]
    X = df.drop(columns=[label_col])
    
    # Remove non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    log.info(f"Feature matrix shape: {X.shape}")
    log.info(f"Target distribution:\n{y.value_counts()}")
    
    # Encode target if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        log.info(f"Encoded labels: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        y_encoded = y.values
        le = None
    
    # Determine task type
    n_unique = len(np.unique(y_encoded))
    if n_unique <= 2:
        ml_task = "binary_classification"
    else:
        ml_task = "multiclass_classification"
    
    log.info(f"ML task: {ml_task} ({n_unique} classes)")
    
    # Minimal MLJAR configuration for feature selection only
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log.info("Initializing MLJAR AutoML for feature selection...")
    
    # Use minimal settings for faster feature selection
    automl = AutoML(
        ml_task=ml_task,
        mode="Perform",  # Fast mode
        algorithms=["Xgboost"],  # Single fast algorithm
        train_ensemble=False,
        stack_models=False,
        features_selection=True,  # Enable feature selection
        results_path=str(output_path),
        total_time_limit=300,  # 5 minutes max
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 3,  # Minimal folds for speed
            "shuffle": True,
            "random_seed": 42
        },
        verbose=1,
        random_state=42
    )
    
    log.info("Training MLJAR for feature selection...")
    automl.fit(X, y_encoded)
    
    log.info("Extracting selected features...")
    
    # Get feature importance from the best model
    best_model = automl._best_model
    
    if hasattr(best_model, 'get_feature_importance'):
        importance_df = best_model.get_feature_importance()
    else:
        # Try to extract from the model learner
        learner = best_model.learner
        if hasattr(learner, 'get_feature_importance'):
            importance_df = learner.get_feature_importance()
        else:
            log.warning("Could not extract feature importance, using all features")
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': np.ones(len(X.columns))
            })
    
    # Filter features by importance threshold (keep features with importance > 0)
    selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
    
    log.info(f"\n{'='*60}")
    log.info(f"MLJAR FEATURE SELECTION RESULTS")
    log.info(f"{'='*60}")
    log.info(f"Original features: {X.shape[1]}")
    log.info(f"Selected features: {len(selected_features)}")
    log.info(f"Reduction: {100 * (1 - len(selected_features)/X.shape[1]):.1f}%")
    log.info(f"\nTop 20 features by importance:")
    log.info(f"\n{importance_df.head(20).to_string()}")
    
    # Save results
    output_file = output_path / "selected_features.csv"
    importance_df.to_csv(output_file, index=False)
    log.info(f"\nFull feature importance saved to: {output_file}")
    
    # Save just the selected feature names
    selected_file = output_path / "selected_feature_names.txt"
    with open(selected_file, 'w') as f:
        f.write('\n'.join(selected_features))
    log.info(f"Selected feature names saved to: {selected_file}")
    
    return selected_features, importance_df

if __name__ == "__main__":
    csv_path = r"antigravity_results\station_52008\features\combined\combined_features_station_52008.csv"
    selected_features, importance_df = run_mljar_feature_selection(csv_path)
    
    print(f"\n{'='*60}")
    print(f"SELECTED FEATURES ({len(selected_features)} total):")
    print(f"{'='*60}")
    for i, feat in enumerate(selected_features, 1):
        importance = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        print(f"{i:3d}. {feat:50s} (importance: {importance:.6f})")
