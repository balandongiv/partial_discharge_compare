"""
Simple Feature Selection (MLJAR-style approach without full AutoML)
Uses variance filtering + correlation pruning + XGBoost feature importance
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_feature_selection(csv_path: str, output_dir: str = "simple_feature_selection", 
                         variance_threshold: float = 0.01,
                         correlation_threshold: float = 0.95,
                         importance_threshold: float = 0.001):
    """
    Run feature selection using variance filtering, correlation pruning, and XGBoost importance.
    
    Args:
        csv_path: Path to the CSV file with features
        output_dir: Directory to save results
        variance_threshold: Minimum variance to keep a feature
        correlation_threshold: Maximum correlation between features
        importance_threshold: Minimum XGBoost importance to keep a feature
    """
    log.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    log.info(f"Data shape: {df.shape}")
    
    # Drop ID columns
    id_cols = [c for c in df.columns if 'id' in c.lower() or 'station' in c.lower() or 'measurement' in c.lower()]
    log.info(f"Dropping ID columns: {id_cols}")
    X = df.drop(columns=id_cols)
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    log.info(f"Starting with {X.shape[1]} features")
    
    # Step 1: Variance Threshold
    log.info(f"\n{'='*60}")
    log.info("STEP 1: Variance Filtering")
    log.info(f"{'='*60}")
    
    selector = VarianceThreshold(threshold=variance_threshold)
    X_var = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    log.info(f"Features after variance filtering: {len(selected_features)} (removed {X.shape[1] - len(selected_features)})")
    
    X = pd.DataFrame(X_var, columns=selected_features)
    
    # Step 2: Correlation Pruning
    log.info(f"\n{'='*60}")
    log.info("STEP 2: Correlation Pruning")
    log.info(f"{'='*60}")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    log.info(f"Dropping {len(to_drop)} highly correlated features (threshold={correlation_threshold})")
    
    X = X.drop(columns=to_drop)
    selected_features = X.columns.tolist()
    
    log.info(f"Features after correlation pruning: {len(selected_features)}")
    
    # Step 3: XGBoost Feature Importance
    log.info(f"\n{'='*60}")
    log.info("STEP 3: XGBoost Feature Importance")
    log.info(f"{'='*60}")
    
    # Create a synthetic target (use the first principal component as a proxy)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    y = pca.fit_transform(X).ravel()
    
    log.info("Training XGBoost model for feature importance...")
    
    # Train XGBoost
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Filter by importance threshold
    selected_features = importance_df[importance_df['importance'] > importance_threshold]['feature'].tolist()
    
    log.info(f"Features after importance filtering (threshold={importance_threshold}): {len(selected_features)}")
    
    # Final Results
    log.info(f"\n{'='*60}")
    log.info(f"FEATURE SELECTION RESULTS")
    log.info(f"{'='*60}")
    log.info(f"Original features: {len(numeric_cols)}")
    log.info(f"Selected features: {len(selected_features)}")
    log.info(f"Reduction: {100 * (1 - len(selected_features)/len(numeric_cols)):.1f}%")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "feature_importance.csv"
    importance_df.to_csv(output_file, index=False)
    log.info(f"\nFull feature importance saved to: {output_file}")
    
    selected_file = output_path / "selected_features.txt"
    with open(selected_file, 'w') as f:
        f.write('\n'.join(selected_features))
    log.info(f"Selected feature names saved to: {selected_file}")
    
    # Print top features
    log.info(f"\nTop 50 features by importance:")
    print("\n" + importance_df.head(50).to_string(index=False))
    
    return selected_features, importance_df

if __name__ == "__main__":
    csv_path = r"antigravity_results\station_52008\features\combined\combined_features_station_52008.csv"
    selected_features, importance_df = run_feature_selection(
        csv_path,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        importance_threshold=0.001
    )
    
    print(f"\n{'='*60}")
    print(f"SELECTED FEATURES ({len(selected_features)} total):")
    print(f"{'='*60}")
    for i, feat in enumerate(selected_features[:100], 1):  # Show top 100
        importance = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        print(f"{i:3d}. {feat:50s} (importance: {importance:.6f})")
    
    if len(selected_features) > 100:
        print(f"\n... and {len(selected_features) - 100} more features")
