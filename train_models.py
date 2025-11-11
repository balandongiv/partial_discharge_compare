"""
Model Training, Evaluation, and Reporting Script for Partial Discharge Classification
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_dataset(data_dir):
    """
    Load parquet files from feature selection directory.
    Combines all tracks if multiple exist.
    """
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.rglob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet file(s)")
    
    # Load and combine all parquet files
    dfs = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        print(f"Loaded {pq_file.name}: {df.shape}")
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df


def identify_target_column(df, preferred_col=None):
    """
    Identify the target column from common names.
    Prioritizes preferred_col if provided.
    """
    if preferred_col and preferred_col in df.columns:
        return preferred_col
    
    target_candidates = [
        'faultAnnotation', 'target', 'label', 'class', 
        'faultAnnotation_y', 'faultAnnotation_x', 
        'fault_annotation', 'annotation'
    ]
    
    for col in target_candidates:
        if col in df.columns:
            return col
    
    # If no standard name found, check for columns with 'annotation' or 'label'
    for col in df.columns:
        if 'annotation' in col.lower() or 'label' in col.lower():
            return col
    
    raise ValueError("Could not identify target column. Please specify manually.")


def prepare_data(df, target_col=None):
    """
    Separate features and target, handle encoding.
    """
    if target_col is None:
        target_col = identify_target_column(df, preferred_col='faultAnnotation')
    else:
        target_col = identify_target_column(df, preferred_col=target_col)
    
    print(f"Using target column: {target_col}")
    
    # Identify feature columns (exclude target and metadata)
    exclude_cols = [
        target_col, 'idStation', 'idMeasurement', 'station_id', 
        'file', 'id', 'index', 'faultAnnotation_x', 'faultAnnotation_y'
    ]
    
    # Remove duplicates from exclude_cols
    exclude_cols = list(set(exclude_cols))
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Drop rows where target is NaN
    valid_mask = ~y.isna()
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    if len(X) == 0:
        raise ValueError("No valid rows found after removing NaN targets")
    
    print(f"Removed {len(df) - len(X)} rows with NaN targets")
    
    # Encode target if it's not numeric
    label_encoder = None
    if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Target encoded. Classes: {label_encoder.classes_}")
    else:
        y = y.values
    
    # Remove any remaining non-numeric columns from features
    X = X.select_dtypes(include=[np.number])
    
    # Handle any infinite or NaN values in features
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Ensure target has no NaN (double check)
    y = pd.Series(y)
    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
    y = y.values
    
    print(f"Feature shape: {X.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    return X, y, feature_cols, label_encoder


def train_svm_model(X_train, y_train, cv, n_jobs=-1):
    """Train SVM models (linear and RBF) with GridSearchCV."""
    print("\n" + "="*60)
    print("Training SVM Models...")
    print("="*60)
    
    param_grids = {
        'linear': {
            'C': [0.1, 1, 10, 100],
            'class_weight': ['balanced']
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': ['balanced']
        }
    }
    
    models = {}
    best_params = {}
    
    # Linear SVM
    print("\nTraining Linear SVM...")
    svm_linear = SVC(kernel='linear', probability=True, random_state=42)
    grid_linear = GridSearchCV(
        svm_linear, param_grids['linear'], cv=cv, 
        scoring='f1_macro', n_jobs=n_jobs, verbose=1
    )
    grid_linear.fit(X_train, y_train)
    models['SVM_Linear'] = grid_linear.best_estimator_
    best_params['SVM_Linear'] = grid_linear.best_params_
    print(f"Best params: {grid_linear.best_params_}")
    print(f"Best CV score: {grid_linear.best_score_:.4f}")
    
    # RBF SVM
    print("\nTraining RBF SVM...")
    svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
    grid_rbf = GridSearchCV(
        svm_rbf, param_grids['rbf'], cv=cv,
        scoring='f1_macro', n_jobs=n_jobs, verbose=1
    )
    grid_rbf.fit(X_train, y_train)
    models['SVM_RBF'] = grid_rbf.best_estimator_
    best_params['SVM_RBF'] = grid_rbf.best_params_
    print(f"Best params: {grid_rbf.best_params_}")
    print(f"Best CV score: {grid_rbf.best_score_:.4f}")
    
    return models, best_params


def train_random_forest(X_train, y_train, cv, n_jobs=-1):
    """Train Random Forest with GridSearchCV."""
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    grid = GridSearchCV(
        rf, param_grid, cv=cv,
        scoring='f1_macro', n_jobs=n_jobs, verbose=1
    )
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    return {'RandomForest': grid.best_estimator_}, {'RandomForest': grid.best_params_}


def train_decision_tree(X_train, y_train, cv, n_jobs=-1):
    """Train Decision Tree with GridSearchCV."""
    print("\n" + "="*60)
    print("Training Decision Tree...")
    print("="*60)
    
    param_grid = {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced']
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(
        dt, param_grid, cv=cv,
        scoring='f1_macro', n_jobs=n_jobs, verbose=1
    )
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    return {'DecisionTree': grid.best_estimator_}, {'DecisionTree': grid.best_params_}


def train_knn(X_train, y_train, cv, n_jobs=-1):
    """Train k-Nearest Neighbors with GridSearchCV."""
    print("\n" + "="*60)
    print("Training k-Nearest Neighbors...")
    print("="*60)
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn = KNeighborsClassifier(n_jobs=n_jobs)
    grid = GridSearchCV(
        knn, param_grid, cv=cv,
        scoring='f1_macro', n_jobs=n_jobs, verbose=1
    )
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    return {'KNN': grid.best_estimator_}, {'KNN': grid.best_params_}


def train_mlp(X_train, y_train, cv, n_jobs=-1):
    """Train Multi-Layer Perceptron (ANN) with GridSearchCV."""
    print("\n" + "="*60)
    print("Training Multi-Layer Perceptron (ANN)...")
    print("="*60)
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }
    
    mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    grid = GridSearchCV(
        mlp, param_grid, cv=cv,
        scoring='f1_macro', n_jobs=n_jobs, verbose=1
    )
    grid.fit(X_train, y_train)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")
    
    return {'MLP': grid.best_estimator_}, {'MLP': grid.best_params_}


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Handle binary vs multiclass
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        # Multiclass - use one-vs-rest
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'roc_auc': roc_auc
    }
    
    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close('all')
    print(f"Saved confusion matrix: {save_path}")


def plot_roc_curves(all_models, X_test, y_test, save_path):
    """Plot ROC curves for all models on the same graph."""
    plt.figure(figsize=(12, 8))
    
    # Handle binary vs multiclass
    is_binary = len(np.unique(y_test)) == 2
    
    for model_name, model in all_models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)
            
            if is_binary:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
            else:
                # Multiclass: use one-vs-rest macro average
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                n_classes = y_test_bin.shape[1]
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute macro-average ROC
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                
                macro_auc = auc(all_fpr, mean_tpr)
                plt.plot(all_fpr, mean_tpr, label=f'{model_name} (AUC = {macro_auc:.3f})', linewidth=2)
        except Exception as e:
            print(f"Warning: Could not plot ROC for {model_name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close('all')
    print(f"Saved ROC curves: {save_path}")


def plot_feature_importance(models, feature_names, save_path):
    """Plot feature importance for tree-based models."""
    tree_models = ['RandomForest', 'DecisionTree']
    
    available_models = [m for m in tree_models if m in models]
    
    if not available_models:
        print("No tree-based models available for feature importance plot.")
        return
    
    fig, axes = plt.subplots(1, len(available_models), figsize=(8 * len(available_models), 6))
    if len(available_models) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(available_models):
        model = models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            axes[idx].barh(range(len(indices)), importances[indices], color='steelblue')
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels([feature_names[i] for i in indices])
            axes[idx].set_xlabel('Importance', fontsize=11)
            axes[idx].set_title(f'Feature Importance - {model_name}', fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close('all')
    print(f"Saved feature importance: {save_path}")


def plot_model_comparison(metrics_df, save_path):
    """Create bar plot comparing models (Accuracy + F1)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = metrics_df['model'].values
    x_pos = np.arange(len(models))
    
    # Accuracy plot
    ax1.bar(x_pos, metrics_df['accuracy'], color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df['accuracy']):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # F1 Score plot
    ax2.bar(x_pos, metrics_df['f1_score'], color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Model F1 Score Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df['f1_score']):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close('all')
    print(f"Saved model comparison: {save_path}")


def save_classification_reports(all_models, X_test, y_test, save_dir):
    """Save classification reports as text and Markdown."""
    save_dir = Path(save_dir)
    
    # Text version
    text_report = []
    text_report.append("="*80)
    text_report.append("CLASSIFICATION REPORTS")
    text_report.append("="*80)
    text_report.append("")
    
    # Markdown version
    md_report = []
    md_report.append("# Classification Reports\n")
    md_report.append("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    md_report.append("---\n")
    
    for model_name, model in all_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        text_report.append(f"\n{'='*80}")
        text_report.append(f"Model: {model_name}")
        text_report.append(f"{'='*80}")
        text_report.append(report)
        text_report.append("")
        
        md_report.append(f"\n## {model_name}\n")
        md_report.append("```\n")
        md_report.append(report)
        md_report.append("```\n")
        md_report.append("---\n")
    
    # Save text file
    text_path = save_dir / "classification_reports.txt"
    with open(text_path, 'w') as f:
        f.write('\n'.join(text_report))
    print(f"Saved text classification reports: {text_path}")
    
    # Save Markdown file
    md_path = save_dir / "classification_reports.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_report))
    print(f"Saved Markdown classification reports: {md_path}")


def generate_training_log(all_models, best_params, metrics_df, save_path):
    """Generate comprehensive training log in Markdown format."""
    log = []
    log.append("# Model Training Log\n")
    log.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.append("---\n\n")
    
    log.append("## Model List\n")
    for i, model_name in enumerate(metrics_df['model'].values, 1):
        log.append(f"{i}. {model_name}\n")
    log.append("\n---\n\n")
    
    log.append("## Hyperparameters Tested\n\n")
    for model_name in metrics_df['model'].values:
        log.append(f"### {model_name}\n")
        log.append("```python\n")
        if model_name == 'SVM_Linear':
            log.append("{'C': [0.1, 1, 10, 100], 'class_weight': ['balanced']}\n")
        elif model_name == 'SVM_RBF':
            log.append("{'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'class_weight': ['balanced']}\n")
        elif model_name == 'RandomForest':
            log.append("{'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'class_weight': ['balanced']}\n")
        elif model_name == 'DecisionTree':
            log.append("{'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'criterion': ['gini', 'entropy'], 'class_weight': ['balanced']}\n")
        elif model_name == 'KNN':
            log.append("{'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}\n")
        elif model_name == 'MLP':
            log.append("{'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate': ['constant', 'adaptive'], 'max_iter': [500, 1000]}\n")
        log.append("```\n\n")
    
    log.append("---\n\n")
    log.append("## Best Hyperparameters Found\n\n")
    for model_name, params in best_params.items():
        log.append(f"### {model_name}\n")
        log.append("```python\n")
        log.append(f"{params}\n")
        log.append("```\n\n")
    
    log.append("---\n\n")
    log.append("## Evaluation Metrics\n\n")
    log.append(metrics_df.to_markdown(index=False))
    log.append("\n\n---\n\n")
    
    log.append("## Performance Comments\n\n")
    best_model = metrics_df.loc[metrics_df['f1_score'].idxmax()]
    log.append(f"**Best Model (by F1 Score):** {best_model['model']}\n")
    log.append(f"- Accuracy: {best_model['accuracy']:.4f}\n")
    log.append(f"- Precision: {best_model['precision']:.4f}\n")
    log.append(f"- Recall: {best_model['recall']:.4f}\n")
    log.append(f"- F1 Score: {best_model['f1_score']:.4f}\n")
    log.append(f"- ROC-AUC: {best_model['roc_auc']:.4f}\n")
    
    log.append("\n### Model Performance Summary\n")
    log.append("- All models were evaluated using stratified 5-fold cross-validation.\n")
    log.append("- Models were trained with class_weight='balanced' where supported to handle class imbalance.\n")
    log.append("- Feature scaling (StandardScaler) was applied to all features.\n")
    log.append("- The best model was selected based on F1 score (macro-averaged).\n")
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(log))
    print(f"Saved training log: {save_path}")


def main():
    """Main execution function."""
    print("="*80)
    print("PARTIAL DISCHARGE CLASSIFICATION - MODEL TRAINING & EVALUATION")
    print("="*80)
    
    # Setup directories
    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data_dir = Path("features/4_feature_selection")
    df = load_dataset(data_dir)
    
    # Prepare data
    print("\n[2] Preparing data...")
    X, y, feature_names, label_encoder = prepare_data(df, target_col='faultAnnotation')
    
    # Train-test split (80/20, stratified)
    print("\n[3] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print("\n[4] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Save scaler and label encoder
    joblib.dump(scaler, models_dir / "scaler.joblib")
    if label_encoder is not None:
        joblib.dump(label_encoder, models_dir / "label_encoder.joblib")
    print("Scaler saved.")
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs = -1  # Use all available cores
    
    # Train all models
    print("\n[5] Training models...")
    all_models = {}
    all_best_params = {}
    
    # SVM models
    svm_models, svm_params = train_svm_model(X_train_scaled, y_train, cv, n_jobs)
    all_models.update(svm_models)
    all_best_params.update(svm_params)
    
    # Random Forest
    rf_models, rf_params = train_random_forest(X_train_scaled, y_train, cv, n_jobs)
    all_models.update(rf_models)
    all_best_params.update(rf_params)
    
    # Decision Tree
    dt_models, dt_params = train_decision_tree(X_train_scaled, y_train, cv, n_jobs)
    all_models.update(dt_models)
    all_best_params.update(dt_params)
    
    # KNN
    knn_models, knn_params = train_knn(X_train_scaled, y_train, cv, n_jobs)
    all_models.update(knn_models)
    all_best_params.update(knn_params)
    
    # MLP
    mlp_models, mlp_params = train_mlp(X_train_scaled, y_train, cv, n_jobs)
    all_models.update(mlp_models)
    all_best_params.update(mlp_params)
    
    # Evaluate all models
    print("\n[6] Evaluating models...")
    all_metrics = []
    all_predictions = {}
    all_probabilities = {}
    
    for model_name, model in all_models.items():
        metrics, y_pred, y_pred_proba = evaluate_model(
            model, X_test_scaled, y_test, model_name
        )
        all_metrics.append(metrics)
        all_predictions[model_name] = y_pred
        all_probabilities[model_name] = y_pred_proba
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values('f1_score', ascending=False)
    
    # Save metrics CSV
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")
    
    # Generate plots and reports
    print("\n[7] Generating plots and reports...")
    
    # Confusion matrices
    for model_name, model in all_models.items():
        y_pred = all_predictions[model_name]
        cm_path = results_dir / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
    
    # ROC curves
    roc_path = results_dir / "roc_curves.png"
    plot_roc_curves(all_models, X_test_scaled, y_test, roc_path)
    
    # Feature importance
    feat_path = results_dir / "feature_importance.png"
    plot_feature_importance(all_models, feature_names, feat_path)
    
    # Model comparison
    comp_path = results_dir / "model_comparison_barplot.png"
    plot_model_comparison(metrics_df, comp_path)
    
    # Classification reports
    save_classification_reports(all_models, X_test_scaled, y_test, results_dir)
    
    # Training log
    log_path = results_dir / "training_log.md"
    generate_training_log(all_models, all_best_params, metrics_df, log_path)
    
    # Select and save best model
    print("\n[8] Selecting best model...")
    best_model_name = metrics_df.iloc[0]['model']
    best_model = all_models[best_model_name]
    
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"Best model ({best_model_name}) saved to: {best_model_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY - MODEL COMPARISON")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("\n" + "="*80)
    print(f"Best Model: {best_model_name}")
    print(f"Best F1 Score: {metrics_df.iloc[0]['f1_score']:.4f}")
    print("="*80)
    
    print("\nAll results saved to 'results/' directory")
    print("Best model saved to 'models/best_model.joblib'")


if __name__ == "__main__":
    main()
