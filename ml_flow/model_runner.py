from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import logging
import pickle
import os


class XGBoostCheckpoint(xgb.callback.TrainingCallback):
    """Callback for saving XGBoost checkpoints."""

    def __init__(self, model_dir, interval, prefix):
        self.model_dir = model_dir
        self.interval = interval
        self.prefix = prefix

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % self.interval == 0:
            os.makedirs(self.model_dir, exist_ok=True)
            path = os.path.join(self.model_dir, f"{self.prefix}_{epoch + 1}.json")
            model.save_model(path)
        return False


def _latest_checkpoint(model_dir, prefix):
    """Return the latest checkpoint path if available."""
    if not os.path.isdir(model_dir):
        return None
    checkpoints = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith(prefix)
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(
    X_train,
    y_train,
    model_type='DecisionTree',
    params=None,
    cv=5,
    scoring='accuracy',
    model_dir='model',
    checkpoint_interval=None,
    resume=False,
):
    """
    Trains and evaluates a machine learning model using cross-validation.
    Saves the trained model to model_dir.
    """
    logging.info(f"Training {model_type} model...")
    if model_type == 'DecisionTree':
        model_cls = DecisionTreeClassifier
    elif model_type == 'SVM':
        model_cls = SVC
    elif model_type == 'RandomForest':
        model_cls = RandomForestClassifier
    elif model_type == 'XGBoost':
        model_cls = xgb.XGBClassifier
    elif model_type == 'LogisticRegression':
        model_cls = LogisticRegression
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    base_model = model_cls(**(params or {}))

    cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring=scoring)
    logging.info(f"Cross-validation scores ({scoring}): {cv_scores}")
    logging.info(f"Mean CV score ({scoring}): {cv_scores.mean():.4f}")

    fit_kwargs = {}
    callbacks = []
    if model_type == 'XGBoost':
        if checkpoint_interval:
            callbacks.append(
                XGBoostCheckpoint(model_dir, checkpoint_interval, f"{model_type}_checkpoint")
            )
        if resume:
            ckpt = _latest_checkpoint(model_dir, f"{model_type}_checkpoint")
            if ckpt:
                logging.info(f"Resuming from checkpoint {ckpt}")
                fit_kwargs['xgb_model'] = ckpt
        if callbacks:
            fit_kwargs['callbacks'] = callbacks

    model = model_cls(**(params or {}))
    model.fit(X_train, y_train, **fit_kwargs)  # Fit on the entire training set after CV

    os.makedirs(model_dir, exist_ok=True) # Ensure model directory exists
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Trained {model_type} model saved to {model_path}")

    return model, cv_scores.mean()

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    """
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
    recall = recall_score(y_test, y_pred, average='weighted')    # Use 'weighted' for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multi-class
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # ROC AUC for binary or multiclass (ovr)
    try:
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') # 'ovr' for multiclass
        logging.info(f"Test ROC AUC (OVR): {roc_auc:.4f}")
    except AttributeError: # Models without predict_proba (e.g., some SVM kernels without probability=True)
        logging.warning("ROC AUC score not available for this model.")
        roc_auc = None

    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1-Score: {f1:.4f}")
    logging.info("Confusion Matrix:\n" + str(conf_matrix))
    logging.info("Evaluation completed.")
    return accuracy, roc_auc, precision, recall, f1, conf_matrix

if __name__ == '__main__':
    from data_loader import load_data
    from data_processor import preprocess_data

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target='target') # Assuming 'target' column exists

    model, cv_mean_score = train_model(X_train, y_train, model_type='RandomForest')
    evaluate_model(model, X_test, y_test)
