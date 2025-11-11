"""
Evaluation and Reporting Module for Partial Discharge Classification Pipeline

This module implements Phase 3 of the PD classification pipeline, responsible for evaluating
trained models and generating comprehensive reports including metrics, confusion matrices,
and performance visualizations.

Step-by-Step Process:
1. Data Loading:
   - Loads selected features from the baseline track (features/4_feature_selection/tracks/baseline_all_feats/)
   - Combines features from all stations into a unified dataset
   - Generates dummy labels based on file naming patterns (1 = PD present, 0 = no PD)
   - Separates feature matrix (X) and target labels (y)

2. Model Discovery:
   - Scans models/tuned_gridsearch/ directory for trained model files (.joblib)
   - Identifies available models (logreg, svm, rf, xgb, etc.)
   - Loads each model using joblib for evaluation

3. Model Evaluation:
   - Generates predictions using each loaded model
   - Computes probability scores for models that support predict_proba()
   - Calculates key performance metrics:
     * Accuracy: Overall classification accuracy
     * F1-Score: Harmonic mean of precision and recall
     * ROC-AUC: Area under ROC curve (when probabilities available)

4. Confusion Matrix Generation:
   - Creates confusion matrices for each model
   - Generates visualization plots with proper formatting
   - Saves confusion matrix plots as PNG files in reports/ directory
   - Uses color-coded heatmaps for easy interpretation

5. Metrics Compilation:
   - Collects all performance metrics into a structured DataFrame
   - Saves comprehensive metrics table as CSV file (reports/metrics.csv)
   - Provides summary statistics for model comparison

6. Report Generation:
   - Creates standardized directory structure for reports
   - Saves all visualizations and metrics files
   - Logs processing progress and completion status

Output Files:
- reports/metrics.csv: Performance metrics for all models
- reports/cm_<model_name>.png: Confusion matrix plots for each model
- reports/pipeline.log: Detailed processing logs

Configuration Parameters:
- station_id: Optional specific station to evaluate (evaluates all if None)
- input_path: Path to processed data (typically partial_discharge_project/)
- output_path: Root directory for report outputs

Dependencies:
- joblib: Model loading and serialization
- numpy: Numerical operations and array handling
- pandas: Data manipulation and CSV export
- sklearn.metrics: Performance metric calculations
- matplotlib: Visualization and plot generation
- pathlib: Cross-platform path handling
"""

from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .logger import get_logger


def _format_metric(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.3f} ± {std:.3f}"


def _extract_feature_importances(artifact: dict[str, object]) -> Optional[pd.DataFrame]:
    estimator = artifact.get("model")
    feature_names = artifact.get("feature_names", [])
    if estimator is None or not feature_names:
        return None
    values: Optional[np.ndarray] = None
    if hasattr(estimator, "feature_importances_"):
        values = getattr(estimator, "feature_importances_")
    elif hasattr(estimator, "coef_"):
        coef = getattr(estimator, "coef_")
        coef = np.asarray(coef)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)
    if values is None:
        return None
    series = pd.Series(np.asarray(values, dtype=np.float32), index=feature_names)
    series = series.sort_values(ascending=False)
    return series.to_frame(name="importance")


def _generate_shap_plot(
    artifact: dict[str, object],
    baseline_df: pd.DataFrame,
    output_path: Path,
    log,
) -> Optional[Path]:
    try:
        import shap
    except ImportError:
        log.info("SHAP not available; skipping explainability plot.")
        return None

    estimator = artifact.get("model")
    feature_names = artifact.get("feature_names", [])
    if estimator is None or not feature_names:
        return None
    if baseline_df.empty:
        log.warning("Baseline feature dataframe empty; skipping SHAP plot.")
        return None
    missing = [f for f in feature_names if f not in baseline_df.columns]
    if missing:
        log.warning("Missing features for SHAP computation: %s", missing)
        return None

    X = baseline_df[feature_names].astype(np.float32, copy=False).to_numpy()
    if X.size == 0:
        return None
    sample_size = min(200, X.shape[0])
    X_sample = X[:sample_size]
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_to_plot = shap_values[0]
        else:
            shap_to_plot = shap_values
        shap.summary_plot(shap_to_plot, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info("Saved SHAP summary plot -> %s", output_path)
        return output_path
    except Exception as exc:  # pragma: no cover
        log.warning("Unable to generate SHAP plot: %s", exc)
        plt.close("all")
        return None


def _write_pdf_report(lines: list[str], output_path: Path, log) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as pdf_canvas_module
    except ImportError:
        log.info("reportlab not available; skipping PDF generation.")
        return

    pdf = pdf_canvas_module(str(output_path), pagesize=letter)
    width, height = letter
    margin = 54
    y = height - margin
    for line in lines:
        if not line.strip():
            y -= 18
        else:
            pdf.drawString(margin, y, line[:120])
            y -= 14
        if y < margin:
            pdf.showPage()
            y = height - margin
    pdf.save()
    log.info("PDF report generated -> %s", output_path)


def run(station_id: Optional[str], input_path: Path, output_path: Path, config: Optional[dict] = None) -> dict:
    """Generate evaluation plots and markdown report based on training outputs."""
    log = get_logger(__name__, log_dir=output_path / "reports")
    log.info("Starting evaluation/reporting stage")

    results_dir = Path("results")
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Model metrics not found; ensure model_training stage completed.")
    metrics_df = pd.read_csv(metrics_path)
    log.info("Loaded model metrics -> %s", metrics_path)

    training_meta_path = results_dir / "training_summary.json"
    training_meta: dict[str, object] = {}
    if training_meta_path.exists():
        training_meta = json.loads(training_meta_path.read_text(encoding="utf-8"))
        log.info("Loaded training metadata -> %s", training_meta_path)
    best_model_info = training_meta.get("best_model", {}) if isinstance(training_meta, dict) else {}
    best_model_name = best_model_info.get("name") if isinstance(best_model_info, dict) else None
    best_model_metrics = best_model_info.get("metrics") if isinstance(best_model_info, dict) else {}
    environment_info = training_meta.get("environment") if isinstance(training_meta, dict) else {}
    config_snapshot = training_meta.get("config_snapshot") if isinstance(training_meta, dict) else None

    prediction_files = sorted(results_dir.glob("predictions_*.parquet"))
    if not prediction_files:
        raise FileNotFoundError("No prediction files found. Re-run model_training to generate cross-validated predictions.")

    figures_info = []
    class_balance: Optional[dict[str, int]] = None

    for pred_path in prediction_files:
        model_name = pred_path.stem.replace("predictions_", "", 1)
        df = pd.read_parquet(pred_path)
        y_true = df["true_label"].astype(str)
        y_pred = df["pred_label"].astype(str)
        labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))

        if class_balance is None:
            counts = y_true.value_counts()
            class_balance = {label: int(counts.get(label, 0)) for label in labels}

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title(f"Confusion Matrix - {model_name}")
        fig_cm.tight_layout()
        cm_path = report_dir / f"confusion_{model_name}.png"
        fig_cm.savefig(cm_path, dpi=200)
        plt.close(fig_cm)
        log.info("Saved confusion matrix -> %s", cm_path)

        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        roc_path = None
        if prob_cols:
            class_labels = [col.replace("prob_", "", 1) for col in prob_cols]
            proba = df[prob_cols].to_numpy()
            if proba.shape[1] < 2:
                log.warning("Skipping ROC curve for %s due to single probability column", model_name)
            else:
                y_binarized = label_binarize(y_true, classes=class_labels)
                if y_binarized.ndim == 1 or y_binarized.shape[1] < min(2, len(class_labels)):
                    log.warning("Skipping ROC curve for %s due to insufficient label variety", model_name)
                else:
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    plotted = False
                    if len(class_labels) == 2 and proba.shape[1] >= 2:
                        pos_idx = 1
                        fpr, tpr, _ = roc_curve(y_binarized[:, pos_idx], proba[:, pos_idx])
                        roc_auc_val = auc(fpr, tpr)
                        ax_roc.plot(fpr, tpr, label=f"{class_labels[pos_idx]} (AUC={roc_auc_val:.3f})")
                        plotted = True
                    else:
                        for idx, label in enumerate(class_labels[: proba.shape[1]]):
                            fpr, tpr, _ = roc_curve(y_binarized[:, idx], proba[:, idx])
                            roc_auc_val = auc(fpr, tpr)
                            ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_val:.3f})")
                            plotted = True
                    if plotted:
                        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
                        ax_roc.set_xlabel("False Positive Rate")
                        ax_roc.set_ylabel("True Positive Rate")
                        ax_roc.set_title(f"ROC Curves - {model_name}")
                        ax_roc.legend(loc="lower right", fontsize="small")
                        ax_roc.grid(True, linestyle="--", alpha=0.4)
                        fig_roc.tight_layout()
                        roc_path = report_dir / f"roc_{model_name}.png"
                        fig_roc.savefig(roc_path, dpi=200)
                        plt.close(fig_roc)
                        log.info("Saved ROC curves -> %s", roc_path)
                    else:
                        plt.close(fig_roc)

        figures_info.append({"model": model_name, "confusion": cm_path, "roc": roc_path})

    # Create comparison plot (F1 means)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df["model"], metrics_df["f1_mean"], yerr=metrics_df["f1_std"], capsize=4)
    ax.set_ylabel("F1 Score (mean)")
    ax.set_title("Model F1 Score Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    comparison_path = report_dir / "model_f1_comparison.png"
    fig.savefig(comparison_path, dpi=200)
    plt.close(fig)
    log.info("Saved model comparison chart -> %s", comparison_path)

    feature_importance_df: Optional[pd.DataFrame] = None
    shap_plot_path: Optional[Path] = None
    baseline_features_path = Path("features") / "4_feature_selection" / "selected_features.parquet"
    baseline_df = pd.read_parquet(baseline_features_path) if baseline_features_path.exists() else pd.DataFrame()

    if best_model_name:
        model_artifact_path = Path("models") / "trained" / f"{best_model_name}.joblib"
        if model_artifact_path.exists():
            artifact = joblib.load(model_artifact_path)
            feature_importance_df = _extract_feature_importances(artifact)
            shap_output_path = report_dir / f"shap_{best_model_name}.png"
            shap_plot_path = _generate_shap_plot(artifact, baseline_df, shap_output_path, log)
        else:
            log.warning("Best model artifact not found at %s", model_artifact_path)

    # Build Markdown report
    report_lines = [
        "# Partial Discharge Classification Report",
        "",
        "## Model Performance Summary",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in metrics_df.iterrows():
        report_lines.append(
            f"| {row['model']} | "
            f"{_format_metric(row['accuracy_mean'], row['accuracy_std'])} | "
            f"{_format_metric(row['precision_mean'], row['precision_std'])} | "
            f"{_format_metric(row['recall_mean'], row['recall_std'])} | "
            f"{_format_metric(row['f1_mean'], row['f1_std'])} | "
            f"{_format_metric(row['roc_auc_mean'], row['roc_auc_std'])} |"
        )

    if class_balance:
        report_lines.extend(
            [
                "",
                "## Class Balance",
                "",
                "| Label | Count |",
                "| --- | --- |",
            ]
        )
        for label, count in class_balance.items():
            report_lines.append(f"| {label} | {count} |")

    if best_model_name:
        report_lines.extend(
            [
                "",
                "## Best Performing Model",
                "",
                f"- Model: **{best_model_name}**",
            ]
        )
        if isinstance(best_model_metrics, dict):
            f1_val = best_model_metrics.get("f1_mean")
            roc_val = best_model_metrics.get("roc_auc_mean")
            if isinstance(f1_val, (int, float)):
                report_lines.append(f"- F1 Score (mean): {float(f1_val):.3f}")
            if isinstance(roc_val, (int, float)):
                report_lines.append(f"- ROC-AUC (mean): {float(roc_val):.3f}")

    if feature_importance_df is not None and not feature_importance_df.empty:
        report_lines.extend(
            [
                "",
                "## Top Feature Importances",
                "",
                "| Feature | Importance |",
                "| --- | --- |",
            ]
        )
        for feature, value in feature_importance_df.head(15)["importance"].items():
            report_lines.append(f"| {feature} | {value:.4f} |")

    if shap_plot_path is not None:
        report_lines.extend(
            [
                "",
                f"![SHAP Summary]({shap_plot_path.as_posix()})",
            ]
        )

    report_lines.extend(
        [
            "",
            f"![Model F1 Comparison]({comparison_path.as_posix()})",
            "",
            "## Model-Level Diagnostics",
        ]
    )

    for info in figures_info:
        report_lines.append(f"### {info['model']}")
        report_lines.append("")
        report_lines.append(f"![Confusion Matrix]({info['confusion'].as_posix()})")
        if info["roc"] is not None:
            report_lines.append("")
            report_lines.append(f"![ROC Curves]({info['roc'].as_posix()})")
        report_lines.append("")

    env_lines = [
        "",
        "## Environment & Reproducibility",
        "",
    ]
    if isinstance(environment_info, dict):
        for key, value in environment_info.items():
            env_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    if config_snapshot:
        env_lines.append(f"- Config snapshot: {config_snapshot}")
    report_lines.extend(env_lines)

    report_path = report_dir / "pipeline_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Markdown report generated -> %s", report_path)

    pdf_lines = [
        "Partial Discharge Classification Report",
        f"Best model: {best_model_name or 'N/A'}",
    ]
    if isinstance(best_model_metrics, dict):
        f1_val = best_model_metrics.get("f1_mean")
        roc_val = best_model_metrics.get("roc_auc_mean")
        if isinstance(f1_val, (int, float)):
            pdf_lines.append(f"F1 (mean): {float(f1_val):.3f}")
        if isinstance(roc_val, (int, float)):
            pdf_lines.append(f"ROC-AUC (mean): {float(roc_val):.3f}")
    if class_balance:
        pdf_lines.append("Class distribution:")
        for label, count in class_balance.items():
            pdf_lines.append(f"  - {label}: {count}")
    _write_pdf_report(pdf_lines, report_dir / "pipeline_report.pdf", log)

    log.info("Evaluation/reporting stage completed")
    return {
        "num_models": len(figures_info),
        "report": str(report_path),
        "pdf": str((report_dir / "pipeline_report.pdf").resolve()),
        "shap_plot": str(shap_plot_path) if shap_plot_path else None,
    }


