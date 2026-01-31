"""
STEP 5: Reporting & Experiment Management for Partial Discharge (PD) Classification

This script generates comprehensive visualizations and reports based on model training results:
- Model performance comparison graphs (F1, Accuracy, ROC-AUC, Precision, Recall)
- Track comparison visualizations
- Training time analysis
- Feature count vs performance analysis
- Comprehensive markdown report with all visualizations

Inputs:
- outputs/step4_model_training/cv_results.csv
- outputs/step4_model_training/cv_metrics_detailed.csv
- outputs/step4_model_training/cv_metrics_summary.csv
- outputs/step4_model_training/track_summary.csv
- outputs/step4_model_training/performance_metrics.csv
- outputs/step4_model_training/best_model.json

Outputs:
- outputs/step5_reporting/visualizations/*.png (all graphs)
- outputs/step5_reporting/model_performance_report.md
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 10


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_results(base_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all training result files.
    
    Args:
        base_path: Base path to outputs directory
        
    Returns:
        Dictionary of DataFrames with training results
    """
    training_dir = base_path / "outputs" / "step4_model_training"
    
    results = {}
    
    # Load detailed metrics
    metrics_path = training_dir / "cv_metrics_detailed.csv"
    if metrics_path.exists():
        results["detailed_metrics"] = pd.read_csv(metrics_path)
        logger.info(f"Loaded detailed metrics: {len(results['detailed_metrics'])} models")
    else:
        logger.warning(f"Detailed metrics not found: {metrics_path}")
        results["detailed_metrics"] = pd.DataFrame()
    
    # Load CV results
    cv_results_path = training_dir / "cv_results.csv"
    if cv_results_path.exists():
        results["cv_results"] = pd.read_csv(cv_results_path)
        logger.info(f"Loaded CV results: {len(results['cv_results'])} models")
    else:
        logger.warning(f"CV results not found: {cv_results_path}")
        results["cv_results"] = pd.DataFrame()
    
    # Load performance metrics
    perf_path = training_dir / "performance_metrics.csv"
    if perf_path.exists():
        results["performance"] = pd.read_csv(perf_path)
        logger.info(f"Loaded performance metrics: {len(results['performance'])} models")
    else:
        logger.warning(f"Performance metrics not found: {perf_path}")
        results["performance"] = pd.DataFrame()
    
    # Load track summary
    track_summary_path = training_dir / "track_summary.csv"
    if track_summary_path.exists():
        results["track_summary"] = pd.read_csv(track_summary_path)
        logger.info("Loaded track summary")
    else:
        logger.warning(f"Track summary not found: {track_summary_path}")
        results["track_summary"] = pd.DataFrame()
    
    # Load best model info
    best_model_path = training_dir / "best_model.json"
    if best_model_path.exists():
        with open(best_model_path, 'r') as f:
            results["best_model"] = json.load(f)
        logger.info("Loaded best model info")
    else:
        logger.warning(f"Best model info not found: {best_model_path}")
        results["best_model"] = {}
    
    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_model_performance_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
    metric: str = "F1_Mean"
) -> None:
    """
    Create bar chart comparing model performance by metric.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_path: Path to save the plot
        metric: Metric to plot (F1_Mean, Accuracy_Mean, ROC_AUC_Mean, etc.)
    """
    if metrics_df.empty or metric not in metrics_df.columns:
        logger.warning(f"Cannot plot {metric}: data not available")
        return
    
    # Sort by metric value
    df_sorted = metrics_df.sort_values(metric, ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get error bars if available
    std_col = metric.replace("_Mean", "_Std")
    has_error = std_col in df_sorted.columns
    
    # Create bars
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(
        y_pos,
        df_sorted[metric],
        xerr=df_sorted[std_col] if has_error else None,
        capsize=5,
        alpha=0.8,
        color=sns.color_palette("husl", len(df_sorted))
    )
    
    # Add labels
    labels = [f"{row['Model']} ({row['Track']})" for _, row in df_sorted.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace("_", " "), fontsize=12, fontweight='bold')
    ax.set_title(f"Model Performance Comparison - {metric.replace('_', ' ')}", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', ha='left' if width > 0 else 'right', 
               va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance comparison plot: {output_path}")


def plot_metric_comparison_grid(
    metrics_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Create a grid of comparison plots for multiple metrics.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_path: Path to save the plot
    """
    if metrics_df.empty:
        logger.warning("Cannot plot metric grid: data not available")
        return
    
    metrics_to_plot = ["F1_Mean", "Accuracy_Mean", "ROC_AUC_Mean", "Precision_Mean", "Recall_Mean"]
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    if not available_metrics:
        logger.warning("No metrics available for grid plot")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    df_sorted = metrics_df.sort_values("F1_Mean" if "F1_Mean" in metrics_df.columns else available_metrics[0], 
                                       ascending=True)
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        std_col = metric.replace("_Mean", "_Std")
        has_error = std_col in df_sorted.columns
        
        y_pos = np.arange(len(df_sorted))
        bars = ax.barh(
            y_pos,
            df_sorted[metric],
            xerr=df_sorted[std_col] if has_error else None,
            capsize=3,
            alpha=0.8
        )
        
        labels = [f"{row['Model']}\n({row['Track']})" for _, row in df_sorted.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(metric.replace("_", " "), fontsize=10)
        ax.set_title(metric.replace("_", " "), fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, val in zip(bars, df_sorted[metric]):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if width > 0 else 'right', 
                   va='center', fontsize=7)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metric comparison grid: {output_path}")


def plot_track_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Compare performance across tracks.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_path: Path to save the plot
    """
    if metrics_df.empty or "Track" not in metrics_df.columns:
        logger.warning("Cannot plot track comparison: data not available")
        return
    
    # Group by track and calculate statistics
    track_stats = metrics_df.groupby("Track").agg({
        "F1_Mean": ["mean", "std", "max"],
        "Accuracy_Mean": ["mean", "std", "max"],
        "ROC_AUC_Mean": ["mean", "std", "max"]
    }).round(4)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ["F1_Mean", "Accuracy_Mean", "ROC_AUC_Mean"]
    metric_labels = ["F1 Score", "Accuracy", "ROC-AUC"]
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        if metric not in metrics_df.columns:
            ax.axis('off')
            continue
        
        tracks = track_stats.index.tolist()
        means = track_stats[(metric, "mean")].values
        stds = track_stats[(metric, "std")].values
        maxs = track_stats[(metric, "max")].values
        
        x = np.arange(len(tracks))
        width = 0.25
        
        ax.bar(x - width, means, width, label='Mean', alpha=0.8, yerr=stds, capsize=5)
        ax.bar(x, maxs, width, label='Max', alpha=0.8)
        
        # Add individual model points
        for track in tracks:
            track_data = metrics_df[metrics_df["Track"] == track][metric]
            track_idx = tracks.index(track)
            ax.scatter([track_idx] * len(track_data), track_data, 
                      s=50, alpha=0.5, color='red', zorder=5)
        
        ax.set_xlabel("Track", fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f"{label} by Track", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tracks)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved track comparison plot: {output_path}")


def plot_training_time_analysis(
    perf_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Analyze and visualize training time vs performance.
    
    Args:
        perf_df: DataFrame with performance metrics including training time
        output_path: Path to save the plot
    """
    if perf_df.empty or "Training_Time_Minutes" not in perf_df.columns:
        logger.warning("Cannot plot training time analysis: data not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training time vs F1 score
    ax1 = axes[0]
    for track in perf_df["Track"].unique():
        track_data = perf_df[perf_df["Track"] == track]
        ax1.scatter(
            track_data["Training_Time_Minutes"],
            track_data["Best_CV_F1_Score"],
            label=track,
            s=100,
            alpha=0.7
        )
        # Add model labels
        for _, row in track_data.iterrows():
            ax1.annotate(
                row["Model"],
                (row["Training_Time_Minutes"], row["Best_CV_F1_Score"]),
                fontsize=8,
                alpha=0.7
            )
    
    ax1.set_xlabel("Training Time (minutes)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("F1 Score", fontsize=11, fontweight='bold')
    ax1.set_title("Training Time vs Performance", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Training time by model type
    ax2 = axes[1]
    model_order = perf_df.groupby("Model")["Training_Time_Minutes"].mean().sort_values(ascending=False).index
    perf_df_sorted = perf_df.sort_values("Training_Time_Minutes", ascending=False)
    
    sns.barplot(
        data=perf_df_sorted,
        x="Model",
        y="Training_Time_Minutes",
        hue="Track",
        ax=ax2,
        order=model_order
    )
    ax2.set_xlabel("Model", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Training Time (minutes)", fontsize=11, fontweight='bold')
    ax2.set_title("Training Time by Model", fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training time analysis: {output_path}")


def plot_feature_count_analysis(
    perf_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Analyze relationship between feature count and performance.
    
    Args:
        perf_df: DataFrame with performance metrics including feature counts
        output_path: Path to save the plot
    """
    if perf_df.empty or "N_Features" not in perf_df.columns:
        logger.warning("Cannot plot feature count analysis: data not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Feature count vs F1 score
    ax1 = axes[0]
    for track in perf_df["Track"].unique():
        track_data = perf_df[perf_df["Track"] == track]
        ax1.scatter(
            track_data["N_Features"],
            track_data["Best_CV_F1_Score"],
            label=track,
            s=100,
            alpha=0.7
        )
        # Add model labels
        for _, row in track_data.iterrows():
            ax1.annotate(
                row["Model"],
                (row["N_Features"], row["Best_CV_F1_Score"]),
                fontsize=8,
                alpha=0.7
            )
    
    ax1.set_xlabel("Number of Features", fontsize=11, fontweight='bold')
    ax1.set_ylabel("F1 Score", fontsize=11, fontweight='bold')
    ax1.set_title("Feature Count vs Performance", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Feature count distribution by track
    ax2 = axes[1]
    track_features = perf_df.groupby("Track")["N_Features"].unique()
    for track, features in track_features.items():
        ax2.barh([track], [features[0]], alpha=0.7, label=track)
        ax2.text(features[0], track, f"{features[0]} features", 
                va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel("Number of Features", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Track", fontsize=11, fontweight='bold')
    ax2.set_title("Feature Count by Track", fontsize=12, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature count analysis: {output_path}")


def plot_model_ranking(
    metrics_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Create a comprehensive ranking visualization.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_path: Path to save the plot
    """
    if metrics_df.empty:
        logger.warning("Cannot plot model ranking: data not available")
        return
    
    # Create composite score (normalized F1, Accuracy, ROC-AUC)
    metrics_to_rank = []
    if "F1_Mean" in metrics_df.columns:
        metrics_to_rank.append("F1_Mean")
    if "Accuracy_Mean" in metrics_df.columns:
        metrics_to_rank.append("Accuracy_Mean")
    if "ROC_AUC_Mean" in metrics_df.columns:
        metrics_to_rank.append("ROC_AUC_Mean")
    
    if not metrics_to_rank:
        logger.warning("No metrics available for ranking")
        return
    
    # Normalize and combine
    df_rank = metrics_df.copy()
    for metric in metrics_to_rank:
        df_rank[f"{metric}_norm"] = (df_rank[metric] - df_rank[metric].min()) / \
                                    (df_rank[metric].max() - df_rank[metric].min())
    
    df_rank["Composite_Score"] = df_rank[[f"{m}_norm" for m in metrics_to_rank]].mean(axis=1)
    df_rank = df_rank.sort_values("Composite_Score", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(df_rank))
    bars = ax.barh(y_pos, df_rank["Composite_Score"], alpha=0.8,
                   color=sns.color_palette("viridis", len(df_rank)))
    
    labels = [f"{row['Model']} ({row['Track']})" for _, row in df_rank.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Composite Score (Normalized)", fontsize=12, fontweight='bold')
    ax.set_title("Model Ranking by Composite Score", fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, score in zip(bars, df_rank["Composite_Score"]):
        ax.text(score, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', ha='left', va='center', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model ranking plot: {output_path}")


def plot_cv_score_distribution(
    cv_results_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Visualize CV score distributions across models.
    
    Args:
        cv_results_df: DataFrame with CV results
        output_path: Path to save the plot
    """
    if cv_results_df.empty or "best_cv_score" not in cv_results_df.columns:
        logger.warning("Cannot plot CV score distribution: data not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Box plot by track
    ax1 = axes[0]
    tracks = cv_results_df["track"].unique()
    data_to_plot = [cv_results_df[cv_results_df["track"] == track]["best_cv_score"].values 
                    for track in tracks]
    
    bp = ax1.boxplot(data_to_plot, labels=tracks, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    
    ax1.set_ylabel("Best CV Score (F1)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Track", fontsize=11, fontweight='bold')
    ax1.set_title("CV Score Distribution by Track", fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Violin plot by model
    ax2 = axes[1]
    if "model_name" in cv_results_df.columns:
        model_order = cv_results_df.groupby("model_name")["best_cv_score"].mean().sort_values(ascending=False).index
        sns.violinplot(
            data=cv_results_df,
            x="model_name",
            y="best_cv_score",
            order=model_order,
            ax=ax2
        )
        ax2.set_ylabel("Best CV Score (F1)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Model", fontsize=11, fontweight='bold')
        ax2.set_title("CV Score Distribution by Model", fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CV score distribution plot: {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    results: Dict[str, Any],
    output_dir: Path,
    viz_dir: Path
) -> Path:
    """
    Generate comprehensive markdown report with all visualizations.
    
    Args:
        results: Dictionary of loaded results
        output_dir: Output directory
        viz_dir: Directory with visualizations
        
    Returns:
        Path to generated report
    """
    report_path = output_dir / "model_performance_report.md"
    
    lines = [
        "# Model Performance Report - Partial Discharge Classification",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Add best model info
    if results.get("best_model"):
        best = results["best_model"]
        lines.extend([
            f"**Best Model**: {best.get('model_name', 'N/A')} ({best.get('track', 'N/A')})",
            f"- Best CV F1 Score: {best.get('best_cv_score', 0):.4f}",
            f"- Number of Features: {best.get('n_features', 'N/A')}",
            f"- Training Time: {best.get('training_time_seconds', 0) / 60:.2f} minutes",
            ""
        ])
    
    # Add track summary
    if not results.get("track_summary", pd.DataFrame()).empty:
        lines.extend([
            "## Track Summary",
            "",
            "| Track | Best F1 (Max) | Mean F1 | Std F1 | Total Training Time (min) |",
            "|-------|----------------|---------|--------|---------------------------|"
        ])
        track_df = results["track_summary"]
        for _, row in track_df.iterrows():
            if "Track" in row.index:
                track = row["Track"]
                max_f1 = row.get("Best_CV_F1_Score", {}).get("max", "N/A") if isinstance(row.get("Best_CV_F1_Score"), dict) else "N/A"
                mean_f1 = row.get("Best_CV_F1_Score", {}).get("mean", "N/A") if isinstance(row.get("Best_CV_F1_Score"), dict) else "N/A"
                std_f1 = row.get("Best_CV_F1_Score", {}).get("std", "N/A") if isinstance(row.get("Best_CV_F1_Score"), dict) else "N/A"
                time_min = row.get("Training_Time_Minutes", {}).get("sum", "N/A") if isinstance(row.get("Training_Time_Minutes"), dict) else "N/A"
                lines.append(f"| {track} | {max_f1} | {mean_f1} | {std_f1} | {time_min} |")
        lines.append("")
    
    # Add detailed metrics table
    if not results.get("detailed_metrics", pd.DataFrame()).empty:
        lines.extend([
            "## Detailed Model Performance",
            "",
            "| Rank | Track | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
            "|------|-------|-------|----------|-----------|--------|-----|---------|"
        ])
        metrics_df = results["detailed_metrics"]
        for _, row in metrics_df.iterrows():
            lines.append(
                f"| {row.get('Rank', 'N/A')} | {row.get('Track', 'N/A')} | {row.get('Model', 'N/A')} | "
                f"{row.get('Accuracy_Mean', 0):.4f} ± {row.get('Accuracy_Std', 0):.4f} | "
                f"{row.get('Precision_Mean', 0):.4f} ± {row.get('Precision_Std', 0):.4f} | "
                f"{row.get('Recall_Mean', 0):.4f} ± {row.get('Recall_Std', 0):.4f} | "
                f"{row.get('F1_Mean', 0):.4f} ± {row.get('F1_Std', 0):.4f} | "
                f"{row.get('ROC_AUC_Mean', 0):.4f} ± {row.get('ROC_AUC_Std', 0):.4f} |"
            )
        lines.append("")
    
    # Add visualizations section
    lines.extend([
        "## Visualizations",
        ""
    ])
    
    # List all visualization files
    viz_files = sorted(viz_dir.glob("*.png"))
    viz_categories = {
        "Performance Comparison": ["model_performance_comparison", "metric_comparison_grid"],
        "Track Analysis": ["track_comparison"],
        "Training Analysis": ["training_time_analysis", "feature_count_analysis"],
        "Ranking": ["model_ranking"],
        "CV Analysis": ["cv_score_distribution"]
    }
    
    for category, patterns in viz_categories.items():
        lines.append(f"### {category}")
        lines.append("")
        for pattern in patterns:
            matching_files = [f for f in viz_files if pattern in f.stem]
            for viz_file in matching_files:
                rel_path = viz_file.relative_to(output_dir)
                lines.append(f"![{viz_file.stem}]({rel_path.as_posix()})")
                lines.append("")
    
    # Add any remaining visualizations
    remaining = [f for f in viz_files if not any(p in f.stem for p in sum(viz_categories.values(), []))]
    if remaining:
        lines.append("### Additional Visualizations")
        lines.append("")
        for viz_file in remaining:
            rel_path = viz_file.relative_to(output_dir)
            lines.append(f"![{viz_file.stem}]({rel_path.as_posix()})")
            lines.append("")
    
    # Add conclusions
    lines.extend([
        "## Conclusions",
        "",
        "### Key Findings:",
        ""
    ])
    
    if not results.get("detailed_metrics", pd.DataFrame()).empty:
        metrics_df = results["detailed_metrics"]
        best_model = metrics_df.loc[metrics_df["F1_Mean"].idxmax()] if "F1_Mean" in metrics_df.columns else None
        if best_model is not None:
            lines.extend([
                f"1. **Best Performing Model**: {best_model.get('Model', 'N/A')} on {best_model.get('Track', 'N/A')} "
                f"with F1 score of {best_model.get('F1_Mean', 0):.4f}",
                ""
            ])
    
    lines.extend([
        "2. Track comparison shows the relative performance of different feature selection strategies.",
        "",
        "3. Training time analysis reveals the computational cost vs performance trade-offs.",
        "",
        "4. Feature count analysis shows the relationship between model complexity and performance.",
        "",
        "---",
        "",
        f"**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Generated markdown report: {report_path}")
    return report_path


# =============================================================================
# MAIN REPORTING PIPELINE
# =============================================================================

def run_reporting(
    base_path: Path,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run the complete reporting pipeline.
    
    Args:
        base_path: Base path to project
        output_dir: Output directory (defaults to outputs/step5_reporting)
        
    Returns:
        Dictionary with report paths and summary
    """
    if output_dir is None:
        output_dir = base_path / "outputs" / "step5_reporting"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("STEP 5: REPORTING & EXPERIMENT MANAGEMENT")
    logger.info("="*70)
    
    # Load results
    logger.info("Loading training results...")
    results = load_training_results(base_path)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Model performance comparison
    if not results.get("detailed_metrics", pd.DataFrame()).empty:
        metrics_df = results["detailed_metrics"]
        
        # F1 Score comparison
        plot_model_performance_comparison(
            metrics_df,
            viz_dir / "model_performance_comparison_f1.png",
            "F1_Mean"
        )
        
        # Accuracy comparison
        plot_model_performance_comparison(
            metrics_df,
            viz_dir / "model_performance_comparison_accuracy.png",
            "Accuracy_Mean"
        )
        
        # ROC-AUC comparison
        plot_model_performance_comparison(
            metrics_df,
            viz_dir / "model_performance_comparison_roc_auc.png",
            "ROC_AUC_Mean"
        )
        
        # Metric comparison grid
        plot_metric_comparison_grid(
            metrics_df,
            viz_dir / "metric_comparison_grid.png"
        )
        
        # Track comparison
        plot_track_comparison(
            metrics_df,
            viz_dir / "track_comparison.png"
        )
        
        # Model ranking
        plot_model_ranking(
            metrics_df,
            viz_dir / "model_ranking.png"
        )
    
    # Training time analysis
    if not results.get("performance", pd.DataFrame()).empty:
        plot_training_time_analysis(
            results["performance"],
            viz_dir / "training_time_analysis.png"
        )
        
        plot_feature_count_analysis(
            results["performance"],
            viz_dir / "feature_count_analysis.png"
        )
    
    # CV score distribution
    if not results.get("cv_results", pd.DataFrame()).empty:
        plot_cv_score_distribution(
            results["cv_results"],
            viz_dir / "cv_score_distribution.png"
        )
    
    # Generate report
    logger.info("Generating markdown report...")
    report_path = generate_markdown_report(results, output_dir, viz_dir)
    
    # Print summary
    logger.info("="*70)
    logger.info("STEP 5: REPORTING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"  - Visualizations: {viz_dir}")
    logger.info(f"  - Report: {report_path}")
    logger.info("="*70)
    
    return {
        "output_dir": str(output_dir),
        "visualizations_dir": str(viz_dir),
        "report_path": str(report_path),
        "num_visualizations": len(list(viz_dir.glob("*.png")))
    }


def main():
    """Main entry point for reporting."""
    base_path = Path(__file__).parent
    
    results = run_reporting(base_path)
    
    print("\n" + "="*70)
    print("REPORTING COMPLETE")
    print("="*70)
    print(f"Report: {results['report_path']}")
    print(f"Visualizations: {results['num_visualizations']} graphs generated")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
