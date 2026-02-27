"""
Feature Expansion Module for Partial Discharge Classification Pipeline

This module implements the feature expansion phase of the PD classification pipeline,
responsible for creating additional features through pairwise arithmetic operations,
polynomial transformations, and feature-engine driven mathematical combinations.
It enhances the feature space to capture non-linear relationships and feature
interactions that may improve model performance.

Step-by-Step Process:
1. Input Data Loading:
   - Reads Parquet files from ``features/2_feature_engineering/classic_stats/``
   - Loads feature matrices for each station
   - Identifies and preserves metadata columns (station_id, measurement_id, etc.)

2. Feature Matrix Preparation:
   - Separates feature columns from metadata columns
   - Creates clean feature matrix (X) for polynomial transformation
   - Maintains traceability to original data through metadata preservation

3. Pairwise Arithmetic Combinations:
   - Addition, subtraction, division, multiplication, and absolute difference
   - Safe guards against division by near-zero values
   - Produces deterministic feature names for reproducibility

4. Polynomial Feature Generation:
   - Applies ``sklearn.preprocessing.PolynomialFeatures`` with degree=2
   - Generates interaction terms between all feature pairs (interaction only)
   - Excludes bias term (``include_bias=False``) for cleaner feature space

5. Feature-Engine Mathematical Combination:
   - Uses ``feature_engine.creation.MathFeatures`` for additional aggregations
   - Currently enabled operations: sum, product, and mean per feature pair
   - Prefixes the generated features with ``mc_`` to avoid name collisions

4. Feature Name Generation:
   - Creates descriptive feature names for expanded features
   - Uses ``get_feature_names_out()`` for consistent naming
   - Maintains interpretability of generated features

6. Feature Space Expansion:
   - Original features: 9 basic statistics
   - Expanded features include pairwise arithmetic ops (+, −, ×, ÷, |Δ|)
   - Polynomial interactions capture non-linear relationships (×)
   - Feature-engine combinations add aggregated statistics (sum, prod, mean)

7. Output Organization:
   - Reconstructs DataFrame with expanded features
   - Re-attaches metadata columns for traceability

8. File Management:
   - Creates ``features/3_feature_comb_expansion/mathematical_combination/`` directory
   - Saves expanded features as Parquet and CSV files
   - Logs expansion statistics (input vs output feature counts)

Feature Expansion Types:
- Original Features: All 9 basic features preserved
- Arithmetic Combinations: Pairwise sums, differences, ratios, products, absolute differences
- Polynomial Interactions: Pairwise products of original features
- Feature-Engine Combinations: Sum, product, and mean aggregations per feature pair

Configuration Parameters:
- station_id: Optional specific station to process (processes all if None)
- input_path: Path to feature engineering outputs
- output_path: Root directory for expanded feature outputs

Dependencies:
- pandas: Data manipulation and CSV I/O
- sklearn.preprocessing: Polynomial feature generation
- pathlib: Cross-platform path handling

Output Structure:
- ``features/3_feature_comb_expansion/mathematical_combination/combined_features.(parquet|csv)``
- ``features/3_feature_comb_expansion/mathematical_combination/feature_combinations.md``
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from feature_engine.creation import MathFeatures
from sklearn.preprocessing import PolynomialFeatures

from .utils.logger import get_logger


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    mask = np.abs(b) > 1e-9
    np.divide(a, b, out=out, where=mask)
    return out


def run(station_id: Optional[str], input_path: Path, output_path: Path, config: Optional[dict] = None) -> dict:
    """Expand features with arithmetic combinations and polynomial interactions.

    Input: ``features/2_feature_engineering/classic_stats/base_features.parquet``
    Output: ``features/3_feature_comb_expansion/mathematical_combination/combined_features.parquet``
    along with documentation ``feature_combinations.md`` within the same directory.
    """
    log = get_logger(__name__, log_dir=output_path / "reports")
    log.info("Starting feature combination stage")

    base_path = Path("features") / "2_feature_engineering" / "classic_stats" / "base_features.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Base features not found at {base_path}. Run feature engineering first.")
    base_df = pd.read_parquet(base_path)
    if base_df.empty:
        raise ValueError("Base features dataframe is empty; cannot perform combinations.")

    meta_cols = ["station_id", "measurement_id", "faultAnnotation", "signal_length"]
    meta_cols = [c for c in meta_cols if c in base_df.columns]
    feature_cols = [c for c in base_df.columns if c not in meta_cols]
    numeric_df = base_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    combination_df = pd.DataFrame(index=base_df.index)
    documentation: list[dict[str, str]] = []

    def _record(name: str, op: str, desc: str, meaning: str) -> None:
        documentation.append(
            {
                "name": name,
                "operation": op,
                "description": desc,
                "meaning": meaning,
            }
        )

    # Pairwise arithmetic combinations
    for col_i, col_j in combinations(feature_cols, 2):
        add_name = f"{col_i}_plus_{col_j}"
        combination_df[add_name] = (numeric_df[col_i] + numeric_df[col_j]).astype(np.float32)
        _record(add_name, "Addition", f"Sum of {col_i} and {col_j}", f"Combined magnitude of {col_i} and {col_j}")

        sub_name = f"{col_i}_minus_{col_j}"
        combination_df[sub_name] = (numeric_df[col_i] - numeric_df[col_j]).astype(np.float32)
        _record(sub_name, "Subtraction", f"Difference between {col_i} and {col_j} (ordered)", f"Directional contrast from {col_i} to {col_j}")

        div_name = f"{col_i}_div_{col_j}"
        div_values = _safe_divide(numeric_df[col_i].to_numpy(), numeric_df[col_j].to_numpy())
        combination_df[div_name] = div_values.astype(np.float32)
        _record(div_name, "Division", f"Ratio of {col_i} to {col_j}", f"Relative scaling of {col_i} to {col_j}")

        abs_name = f"{col_i}_abs_diff_{col_j}"
        combination_df[abs_name] = (np.abs(numeric_df[col_i] - numeric_df[col_j])).astype(np.float32)
        _record(abs_name, "Absolute Difference", f"Absolute difference between {col_i} and {col_j}", f"Distance between {col_i} and {col_j}")

        prod_name = f"{col_i}_times_{col_j}"
        combination_df[prod_name] = (numeric_df[col_i] * numeric_df[col_j]).astype(np.float32)
        _record(prod_name, "Multiplication", f"Product of {col_i} and {col_j}", f"Interaction energy of {col_i} with {col_j}")

    # Polynomial interactions (pairwise products only)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_data = poly.fit_transform(numeric_df)
    poly_names = poly.get_feature_names_out(feature_cols)
    # Skip the first len(feature_cols) entries (original features)
    interaction_start = len(feature_cols)
    for idx, name in enumerate(poly_names[interaction_start:], start=interaction_start):
        terms = name.split(" ")
        if len(terms) != 2:
            continue
        col_i, col_j = terms
        mult_name = f"{col_i}_mult_{col_j}"
        combination_df[mult_name] = poly_data[:, idx].astype(np.float32)
        _record(mult_name, "Polynomial Interaction", f"Product of {col_i} and {col_j}", f"Second-order interaction between {col_i} and {col_j}")

    # Feature-engine MathFeatures (sum, prod, mean) per pair
    math_operations = ["sum", "prod", "mean"]
    for col_i, col_j in combinations(feature_cols, 2):
        math_combiner = MathFeatures(
            variables=[col_i, col_j],
            func=math_operations,
            new_variables_names=[
                f"sum_{col_i}_{col_j}",
                f"prod_{col_i}_{col_j}",
                f"mean_{col_i}_{col_j}",
            ],
            missing_values="ignore",
            drop_original=False,
        )
        pair_df = numeric_df[[col_i, col_j]].copy()
        mc_df = math_combiner.fit_transform(pair_df)
        new_cols = [c for c in mc_df.columns if c not in pair_df.columns]
        for original_name in new_cols:
            mc_name = f"mc_{original_name}"
            combination_df[mc_name] = mc_df[original_name].astype(np.float32)
            op_type = original_name.split("_", maxsplit=1)[0]
            meaning = {
                "sum": f"Feature-engine aggregate sum of {col_i} and {col_j}",
                "prod": f"Feature-engine aggregate product of {col_i} and {col_j}",
                "mean": f"Feature-engine aggregate mean of {col_i} and {col_j}",
            }.get(op_type, f"feature-engine MathFeatures derived value for {col_i} and {col_j}")
            _record(mc_name, f"feature-engine {op_type}", f"feature-engine MathFeatures generated column ({original_name})", meaning)

    combination_df = combination_df.loc[:, ~combination_df.columns.duplicated()]
    mc_feature_count = sum(1 for col in combination_df.columns if col.startswith("mc_"))

    combined_features = pd.concat([numeric_df, combination_df], axis=1)
    # Ensure metadata columns are at front
    final_df = pd.concat([base_df[meta_cols], combined_features], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    expansion_root = Path("features") / "3_feature_comb_expansion"
    math_dir = expansion_root / "mathematical_combination"
    math_dir.mkdir(parents=True, exist_ok=True)
    combined_path = math_dir / "combined_features.parquet"
    combined_csv = math_dir / "combined_features.csv"
    final_df.to_parquet(combined_path, index=False)
    final_df.to_csv(combined_csv, index=False)
    log.info("Combined feature dataset -> %s", combined_path)

    doc_path = math_dir / "feature_combinations.md"
    with doc_path.open("w", encoding="utf-8") as handle:
        handle.write("| Combination Feature Name | Operation Type | Description | Meaning |\n")
        handle.write("| --- | --- | --- | --- |\n")
        for entry in documentation:
            handle.write(f"| {entry['name']} | {entry['operation']} | {entry['description']} | {entry['meaning']} |\n")
        handle.write("\n**Summary**\n")
        handle.write(f"\n- Base features: {len(feature_cols)}\n")
        handle.write(f"- Combination features: {combination_df.shape[1]}\n")
        handle.write(f"- feature-engine features: {mc_feature_count}\n")
        handle.write(f"- Total features (including base): {final_df.shape[1] - len(meta_cols)}\n")
    log.info("Documented %d combination features -> %s", len(documentation), doc_path)

    summary = {
        "base_features": len(feature_cols),
        "combination_features": combination_df.shape[1],
        "feature_engine_features": mc_feature_count,
        "total_features": final_df.shape[1] - len(meta_cols),
    }
    log.info("Feature combination summary: %s", summary)
    return summary


