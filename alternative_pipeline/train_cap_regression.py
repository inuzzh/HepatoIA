from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_DIR / "artifacts" / "hepatoia_final_dataset_v2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts"

FEATURES = [
    "Waist_cm",
    "TG_HDL_ratio",
    "GGT",
    "Uric_Acid",
    "AST_ALT_ratio",
    "TyG_index",
    "Age",
    "HDL_BMI_ratio",
    "NHHR",
]

FORBIDDEN_PREDICTORS = {
    "SEQN",
    "CAP_dBm",
    "LSM_kPa",
    "Steatosis_Grade_CAP",
    "Fibrosis_Grade_LSM",
}

LOG1P_FEATURES = {"TG_HDL_ratio", "GGT", "NHHR"}
TARGET_COLUMN = "CAP_dBm"
MODEL_VERSION = "hepatoia-cap-linreg-v0.1"


def stratified_split_continuous(n_rows: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_test = int(round(n_rows * test_size))
    return idx[n_test:], idx[:n_test]


def fit_preprocessor(df: pd.DataFrame) -> dict:
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    leaked = sorted(set(FEATURES) & FORBIDDEN_PREDICTORS)
    if leaked:
        raise ValueError(f"Forbidden leakage predictors requested: {leaked}")

    x = df[FEATURES].astype(float).copy()
    for feature in LOG1P_FEATURES:
        x[feature] = np.log1p(x[feature].clip(lower=0))

    medians = x.median(axis=0)
    x = x.fillna(medians)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0, 1.0)
    return {
        "features": FEATURES,
        "log1p_features": sorted(LOG1P_FEATURES),
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
    }


def transform(df: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    x = df[preprocessor["features"]].astype(float).copy()
    for feature in preprocessor["log1p_features"]:
        x[feature] = np.log1p(x[feature].clip(lower=0))
    x = x.fillna(pd.Series(preprocessor["medians"]))
    means = pd.Series(preprocessor["means"])
    stds = pd.Series(preprocessor["stds"])
    return ((x - means) / stds).to_numpy(dtype=float)


def train_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 0.02,
    lr: float = 0.08,
    epochs: int = 6000,
) -> tuple[float, np.ndarray]:
    n_rows, n_features = x.shape
    bias = float(np.mean(y))
    weights = np.zeros(n_features, dtype=float)

    for _ in range(epochs):
        pred = bias + x @ weights
        error = pred - y
        grad_bias = float(np.mean(error))
        grad_weights = (x.T @ error) / n_rows + l2 * weights
        bias -= lr * grad_bias
        weights -= lr * grad_weights
    return bias, weights


def predict_cap(x: np.ndarray, model: dict) -> np.ndarray:
    return model["bias"] + x @ np.array(model["weights"], dtype=float)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    residuals = y - pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": r2,
        "n": int(len(y)),
    }


def coefficient_table(model: dict) -> list[dict]:
    rows = []
    for feature, weight in zip(model["preprocessor"]["features"], model["weights"]):
        rows.append(
            {
                "feature": feature,
                "standardized_coefficient": weight,
                "direction": "raises_cap" if weight > 0 else "lowers_cap",
                "absolute_strength": abs(weight),
            }
        )
    return sorted(rows, key=lambda item: item["absolute_strength"], reverse=True)


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HepatoIA CAP regression model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[df[TARGET_COLUMN].notna()].reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(float).to_numpy()

    train_idx, test_idx = stratified_split_continuous(len(df), test_size=0.20, seed=args.seed)

    preprocessor = fit_preprocessor(df.iloc[train_idx])
    x_train = transform(df.iloc[train_idx], preprocessor)
    x_test = transform(df.iloc[test_idx], preprocessor)

    bias, weights = train_linear_regression(x_train, y[train_idx])

    model = {
        "model_version": MODEL_VERSION,
        "model_type": "regularized_linear_regression_numpy",
        "target": {
            "column": TARGET_COLUMN,
            "unit": "dB/m",
            "definition": "Controlled Attenuation Parameter (CAP) from VCTE/FibroScan, continuous estimate.",
        },
        "forbidden_predictors": sorted(FORBIDDEN_PREDICTORS),
        "preprocessor": preprocessor,
        "bias": bias,
        "weights": weights.tolist(),
    }
    model["coefficient_table"] = coefficient_table(model)

    train_pred = predict_cap(x_train, model)
    test_pred = predict_cap(x_test, model)

    report = {
        "dataset": str(args.dataset),
        "n_rows": int(len(df)),
        "splits": {"train": int(len(train_idx)), "test": int(len(test_idx))},
        "metrics": {
            "train": regression_metrics(y[train_idx], train_pred),
            "test": regression_metrics(y[test_idx], test_pred),
        },
        "coefficient_table": model["coefficient_table"],
        "clinical_guardrails": [
            "This regression estimates CAP (dB/m), not a diagnosis by itself.",
            "CAP_dBm, LSM_kPa, steatosis grade, fibrosis grade, and SEQN are blocked as predictors.",
            "Combine with the rule-based severity_rules.py classifier to get a clinical severity label.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hepatoia_cap_regression.json"
    report_path = args.output_dir / "cap_regression_report.json"
    model_path.write_text(json.dumps(to_builtin(model), indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(to_builtin(report), indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(json.dumps(to_builtin(report["metrics"]), indent=2))


if __name__ == "__main__":
    main()
