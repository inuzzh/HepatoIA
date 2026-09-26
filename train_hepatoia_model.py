from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(__file__).resolve().parent / "artifacts" / "hepatoia_final_dataset_v2.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"

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
TARGET_COLUMN = "Steatosis_Grade_CAP"
MODEL_VERSION = "hepatoia-severity3-softmax-v0.1"

# 3-tier severity grouping: S1_Leve and S2_Moderada are merged so the
# minority middle grades don't leave a severely underrepresented class
# (raw grade counts were S0=3333, S1=848, S2=610, S3=2976).
CLASS_LABELS = ["Normal", "Leve_Moderada", "Severa"]
GRADE_TO_CLASS = {
    "S0_Normal": 0,
    "S1_Leve": 1,
    "S2_Moderada": 1,
    "S3_Severa": 2,
}
N_CLASSES = len(CLASS_LABELS)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def stratified_split(
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def prepare_target(df: pd.DataFrame) -> np.ndarray:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    mapped = df[TARGET_COLUMN].map(GRADE_TO_CLASS)
    if mapped.isna().any():
        unknown = sorted(set(df[TARGET_COLUMN].unique()) - set(GRADE_TO_CLASS))
        raise ValueError(f"Unmapped target values: {unknown}")
    return mapped.to_numpy(dtype=int)


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


def train_softmax_regression(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 0.03,
    lr: float = 0.06,
    epochs: int = 7000,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_features = x.shape
    bias = np.zeros(N_CLASSES, dtype=float)
    weights = np.zeros((n_features, N_CLASSES), dtype=float)
    y_onehot = np.eye(N_CLASSES)[y]

    # Same idea as the binary pos/neg weighting before, generalized to
    # N classes: each class contributes equally to the gradient regardless
    # of how many rows it has.
    counts = np.maximum(np.bincount(y, minlength=N_CLASSES), 1)
    class_weight = n_rows / (N_CLASSES * counts)
    sample_weights = class_weight[y]

    for _ in range(epochs):
        logits = bias + x @ weights
        probs = softmax(logits)
        error = (probs - y_onehot) * sample_weights[:, None]
        grad_bias = np.mean(error, axis=0)
        grad_weights = (x.T @ error) / n_rows + l2 * weights
        bias -= lr * grad_bias
        weights -= lr * grad_weights
    return bias, weights


def fit_temperature_calibrator(logits: np.ndarray, y: np.ndarray, epochs: int = 3000, lr: float = 0.05) -> dict:
    """Multiclass generalization of the binary Platt scaling used before:
    a single scalar temperature that rescales all logits before softmax,
    fit by gradient descent on the calibration split.
    """
    y_onehot = np.eye(N_CLASSES)[y]
    temperature = 1.0
    for _ in range(epochs):
        scaled = logits / temperature
        probs = softmax(scaled)
        error = probs - y_onehot
        grad_temperature = float(np.mean(np.sum(error * (-logits / (temperature ** 2)), axis=1)))
        temperature -= lr * grad_temperature
        temperature = max(temperature, 1e-3)
    return {"method": "temperature_scaling", "temperature": temperature}


def predict_proba(x: np.ndarray, model: dict, calibrated: bool = True) -> np.ndarray:
    logits = model["bias"] + x @ np.array(model["weights"], dtype=float)
    if calibrated:
        logits = logits / model["calibrator"]["temperature"]
    return softmax(logits)


def auc_score(y_binary: np.ndarray, probs: np.ndarray) -> float:
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    pos = y_binary == 1
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(~pos))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    """Same metric family as the binary model (AUC, accuracy, sensitivity,
    specificity, precision, F1, Brier score, confusion matrix), extended to
    3 classes the standard way: one-vs-rest per class, macro-averaged.
    Decision rule is argmax over the calibrated class probabilities
    (there's no single Youden threshold once there are 3 classes).
    """
    pred = np.argmax(probs, axis=1)
    y_onehot = np.eye(N_CLASSES)[y]

    per_class = {}
    aucs, precisions, recalls, specificities, f1s = [], [], [], [], []
    for c in range(N_CLASSES):
        y_bin = (y == c).astype(int)
        pred_bin = (pred == c).astype(int)
        tp = int(np.sum((pred_bin == 1) & (y_bin == 1)))
        tn = int(np.sum((pred_bin == 0) & (y_bin == 0)))
        fp = int(np.sum((pred_bin == 1) & (y_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (y_bin == 1)))
        precision = tp / max(1, tp + fp)
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
        auc = auc_score(y_bin, probs[:, c])

        aucs.append(auc)
        precisions.append(precision)
        recalls.append(sensitivity)
        specificities.append(specificity)
        f1s.append(f1)
        per_class[CLASS_LABELS[c]] = {
            "auc": auc,
            "precision": precision,
            "sensitivity_recall": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    confusion_matrix = [
        [int(np.sum((y == true_c) & (pred == pred_c))) for pred_c in range(N_CLASSES)]
        for true_c in range(N_CLASSES)
    ]

    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_auc": float(np.nanmean(aucs)),
        "macro_precision": float(np.mean(precisions)),
        "macro_sensitivity_recall": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_f1": float(np.mean(f1s)),
        "brier_score": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASS_LABELS, "matrix": confusion_matrix},
    }


def coefficient_table(model: dict) -> list[dict]:
    weights = np.array(model["weights"], dtype=float)  # (n_features, n_classes)
    rows = []
    for i, feature in enumerate(model["preprocessor"]["features"]):
        per_class_weight = {CLASS_LABELS[c]: float(weights[i, c]) for c in range(N_CLASSES)}
        strongest_class = max(per_class_weight, key=lambda c: abs(per_class_weight[c]))
        rows.append(
            {
                "feature": feature,
                "coefficients_by_class": per_class_weight,
                "strongest_class": strongest_class,
                "absolute_strength": abs(per_class_weight[strongest_class]),
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
    parser = argparse.ArgumentParser(description="Train HepatoIA 3-tier severity classification model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[df[TARGET_COLUMN].notna()].copy()
    y = prepare_target(df)

    train_cal_idx, test_idx = stratified_split(y, test_size=0.20, seed=args.seed)
    train_cal_y = y[train_cal_idx]
    train_rel_idx, cal_rel_idx = stratified_split(train_cal_y, test_size=0.25, seed=args.seed + 1)
    train_idx = train_cal_idx[train_rel_idx]
    cal_idx = train_cal_idx[cal_rel_idx]

    preprocessor = fit_preprocessor(df.iloc[train_idx])
    x_train = transform(df.iloc[train_idx], preprocessor)
    x_cal = transform(df.iloc[cal_idx], preprocessor)
    x_test = transform(df.iloc[test_idx], preprocessor)

    bias, weights = train_softmax_regression(x_train, y[train_idx])
    raw_cal_logits = bias + x_cal @ weights
    calibrator = fit_temperature_calibrator(raw_cal_logits, y[cal_idx])

    model = {
        "model_version": MODEL_VERSION,
        "model_type": "regularized_multinomial_logistic_regression_numpy",
        "target": {
            "column": TARGET_COLUMN,
            "classes": CLASS_LABELS,
            "grade_to_class": GRADE_TO_CLASS,
            "definition": "3-tier severity: Normal (S0), Leve_Moderada (S1+S2), Severa (S3).",
        },
        "forbidden_predictors": sorted(FORBIDDEN_PREDICTORS),
        "preprocessor": preprocessor,
        "bias": bias.tolist(),
        "weights": weights.tolist(),
        "calibrator": calibrator,
    }
    model["coefficient_table"] = coefficient_table(model)

    train_probs = predict_proba(x_train, model, calibrated=True)
    cal_probs = predict_proba(x_cal, model, calibrated=True)
    test_probs = predict_proba(x_test, model, calibrated=True)

    report = {
        "dataset": str(args.dataset),
        "n_rows": int(len(df)),
        "class_counts": {label: int(np.sum(y == c)) for c, label in enumerate(CLASS_LABELS)},
        "splits": {
            "train": int(len(train_idx)),
            "calibration": int(len(cal_idx)),
            "test": int(len(test_idx)),
        },
        "metrics": {
            "train": metrics(y[train_idx], train_probs),
            "calibration": metrics(y[cal_idx], cal_probs),
            "test": metrics(y[test_idx], test_probs),
        },
        "coefficient_table": model["coefficient_table"],
        "clinical_guardrails": [
            "This is non-invasive clinical decision support, not an autonomous diagnosis.",
            "CAP_dBm, LSM_kPa, steatosis grade, fibrosis grade, and SEQN are blocked as predictors.",
            "The output is a calibrated severity classification: Normal / Leve_Moderada / Severa.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hepatoia_model.json"
    report_path = args.output_dir / "training_report.json"
    model_path.write_text(json.dumps(to_builtin(model), indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(to_builtin(report), indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(json.dumps(to_builtin(report["metrics"]["test"]), indent=2))


if __name__ == "__main__":
    main()
