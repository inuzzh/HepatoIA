from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "hepatoia_model.json"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def load_model(model_path: str | Path = MODEL_PATH) -> dict:
    return json.loads(Path(model_path).read_text(encoding="utf-8"))


def compute_derived_inputs(values: dict) -> dict:
    result = dict(values)
    if "TG_HDL_ratio" not in result and {"Triglycerides", "HDL"} <= result.keys():
        result["TG_HDL_ratio"] = result["Triglycerides"] / result["HDL"]
    if "AST_ALT_ratio" not in result and {"AST", "ALT"} <= result.keys():
        result["AST_ALT_ratio"] = result["AST"] / result["ALT"] if result["ALT"] else math.nan
    if "TyG_index" not in result and {"Triglycerides", "Glucose"} <= result.keys():
        result["TyG_index"] = math.log(result["Triglycerides"] * result["Glucose"] / 2)
    if "HDL_BMI_ratio" not in result and {"HDL", "BMI"} <= result.keys():
        result["HDL_BMI_ratio"] = result["HDL"] / result["BMI"] if result["BMI"] else math.nan
    if "NHHR" not in result and {"TotalCholesterol", "HDL"} <= result.keys():
        result["NHHR"] = (result["TotalCholesterol"] - result["HDL"]) / result["HDL"] if result["HDL"] else math.nan
    return result


def _transform_one(values: dict, model: dict) -> tuple[np.ndarray, list[str]]:
    values = compute_derived_inputs(values)
    prep = model["preprocessor"]
    row = {}
    missing = []
    for feature in prep["features"]:
        raw_value = values.get(feature)
        if raw_value is None or pd.isna(raw_value):
            raw_value = prep["medians"][feature]
            missing.append(feature)
        if feature in prep["log1p_features"]:
            raw_value = math.log1p(max(0.0, float(raw_value)))
        row[feature] = (float(raw_value) - prep["means"][feature]) / prep["stds"][feature]
    return np.array([row[feature] for feature in prep["features"]], dtype=float), missing


def predict(values: dict, model: dict | None = None) -> dict:
    model = model or load_model()
    x, imputed_features = _transform_one(values, model)
    weights = np.array(model["weights"], dtype=float)  # (n_features, n_classes)
    bias = np.array(model["bias"], dtype=float)  # (n_classes,)
    raw_logits = bias + x @ weights

    calibrator = model["calibrator"]
    calibrated_logits = raw_logits / calibrator["temperature"]
    probabilities = _softmax(calibrated_logits)

    class_labels = model["target"]["classes"]
    predicted_idx = int(np.argmax(probabilities))
    probability_by_class = {
        label: round(float(probabilities[i]), 4) for i, label in enumerate(class_labels)
    }

    # Per-class feature contributions in log-odds space, same idea as the
    # binary model's explanation, reported for the predicted class.
    contributions = []
    for feature, z_value, weight_row in zip(model["preprocessor"]["features"], x, weights):
        contribution = float(z_value * weight_row[predicted_idx])
        contributions.append(
            {
                "feature": feature,
                "contribution_log_odds": contribution,
                "effect": "raises_risk" if contribution > 0 else "lowers_risk",
            }
        )
    contributions.sort(key=lambda item: abs(item["contribution_log_odds"]), reverse=True)

    return {
        "model_version": model["model_version"],
        "severity_classification": class_labels[predicted_idx],
        "probability_by_class": probability_by_class,
        "top_contributors": contributions[:5],
        "imputed_features": imputed_features,
        "intended_use": "Clinical decision support for non-invasive severity screening (Normal / Leve-Moderada / Severa).",
    }


if __name__ == "__main__":
    example = {
        "Age": 49,
        "Waist_cm": 120.4,
        "GGT": 12,
        "Uric_Acid": 5.0,
        "AST": 14,
        "ALT": 8,
        "Triglycerides": 101,
        "Glucose": 95,
        "HDL": 33,
        "BMI": 29.7,
        "TotalCholesterol": 180,
    }
    print(json.dumps(predict(example), indent=2, ensure_ascii=False))
