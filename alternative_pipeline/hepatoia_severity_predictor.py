"""End-to-end HepatoIA severity pipeline.

Stage 1 (ML): a regularized linear regression estimates CAP (dB/m) from
routine labs and anthropometrics — see train_cap_regression.py.

Stage 2 (rule-based, no ML): the estimated CAP is passed through
severity_rules.py, which applies literature-cited thresholds to produce a
severity label. Stage 2 has no coefficients and nothing to train — every
branch is a clinical cutoff you can look up and verify by hand.

Keeping the two stages separate means the severity labels stay meaningful
even as the regression model gets retrained or improved: the clinical
definition of "severe" doesn't drift with the model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from severity_rules import MetabolicContext, assess_severity

REGRESSION_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "hepatoia_cap_regression.json"


def load_regression_model(model_path: str | Path = REGRESSION_MODEL_PATH) -> dict:
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
        if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
            raw_value = prep["medians"][feature]
            missing.append(feature)
        if feature in prep["log1p_features"]:
            raw_value = math.log1p(max(0.0, float(raw_value)))
        row[feature] = (float(raw_value) - prep["means"][feature]) / prep["stds"][feature]
    return np.array([row[feature] for feature in prep["features"]], dtype=float), missing


def predict_cap(values: dict, model: dict | None = None) -> dict:
    model = model or load_regression_model()
    x, imputed_features = _transform_one(values, model)
    weights = np.array(model["weights"], dtype=float)
    cap_estimate = float(model["bias"] + x @ weights)
    cap_estimate = max(100.0, min(400.0, cap_estimate))  # clamp to the FibroScan CAP sensor's physical range

    contributions = []
    for feature, z_value, weight in zip(model["preprocessor"]["features"], x, weights):
        contribution = float(z_value * weight)
        contributions.append(
            {
                "feature": feature,
                "contribution_dbm": contribution,
                "effect": "raises_cap" if contribution > 0 else "lowers_cap",
            }
        )
    contributions.sort(key=lambda item: abs(item["contribution_dbm"]), reverse=True)

    return {
        "model_version": model["model_version"],
        "cap_estimate_dbm": round(cap_estimate, 1),
        "top_contributors": contributions[:5],
        "imputed_features": imputed_features,
    }


def predict_severity(values: dict, model: dict | None = None) -> dict:
    """Full pipeline: raw clinical values -> estimated CAP (ML) -> severity (rules)."""
    cap_result = predict_cap(values, model)

    metabolic = None
    if any(k in values for k in ("BMI", "has_diabetes", "has_hypertension")):
        metabolic = MetabolicContext(
            bmi=values.get("BMI"),
            has_diabetes=values.get("has_diabetes"),
            has_hypertension=values.get("has_hypertension"),
        )

    assessment = assess_severity(
        cap_dbm=cap_result["cap_estimate_dbm"],
        lsm_kpa=values.get("LSM_kPa"),  # only used if you actually measured it
        metabolic=metabolic,
    )

    return {
        "model_version": cap_result["model_version"],
        "cap_estimate_dbm": cap_result["cap_estimate_dbm"],
        "steatosis_grade": assessment.steatosis_grade,
        "steatosis_presence": assessment.steatosis_presence,
        "fibrosis_grade": assessment.fibrosis_grade,
        "combined_severity": assessment.combined_severity,
        "mafld": assessment.mafld,
        "top_contributors": cap_result["top_contributors"],
        "imputed_features": cap_result["imputed_features"],
        "intended_use": "Clinical decision support for non-invasive severity screening. Not an autonomous diagnosis.",
        "method_note": (
            "cap_estimate_dbm comes from a trained regression model (ML). "
            "steatosis_grade, fibrosis_grade, combined_severity, and mafld come from "
            "fixed clinical cutoffs in severity_rules.py, not from a learned model."
        ),
    }


if __name__ == "__main__":
    example = {
        "Age": 55,
        "Waist_cm": 118,
        "GGT": 95,
        "Uric_Acid": 8.1,
        "AST": 60,
        "ALT": 85,
        "Triglycerides": 320,
        "Glucose": 160,
        "HDL": 28,
        "BMI": 34.5,
        "TotalCholesterol": 210,
        "has_diabetes": False,
        "has_hypertension": True,
    }
    print(json.dumps(predict_severity(example), indent=2, ensure_ascii=False))
