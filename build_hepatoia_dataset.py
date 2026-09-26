from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "artifacts" / "hepatoia_final_dataset_v2.csv"

# Eddowes et al. 2019 (Gastroenterology) VCTE cutoffs, used by NHANES documentation
# for the FibroScan CAP/LSM exam.
CAP_CUTOFFS = [(248, "S0_Normal"), (268, "S1_Leve"), (281, "S2_Moderada")]
CAP_LABEL_ABOVE = "S3_Severa"

LSM_CUTOFFS = [(8.2, "F0_F1"), (9.7, "F2"), (13.6, "F3")]
LSM_LABEL_ABOVE = "F4"

LSM_IQR_RATIO_MAX = 30.0  # % — standard VCTE reliability threshold (IQR/median < 30%)

# Excessive-alcohol exclusion, same definition Lin et al. 2025 (pone.0319851)
# cite: average >=15 drinks/day, or drinking >=3-4 times/week.
ALQ121_EXCESSIVE_CODES = {1, 2, 3}
ALQ130_EXCESSIVE_MIN = 15.0
ALQ130_SENTINELS = {777.0, 999.0}


@dataclass
class CycleSpec:
    """One NHANES release cycle: where its raw files live, what they're
    called, and any column renames vs. the canonical names used below.
    Two cycles can use different filenames and even different variable
    names for the same measurement (e.g. triglycerides was LBXTR in the
    2017-2020 pre-pandemic files and LBXTLG in the 2021-2023 'L' cycle).
    """

    label: str
    raw_dir: Path
    file_names: dict[str, str]  # logical name -> actual filename
    column_aliases: dict[str, str] = field(default_factory=dict)  # canonical -> actual


CYCLES = [
    CycleSpec(
        label="2017-2020_prepandemic",
        raw_dir=PROJECT_DIR / "data" / "raw" / "nhanes_2017_2020",
        file_names={
            "demo": "P_DEMO.xpt",
            "bmx": "P_BMX.xpt",
            "biopro": "P_BIOPRO.xpt",
            "hdl": "P_HDL.xpt",
            "trigly": "P_TRIGLY.xpt",
            "lux": "P_LUX.xpt",
            "alq": "P_ALQ.xpt",
        },
    ),
    CycleSpec(
        label="2021-2023",
        raw_dir=PROJECT_DIR / "data" / "raw" / "nhanes_2021_2023",
        file_names={
            "demo": "DEMO_L.xpt",
            "bmx": "BMX_L.xpt",
            "biopro": "BIOPRO_L.xpt",
            "hdl": "HDL_L.xpt",
            "trigly": "TRIGLY_L.xpt",
            "lux": "LUX_L.xpt",
            "alq": "ALQ_L.xpt",
        },
        column_aliases={"LBXTR": "LBXTLG"},
    ),
]


def _grade(value: float, cutoffs: list[tuple[float, str]], label_above: str) -> str | float:
    if pd.isna(value):
        return np.nan
    for threshold, label in cutoffs:
        if value < threshold:
            return label
    return label_above


def load_xpt(raw_dir: Path, filename: str, columns: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    path = raw_dir / filename
    df = pd.read_sas(path, format="xport")
    actual_columns = [aliases.get(c, c) for c in columns]
    missing = [c for c in actual_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing expected columns: {missing}")
    df = df[actual_columns].copy()
    # Rename aliased columns back to their canonical name so downstream
    # code never has to know which cycle it's looking at.
    rename_map = {aliases[c]: c for c in columns if c in aliases}
    return df.rename(columns=rename_map)


def build_cycle_dataset(cycle: CycleSpec) -> pd.DataFrame:
    demo = load_xpt(cycle.raw_dir, cycle.file_names["demo"], ["SEQN", "RIDAGEYR", "RIDEXPRG"], cycle.column_aliases)
    bmx = load_xpt(cycle.raw_dir, cycle.file_names["bmx"], ["SEQN", "BMXWAIST", "BMXBMI"], cycle.column_aliases)
    biopro = load_xpt(
        cycle.raw_dir,
        cycle.file_names["biopro"],
        ["SEQN", "LBXSASSI", "LBXSATSI", "LBXSGTSI", "LBXSUA", "LBXSGL", "LBXSCH"],
        cycle.column_aliases,
    )
    hdl = load_xpt(cycle.raw_dir, cycle.file_names["hdl"], ["SEQN", "LBDHDD"], cycle.column_aliases)
    trigly = load_xpt(cycle.raw_dir, cycle.file_names["trigly"], ["SEQN", "LBXTR"], cycle.column_aliases)
    lux = load_xpt(
        cycle.raw_dir,
        cycle.file_names["lux"],
        ["SEQN", "LUAXSTAT", "LUXCAPM", "LUXCPIQR", "LUXSMED", "LUXSIQR", "LUXSIQRM"],
        cycle.column_aliases,
    )
    alq = load_xpt(cycle.raw_dir, cycle.file_names["alq"], ["SEQN", "ALQ121", "ALQ130"], cycle.column_aliases)

    # Triglycerides (fasting subsample) is left-joined so non-fasted subjects
    # stay in the dataset; median imputation at training time covers the gaps.
    df = demo.merge(bmx, on="SEQN", how="inner")
    df = df.merge(biopro, on="SEQN", how="inner")
    df = df.merge(hdl, on="SEQN", how="inner")
    df = df.merge(lux, on="SEQN", how="inner")
    df = df.merge(trigly, on="SEQN", how="left")
    df = df.merge(alq, on="SEQN", how="left")

    # Cohort filters: adults, not pregnant, complete + reliable elastography exam.
    df = df[df["RIDAGEYR"] >= 18]
    df = df[df["RIDEXPRG"] != 1]
    df = df[df["LUAXSTAT"] == 1]
    df = df[df["LUXSIQRM"] < LSM_IQR_RATIO_MAX]

    # Exclude excessive alcohol use so "steatosis" here isn't contaminated by
    # alcohol-related fatty liver. Subjects who didn't answer P_ALQ are kept.
    alq130_valid = df["ALQ130"].where(~df["ALQ130"].isin(ALQ130_SENTINELS))
    excessive_alcohol = (alq130_valid >= ALQ130_EXCESSIVE_MIN) | (df["ALQ121"].isin(ALQ121_EXCESSIVE_CODES))
    df = df[~excessive_alcohol.fillna(False)]

    df["Age"] = df["RIDAGEYR"]
    df["Waist_cm"] = df["BMXWAIST"]
    df["GGT"] = df["LBXSGTSI"]
    df["Uric_Acid"] = df["LBXSUA"]
    df["AST_ALT_ratio"] = df["LBXSASSI"] / df["LBXSATSI"]
    df["TG_HDL_ratio"] = df["LBXTR"] / df["LBDHDD"]
    df["TyG_index"] = np.log(df["LBXTR"] * df["LBXSGL"] / 2)
    df["HDL_BMI_ratio"] = df["LBDHDD"] / df["BMXBMI"]
    df["NHHR"] = (df["LBXSCH"] - df["LBDHDD"]) / df["LBDHDD"]

    df["CAP_dBm"] = df["LUXCAPM"]
    df["LSM_kPa"] = df["LUXSMED"]
    df["Steatosis_Grade_CAP"] = df["CAP_dBm"].apply(lambda v: _grade(v, CAP_CUTOFFS, CAP_LABEL_ABOVE))
    df["Fibrosis_Grade_LSM"] = df["LSM_kPa"].apply(lambda v: _grade(v, LSM_CUTOFFS, LSM_LABEL_ABOVE))

    df = df[df["Steatosis_Grade_CAP"].notna()]
    df = df.replace([np.inf, -np.inf], np.nan)
    df["NHANES_Cycle"] = cycle.label

    output_columns = [
        "SEQN",
        "NHANES_Cycle",
        "Age",
        "Waist_cm",
        "TG_HDL_ratio",
        "GGT",
        "Uric_Acid",
        "AST_ALT_ratio",
        "TyG_index",
        "HDL_BMI_ratio",
        "NHHR",
        "CAP_dBm",
        "LSM_kPa",
        "Steatosis_Grade_CAP",
        "Fibrosis_Grade_LSM",
    ]
    return df[output_columns].reset_index(drop=True)


def build_dataset(cycles: list[CycleSpec] = CYCLES) -> pd.DataFrame:
    frames = [build_cycle_dataset(cycle) for cycle in cycles]
    combined = pd.concat(frames, ignore_index=True)
    duplicate_seqn = combined["SEQN"].duplicated().sum()
    if duplicate_seqn:
        raise ValueError(f"{duplicate_seqn} duplicate SEQN across cycles — cycles are not disjoint populations")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify raw NHANES cycles into the HepatoIA training dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"Saved dataset: {args.output}")
    print(f"Rows: {len(dataset)}")
    print(dataset.groupby("NHANES_Cycle").size())
    print()
    print(dataset["Steatosis_Grade_CAP"].value_counts())


if __name__ == "__main__":
    main()
