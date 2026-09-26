"""Rule-based (non-ML) severity classification for hepatic steatosis.

Deterministic, literature-cited cutoffs applied to a CAP value (dB/m),
optionally combined with liver stiffness (LSM, kPa) and simple metabolic
flags. No coefficients, no training data, no probabilities — every branch
is a threshold a clinician can look up and verify by hand.

References:
- Eddowes PJ, et al. Accuracy of FibroScan Controlled Attenuation Parameter
  and Liver Stiffness Measurement in Assessing Steatosis and Fibrosis in
  Patients With NAFLD. Gastroenterology 2019. (CAP grading cutoffs)
- Sasso M, et al. Controlled attenuation parameter (CAP): a novel VCTE
  guided ultrasonic attenuation measurement. Ultrasound Med Biol 2010.
  (single-cutoff "any steatosis" convention, 238-248 dB/m range)
- Eslam M, et al. A new definition for metabolic dysfunction-associated
  fatty liver disease: an international expert consensus statement.
  J Hepatol 2020. (MAFLD combined criterion)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Eddowes et al. 2019 four-tier CAP grading.
CAP_GRADE_CUTOFFS: list[tuple[float, str]] = [
    (248.0, "S0_Normal"),
    (268.0, "S1_Leve"),
    (281.0, "S2_Moderada"),
]
CAP_GRADE_LABEL_ABOVE = "S3_Severa"

# Sasso et al. 2010 / widely used single cutoff for "any steatosis present".
CAP_PRESENCE_CUTOFF = 238.0

# Eddowes et al. 2019 four-tier LSM (fibrosis) grading, kPa.
LSM_GRADE_CUTOFFS: list[tuple[float, str]] = [
    (8.2, "F0_F1"),
    (9.7, "F2"),
    (13.6, "F3"),
]
LSM_GRADE_LABEL_ABOVE = "F4"

STEATOSIS_SEVERITY_ORDER = ["S0_Normal", "S1_Leve", "S2_Moderada", "S3_Severa"]
FIBROSIS_SEVERITY_ORDER = ["F0_F1", "F2", "F3", "F4"]


def _grade(value: float, cutoffs: list[tuple[float, str]], label_above: str) -> str:
    for threshold, label in cutoffs:
        if value < threshold:
            return label
    return label_above


def classify_steatosis_grade(cap_dbm: float) -> str:
    """4-tier severity grade from a CAP value, per Eddowes et al. 2019."""
    return _grade(cap_dbm, CAP_GRADE_CUTOFFS, CAP_GRADE_LABEL_ABOVE)


def classify_steatosis_presence(cap_dbm: float) -> str:
    """Binary presence/absence of steatosis at the 238 dB/m cutoff."""
    return "Esteatosis" if cap_dbm >= CAP_PRESENCE_CUTOFF else "Normal"


def classify_fibrosis_grade(lsm_kpa: float) -> str:
    """4-tier fibrosis grade from an LSM value, per Eddowes et al. 2019."""
    return _grade(lsm_kpa, LSM_GRADE_CUTOFFS, LSM_GRADE_LABEL_ABOVE)


@dataclass
class MetabolicContext:
    """Simple flags for the MAFLD consensus criterion (Eslam et al. 2020)."""

    bmi: float | None = None
    has_diabetes: bool | None = None
    has_hypertension: bool | None = None

    def meets_metabolic_dysfunction(self) -> bool | None:
        checks = []
        if self.bmi is not None:
            checks.append(self.bmi >= 25.0)
        if self.has_diabetes is not None:
            checks.append(bool(self.has_diabetes))
        if self.has_hypertension is not None:
            checks.append(bool(self.has_hypertension))
        if not checks:
            return None
        return any(checks)


def classify_mafld(cap_dbm: float, metabolic: MetabolicContext) -> dict:
    """MAFLD per Eslam et al. 2020: steatosis (CAP>=238) AND >=1 metabolic
    dysfunction criterion (BMI>=25, diabetes, or hypertension).
    Returns a dict because, unlike the other classifiers, this one can be
    'indeterminate' when there isn't enough metabolic data to evaluate.
    """
    has_steatosis = cap_dbm >= CAP_PRESENCE_CUTOFF
    metabolic_dysfunction = metabolic.meets_metabolic_dysfunction()

    if not has_steatosis:
        return {"classification": "No_MAFLD", "reason": "CAP below steatosis cutoff (238 dB/m)"}
    if metabolic_dysfunction is None:
        return {"classification": "Indeterminate", "reason": "Steatosis present but no metabolic data supplied"}
    if metabolic_dysfunction:
        return {"classification": "MAFLD", "reason": "Steatosis + at least one metabolic dysfunction criterion"}
    return {"classification": "No_MAFLD", "reason": "Steatosis present but no metabolic dysfunction criterion met"}


@dataclass
class SeverityAssessment:
    cap_dbm: float
    steatosis_grade: str
    steatosis_presence: str
    lsm_kpa: float | None = None
    fibrosis_grade: str | None = None
    mafld: dict | None = None
    combined_severity: str | None = None


def _combine_steatosis_fibrosis(steatosis_grade: str, fibrosis_grade: str) -> str:
    # Overall severity takes whichever axis (fat content or scarring) ranks
    # higher, since advanced fibrosis drives progression risk regardless of
    # how much fat is currently present.
    steatosis_rank = STEATOSIS_SEVERITY_ORDER.index(steatosis_grade)
    fibrosis_rank = FIBROSIS_SEVERITY_ORDER.index(fibrosis_grade)
    overall_rank = max(steatosis_rank, fibrosis_rank)
    labels = ["Normal", "Leve", "Moderada", "Severa"]
    return labels[overall_rank]


def assess_severity(
    cap_dbm: float,
    lsm_kpa: float | None = None,
    metabolic: MetabolicContext | None = None,
) -> SeverityAssessment:
    """Single entry point: takes a CAP value (and optionally LSM + metabolic
    context) and returns every rule-based classification in one object.
    """
    steatosis_grade = classify_steatosis_grade(cap_dbm)
    steatosis_presence = classify_steatosis_presence(cap_dbm)

    fibrosis_grade = classify_fibrosis_grade(lsm_kpa) if lsm_kpa is not None else None
    combined = _combine_steatosis_fibrosis(steatosis_grade, fibrosis_grade) if fibrosis_grade else None

    mafld_result = classify_mafld(cap_dbm, metabolic) if metabolic is not None else None

    return SeverityAssessment(
        cap_dbm=cap_dbm,
        steatosis_grade=steatosis_grade,
        steatosis_presence=steatosis_presence,
        lsm_kpa=lsm_kpa,
        fibrosis_grade=fibrosis_grade,
        mafld=mafld_result,
        combined_severity=combined,
    )


if __name__ == "__main__":
    import json

    examples = [
        {"cap_dbm": 210, "lsm_kpa": 5.5},
        {"cap_dbm": 260, "lsm_kpa": 9.0, "metabolic": MetabolicContext(bmi=31, has_diabetes=False, has_hypertension=True)},
        {"cap_dbm": 320, "lsm_kpa": 15.0, "metabolic": MetabolicContext(bmi=34, has_diabetes=True)},
    ]
    for ex in examples:
        result = assess_severity(**ex)
        print(json.dumps(result.__dict__, indent=2, default=str, ensure_ascii=False))
        print()
