"""
tools/risk_scores.py
 
Validated clinical risk score calculators.
These are implemented as pure Python functions — no AI involved.
 
Design principle: validated scoring tools should never be delegated to an LLM.
The logic is deterministic, well-established, and must behave identically
every time. Hard-coded rules are safer and more auditable than AI inference.
 
Scores implemented:
- HEART score (chest pain / ACS risk stratification)
- Wells PE score (pulmonary embolism probability)
- CURB-65 (pneumonia severity) — coming Week 4
"""
 
 
# ---------------------------------------------------------------------------
# HEART Score
# ---------------------------------------------------------------------------
 
def calculate_heart_score(
    history: int,
    ecg: int,
    age: int,
    risk_factors: int,
    troponin: int,
) -> dict:
    """
    Calculates the HEART score for chest pain risk stratification.
 
    Each parameter is scored 0, 1, or 2:
 
    History (how suspicious is the history for ACS?):
        0 = Slightly suspicious
        1 = Moderately suspicious
        2 = Highly suspicious
 
    ECG:
        0 = Normal
        1 = Non-specific repolarisation disturbance
        2 = Significant ST deviation
 
    Age:
        0 = < 45
        1 = 45-64
        2 = >= 65
 
    Risk factors (HTN, hypercholesterolaemia, DM, obesity, smoking,
                  family history, atherosclerotic disease):
        0 = No known risk factors
        1 = 1-2 risk factors
        2 = >= 3 risk factors OR history of atherosclerotic disease
 
    Troponin:
        0 = <= normal limit
        1 = 1-3x normal limit
        2 = > 3x normal limit
 
    Returns:
        dict with score, risk category, and recommended action
    """
    # Validate inputs
    for name, value in [("history", history), ("ecg", ecg), ("age", age),
                         ("risk_factors", risk_factors), ("troponin", troponin)]:
        if value not in [0, 1, 2]:
            raise ValueError(f"{name} must be 0, 1, or 2. Got: {value}")
 
    total = history + ecg + age + risk_factors + troponin
 
    if total <= 3:
        risk = "Low"
        action = "Early discharge may be appropriate. Outpatient follow-up recommended."
        mace_risk = "1.7% risk of MACE at 6 weeks"
    elif total <= 6:
        risk = "Moderate"
        action = "Admit for observation. Serial troponins. Cardiology review."
        mace_risk = "12% risk of MACE at 6 weeks"
    else:
        risk = "High"
        action = "Early invasive strategy. Urgent cardiology review."
        mace_risk = "65% risk of MACE at 6 weeks"
 
    return {
        "score": total,
        "risk_category": risk,
        "recommended_action": action,
        "mace_risk": mace_risk,
        "breakdown": {
            "history": history,
            "ecg": ecg,
            "age": age,
            "risk_factors": risk_factors,
            "troponin": troponin,
        }
    }
 
 
# ---------------------------------------------------------------------------
# Wells PE Score
# ---------------------------------------------------------------------------
 
def calculate_wells_pe_score(
    clinical_signs_dvt: bool,
    pe_most_likely_diagnosis: bool,
    heart_rate_over_100: bool,
    immobilisation_or_surgery: bool,
    previous_dvt_or_pe: bool,
    haemoptysis: bool,
    malignancy: bool,
) -> dict:
    """
    Calculates the Wells score for PE probability.
 
    Parameters (each True = points awarded):
        clinical_signs_dvt: Clinical signs of DVT (3 points)
        pe_most_likely_diagnosis: PE is #1 diagnosis or equally likely (3 points)
        heart_rate_over_100: Heart rate > 100 bpm (1.5 points)
        immobilisation_or_surgery: Immobilisation >= 3 days or surgery in last 4 weeks (1.5 points)
        previous_dvt_or_pe: Previous DVT or PE (1.5 points)
        haemoptysis: Haemoptysis (1 point)
        malignancy: Malignancy on treatment, treated in last 6 months, or palliative (1 point)
 
    Returns:
        dict with score, probability category, and recommended action
    """
    score = 0.0
    score += 3.0 if clinical_signs_dvt else 0
    score += 3.0 if pe_most_likely_diagnosis else 0
    score += 1.5 if heart_rate_over_100 else 0
    score += 1.5 if immobilisation_or_surgery else 0
    score += 1.5 if previous_dvt_or_pe else 0
    score += 1.0 if haemoptysis else 0
    score += 1.0 if malignancy else 0
 
    if score <= 1:
        probability = "Low"
        action = "D-dimer testing appropriate. If negative, PE excluded."
        pe_prevalence = "~1% prevalence of PE"
    elif score <= 6:
        probability = "Moderate"
        action = "D-dimer testing. If positive or score borderline, consider CTPA."
        pe_prevalence = "~17% prevalence of PE"
    else:
        probability = "High"
        action = "Proceed directly to CTPA. Do not delay with D-dimer."
        pe_prevalence = "~37% prevalence of PE"
 
    return {
        "score": score,
        "probability": probability,
        "recommended_action": action,
        "pe_prevalence": pe_prevalence,
        "breakdown": {
            "clinical_signs_dvt": clinical_signs_dvt,
            "pe_most_likely_diagnosis": pe_most_likely_diagnosis,
            "heart_rate_over_100": heart_rate_over_100,
            "immobilisation_or_surgery": immobilisation_or_surgery,
            "previous_dvt_or_pe": previous_dvt_or_pe,
            "haemoptysis": haemoptysis,
            "malignancy": malignancy,
        }
    }
 
 
# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    print("=== HEART Score Test ===")
    result = calculate_heart_score(
        history=2,       # Highly suspicious — classic ACS presentation
        ecg=1,           # Non-specific changes
        age=1,           # 45-64
        risk_factors=2,  # DM + HTN + smoking = 3 risk factors
        troponin=0,      # Normal on arrival
    )
    print(f"Score: {result['score']}")
    print(f"Risk: {result['risk_category']}")
    print(f"Action: {result['recommended_action']}")
    print(f"MACE risk: {result['mace_risk']}")
 
    print("\n=== Wells PE Score Test ===")
    result = calculate_wells_pe_score(
        clinical_signs_dvt=False,
        pe_most_likely_diagnosis=True,   # PE is top differential
        heart_rate_over_100=True,        # Tachycardic
        immobilisation_or_surgery=True,  # Long-haul flight
        previous_dvt_or_pe=False,
        haemoptysis=False,
        malignancy=False,
    )
    print(f"Score: {result['score']}")
    print(f"Probability: {result['probability']}")
    print(f"Action: {result['recommended_action']}")
    print(f"PE prevalence: {result['pe_prevalence']}")