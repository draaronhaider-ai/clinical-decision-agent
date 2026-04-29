"""
agent/graph.py

Defines the LangGraph state machine for the clinical decision agent.

Architecture:
    intake → red_flag_check → guideline_retrieval → risk_score → drug_check → recommendation → output

Each node is a pure function: takes AgentState, returns partial AgentState.
The graph controls flow — nodes don't call each other directly.
"""

import json
import anthropic as anthropic_sdk
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from tools.guidelines import retrieve_guideline
from tools.risk_scores import calculate_heart_score, calculate_wells_pe_score
from tools.drug_check import extract_medications, check_drug_interactions, format_drug_check_result

llm_client = anthropic_sdk.Anthropic()


def llm_complete(prompt: str, max_tokens: int = 500) -> str:
    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def intake_node(state: AgentState) -> dict:
    print(f"[intake] Received presentation: {state['presentation'][:80]}...")
    return {
        "reasoning_steps": [f"Received presentation: {state['presentation']}"],
        "retrieved_guidelines": "",
        "risk_score_result": {},
        "drug_interaction_result": "",
        "red_flags_detected": [],
        "escalate_immediately": False,
        "needs_drug_check": False,
        "final_output": "",
        "confidence": "moderate",
    }


def red_flag_check_node(state: AgentState) -> dict:
    presentation = state["presentation"].lower()
    RED_FLAGS = {
        "haemodynamic instability": ["hypotensive", "bp 80", "bp 70", "shocked", "unresponsive"],
        "loss of consciousness": ["collapsed", "unresponsive", "unconscious", "gcs 3", "gcs <8"],
        "severe respiratory distress": ["sats 80", "sats 70", "unable to speak", "cyanosed"],
        "stemi features": ["st elevation", "stemi", "tombstoning"],
    }
    detected = []
    for flag_name, keywords in RED_FLAGS.items():
        if any(kw in presentation for kw in keywords):
            detected.append(flag_name)
    escalate = len(detected) > 0
    step = f"Red flag check: {'ESCALATE — ' + ', '.join(detected) if escalate else 'No red flags detected'}"
    return {"red_flags_detected": detected, "escalate_immediately": escalate, "reasoning_steps": [step]}


def guideline_retrieval_node(state: AgentState) -> dict:
    presentation_lower = state["presentation"].lower()
    pneumonia_indicators = ["pneumonia", "productive cough", "fever", "cough", "sputum", "lrti"]
    pe_indicators = ["pleuritic", "flight", "ocp", "dvt"]
    if any(ind in presentation_lower for ind in pneumonia_indicators):
        query = f"NICE pneumonia severity assessment CURB-65 antibiotic recommendations: {state['presentation']}"
    elif any(ind in presentation_lower for ind in pe_indicators):
        query = f"NICE pulmonary embolism Wells score assessment: {state['presentation']}"
    else:
        query = f"NICE chest pain ACS assessment recommendations: {state['presentation']}"
    print(f"[guideline_retrieval] Querying guidelines...")
    result = retrieve_guideline(query)
    return {"retrieved_guidelines": result, "reasoning_steps": [f"Guideline retrieval: retrieved {len(result)} characters of relevant NICE guidance"]}


def risk_score_node(state: AgentState) -> dict:
    print(f"[risk_score] Extracting variables and calculating score...")
    presentation = state["presentation"].lower()
    pe_indicators = ["pleuritic", "flight", "ocp", "dvt", "haemoptysis", "leg swelling"]
    pneumonia_indicators = ["pneumonia", "consolidation", "productive cough", "lobar", "cap", "lrti", "lower respiratory", "fever", "cough", "sputum", "crackles", "bronchial breathing"]
    use_wells = any(indicator in presentation for indicator in pe_indicators)
    use_curb65 = any(indicator in presentation for indicator in pneumonia_indicators)

    if use_wells:
        prompt = (
            "You are a clinical assistant. Extract Wells PE score variables from this presentation. "
            "Return ONLY a JSON object with these exact boolean fields (true/false): "
            '{"clinical_signs_dvt": false, "pe_most_likely_diagnosis": false, '
            '"heart_rate_over_100": false, "immobilisation_or_surgery": false, '
            '"previous_dvt_or_pe": false, "haemoptysis": false, "malignancy": false} '
            f"Presentation: {state['presentation']} Return only the JSON, no explanation."
        )
        response = llm_complete(prompt)
        try:
            clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = calculate_wells_pe_score(**json.loads(clean))
            score_name = "Wells PE Score"
        except Exception as e:
            return {"reasoning_steps": [f"Risk score extraction failed: {e}"], "risk_score_result": {}}
    elif use_curb65:
        prompt = (
            "You are a clinical assistant. Extract CURB-65 score variables from this presentation. "
            "Return ONLY a JSON object with these exact boolean fields (true/false): "
            '{"confusion": false, "urea_over_7": false, "respiratory_rate_over_30": false, '
            '"low_blood_pressure": false, "age_over_65": false} '
            "confusion: new onset confusion or AMTS <= 8. "
            "urea_over_7: blood urea > 7 mmol/L (use false if not mentioned). "
            "respiratory_rate_over_30: RR >= 30 breaths/min. "
            "low_blood_pressure: systolic BP < 90 or diastolic <= 60. "
            "age_over_65: age 65 or over. "
            f"Presentation: {state['presentation']} Return only the JSON, no explanation."
        )
        response = llm_complete(prompt)
        try:
            clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            from tools.risk_scores import calculate_curb65
            result = calculate_curb65(**json.loads(clean))
            score_name = "CURB-65"
        except Exception as e:
            return {"reasoning_steps": [f"Risk score extraction failed: {e}"], "risk_score_result": {}}
    else:
        prompt = (
            "You are a clinical assistant. Extract HEART score variables from this presentation. "
            "Return ONLY a JSON object with these exact integer fields (0, 1, or 2 only): "
            '{"history": 0, "ecg": 0, "age": 0, "risk_factors": 0, "troponin": 0} '
            "history: 0=slightly suspicious, 1=moderately suspicious, 2=highly suspicious for ACS. "
            "ecg: 0=normal, 1=non-specific changes, 2=significant ST deviation. "
            "age: 0=under 45, 1=45 to 64, 2=65 or over. "
            "risk_factors: 0=none, 1=one or two, 2=three or more or known atherosclerotic disease. "
            "troponin: 0=normal or not mentioned, 1=one to three times normal, 2=more than three times normal. "
            f"Presentation: {state['presentation']} Return only the JSON, no explanation."
        )
        response = llm_complete(prompt)
        try:
            clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = calculate_heart_score(**json.loads(clean))
            score_name = "HEART Score"
        except Exception as e:
            return {"reasoning_steps": [f"Risk score extraction failed: {e}"], "risk_score_result": {}}

    risk_level = result.get("risk_category") or result.get("probability") or result.get("severity", "N/A")
    result["score_name"] = score_name
    return {"risk_score_result": result, "reasoning_steps": [f"{score_name}: {result['score']} — {risk_level} risk"]}


def drug_check_node(state: AgentState) -> dict:
    print(f"[drug_check] Checking medications...")
    medications = extract_medications(state["presentation"], llm_client)
    if not medications:
        return {"drug_interaction_result": "No medications identified in presentation.", "reasoning_steps": ["Drug check: no medications identified"]}
    result = check_drug_interactions(medications)
    formatted = format_drug_check_result(result)
    return {"drug_interaction_result": formatted, "reasoning_steps": [f"Drug check: reviewed {len(medications)} medication(s) — {', '.join(medications)}"]}


def recommendation_node(state: AgentState) -> dict:
    print(f"[recommendation] Generating clinical recommendation...")
    risk = state.get("risk_score_result", {})
    score_name = risk.get("score_name", "Risk Score")
    score_value = risk.get("score", "N/A")
    risk_level = risk.get("risk_category") or risk.get("probability") or risk.get("severity", "N/A")
    action = risk.get("recommended_action", "N/A")
    prompt = (
        "You are a senior clinical decision support system designed for use within the UK NHS. "
        "Use UK drug names, NHS referral pathways, and NICE guideline frameworks only. "
        "Do not reference US or international guidelines. "
        "Generate a concise clinical recommendation based on the following information. "
        "Write in clear, professional clinical language. Be specific and actionable. Maximum 150 words. "
        "Do not repeat the situation or background — focus only on the recommendation. "
        "End with a one-sentence note reminding the clinician this is decision support only.\n\n"
        f"Presentation: {state['presentation']}\n"
        f"NICE Guidance summary: {state['retrieved_guidelines'][:500]}\n"
        f"{score_name}: {score_value} ({risk_level} risk) — {action}\n"
        f"Medications: {state['drug_interaction_result']}\n"
    )
    try:
        recommendation = llm_complete(prompt, max_tokens=300)
    except Exception as e:
        recommendation = f"Recommendation generation failed: {e}"
    return {"reasoning_steps": ["Clinical recommendation generated"], "final_output": recommendation}


def output_node(state: AgentState) -> dict:
    if state["escalate_immediately"]:
        flags = ", ".join(state["red_flags_detected"])
        output = f"⚠️  IMMEDIATE ESCALATION REQUIRED\n\nRed flags detected: {flags}\n\nThis presentation requires immediate clinical review.\nDo not proceed with AI-assisted management planning.\nCall for senior support / emergency services now."
        confidence = "high"
    else:
        risk = state.get("risk_score_result", {})
        score_name = risk.get("score_name", "Risk Score")
        score_value = risk.get("score", "N/A")
        risk_level = risk.get("risk_category") or risk.get("probability") or risk.get("severity", "N/A")
        action = risk.get("recommended_action", "N/A")
        mace_or_prev = risk.get("mace_risk") or risk.get("pe_prevalence", "")
        output = f"""SITUATION
{state['presentation']}

BACKGROUND — NICE Guidance
{state['retrieved_guidelines'] or 'No guideline content retrieved.'}

ASSESSMENT
{score_name}: {score_value} ({risk_level} risk)
{mace_or_prev}
Recommended action: {action}

Drug check:
{state['drug_interaction_result'] or 'Not checked.'}

RECOMMENDATION
{state['final_output'] or 'Not generated.'}

---
Confidence: {state['confidence']}
This output is decision support only. Clinical judgement takes precedence."""
        confidence = state["confidence"]
    return {"final_output": output, "confidence": confidence, "reasoning_steps": ["Output generated"]}


def should_escalate(state: AgentState) -> str:
    return "output" if state["escalate_immediately"] else "guideline_retrieval"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("red_flag_check", red_flag_check_node)
    graph.add_node("guideline_retrieval", guideline_retrieval_node)
    graph.add_node("risk_score", risk_score_node)
    graph.add_node("drug_check", drug_check_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("output", output_node)
    graph.set_entry_point("intake")
    graph.add_edge("intake", "red_flag_check")
    graph.add_conditional_edges("red_flag_check", should_escalate, {"output": "output", "guideline_retrieval": "guideline_retrieval"})
    graph.add_edge("guideline_retrieval", "risk_score")
    graph.add_edge("risk_score", "drug_check")
    graph.add_edge("drug_check", "recommendation")
    graph.add_edge("recommendation", "output")
    graph.add_edge("output", END)
    return graph.compile()