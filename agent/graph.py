"""
agent/graph.py

Defines the LangGraph state machine for the clinical decision agent.

Architecture:
    intake → red_flag_check → router → [guideline_retrieval, risk_score] → output

Each node is a pure function: takes AgentState, returns partial AgentState.
The graph controls flow — nodes don't call each other directly.
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def intake_node(state: AgentState) -> dict:
    """
    Entry point. Validates the presentation exists and initialises
    fields that downstream nodes will populate.
    """
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
    """
    Rule-based red flag detection. Intentionally NOT AI-driven —
    hard-coded rules are safer and more auditable for critical safety checks.

    If red flags are found, sets escalate_immediately = True.
    The router will then bypass all further reasoning and go straight to output.
    """
    presentation = state["presentation"].lower()

    # These are the non-negotiable stop-everything flags
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

    step = (
        f"Red flag check: {'ESCALATE — ' + ', '.join(detected) if escalate else 'No red flags detected'}"
    )

    return {
        "red_flags_detected": detected,
        "escalate_immediately": escalate,
        "reasoning_steps": [step],
    }


def router_node(state: AgentState) -> dict:
    """
    Placeholder routing node. In the full agent this will use an LLM
    to decide which tools to invoke. For now it just logs its decision.

    Real routing logic arrives in Day 4 when tools are wired in.
    """
    if state["escalate_immediately"]:
        step = "Router: escalation flagged — bypassing further reasoning"
    else:
        step = "Router: no escalation — proceeding to guideline retrieval and risk scoring (tools not yet wired)"

    return {"reasoning_steps": [step]}


def output_node(state: AgentState) -> dict:
    """
    Produces the final structured output.
    Format follows SBAR: Situation, Background, Assessment, Recommendation.
    """
    if state["escalate_immediately"]:
        flags = ", ".join(state["red_flags_detected"])
        output = f"""
⚠️  IMMEDIATE ESCALATION REQUIRED

Red flags detected: {flags}

This presentation requires immediate clinical review.
Do not proceed with AI-assisted management planning.
Call for senior support / emergency services now.
        """.strip()
        confidence = "high"

    else:
        # Placeholder output — will be replaced with real LLM-generated SBAR in Week 2
        output = f"""
SITUATION
Patient presentation: {state['presentation']}

BACKGROUND
Guideline retrieval: {state['retrieved_guidelines'] or '[not yet implemented]'}

ASSESSMENT
Risk score: {state['risk_score_result'] or '[not yet implemented]'}

RECOMMENDATION
[Full recommendation will be generated once tools are wired in — Day 4]

---
Confidence: {state['confidence']}
This output is decision support only. Clinical judgement takes precedence.
        """.strip()
        confidence = state["confidence"]

    return {
        "final_output": output,
        "confidence": confidence,
        "reasoning_steps": ["Output generated"],
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_escalate(state: AgentState) -> str:
    """
    Conditional edge: if red flags detected, skip all reasoning and go to output.
    This is the safety-critical branch — it must be fast and reliable.
    """
    if state["escalate_immediately"]:
        return "output"
    return "router"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("red_flag_check", red_flag_check_node)
    graph.add_node("router", router_node)
    graph.add_node("output", output_node)

    # Add edges
    graph.set_entry_point("intake")
    graph.add_edge("intake", "red_flag_check")

    # Conditional edge: escalate immediately or continue
    graph.add_conditional_edges(
        "red_flag_check",
        should_escalate,
        {
            "output": "output",
            "router": "router",
        }
    )

    graph.add_edge("router", "output")
    graph.add_edge("output", END)

    return graph.compile()
