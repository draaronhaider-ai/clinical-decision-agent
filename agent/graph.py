"""
agent/graph.py
 
Defines the LangGraph state machine for the clinical decision agent.
 
Architecture:
    intake → red_flag_check → router → guideline_retrieval → output
 
Each node is a pure function: takes AgentState, returns partial AgentState.
The graph controls flow — nodes don't call each other directly.
"""
 
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from tools.guidelines import retrieve_guideline
 
 
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
 
 
def guideline_retrieval_node(state: AgentState) -> dict:
    """
    Queries the NICE guideline vector store based on the presentation.
    Constructs a focused clinical query from the presentation text.
    """
    presentation = state["presentation"]
 
    # Build a focused query for the retrieval
    query = f"Assessment and management recommendations for: {presentation}"
 
    print(f"[guideline_retrieval] Querying guidelines...")
    result = retrieve_guideline(query)
 
    step = f"Guideline retrieval: retrieved {len(result)} characters of relevant NICE guidance"
 
    return {
        "retrieved_guidelines": result,
        "reasoning_steps": [step],
    }
 
 
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
        guidelines_section = (
            state["retrieved_guidelines"]
            if state["retrieved_guidelines"]
            else "No guideline content retrieved."
        )
 
        output = f"""
SITUATION
{state['presentation']}
 
BACKGROUND — NICE Guidance
{guidelines_section}
 
ASSESSMENT
Risk score calculation: [coming in Day 4 — HEART score]
Drug interaction check: [coming in Day 4 — OpenFDA]
 
RECOMMENDATION
[Full LLM-generated recommendation coming in Week 2]
 
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
    """
    if state["escalate_immediately"]:
        return "output"
    return "guideline_retrieval"
 
 
# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
 
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
 
    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("red_flag_check", red_flag_check_node)
    graph.add_node("guideline_retrieval", guideline_retrieval_node)
    graph.add_node("output", output_node)
 
    # Add edges
    graph.set_entry_point("intake")
    graph.add_edge("intake", "red_flag_check")
 
    # Conditional edge: escalate immediately or retrieve guidelines
    graph.add_conditional_edges(
        "red_flag_check",
        should_escalate,
        {
            "output": "output",
            "guideline_retrieval": "guideline_retrieval",
        }
    )
 
    graph.add_edge("guideline_retrieval", "output")
    graph.add_edge("output", END)
 
    return graph.compile()