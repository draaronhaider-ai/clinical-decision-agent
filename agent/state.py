"""
agent/state.py

Defines the state schema that flows through the LangGraph agent.
Every node reads from and writes to this state dict.
"""

from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    """
    The central state object passed between all nodes in the graph.

    Design principle: every piece of information the agent generates
    lives here — nothing is hidden inside a node. This makes the
    reasoning chain fully inspectable and auditable.
    """

    # --- Input ---
    presentation: str
    # The raw clinical text entered by the user.
    # Example: "67M, chest pain radiating to jaw, diaphoresis, onset 2hrs ago"

    # --- Reasoning chain ---
    reasoning_steps: Annotated[list[str], operator.add]
    # Append-only log of every action the agent takes.
    # Using operator.add means nodes append rather than overwrite.
    # This is what gets displayed in the UI as the agent's visible reasoning.

    # --- Tool outputs ---
    retrieved_guidelines: str
    # Raw text returned from the NICE guideline RAG retrieval.

    risk_score_result: dict
    # Output from whichever validated risk score was calculated.
    # e.g. {"score": 7, "risk_category": "High — early invasive strategy"}

    drug_interaction_result: str
    # Output from the OpenFDA drug interaction check, if triggered.

    red_flags_detected: list[str]
    # List of any red flags identified. Empty list = none detected.

    # --- Control flow ---
    escalate_immediately: bool
    # If True, the agent stops and escalates rather than continuing.
    # Set by the red flag detection node.

    needs_drug_check: bool
    # Whether a medication list was detected in the presentation.

    # --- Output ---
    final_output: str
    # The structured SBAR-format response returned to the clinician.

    confidence: str
    # "high" | "moderate" | "low" — the agent's self-assessed confidence.
    # Low confidence triggers an explicit caveat in the output.
