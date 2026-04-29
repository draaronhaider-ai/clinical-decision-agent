"""
main.py

Entry point for running the agent from the command line.
Use this during development to test the agent without the Streamlit UI.

Usage:
    python main.py
    python main.py --presentation "Your custom presentation here"
"""

import argparse
from dotenv import load_dotenv
from agent.graph import build_graph

load_dotenv()

# Test presentations — add your own as you develop
TEST_PRESENTATIONS = [
    "45M, central chest pain radiating to left arm, onset 90 minutes ago, diaphoresis, PMH: T2DM, HTN, ex-smoker. Current meds: metformin, ramipril.",
    "28F, sharp pleuritic chest pain, SOB, recent long-haul flight, no PMH, on OCP.",
    "72M, hypotensive BP 70/40, central chest pain, diaphoretic, unresponsive to voice.",  # Should trigger escalation
]


def run_agent(presentation: str, verbose: bool = True) -> str:
    graph = build_graph()

    initial_state = {
        "presentation": presentation,
        "reasoning_steps": [],
        "retrieved_guidelines": "",
        "risk_score_result": {},
        "drug_interaction_result": "",
        "red_flags_detected": [],
        "escalate_immediately": False,
        "needs_drug_check": False,
        "final_output": "",
        "confidence": "moderate",
    }

    result = graph.invoke(initial_state)

    if verbose:
        print("\n" + "="*60)
        print("REASONING CHAIN")
        print("="*60)
        for i, step in enumerate(result["reasoning_steps"], 1):
            print(f"{i}. {step}")

        print("\n" + "="*60)
        print("FINAL OUTPUT")
        print("="*60)
        print(result["final_output"])

    return result["final_output"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Decision Agent")
    parser.add_argument(
        "--presentation",
        type=str,
        default=None,
        help="Clinical presentation text. Defaults to test presentations if not provided."
    )
    args = parser.parse_args()

    if args.presentation:
        run_agent(args.presentation)
    else:
        # Run all test presentations
        for i, presentation in enumerate(TEST_PRESENTATIONS, 1):
            print(f"\n{'#'*60}")
            print(f"TEST CASE {i}")
            print(f"{'#'*60}")
            print(f"Input: {presentation}\n")
            run_agent(presentation)
