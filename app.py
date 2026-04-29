"""
app.py

Streamlit frontend for the Clinical Decision Support Agent.
Run with: streamlit run app.py
"""

import streamlit as st
from agent.graph import build_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #090c12;
        border-right: 1px solid #1e2535;
    }

    /* Header */
    .main-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 500;
        color: #4a9eff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .main-subheader {
        font-size: 0.8rem;
        color: #556070;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        margin-bottom: 2rem;
    }

    /* Input area */
    .stTextArea textarea {
        background-color: #141824 !important;
        border: 1px solid #1e2d45 !important;
        border-radius: 4px !important;
        color: #c8d0e0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #4a9eff !important;
        box-shadow: 0 0 0 1px #4a9eff22 !important;
    }

    /* Button */
    .stButton > button {
        background-color: #4a9eff !important;
        color: #0a0e18 !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.5rem !important;
        width: 100% !important;
        transition: background-color 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: #6ab4ff !important;
    }

    /* Reasoning chain */
    .reasoning-step {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #556070;
        padding: 0.35rem 0;
        border-bottom: 1px solid #1a1f2e;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }

    .reasoning-step-num {
        color: #4a9eff;
        min-width: 1.2rem;
        font-weight: 500;
    }

    .reasoning-step-text {
        color: #7a8899;
    }

    /* SBAR sections */
    .sbar-section {
        margin-bottom: 1.5rem;
    }

    .sbar-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4a9eff;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #1e2d45;
    }

    .sbar-content {
        font-size: 0.9rem;
        color: #c8d0e0;
        line-height: 1.7;
    }

    /* Escalation alert */
    .escalation-alert {
        background-color: #1a0a0a;
        border: 1px solid #cc3333;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .escalation-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 500;
        color: #ff4444;
        margin-bottom: 0.8rem;
    }

    .escalation-body {
        font-size: 0.9rem;
        color: #cc8888;
        line-height: 1.6;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background-color: #141e2e;
        border: 1px solid #1e3450;
        border-radius: 3px;
        padding: 0.2rem 0.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #4a9eff;
        margin-right: 0.5rem;
    }

    /* Divider */
    .output-divider {
        border: none;
        border-top: 1px solid #1a1f2e;
        margin: 1.5rem 0;
    }

    /* Disclaimer */
    .disclaimer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #3a4455;
        padding: 0.8rem;
        border: 1px solid #1a1f2e;
        border-radius: 3px;
        margin-top: 1.5rem;
        line-height: 1.5;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="main-header">ClinicalAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subheader">NHS Decision Support // v0.1</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 0.75rem; color: #556070; line-height: 1.7; font-family: 'IBM Plex Mono', monospace;">
    <strong style="color: #7a8899;">GUIDELINES</strong><br>
    NICE CG95 — Chest Pain<br>
    NICE NG250 - Pneumonia<br>
    <br>
    <strong style="color: #7a8899;">RISK SCORES</strong><br>
    HEART Score<br>
    Wells PE Score<br>
    CURB-65<br>
    <br>
    <strong style="color: #7a8899;">TOOLS</strong><br>
    OpenFDA Drug Check<br>
    Red Flag Detection<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 0.7rem; color: #3a4455; font-family: 'IBM Plex Mono', monospace; line-height: 1.6;">
    ⚠ For clinical decision support only.<br>
    Not validated for clinical use.<br>
    UK NHS guidelines only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.65rem; color: #2a3040; font-family: 'IBM Plex Mono', monospace;">
    Built by Aaron · github.com/aaron
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col_input, col_output = st.columns([1, 1.4], gap="large")

with col_input:
    st.markdown("""
    <div style="font-size: 0.7rem; color: #556070; font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;">
    Clinical Presentation
    </div>
    """, unsafe_allow_html=True)

    presentation = st.text_area(
        label="presentation",
        label_visibility="collapsed",
        placeholder="e.g. 45M, central chest pain radiating to left arm, onset 90 min, diaphoresis. PMH: T2DM, HTN, ex-smoker. Meds: metformin, ramipril.",
        height=180,
    )

    run = st.button("Analyse Presentation →")

    # Example cases
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.68rem; color: #3a4455; font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;">
    Example Cases
    </div>
    """, unsafe_allow_html=True)

    examples = {
        "ACS (moderate risk)": "45M, central chest pain radiating to left arm, onset 90 minutes ago, diaphoresis, PMH: T2DM, HTN, ex-smoker. Current meds: metformin, ramipril.",
        "Suspected PE": "28F, sharp pleuritic chest pain, SOB, recent long-haul flight, no PMH, on OCP.",
        "Pneumonia (high severity)": "78F, productive cough, fever 38.9, RR 32, BP 85/50, confused, PMH: COPD. No recent antibiotics.",
        "Immediate escalation": "72M, hypotensive BP 70/40, central chest pain, diaphoretic, unresponsive to voice.",
    }

    for label, text in examples.items():
        if st.button(label, key=label):
            st.session_state["example_text"] = text
            st.rerun()

    if "example_text" in st.session_state:
        st.info(f"Example loaded — paste into the text box above:\n\n{st.session_state['example_text']}")

    # Reasoning chain
    if "reasoning_steps" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.68rem; color: #556070; font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.6rem;">
        Reasoning Chain
        </div>
        """, unsafe_allow_html=True)

        for i, step in enumerate(st.session_state["reasoning_steps"], 1):
            st.markdown(f"""
            <div class="reasoning-step">
                <span class="reasoning-step-num">{i}.</span>
                <span class="reasoning-step-text">{step}</span>
            </div>
            """, unsafe_allow_html=True)

with col_output:
    if not run and "output" not in st.session_state:
        st.markdown("""
        <div style="height: 300px; display: flex; align-items: center; justify-content: center;
        border: 1px dashed #1a2030; border-radius: 4px; margin-top: 1.5rem;">
            <div style="text-align: center;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
                color: #2a3545; letter-spacing: 0.08em; text-transform: uppercase;">
                    Awaiting presentation
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if run and presentation.strip():
        with st.spinner("Analysing..."):
            graph = build_graph()
            initial_state = {
                "presentation": presentation.strip(),
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

        st.session_state["output"] = result
        st.session_state["reasoning_steps"] = result["reasoning_steps"]
        st.rerun()

    elif run and not presentation.strip():
        st.warning("Please enter a clinical presentation.")

    if "output" in st.session_state:
        result = st.session_state["output"]

        if result.get("escalate_immediately"):
            flags = ", ".join(result.get("red_flags_detected", []))
            st.markdown(f"""
            <div class="escalation-alert">
                <div class="escalation-title">⚠ IMMEDIATE ESCALATION REQUIRED</div>
                <div class="escalation-body">
                    <strong>Red flags detected:</strong> {flags}<br><br>
                    This presentation requires immediate clinical review.<br>
                    Do not proceed with AI-assisted management planning.<br>
                    Call for senior support / emergency services now.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            risk = result.get("risk_score_result", {})
            score_name = risk.get("score_name", "")
            score_value = risk.get("score", "")
            risk_level = risk.get("risk_category") or risk.get("probability", "")
            action = risk.get("recommended_action", "")
            mace_or_prev = risk.get("mace_risk") or risk.get("pe_prevalence", "")

            # SITUATION
            st.markdown(f"""
            <div class="sbar-section">
                <div class="sbar-label">Situation</div>
                <div class="sbar-content">{result.get('presentation', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<hr class="output-divider">', unsafe_allow_html=True)

            # BACKGROUND
            guidelines = result.get("retrieved_guidelines", "No guideline content retrieved.")
            st.markdown(f"""
            <div class="sbar-section">
                <div class="sbar-label">Background — NICE Guidance</div>
                <div class="sbar-content">
            </div>
            """, unsafe_allow_html=True)

            st.markdown(guidelines)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<hr class="output-divider">', unsafe_allow_html=True)

            # ASSESSMENT
            st.markdown('<div class="sbar-label">Assessment</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="score-badge">{score_name}: {score_value}</span> **{risk_level} risk** · {mace_or_prev}', unsafe_allow_html=True)
            st.markdown(f"**Recommended action:** {action}")
            st.markdown(f"**Drug check:**\n\n{result.get('drug_interaction_result', 'Not checked.')}")
            st.markdown('<hr class="output-divider">', unsafe_allow_html=True)

            # RECOMMENDATION
            st.markdown('<div class="sbar-label">Recommendation</div>', unsafe_allow_html=True)
            recommendation = result.get("final_output", "Not generated.")
            if "RECOMMENDATION" in recommendation:
                recommendation = recommendation.split("RECOMMENDATION")[-1].strip()
                recommendation = recommendation.split("---")[0].strip()
            st.markdown(recommendation)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                ⚠ CLINICAL DECISION SUPPORT ONLY · NOT VALIDATED FOR CLINICAL USE ·
                UK NHS GUIDELINES · NICE CG95 · CLINICAL JUDGEMENT TAKES PRECEDENCE
            </div>
            """, unsafe_allow_html=True)