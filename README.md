# Clinical Decision Support Agent

An agentic AI system for supporting clinical decision-making in acute presentations. Built to demonstrate human-in-the-loop AI architecture, RAG-grounded reasoning, and safety-first system design.

> ⚠️ **This is a portfolio/research project and is not validated for clinical use. It does not constitute medical advice.**

---

## What this is

A multi-step AI agent that takes a free-text clinical presentation and autonomously orchestrates a structured reasoning process — retrieving NICE guidelines, calculating validated risk scores, checking drug interactions, and flagging red flags — before returning a structured SBAR recommendation with its full reasoning chain exposed.

The key design principle: the agent decides *what to do and in what order*, but every decision is visible, auditable, and ultimately deferential to clinical judgement.

---

## Architecture

```
Presentation (free text)
        │
        ▼
   [intake_node]           — Initialises state
        │
        ▼
[red_flag_check_node]      — Rule-based, NOT AI. Hard stops for critical presentations.
        │
   ┌────┴────┐
escalate?   continue
   │             │
   ▼             ▼
[output]     [guideline_retrieval_node]  — Queries NICE guideline vector store
                 │
                 ▼
          [risk_score_node]  — LLM extracts variables → pure Python calculator
                 │
                 ▼
          [drug_check_node]  — OpenFDA API
                 │
                 ▼
       [recommendation_node]  — LLM generates UK NHS SBAR recommendation
                 │
                 ▼
           [output_node]   — SBAR-format structured output
```

### Why rule-based red flag detection?

The red flag check is intentionally not AI-driven. Hard-coded rules are:
- **Faster** — no LLM call latency on the critical safety path
- **Auditable** — exact trigger conditions are visible in code, not a black box
- **Reliable** — no hallucination risk on the most consequential decision in the system

This is a deliberate product decision, not a technical limitation.

---

## Tech stack

| Component | Tool | Reason |
|---|---|---|
| Agent orchestration | LangGraph | Explicit state machine — reasoning flow is visible and controllable |
| LLM | Claude Sonnet (Anthropic) | Strong instruction following, UK NHS framing |
| Guideline RAG | LlamaIndex + ChromaDB | Persistent vector store, clean ingestion pipeline |
| Embeddings | HuggingFace sentence-transformers | Free, runs locally, no API key required |
| Risk scores | Pure Python | Validated logic should not be delegated to an LLM |
| Drug interactions | OpenFDA API | Free, no auth, programmatic |
| Frontend | Streamlit | Fast iteration, easy deployment |

---

## Project structure

```
clinical-decision-agent/
├── agent/
│   ├── graph.py        # LangGraph state machine
│   └── state.py        # AgentState TypedDict
├── tools/
│   ├── risk_scores.py  # HEART, CURB-65, Wells — pure Python
│   ├── drug_check.py   # OpenFDA integration
│   └── guidelines.py   # RAG retrieval tool
├── rag/
│   └── ingest.py       # NICE guideline ingestion pipeline
├── guidelines/         # NICE guideline text files
├── chroma_db/          # Pre-built vector store
├── main.py             # CLI entry point
├── app.py              # Streamlit frontend
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/draaronhaider-ai/clinical-decision-agent
cd clinical-decision-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your Anthropic API key to .env

# 5. Run the agent (CLI)
python main.py

# 6. Run the frontend
streamlit run app.py
```

---

## Development log

### Completed
- [x] Project structure and environment setup
- [x] Agent state schema defined
- [x] LangGraph state machine with red flag escalation
- [x] NICE guideline RAG pipeline (CG95 chest pain + NG250 pneumonia)
- [x] Guideline retrieval wired as agent tool
- [x] HEART score calculator
- [x] Wells PE score calculator
- [x] CURB-65 / pneumonia presentation
- [x] Full reasoning loop across all presentation types
- [x] Drug interaction check (OpenFDA)
- [x] Streamlit frontend with visible reasoning chain
- [x] Escalation UI with immediate stop logic
- [x] Safety architecture documentation
- [x] Public deployment (Streamlit Cloud)

---

## Architectural decisions

*Each entry records what was decided, why, and what was considered and rejected.*

---

### 1. LangGraph over vanilla LangChain

**Decision:** Used LangGraph to implement the agent as an explicit state machine rather than a linear LangChain pipeline.

**Reasoning:** Clinical reasoning has non-linear branching logic — the agent needs to escalate immediately on red flags, choose between different risk scores depending on presentation type, and halt rather than continue when safety thresholds are crossed. A linear chain cannot express this cleanly. LangGraph's node-and-edge model makes the reasoning flow explicit, inspectable, and controllable. Every decision point is visible in the graph structure rather than buried inside prompt logic.

**Rejected:** A single LLM prompt that attempts to do everything in one call. This would be faster to build but produces a black box — no auditability, no controlled failure modes, and no way to enforce the safety-critical escalation logic deterministically.

---

### 2. Rule-based red flag detection

**Decision:** The red flag check is implemented as hard-coded keyword matching, not an LLM call.

**Reasoning:** This is the most safety-critical decision in the system — whether to stop everything and escalate rather than reason further. Hard-coded rules are: (a) deterministic, behaving identically every time; (b) fast, adding no LLM latency on the critical path; (c) auditable, with exact trigger conditions visible in code. AI models can hallucinate, misinterpret context, or behave unpredictably under distribution shift. None of those failure modes are acceptable when the output is "call 999."

**Rejected:** Asking the LLM to assess red flags as part of its reasoning chain. Even a highly capable model introduces non-zero hallucination risk on the most consequential branch of the system.

---

### 3. Pure Python risk score calculators

**Decision:** HEART score, Wells PE score, and CURB-65 are implemented as deterministic Python functions. The LLM is used only to extract variables from free text — the calculation itself is never delegated to the model.

**Reasoning:** Validated clinical scoring tools have fixed, well-defined logic. There is no ambiguity in adding up integers. Delegating arithmetic to an LLM introduces unnecessary hallucination risk for a task that is trivially solved in code. The separation also makes the system more auditable — a clinician can verify the extracted variables and recalculate the score independently.

**Rejected:** Asking the LLM to calculate the score directly from the presentation. This is tempting because it reduces code, but it conflates two tasks that are best handled separately: natural language understanding (LLM's strength) and deterministic arithmetic (code's strength).

---

### 4. RAG over NICE guidelines rather than baking knowledge into the prompt

**Decision:** NICE guidelines are ingested into a ChromaDB vector store and retrieved at runtime using semantic search. The LLM is not relied upon to recall guideline content from training data.

**Reasoning:** LLM training data has a knowledge cutoff, may contain outdated guideline versions, and cannot be audited for accuracy. By grounding every recommendation in retrieved text from the actual current guidelines, the system produces outputs that are traceable to a specific source document. This is essential for clinical credibility and directly maps to the regulatory requirement for AI medical devices to have auditable evidence bases.

**Rejected:** Including guideline content directly in the system prompt. This would work for a small number of guidelines but does not scale, cannot be updated without redeployment, and does not provide chunk-level traceability.

---

### 5. UK NHS framing enforced at the prompt level

**Decision:** The recommendation prompt explicitly instructs the LLM to use UK drug names, NHS referral pathways, and NICE guideline frameworks only, and to not reference US or international guidelines.

**Reasoning:** Clinical decision support tools must be jurisdiction-specific. A tool that mixes NICE guidance with AHA/ACC guidelines, or uses US drug names (epinephrine vs adrenaline), creates confusion and potential patient safety risk. Enforcing UK framing at the prompt level is a lightweight but meaningful safety control.

**Rejected:** Relying on the LLM to infer the appropriate jurisdiction from context. This is unreliable — models trained on predominantly US medical literature will default to US conventions without explicit instruction.

---

### 6. SBAR output format

**Decision:** The structured output follows SBAR (Situation, Background, Assessment, Recommendation), the standard clinical communication framework used across UK acute medicine.

**Reasoning:** SBAR is the format NHS clinicians already use to communicate about patients — in referral calls, handover notes, and escalation conversations. Using it makes the tool's output immediately legible to its target users without requiring them to learn a new schema. It also enforces a discipline on the model: the recommendation section cannot bleed into background context, keeping outputs actionable rather than discursive.

**Rejected:** Free-form clinical narrative. While potentially more readable, unstructured output makes it harder to audit individual claims and harder for clinicians to quickly locate the actionable recommendation.

---

### 7. Visible reasoning chain

**Decision:** Every step the agent takes is logged to a `reasoning_steps` list in the state and displayed in the UI alongside the output.

**Reasoning:** A clinical AI tool that shows only its conclusion is not trustworthy in a clinical setting. Clinicians need to be able to see what the agent did, in what order, and what it retrieved — so they can identify if something went wrong and decide whether to trust the output. Transparency is not just a nice-to-have; it is a prerequisite for responsible deployment and a core requirement under MHRA guidance on AI as a medical device.

**Rejected:** Hiding the reasoning chain and showing only the SBAR output. This produces a cleaner UI but a less trustworthy tool.