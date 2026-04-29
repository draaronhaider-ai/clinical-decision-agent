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
[output]     [router_node]  — LLM decides which tools to invoke
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
 [guideline]  [risk_    [drug_
 [retrieval]  score]    check]
        └────────┬────────┘
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
| LLM | GPT-4o | Strong instruction following for structured clinical output |
| Guideline RAG | LlamaIndex + ChromaDB | Persistent vector store, clean ingestion pipeline |
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
├── utils/
│   └── formatting.py   # SBAR output formatting
├── guidelines/         # NICE PDFs (not committed to git)
├── chroma_db/          # Vector store (not committed to git)
├── main.py             # CLI entry point
├── app.py              # Streamlit frontend (Week 3)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/clinical-decision-agent
cd clinical-decision-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your OpenAI API key to .env

# 5. Run the agent
python main.py
```

---

## Development log

### Week 1 (current)
- [x] Project structure and environment setup
- [x] Agent state schema defined
- [x] Minimal graph skeleton with intake → red flag check → router → output
- [ ] NICE guideline RAG pipeline
- [ ] Guideline retrieval wired as agent tool

### Week 2
- [ ] HEART score calculator
- [ ] Wells PE score calculator
- [ ] Full chest pain reasoning loop

### Week 3
- [ ] Drug interaction check (OpenFDA)
- [ ] Streamlit frontend with visible reasoning chain
- [ ] Escalation UI

### Week 4
- [ ] Safety architecture documentation
- [ ] Confidence calibration
- [ ] CURB-65 / pneumonia presentation

### Week 5
- [ ] Public deployment (Hugging Face Spaces / Streamlit Cloud)
- [ ] Portfolio writeup

---

## Architectural decisions

*This section is updated as decisions are made. Each entry records what was decided, why, and what was rejected.*

**1. LangGraph over vanilla LangChain**
Chose LangGraph because the clinical reasoning process has explicit branching logic (escalate vs. continue, which risk score to use) that is better expressed as a state machine than a linear chain. The graph structure also makes the reasoning process inspectable — you can see exactly which nodes were visited.

**2. Rule-based red flag detection**
See architecture section above.

**3. SBAR output format**
SBAR (Situation, Background, Assessment, Recommendation) is the standard structured communication format in UK acute medicine. Using it makes the output immediately legible to the clinical users this tool is designed for.
