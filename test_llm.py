import anthropic
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()

prompt = (
    "You are a clinical assistant. Extract HEART score variables from this presentation. "
    "Return ONLY a JSON object with these exact integer fields (0, 1, or 2 only): "
    '{"history": 0, "ecg": 0, "age": 0, "risk_factors": 0, "troponin": 0} '
    "Presentation: 45M, central chest pain radiating to left arm, onset 90 minutes ago, "
    "diaphoresis, PMH: T2DM, HTN, ex-smoker. "
    "Return only the JSON, no explanation."
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}]
)

print(repr(response.content[0].text))