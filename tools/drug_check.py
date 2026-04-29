"""
tools/drug_check.py

Drug interaction checker using the OpenFDA API.
Free to use, no API key required.

Design principle: this tool checks for known interactions between medications
listed in the presentation. It does not recommend changes — it flags concerns
for clinical review. The final decision always rests with the clinician.
"""

import requests


OPENFDA_URL = "https://api.fda.gov/drug/label.json"


def extract_medications(presentation: str, llm_client) -> list[str]:
    """
    Uses the LLM to extract medication names from free text.
    Returns a list of drug names.
    """
    prompt = (
        "Extract all medication names from this clinical presentation. "
        "Return ONLY a JSON array of drug name strings, nothing else. "
        "If no medications are mentioned, return an empty array []. "
        "Use generic names where possible. "
        f"Presentation: {presentation}"
    )

    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    import json
    try:
        medications = json.loads(text)
        return [m.lower() for m in medications if isinstance(m, str)]
    except Exception:
        return []


def check_drug_interactions(medications: list[str]) -> dict:
    """
    Checks OpenFDA for warnings associated with each medication.
    Returns a dict of findings per drug.

    Note: OpenFDA returns label warnings, not pairwise interaction data.
    For a production system, a dedicated interaction API (e.g. DrugBank)
    would be more appropriate. This demonstrates the integration pattern.
    """
    if not medications:
        return {"status": "no_medications", "findings": []}

    findings = []

    for drug in medications:
        try:
            params = {
                "search": f"openfda.generic_name:{drug}",
                "limit": 1,
            }
            response = requests.get(OPENFDA_URL, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                if results:
                    label = results[0]
                    warnings = label.get("warnings", ["No specific warnings found"])
                    contraindications = label.get("contraindications", [])

                    findings.append({
                        "drug": drug,
                        "warnings_summary": warnings[0][:300] if warnings else "None found",
                        "has_contraindications": len(contraindications) > 0,
                    })
                else:
                    findings.append({
                        "drug": drug,
                        "warnings_summary": "Drug not found in FDA database",
                        "has_contraindications": False,
                    })
            else:
                findings.append({
                    "drug": drug,
                    "warnings_summary": f"FDA API returned status {response.status_code}",
                    "has_contraindications": False,
                })

        except requests.exceptions.Timeout:
            findings.append({
                "drug": drug,
                "warnings_summary": "FDA API timeout — check manually",
                "has_contraindications": False,
            })
        except Exception as e:
            findings.append({
                "drug": drug,
                "warnings_summary": f"Error checking drug: {str(e)}",
                "has_contraindications": False,
            })

    return {
        "status": "checked",
        "medications_checked": medications,
        "findings": findings,
    }


def format_drug_check_result(result: dict) -> str:
    """Formats the drug check result for display in the SBAR output."""
    if result["status"] == "no_medications":
        return "No medications identified in presentation."

    lines = [f"Medications checked: {', '.join(result['medications_checked'])}"]
    for finding in result["findings"]:
        flag = "⚠️" if finding["has_contraindications"] else "ℹ️"
        lines.append(f"{flag} {finding['drug'].title()}: {finding['warnings_summary'][:200]}")

    return "\n".join(lines)