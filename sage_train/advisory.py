"""Post-prediction environmental advice from a search-connected LLM.

The fine-tuned model never sees these fields. After it predicts a disease, the frontend's
context goes to whichever search-enabled LLM you connect. A statement may be presented
as established only with a source URL; anything else is shown as a hedged possibility.
"""
import json

CONTEXT_FIELDS = ("soil", "weather", "leaf_stage", "pathogen")

DISCLAIMER = ("The diagnosis comes from the photo only. These environmental notes are "
              "suggestions and may not explain what you are seeing.")


def build_advisory_prompt(crop, disease, context):
    """Prompt for the connected LLM. Fields left empty in `context` are omitted."""
    given = {k: context[k] for k in CONTEXT_FIELDS if context.get(k)}
    fields = "\n".join(f"- {k}: {v}" for k, v in given.items()) or "- none provided"
    return (
        f"A photo-based model diagnosed {disease} on {crop} in India. Do not change or re-judge "
        "this diagnosis.\n\n"
        f"Conditions reported by the farmer:\n{fields}\n\n"
        "Search the web for how each reported condition affects this disease on this crop. "
        "Prefer university extension services, ICAR, peer-reviewed papers and government sources.\n\n"
        "Return only JSON in this shape:\n"
        '{"notes": [{"attribute": "<reported field>", "effect": "<one sentence>", '
        '"certainty": "possible" or "established", "sources": ["<url>"]}]}\n\n'
        'Use "established" only if a source you actually read states the effect; put that page\'s '
        'URL in "sources". Otherwise use "possible" and word the effect tentatively ("may", '
        '"might"). Never invent a URL. Skip attributes for which you find nothing relevant.'
    )


def validate_advisory(raw, context):
    """Parse the LLM's reply and enforce the rules; returns what the frontend should show.
    An "established" note with no http(s) source is downgraded to "possible"."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    reported = {k for k in CONTEXT_FIELDS if context.get(k)}
    notes = []
    for note in data.get("notes", []):
        attribute = note.get("attribute")
        effect = str(note.get("effect", "")).strip()
        if attribute not in reported or not effect:
            continue
        sources = [s for s in note.get("sources", [])
                   if isinstance(s, str) and s.startswith(("http://", "https://"))]
        established = note.get("certainty") == "established" and bool(sources)
        notes.append({"attribute": attribute, "effect": effect,
                      "certainty": "established" if established else "possible",
                      "sources": sources})
    return {"disclaimer": DISCLAIMER, "notes": notes}
