"""Training/inference prompt. Inputs are image + crop only.

Soil, weather, leaf stage and pathogen deliberately never reach this prompt: they go to
the separate advisory LLM after the prediction (see advisory.py).
"""
import re

from .catalog import CROPS, disease_names

ANSWER_PREFIX = "Disease:"


def system_prompt(crop_key):
    display = CROPS[crop_key]["display"]
    return (
        f"You are a plant pathologist specialising in {display} diseases in India. "
        "Diagnose only from the symptoms visible in the photo."
    )


def user_prompt(crop_key):
    crop = CROPS[crop_key]
    options = "\n".join(f"- {name}: {d['cue']}" for name, d in crop["diseases"].items())
    return (
        f"Crop: {crop['display']}\n"
        "Which disease does this photo show? Choose exactly one:\n"
        f"{options}\n"
        f"Reply on one line, in this format: {ANSWER_PREFIX} <name>"
    )


def format_answer(disease):
    return f"{ANSWER_PREFIX} {disease}"


def build_messages(crop_key, answer=None):
    """Chat messages for the processor's template; the image is passed separately."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(crop_key)}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt(crop_key)}]},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": format_answer(answer)}]})
    return messages


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_prediction(text, labels):
    """Return the label named by the model's first line, or None if it names none or several."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    line = lines[0]
    if ANSWER_PREFIX.lower() in line.lower():
        line = line[line.lower().index(ANSWER_PREFIX.lower()) + len(ANSWER_PREFIX):]
    answer = _norm(line)
    by_norm = {_norm(label): label for label in labels}
    if answer in by_norm:
        return by_norm[answer]
    found = [label for key, label in by_norm.items() if re.search(rf"\b{re.escape(key)}\b", answer)]
    return found[0] if len(found) == 1 else None


def prediction_details(crop_key, disease):
    """Curated facts to show next to a prediction."""
    entry = CROPS[crop_key]["diseases"][disease]
    return {"crop": CROPS[crop_key]["display"], "disease": disease,
            "pathogen": entry["pathogen"], "type": entry["type"]}


__all__ = ["build_messages", "disease_names", "format_answer", "parse_prediction",
           "prediction_details", "system_prompt", "user_prompt"]
