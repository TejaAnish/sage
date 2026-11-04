"""Kaggle paths, secrets and model discovery. Call setup_env() before importing HF libraries."""
import json
import os
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")


def setup_env():
    """Load HF_TOKEN from Kaggle Secrets and put caches where restarts can reuse them."""
    if not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:
            raise SystemExit("HF_TOKEN not found. In Kaggle: Add-ons > Secrets > add HF_TOKEN "
                             f"(a write token) and attach it to this notebook. ({type(e).__name__})")
    os.environ.setdefault("HF_HOME", str(WORK / ".hf_cache"))
    # Plain HTTP downloads resume from .incomplete files and leave no multi-GB chunk cache.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    WORK.mkdir(parents=True, exist_ok=True)


def discover_model(explicit=None):
    """Find the model attached through Kaggle's input bar."""
    if explicit:
        return explicit
    found = []
    for config in KAGGLE_INPUT.rglob("config.json"):
        try:
            if json.loads(config.read_text()).get("model_type") == "qwen2_5_vl":
                found.append(config.parent)
        except (OSError, ValueError):
            continue
    if len(found) != 1:
        raise SystemExit(f"Expected exactly 1 Qwen2.5-VL model under {KAGGLE_INPUT}, found {len(found)}: "
                         f"{[str(p) for p in found]}. Attach one with 'Add Input', or pass --model-path.")
    print(f"Model: {found[0]}", flush=True)
    return str(found[0])


def with_retries(fn, what, attempts=6, wait=30):
    """Run a network call, tolerating temporary Hub or connection failures."""
    import time
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            print(f"{what} failed ({type(e).__name__}: {e}); retry {attempt}/{attempts - 1} in {wait * attempt}s",
                  flush=True)
            time.sleep(wait * attempt)
