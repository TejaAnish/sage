"""Kaggle paths, secrets and model discovery. Call setup_env() before importing HF libraries."""
import json
import os
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

IMAGE_PROCESSOR_TYPE = "Qwen2VLImageProcessor"
KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")


def check_versions(requirements):
    """Stop if the exact versions pinned in the requirements file aren't the ones installed.

    A new Kaggle session resets packages to its defaults, and those can break loading or quietly
    change training partway through a resumed run.
    """
    wrong = []
    for line in Path(requirements).read_text().splitlines():
        if "==" not in line:
            continue
        name, pinned = line.strip().split("==")
        try:
            found = version(name)
        except PackageNotFoundError:
            found = "not installed"
        if found != pinned:
            wrong.append(f"{name} {found} (needs {pinned})")
    if wrong:
        raise SystemExit("Wrong package versions: " + ", ".join(wrong) + ".\nA new Kaggle session resets packages. "
                         "Run this first, then start again:\n  pip install -q -r requirements-kaggle.txt")


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


def prepare_processor_files(model_path, dest, min_pixels, max_pixels):
    """Copy the processor files to a writable folder with the image processor settings corrected.

    Some Kaggle snapshots of Qwen2.5-VL name an image processor class that current transformers
    no longer has, and /kaggle/input is read-only. Writing the pixel limits here also means
    they can't be silently ignored at load time.
    """
    source, dest = Path(model_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}:
            shutil.copy(path, dest / path.name)
    config_path = dest / "preprocessor_config.json"
    config = json.loads(config_path.read_text())
    found = config.get("image_processor_type")
    if found != IMAGE_PROCESSOR_TYPE:
        print(f"Image processor type {found!r} replaced with {IMAGE_PROCESSOR_TYPE!r}", flush=True)
    config.update(image_processor_type=IMAGE_PROCESSOR_TYPE, min_pixels=min_pixels, max_pixels=max_pixels,
                  size={"shortest_edge": min_pixels, "longest_edge": max_pixels})
    config_path.write_text(json.dumps(config, indent=2))
    return str(dest)


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
