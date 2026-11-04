"""Build the per-crop dataset on Kaggle and push it to your Hugging Face account.

  python scripts/build_dataset.py fetch      # stream SAGE shards; resumable
  python scripts/build_dataset.py finalize   # de-duplicate, split, publish; resumable

Resume state lives on the Hub, so it survives a lost Kaggle session: a SAGE shard counts as
done once staging/<shard>.parquet exists in your repo, and a crop once its manifest does.
Rerun the same command after any disconnect.
"""
import argparse
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_train.kaggle_env import WORK, setup_env, with_retries

setup_env()

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image as HFImage, Value
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image, ImageOps

from sage_train.catalog import CROPS, EXCLUDE_SOURCES, HF_SOURCE_DATASET
from sage_train.selection import assign_splits, deduplicate, row_key, select_candidates

BUILD = WORK / "build"
MAX_SIDE = 512
JPEG_QUALITY = 90
STAGING_SCHEMA = pa.schema([("key", pa.string()), ("phash", pa.string()), ("jpeg", pa.binary())])
FEATURES = Features({
    "image": HFImage(), "crop": Value("string"), "disease": Value("string"), "source": Value("string"),
    "raw_label": Value("string"), "filename": Value("string"), "phash": Value("string"),
})


def dihedral_dhash(image):
    """A 64-bit difference hash per flip/90-degree orientation, unrotated first, comma-separated.
    Those transforms are how the augmented PlantVillage copies in SAGE were made;
    selection.cluster_hashes compares them with a tolerance for re-encoding noise."""
    base = ImageOps.grayscale(image).resize((36, 36), Image.BILINEAR)
    hashes = []
    for variant in (None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.FLIP_LEFT_RIGHT,
                    Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE, Image.TRANSVERSE):
        pixels = list((base if variant is None else base.transpose(variant)).resize((9, 8), Image.BILINEAR).getdata())
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
        hashes.append(f"{bits:016x}")
    return ",".join(hashes)


def process_image(data):
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return dihedral_dhash(image), buffer.getvalue()


def process_shard(path, wanted_keys):
    """Return a staging table with the wanted rows of one SAGE shard, plus the unreadable count."""
    pf = pq.ParquetFile(path)
    ids = pf.read(columns=["raw_label", "filename"]).to_pydict()
    keys = [row_key(r, f) for r, f in zip(ids["raw_label"], ids["filename"])]
    indices = [i for i, k in enumerate(keys) if k in wanted_keys]
    out = {"key": [], "phash": [], "jpeg": []}
    corrupt = 0
    if indices:
        images = pf.read(columns=["image"]).column("image").take(pa.array(indices)).to_pylist()
        for i, cell in zip(indices, images):
            try:
                phash, jpeg = process_image(cell["bytes"])
            except Exception:
                corrupt += 1
                continue
            out["key"].append(keys[i])
            out["phash"].append(phash)
            out["jpeg"].append(jpeg)
    return pa.table(out, schema=STAGING_SCHEMA), corrupt


def fetch(args, api, repo):
    meta = pd.read_parquet(ROOT / "metadata" / "sage_metadata.parquet")
    wanted = set(select_candidates(meta, args.candidates_per_class)["key"])
    source_files = sorted(f for f in api.list_repo_files(HF_SOURCE_DATASET, repo_type="dataset")
                          if f.startswith("data/train-") and f.endswith(".parquet"))
    done = {Path(f).name for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("staging/")}
    pending = [f for f in source_files if Path(f).name not in done]
    print(f"{len(wanted):,} images wanted | {len(source_files)} shards | {len(done)} done | "
          f"{len(pending)} pending", flush=True)
    if args.max_shards:
        pending = pending[:args.max_shards]

    shards = BUILD / "shards"
    total = 0
    for number, name in enumerate(pending, 1):
        local = with_retries(lambda: hf_hub_download(HF_SOURCE_DATASET, name, repo_type="dataset", local_dir=shards),
                             f"download {name}")
        table, corrupt = process_shard(local, wanted)
        part = BUILD / Path(name).name
        pq.write_table(table, part, row_group_size=512)
        with_retries(lambda: api.upload_file(path_or_fileobj=str(part), path_in_repo=f"staging/{part.name}",
                                             repo_id=repo, repo_type="dataset"), f"upload {part.name}")
        part.unlink()
        Path(local).unlink()
        total += table.num_rows
        print(f"[{number}/{len(pending)}] {Path(name).name}: {table.num_rows:,} kept, {corrupt} unreadable, "
              f"{total:,} this session", flush=True)
    shutil.rmtree(shards, ignore_errors=True)
    remaining = len(source_files) - len(done) - len(pending)
    print(f"Fetch stopped early by --max-shards: {remaining} shards remain." if remaining else
          "Fetch complete. Next: python scripts/build_dataset.py finalize", flush=True)


def write_crop(crop_key, plan, part_paths, out_dir):
    """Write data/<crop>/<split>.parquet for the planned rows using the HF image schema."""
    schema = FEATURES.arrow_schema
    writers = {}
    planned = plan.set_index("key")
    for path in part_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["key", "jpeg"]):
            rows = {}
            for key, jpeg in zip(batch.column("key").to_pylist(), batch.column("jpeg").to_pylist()):
                if key not in planned.index:
                    continue
                r = planned.loc[key]
                rows.setdefault(r["split"], []).append({
                    "image": {"bytes": jpeg, "path": None}, "crop": CROPS[crop_key]["display"],
                    "disease": r["label"], "source": r["source"], "raw_label": r["raw_label"],
                    "filename": r["filename"], "phash": r["phash"].split(",")[0],
                })
            for split, items in rows.items():
                if split not in writers:
                    target = out_dir / "data" / crop_key / f"{split}.parquet"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    writers[split] = pq.ParquetWriter(target, schema)
                writers[split].write_table(pa.Table.from_pylist(items, schema=schema))
    for writer in writers.values():
        writer.close()


def dataset_card(manifests):
    configs = "".join(
        f"- config_name: {key}\n  data_files:\n" + "".join(
            f"  - split: {split}\n    path: data/{key}/{split}.parquet\n" for split in ("train", "validation", "test"))
        for key in manifests)
    lines = [f"---\nlicense: cc-by-4.0\ntask_categories:\n- image-classification\nconfigs:\n{configs}---\n",
             "# SAGE India Crops\n",
             f"Selected, de-duplicated and split subset of [{HF_SOURCE_DATASET}]"
             f"(https://huggingface.co/datasets/{HF_SOURCE_DATASET}) (CC-BY-4.0; Arshad et al., "
             "arXiv:2605.09768). Images are resized to at most 512 px.\n",
             "| Crop | Disease | Train | Validation | Test |", "|---|---|---|---|---|"]
    for key, m in manifests.items():
        for disease, counts in m["classes"].items():
            lines.append(f"| {key} | {disease} | {counts.get('train', 0)} | {counts.get('validation', 0)} | "
                         f"{counts.get('test', 0)} |")
    return "\n".join(lines) + "\n"


def finalize(args, api, repo):
    meta = pd.read_parquet(ROOT / "metadata" / "sage_metadata.parquet")
    candidates = select_candidates(meta, args.candidates_per_class)
    expected = {Path(f).name for f in api.list_repo_files(HF_SOURCE_DATASET, repo_type="dataset")
                if f.startswith("data/train-") and f.endswith(".parquet")}
    repo_files = set(api.list_repo_files(repo, repo_type="dataset"))
    missing = expected - {Path(f).name for f in repo_files if f.startswith("staging/")}
    if missing:
        raise SystemExit(f"{len(missing)} shards not fetched yet ({sorted(missing)[:3]}...). Run fetch first.")

    staging = Path(with_retries(lambda: snapshot_download(repo, repo_type="dataset", allow_patterns="staging/*",
                                                          local_dir=BUILD / "staging_repo"), "download staging"))
    parts = sorted((staging / "staging").glob("*.parquet"))
    hashes = pd.concat([pq.read_table(p, columns=["key", "phash"]).to_pandas() for p in parts], ignore_index=True)
    all_rows = candidates.merge(hashes, on="key", how="inner")
    print(f"{len(all_rows):,} of {len(candidates):,} wanted images present in SAGE", flush=True)

    manifests = {}
    for crop_key in CROPS:
        manifest_path = f"manifests/{crop_key}.json"
        if manifest_path in repo_files:
            manifests[crop_key] = json.loads(Path(hf_hub_download(repo, manifest_path, repo_type="dataset")).read_text())
            print(f"{crop_key}: already published, skipping", flush=True)
            continue
        rows = all_rows[all_rows.crop_key == crop_key]
        kept, duplicates, conflicts = deduplicate(rows)
        plan = assign_splits(kept, args.val_frac, args.test_frac, args.train_cap, args.eval_cap)
        out_dir = BUILD / "publish" / crop_key
        write_crop(crop_key, plan, parts, out_dir)
        manifest = {
            "crop": crop_key,
            "excluded_sources": sorted(EXCLUDE_SOURCES),
            "fetched": len(rows), "duplicates_removed": duplicates, "label_conflicts_dropped": conflicts,
            "classes": {label: group.groupby("split").size().to_dict() for label, group in plan.groupby("label")},
            "test_by_source": plan[plan.split == "test"].groupby("source").size().to_dict(),
        }
        small = {k: v.get("train", 0) for k, v in manifest["classes"].items() if v.get("train", 0) < args.min_train}
        if small:
            print(f"WARNING {crop_key}: classes under {args.min_train} train images: {small}", flush=True)
        (out_dir / "manifests").mkdir(exist_ok=True)
        (out_dir / "manifests" / f"{crop_key}.json").write_text(json.dumps(manifest, indent=2))
        with_retries(lambda: api.upload_folder(folder_path=str(out_dir), repo_id=repo, repo_type="dataset",
                                               commit_message=f"Add {crop_key}"), f"upload {crop_key}")
        shutil.rmtree(out_dir)
        manifests[crop_key] = manifest
        print(json.dumps(manifest, indent=2), flush=True)

    with_retries(lambda: api.upload_file(path_or_fileobj=dataset_card(manifests).encode(), path_in_repo="README.md",
                                         repo_id=repo, repo_type="dataset"), "upload card")
    if args.delete_staging:
        api.delete_folder("staging", repo_id=repo, repo_type="dataset", commit_message="Remove build staging")
    shutil.rmtree(BUILD, ignore_errors=True)
    print(f"Published https://huggingface.co/datasets/{repo}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fetch", "finalize"])
    parser.add_argument("--repo", help="Target dataset repo (default <your-username>/sage-india-crops)")
    parser.add_argument("--public", action="store_true", help="Create the repo as public (default private)")
    parser.add_argument("--max-shards", type=int, help="fetch: stop after N shards; use 1 for a first check")
    parser.add_argument("--candidates-per-class", type=int, default=5000)
    parser.add_argument("--train-cap", type=int, default=2500, help="Maximum train images per class")
    parser.add_argument("--eval-cap", type=int, default=300, help="Maximum validation/test images per class")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--min-train", type=int, default=150, help="Warn when a class ends up below this")
    parser.add_argument("--delete-staging", action="store_true",
                        help="finalize: remove staging/ from the repo after publishing")
    args = parser.parse_args()

    api = HfApi()
    repo = args.repo or f"{api.whoami()['name']}/sage-india-crops"
    api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"Target: {repo}", flush=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    (fetch if args.stage == "fetch" else finalize)(args, api, repo)


if __name__ == "__main__":
    main()
