"""Choose which SAGE rows to fetch and how to split them after de-duplication.

The same photos appear under several SAGE sources (PlantVillage re-uploads and offline
augmentations), so splitting before de-duplication would leak test images into training.
"""
import hashlib
from collections import defaultdict

import pandas as pd

from .catalog import CROPS, EXCLUDE_SOURCES


def stable_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()


def row_key(raw_label, filename):
    """(raw_label, filename) is unique in SAGE; filename alone repeats across labels."""
    return f"{raw_label}|{filename}"  # never NUL: pandas truncates strings at it when hashing


def _round_robin(df, cap):
    """Keep up to `cap` rows, taking evenly across sources so small sources survive whole."""
    df = df.assign(_h=[stable_hash(k) for k in df["key"]]).sort_values("_h")
    df = df.assign(_rank=df.groupby("source").cumcount())
    return df.sort_values(["_rank", "_h"]).head(cap).drop(columns=["_h", "_rank"])


def select_candidates(meta, per_class_cap):
    """Rows to fetch from SAGE, with columns crop_key, label, source, raw_label, filename, key."""
    parts = []
    for crop_key, crop in CROPS.items():
        rows = meta[(meta["crop"] == crop["dataset_crop"])
                    & meta["disease"].isin(list(crop["diseases"]))
                    & ~meta["source"].isin(EXCLUDE_SOURCES)
                    & ~meta["raw_label"].isin(crop["exclude_raw_labels"])]
        for label, group in rows.groupby("disease"):
            group = group.assign(crop_key=crop_key, label=label,
                                 key=[row_key(r, f) for r, f in zip(group["raw_label"], group["filename"])])
            parts.append(_round_robin(group, per_class_cap))
    columns = ["crop_key", "label", "source", "raw_label", "filename", "key"]
    return pd.concat(parts, ignore_index=True)[columns]


def cluster_hashes(variants, max_distance=3):
    """Group images whose hashes differ by at most `max_distance` bits in any orientation.

    `variants[i]` holds image i's 64-bit hash for each orientation, with its unrotated hash
    first. Hashes within 3 bits must agree exactly on at least one of 4 16-bit bands, so
    comparing within band buckets finds every match. Returns each image's cluster root index.
    """
    parent = list(range(len(variants)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets = defaultdict(list)
    for i, hashes in enumerate(variants):
        for band in range(4):
            buckets[(band, (hashes[0] >> (16 * band)) & 0xFFFF)].append(i)
    for i, hashes in enumerate(variants):
        for value in hashes:
            for band in range(4):
                for j in buckets.get((band, (value >> (16 * band)) & 0xFFFF), ()):
                    if j > i and bin(value ^ variants[j][0]).count("1") <= max_distance:
                        parent[find(j)] = find(i)
    return [find(i) for i in range(len(variants))]


def deduplicate(rows):
    """One row per image cluster (see cluster_hashes). A cluster seen under two labels is dropped
    altogether as a label conflict. The copy from the rarest source is kept, preserving source
    diversity. `phash` holds comma-separated hex hashes, unrotated first.
    Returns (kept, n_duplicates, n_label_conflicts)."""
    rows = rows.sort_values("key").reset_index(drop=True)
    variants = [[int(h, 16) for h in p.split(",")] for p in rows["phash"]]
    roots = cluster_hashes(variants)
    rows = rows.assign(cluster=[f"{variants[r][0]:016x}" for r in roots])
    labels_per_cluster = rows.groupby("cluster")["label"].transform("nunique")
    conflicts = rows[labels_per_cluster > 1]
    clean = rows[labels_per_cluster == 1]
    source_size = clean["source"].map(clean["source"].value_counts())
    kept = (clean.assign(_size=source_size)
            .sort_values(["_size", "source", "key"])
            .drop_duplicates("cluster")
            .drop(columns="_size"))
    return kept, len(clean) - len(kept), int(conflicts["cluster"].nunique())


def assign_splits(rows, val_frac, test_frac, train_cap, eval_cap):
    """Split within each (label, source) so every source is represented where possible, then
    cap each class per split, taking evenly across sources. Strata of 3 rows or fewer stay in train."""
    rows = rows.assign(_h=[stable_hash(c) for c in rows["cluster"]]).sort_values("_h")
    size = rows.groupby(["label", "source"])["_h"].transform("size")
    rank = rows.groupby(["label", "source"]).cumcount()
    position = (rank + 0.5) / size
    split = pd.Series("train", index=rows.index)
    eligible = size > 3
    split[eligible & (position < test_frac)] = "test"
    split[eligible & (position >= test_frac) & (position < test_frac + val_frac)] = "validation"
    rows = rows.drop(columns="_h").assign(split=split)

    kept = []
    for (_, name), group in rows.groupby(["label", "split"]):
        kept.append(_round_robin(group, train_cap if name == "train" else eval_cap))
    return pd.concat(kept, ignore_index=True)
