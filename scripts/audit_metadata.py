"""Write reports/metadata_audit.md from the metadata parquet. Needs only pandas + pyarrow."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_train.catalog import CROPS, EXCLUDE_SOURCES
from sage_train.selection import select_candidates

CANDIDATE_CAP = 5000


def table(df):
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main():
    meta = pd.read_parquet(ROOT / "metadata" / "sage_metadata.parquet")
    candidates = select_candidates(meta, CANDIDATE_CAP)
    out = [
        "# SAGE metadata audit", "",
        f"Rows: {len(meta):,} | crops: {meta.crop.nunique()} | disease labels: {meta.disease.nunique()} | "
        f"sources: {meta.source.nunique()}",
        f"Unique on (raw_label, filename): {not meta.duplicated(['raw_label', 'filename']).any()} | "
        f"unique on filename alone: {not meta.duplicated(['filename']).any()}",
        f"Rows with a pathogen: {meta.pathogen.notna().mean():.1%}. Sources excluded here: {sorted(EXCLUDE_SOURCES)}.",
        "Counts are before perceptual de-duplication, so real class sizes will be lower. "
        "`build_dataset.py finalize` publishes the final counts in each crop's manifest.", "",
    ]
    for key, crop in CROPS.items():
        rows = meta[(meta.crop == crop["dataset_crop"]) & meta.disease.isin(list(crop["diseases"]))
                    & ~meta.source.isin(EXCLUDE_SOURCES) & ~meta.raw_label.isin(crop["exclude_raw_labels"])]
        fetched = candidates[candidates.crop_key == key].groupby("label").size()
        summary = rows.groupby("disease").agg(rows=("source", "size"), sources=("source", "nunique"),
                                              largest_source=("source", lambda s: s.value_counts().index[0]),
                                              largest_share=("source", lambda s: f"{s.value_counts(normalize=True).iloc[0]:.0%}")
                                              ).reset_index()
        summary["fetched"] = summary["disease"].map(fetched)
        out += [f"## {crop['display']}", "", table(summary), ""]
    path = ROOT / "reports" / "metadata_audit.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
