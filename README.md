# SAGE crop disease models

One Qwen2.5-VL-3B-Instruct LoRA adapter per crop, trained on image + crop to name a disease.
Built to run on Kaggle; data comes from [tirtho149/SAGE](https://huggingface.co/datasets/tirtho149/SAGE).

| Model | Diseases |
|---|---|
| `rice` | Blast, Brown Spot, Bacterial Leaf Blight, Tungro |
| `corn` (also serves "maize") | Common Rust, Northern Leaf Blight, Gray Leaf Spot, Southern Rust, Smut |
| `soybean` | Frogeye Leaf Spot, Bacterial Blight, Bacterial Pustule, Brown Spot, Rust |
| `tomato` | Tomato Leaf Curl Virus, Early Blight, Late Blight, Septoria Leaf Blotch, Bacterial Leaf Spot |
| `wheat` | Rust, Smut, Septoria Leaf Blotch, Powdery Mildew, Head Scab |

Selection rationale and counts: [reports/metadata_audit.md](reports/metadata_audit.md) and `sage_train/catalog.py`.

## Kaggle setup

- **Settings:** Internet on. Accelerator **GPU T4** (the code uses one GPU, so T4 x2 adds nothing).
- **Add-ons > Secrets:** `HF_TOKEN`, a Hugging Face token with write access.
- **Add Input** (training only): Qwen2.5-VL-3B-Instruct. It is found automatically under `/kaggle/input`.

Run this first in every session:

```
!git clone https://github.com/<you>/sage-train.git || git -C sage-train pull
%cd sage-train
!pip install -q -r requirements-kaggle.txt
```

## 1. Build the dataset (once; no GPU needed)

```
!python scripts/build_dataset.py fetch --max-shards 1   # check that one shard works
!python scripts/build_dataset.py fetch                  # the remaining 47 shards, about 20 GB in total
!python scripts/build_dataset.py finalize --delete-staging
```

This publishes a private `<you>/sage-india-crops` with one config per crop. Read the WARNING lines
and the manifests: de-duplication may shrink classes a lot. If a session dies, rerun the same
command; finished shards and crops are skipped.

## 2. Train (one crop per session)

Do the smoke test once. It takes about 10 minutes and covers training, the simulated disconnect,
resume, evaluation and Hub pushes, using a separate `-smoke` repo you can delete afterwards.

```
!python scripts/train.py --crop rice --smoke --stop-after-steps 2
!python scripts/train.py --crop rice --smoke
```

Then for each crop:

```
!python scripts/train.py --crop rice
```

Use **Save Version > Save & Run All** so the run continues with the browser closed. If it stops,
rerun the same command: it resumes from the newest checkpoint, locally or on the Hub, at the exact
batch. The 8-hour budget counts across sessions (`--hours` changes it).

### What happens during a run

- **Progress:** a line every 20 optimizer steps with loss, images/s, epoch ETA, budget used and GPU memory.
- **Each epoch:** accuracy, balanced accuracy, macro F1, per-class precision/recall/F1, confusion
  matrix, invalid-answer rate and per-source accuracy, saved to `metrics/` in the adapter repo.
- **Best adapter:** pushed to the root of `<you>/sage-qwen25vl3b-<crop>-lora` whenever validation improves.
- **Checkpoints:** pushed to `checkpoints/last` every 30 minutes and after each epoch.

### Target and fallback loop

The target is **75% validation balanced accuracy** (mean per-class recall). It is stricter than plain
accuracy, which a large class can inflate; plain accuracy is reported too.

1. Train 3 base epochs. If the target is met, stop and save the remaining GPU quota.
2. If not, add epochs at a lower learning rate.
3. After 2 epochs with no improvement, reload the best adapter and halve the learning rate (up to 2 times).
4. Stop at a further plateau, at 10 epochs, or when the budget is used. Then test the best adapter once.

Missing the target is reported as `TARGET NOT MET`; the best adapter is still pushed. Delete the
adapter repo to retrain from scratch. Options to try, in order: read the confusion matrix, raise
`--max-pixels` (with a lower `--batch-size` if memory runs out), then merge or remove a confusable
class in `catalog.py` (Southern Rust vs Common Rust is the likeliest). Changing classes means
rebuilding that crop's dataset.

## Prompt

The prompt lives in `sage_train/prompts.py`. Each crop gets a pathologist system prompt, and the
user turn lists that crop's diseases with a visual cue for each. For rice:

```
Crop: Rice
Which disease does this photo show? Choose exactly one:
- Blast: spindle-shaped lesions with grey centres and brown margins; neck or node rot
- Brown Spot: oval brown spots with grey centres, scattered over the leaf like sesame seeds
- ...
Reply on one line, in this format: Disease: <name>
```

Target output: `Disease: Blast`. Soil, weather, leaf stage and pathogen are never in this prompt.

## Environmental context (after the prediction)

`sage_train/advisory.py` holds the contract for your search-connected LLM:

1. `build_advisory_prompt(crop, disease, context)` makes the prompt from the predicted disease and frontend fields.
2. Send it to your LLM, which searches and returns JSON.
3. `validate_advisory(reply, context)` makes it safe to display. Any note marked `established` without
   a source URL becomes `possible`, and a disclaimer is attached.

The LLM and search provider are not wired in. Validation checks that a URL is present, not that the
page supports the claim.

## Known limits of the data

- **No healthy class.** SAGE has no healthy labels for these crops, so a healthy leaf still gets a
  disease name. The frontend should say so.
- **Source bias.** About 95% of the four main soybean classes come from one source (`Soybean_PNAS`),
  while soybean Rust comes only from others. Corn's Southern Rust is 95% from `Corn_Disease`. The model
  can learn a source's photo style rather than the disease, so check per-source test accuracy.
- **Lab-style images.** Tomato and corn lean on PlantVillage-style photos. Expect lower accuracy on
  field photos than the headline number; `PlantDoc` and `Bugwood` rows in per-source results are the
  best guide.
- **SAGE metadata isn't used for labels' facts.** Its pathogen and symptom columns are often empty and
  sometimes wrong, so pathogen and type come from `catalog.py`. These curated values should be
  reviewed by an agronomist before farmers see them.

## Development

Pure logic is tested without ML libraries; this needs only pandas and pyarrow:

```
python -m unittest discover -s tests
python scripts/audit_metadata.py   # regenerates the audit report
```
