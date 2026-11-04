"""Fine-tune a per-crop LoRA adapter for Qwen2.5-VL on Kaggle.

  python scripts/train.py --crop rice --smoke   # ~10 min end-to-end check, pushes to a -smoke repo
  python scripts/train.py --crop rice           # real run; rerun the same command to resume

The compute budget (default 8 h) is cumulative across sessions. Checkpoints go to the Hub
every 30 minutes, so a lost Kaggle session costs at most that much training.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_train.kaggle_env import WORK, discover_model, setup_env, with_retries

setup_env()

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from PIL import ImageEnhance, ImageOps
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from sage_train.catalog import CROPS, disease_names, resolve_crop
from sage_train.metrics import classification_metrics, format_report
from sage_train.prompts import build_messages, parse_prediction, prediction_details
from sage_train.training_utils import (FallbackPolicy, epoch_order, format_duration, learning_rate, mask_labels,
                                       next_action)

ASSISTANT_HEADER = "<|im_start|>assistant\n"
# Text attention in every layer, plus the last 8 of Qwen2.5-VL's 32 vision blocks and the
# patch merger, so lesion-level features can adapt. Tolerates both transformers module layouts.
LORA_TARGETS = (r".*(language_model\.layers|model\.layers)\.\d+\.self_attn\.(q|k|v|o)_proj"
                r"|.*visual\.blocks\.(2[4-9]|3[01])\.attn\.(q|k|v|qkv|proj)"
                r"|.*visual\.merger\.mlp\.(0|2)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True, help="rice, corn (or maize), soybean, tomato, wheat")
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end check using a separate -smoke repo")
    p.add_argument("--model-path", help="Default: the Qwen2.5-VL model attached through Kaggle's input bar")
    p.add_argument("--dataset", help="Default: <your-username>/sage-india-crops")
    p.add_argument("--repo", help="Adapter repo. Default: <your-username>/sage-qwen25vl3b-<crop>-lora")
    p.add_argument("--public", action="store_true", help="Create the adapter repo as public (default private)")
    p.add_argument("--hours", type=float, default=8.0, help="Cumulative compute budget for this crop")
    p.add_argument("--reserve-minutes", type=float, default=30, help="Kept back for the final test and push")
    p.add_argument("--target", type=float, default=0.75, help="Validation balanced accuracy to reach")
    p.add_argument("--base-epochs", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=10, help="Upper limit including fallback epochs")
    p.add_argument("--patience", type=int, default=2, help="Fallback epochs without improvement before acting")
    p.add_argument("--fallback-rounds", type=int, default=2, help="Reloads of the best adapter at a lower LR")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-pixels", type=int, default=448 * 448)
    p.add_argument("--min-pixels", type=int, default=224 * 224)
    p.add_argument("--oversample-cap", type=int, default=4)
    p.add_argument("--checkpoint-minutes", type=float, default=30)
    p.add_argument("--log-every", type=int, default=20, help="Optimizer steps between progress lines")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop-after-steps", type=int,
                   help="Checkpoint and exit at this step, simulating a disconnect to test resume")
    return p.parse_args()


class CropImages(Dataset):
    def __init__(self, split, augment):
        self.split, self.augment = split, augment

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        row = self.split[index]
        image = row["image"].convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        return image, row["disease"], row["source"]


class TrainCollator:
    def __init__(self, processor, crop):
        self.processor, self.crop = processor, crop
        self.header = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

    def __call__(self, items):
        images, labels, _ = zip(*items)
        texts = [self.processor.apply_chat_template(build_messages(self.crop, label), tokenize=False)
                 for label in labels]
        self.processor.tokenizer.padding_side = "right"
        batch = self.processor(text=texts, images=list(images), padding=True, return_tensors="pt")
        pad = self.processor.tokenizer.pad_token_id
        batch["labels"] = torch.tensor([mask_labels(row, self.header, pad) for row in batch["input_ids"].tolist()])
        return batch


class EvalCollator:
    def __init__(self, processor, crop):
        self.processor = processor
        self.text = processor.apply_chat_template(build_messages(crop), tokenize=False, add_generation_prompt=True)

    def __call__(self, items):
        images, labels, sources = zip(*items)
        self.processor.tokenizer.padding_side = "left"
        batch = self.processor(text=[self.text] * len(items), images=list(images), padding=True, return_tensors="pt")
        return batch, list(labels), list(sources)


class Hub:
    """Pushes that never kill a training run: a failed upload is logged and retried next time."""

    def __init__(self, api, repo):
        self.api, self.repo = api, repo

    def push(self, folder, path_in_repo, message):
        try:
            with_retries(lambda: self.api.upload_folder(folder_path=str(folder), path_in_repo=path_in_repo or None,
                                                        repo_id=self.repo, commit_message=message),
                         f"push {message}", attempts=3, wait=20)
        except Exception as e:
            print(f"WARNING: push failed, continuing locally: {type(e).__name__}: {e}", flush=True)


def evaluate(model, processor, split, crop, args, desc):
    labels = disease_names(crop)
    loader = DataLoader(CropImages(split, augment=False), batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=EvalCollator(processor, crop))
    y_true, y_pred, sources = [], [], []
    samples = []
    model.eval()
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for number, (batch, true, source) in enumerate(loader, 1):
            batch = batch.to("cuda")
            output = model.generate(**batch, max_new_tokens=16, do_sample=False, use_cache=True, temperature=None,
                                    top_p=None, top_k=None, pad_token_id=processor.tokenizer.pad_token_id)
            texts = processor.tokenizer.batch_decode(output[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            y_pred += [parse_prediction(t, labels) for t in texts]
            y_true += true
            sources += source
            if len(samples) < 5:
                samples += [{"true": a, "output": b} for a, b in zip(true, texts)][:5 - len(samples)]
            if number % 25 == 0:
                print(f"  {desc}: {len(y_true)}/{len(split)} images", flush=True)
    model.train()
    metrics = classification_metrics(y_true, y_pred, labels, sources)
    metrics.update(samples=samples, eval_seconds=time.time() - started)
    return metrics


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    temp = Path(path).with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def restore_from_hub(api, repo, out, last, best):
    """Bring in Hub checkpoints if they are newer than what this session has on disk."""
    files = set(api.list_repo_files(repo))
    remote_state = "checkpoints/last/training_state.json"
    local_step = read_json(last / "training_state.json")["global_step"] if (last / "training_state.json").exists() else -1
    if remote_state in files:
        remote_step = read_json(hf_hub_download(repo, remote_state))["global_step"]
        if remote_step > local_step:
            print(f"Restoring checkpoint at step {remote_step} from the Hub", flush=True)
            snapshot = Path(snapshot_download(repo, allow_patterns=["checkpoints/last/*", "metrics/*"]))
            shutil.rmtree(last, ignore_errors=True)
            shutil.copytree(snapshot / "checkpoints" / "last", last)
            if (snapshot / "metrics").exists():
                shutil.copytree(snapshot / "metrics", out / "metrics", dirs_exist_ok=True)
    if "adapter_model.safetensors" in files and not (best / "adapter_model.safetensors").exists():
        best.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(hf_hub_download(repo, name), best / name)


def model_card(crop, repo_dataset, state, test):
    rows = "\n".join(f"| {name} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |"
                     for name, c in test["per_class"].items())
    sources = "\n".join(f"| {name} | {s['accuracy']:.3f} | {s['n']} |" for name, s in test["per_source"].items())
    return f"""---
library_name: peft
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags: [lora, plant-disease, {crop}]
---
# {CROPS[crop]['display']} disease adapter

LoRA adapter for Qwen2.5-VL 3B, trained on image + crop only using [{repo_dataset}](https://huggingface.co/datasets/{repo_dataset}).
It always answers with one of the diseases below: it has no "healthy" or "other" class.

Status: **{state['stop_reason']}** after {state['epochs_done']} epochs, best validation balanced accuracy {state['best_score']:.3f}.

Test: accuracy **{test['accuracy']:.3f}**, balanced accuracy **{test['balanced_accuracy']:.3f}**, macro F1 **{test['macro_f1']:.3f}** (n={test['n']}).

| Disease | Precision | Recall | F1 | n |
|---|---|---|---|---|
{rows}

Test accuracy by original source. A large gap between lab-style and field-style sources means real-world accuracy will be lower than the headline number.

| Source | Accuracy | n |
|---|---|---|
{sources}
"""


def main():
    args = parse_args()
    crop = resolve_crop(args.crop)
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Set Kaggle's Accelerator to GPU T4.")
    if args.smoke:
        args.base_epochs, args.max_epochs, args.checkpoint_minutes, args.log_every = 1, 1, 0, 1

    api = HfApi()
    user = api.whoami()["name"]
    suffix = "-smoke" if args.smoke else ""
    repo = args.repo or f"{user}/sage-qwen25vl3b-{crop}-lora{suffix}"
    dataset_repo = args.dataset or f"{user}/sage-india-crops"
    api.create_repo(repo, private=not args.public, exist_ok=True)
    hub = Hub(api, repo)

    out = WORK / "adapters" / f"{crop}{suffix}"
    last, best, metrics_dir = out / "last", out / "best", out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    restore_from_hub(api, repo, out, last, best)

    state = {"global_step": 0, "epoch": 0, "epochs_done": 0, "batches_done": 0, "best_score": -1.0,
             "epochs_since_best": 0, "fallback_round": 0, "past_base": False, "elapsed_s": 0.0,
             "stop_reason": None, "done": False}
    resuming = (last / "training_state.json").exists()
    if resuming:
        state.update(read_json(last / "training_state.json"))
        print(f"Resuming: epoch {state['epoch'] + 1}, batch {state['batches_done']}, step {state['global_step']}, "
              f"{format_duration(state['elapsed_s'])} already used", flush=True)
    if state["done"]:
        print(f"{crop} already finished ({state['stop_reason']}). Delete {repo} to retrain.", flush=True)
        return

    session_start = time.time()
    elapsed = lambda: state["elapsed_s"] + time.time() - session_start
    train_deadline = args.hours * 3600 - args.reserve_minutes * 60

    random.seed(args.seed + state["global_step"])
    torch.manual_seed(args.seed + state["global_step"])

    data = with_retries(lambda: load_dataset(dataset_repo, crop), "load dataset")
    train_split, val_split, test_split = data["train"], data["validation"], data["test"]
    if args.smoke:
        train_split = train_split.shuffle(seed=args.seed).select(range(min(32, len(train_split))))
        val_split = val_split.shuffle(seed=args.seed).select(range(min(16, len(val_split))))
        test_split = test_split.shuffle(seed=args.seed).select(range(min(16, len(test_split))))
    train_labels = list(train_split["disease"])
    print(f"{crop}: train={len(train_split)} val={len(val_split)} test={len(test_split)} "
          f"classes={disease_names(crop)}", flush=True)

    model_path = discover_model(args.model_path)
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if resuming:
        model = PeftModel.from_pretrained(model, str(last), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, bias="none",
                                                 lora_dropout=args.lora_dropout, target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    vision_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "visual" in n)
    if vision_params == 0:
        raise SystemExit("LoRA matched no vision modules: update LORA_TARGETS for this transformers version.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    if resuming:
        saved = torch.load(last / "optimizer.pt", map_location="cuda", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])

    policy = FallbackPolicy(args.target, args.base_epochs, args.max_epochs, args.patience, args.fallback_rounds)
    batches_per_epoch = len(epoch_order(train_labels, args.seed, args.oversample_cap)) // args.batch_size
    steps_per_epoch = max(1, batches_per_epoch // args.accum)
    base_steps = steps_per_epoch * args.base_epochs
    warmup_steps = max(1, base_steps // 20)
    print(f"{batches_per_epoch} batches/epoch, {steps_per_epoch} optimizer steps/epoch; "
          f"budget {args.hours}h, training stops by {format_duration(train_deadline)}", flush=True)

    def save_checkpoint():
        """Write atomically so a session dying mid-save leaves the previous checkpoint intact."""
        temp = out / "last.tmp"
        shutil.rmtree(temp, ignore_errors=True)
        model.save_pretrained(temp)
        torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, temp / "optimizer.pt")
        state["elapsed_s"] = elapsed()
        write_json(temp / "training_state.json", {**state, "args": vars(args)})
        shutil.rmtree(last, ignore_errors=True)
        temp.rename(last)

    def push_checkpoint():
        save_checkpoint()
        hub.push(last, "checkpoints/last", f"Checkpoint step {state['global_step']}")
        hub.push(metrics_dir, "metrics", f"Metrics step {state['global_step']}")

    def restore_best():
        path = best / "adapter_model.safetensors"
        if path.exists():
            set_peft_model_state_dict(model, load_file(path))
        else:
            print("WARNING: no saved best adapter found; keeping current weights", flush=True)

    model.train()
    last_push = time.time()
    oom_count = 0
    stop = False
    while not stop:
        order = epoch_order(train_labels, args.seed + state["epoch"], args.oversample_cap)
        remaining = order[state["batches_done"] * args.batch_size:]
        loader = DataLoader(Subset(CropImages(train_split, augment=True), remaining), batch_size=args.batch_size,
                            shuffle=False, drop_last=True, num_workers=args.workers, pin_memory=True,
                            collate_fn=TrainCollator(processor, crop))
        optimizer.zero_grad(set_to_none=True)
        loss_sum, loss_count, images_seen = 0.0, 0, 0
        epoch_start = window_start = time.time()
        window_images = 0
        for batch in loader:
            batch = batch.to("cuda")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
                scaler.scale(loss / args.accum).backward()
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                del batch
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"WARNING: out of GPU memory, batch skipped ({oom_count} so far)", flush=True)
                if oom_count > 20:
                    raise SystemExit("Too many out-of-memory errors: lower --batch-size or --max-pixels.")
                state["batches_done"] += 1
                continue
            state["batches_done"] += 1
            loss_sum += loss.item()
            loss_count += 1
            images_seen += args.batch_size
            window_images += args.batch_size
            if state["batches_done"] % args.accum:
                continue

            lr = learning_rate(state["global_step"], warmup_steps, base_steps, args.lr,
                               state["fallback_round"], state["past_base"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1

            if state["global_step"] % args.log_every == 0:
                speed = window_images / max(1e-6, time.time() - window_start)
                left = (batches_per_epoch - state["batches_done"]) * args.batch_size / max(1e-6, speed)
                print(f"[{crop}] epoch {state['epoch'] + 1} batch {state['batches_done']}/{batches_per_epoch} "
                      f"({state['batches_done'] / batches_per_epoch:.1%}) loss={loss_sum / loss_count:.4f} lr={lr:.2e} "
                      f"{speed:.2f} img/s epoch_eta={format_duration(left)} "
                      f"budget_used={format_duration(elapsed())}/{args.hours}h "
                      f"gpu_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
                window_start, window_images = time.time(), 0

            if args.stop_after_steps and state["global_step"] >= args.stop_after_steps:
                push_checkpoint()
                print(f"Stopped at step {state['global_step']} as requested. Rerun without "
                      "--stop-after-steps to resume.", flush=True)
                return
            if args.checkpoint_minutes and time.time() - last_push > args.checkpoint_minutes * 60:
                push_checkpoint()
                last_push = time.time()
            if elapsed() > train_deadline:
                print("Compute budget reached: stopping training for final evaluation", flush=True)
                state["stop_reason"] = "stop_compute_budget"
                stop = True
                break

        partial = stop and state["batches_done"] < batches_per_epoch
        metrics = evaluate(model, processor, val_split, crop, args, "validation")
        score = metrics["balanced_accuracy"]
        improved = score > state["best_score"]
        record = {
            "epoch": state["epoch"] + 1, "partial_epoch": partial, "global_step": state["global_step"],
            "train_loss": loss_sum / loss_count if loss_count else None,
            "train_images_per_second": images_seen / max(1e-6, time.time() - epoch_start - metrics["eval_seconds"]),
            "budget_used_seconds": elapsed(), "fallback_round": state["fallback_round"], "best": improved,
            "validation": metrics,
        }
        write_json(metrics_dir / f"epoch_{state['epoch'] + 1:02d}.json", record)
        with open(metrics_dir / "history.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in record.items() if k != "validation"}
                               | {k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "invalid_rate")}) + "\n")
        print(f"\n=== {crop} epoch {state['epoch'] + 1}{' (partial)' if partial else ''} | "
              f"train_loss={record['train_loss']} | {'NEW BEST' if improved else 'no improvement'}\n"
              f"{format_report(metrics)}\nsample outputs: {metrics['samples'][:3]}\n", flush=True)

        if improved:
            state["best_score"], state["epochs_since_best"] = score, 0
            shutil.rmtree(best, ignore_errors=True)
            model.save_pretrained(best)
            hub.push(best, "", f"Best adapter: epoch {state['epoch'] + 1}, balanced accuracy {score:.4f}")
        else:
            state["epochs_since_best"] += 1

        if not stop:
            state["epochs_done"] += 1
            action = next_action(state["epochs_done"], state["best_score"], state["epochs_since_best"],
                                 state["fallback_round"], policy)
            state["epoch"] += 1
            state["batches_done"] = 0
            if state["epochs_done"] >= args.base_epochs and action == "continue":
                print(f"Validation {state['best_score']:.3f} is below target {args.target}: adding an epoch", flush=True)
                state["past_base"] = True
            elif action == "reload_best":
                state["fallback_round"] += 1
                state["epochs_since_best"], state["past_base"] = 0, True
                restore_best()
                # Fresh optimizer: its momentum belongs to the weights just discarded.
                optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
                print(f"Plateau below target: reloaded best adapter, fallback round {state['fallback_round']} "
                      f"with lower learning rate", flush=True)
            elif action.startswith("stop"):
                state["stop_reason"] = action
                stop = True
        else:
            state["epochs_done"] += 0 if partial else 1
        push_checkpoint()
        last_push = time.time()

    print(f"Training ended: {state['stop_reason']}. Testing the best adapter.", flush=True)
    restore_best()
    test = evaluate(model, processor, test_split, crop, args, "test")
    write_json(metrics_dir / "test.json", test)
    print(f"\n=== {crop} TEST\n{format_report(test)}\n", flush=True)

    state["done"] = True
    state["target_met_on_test"] = test["balanced_accuracy"] >= args.target
    summary = {"crop": crop, "stop_reason": state["stop_reason"], "epochs_done": state["epochs_done"],
               "best_validation_balanced_accuracy": state["best_score"],
               "test_accuracy": test["accuracy"], "test_balanced_accuracy": test["balanced_accuracy"],
               "test_macro_f1": test["macro_f1"], "target": args.target,
               "target_met_on_test": state["target_met_on_test"], "budget_used_seconds": elapsed(),
               "labels": {d: prediction_details(crop, d) for d in disease_names(crop)}}
    best.mkdir(parents=True, exist_ok=True)
    write_json(best / "training_summary.json", summary)
    (best / "README.md").write_text(model_card(crop, dataset_repo, state, test), encoding="utf-8")
    processor.save_pretrained(best)
    hub.push(best, "", f"Final best adapter, test balanced accuracy {test['balanced_accuracy']:.4f}")
    push_checkpoint()
    print(json.dumps(summary, indent=2), flush=True)
    if not state["target_met_on_test"]:
        print(f"TARGET NOT MET: test balanced accuracy {test['balanced_accuracy']:.3f} < {args.target}. "
              "The best adapter was still pushed; see README for next steps.", flush=True)
    print(f"Done: https://huggingface.co/{repo}", flush=True)


if __name__ == "__main__":
    main()
