"""Training decisions with no ML dependencies, so they can be unit-tested anywhere."""
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class FallbackPolicy:
    target: float = 0.75
    base_epochs: int = 3
    max_epochs: int = 10
    patience: int = 2
    max_fallback_rounds: int = 2


def next_action(epochs_done, best_score, epochs_since_best, fallback_round, policy):
    """What to do after an epoch's validation. Below target, epochs keep being added until
    the score plateaus; each plateau reloads the best adapter at a lower learning rate."""
    if epochs_done < policy.base_epochs:
        return "continue"
    if best_score >= policy.target:
        return "stop_target_met"
    if epochs_done >= policy.max_epochs:
        return "stop_max_epochs"
    if epochs_since_best >= policy.patience:
        return "reload_best" if fallback_round < policy.max_fallback_rounds else "stop_plateau"
    return "continue"


def learning_rate(step, warmup_steps, base_steps, base_lr, fallback_round=0, past_base=False):
    """Warmup then cosine over the base epochs; a low constant rate during added epochs,
    halved at each fallback round."""
    if past_base or step >= base_steps:
        return base_lr * 0.3 * 0.5 ** fallback_round
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, base_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def epoch_order(labels, seed, oversample_cap=4):
    """Shuffled dataset indices for one epoch, with rare classes repeated up to
    `oversample_cap` times towards the largest class. Deterministic per seed so a
    resumed run can skip the batches it already trained on."""
    rng = random.Random(seed)
    by_class = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    largest = max(len(v) for v in by_class.values())
    order = []
    for label in sorted(by_class):
        indices = by_class[label]
        want = min(largest, len(indices) * oversample_cap)
        taken = []
        while len(taken) < want:
            batch = indices[:]
            rng.shuffle(batch)
            taken.extend(batch)
        order.extend(taken[:want])
    rng.shuffle(order)
    return order


def mask_labels(input_ids, header_ids, pad_id, ignore=-100):
    """Labels for one row: only tokens after the last assistant header are trained on."""
    n, m = len(input_ids), len(header_ids)
    for start in range(n - m, -1, -1):
        if input_ids[start:start + m] == header_ids:
            answer_start = start + m
            return [ignore if i < answer_start or t == pad_id else t for i, t in enumerate(input_ids)]
    raise ValueError("Assistant header not found in input_ids; the chat template has changed")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
