"""Classification metrics in plain Python. An unparseable answer counts as wrong."""
from collections import defaultdict

INVALID = "<invalid>"


def classification_metrics(y_true, y_pred, labels, sources=None):
    """y_pred items may be None (unparseable). Balanced accuracy is the mean per-class recall."""
    pred = [INVALID if p is None else p for p in y_pred]
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, pred))
    confusion = {t: {p: 0 for p in [*labels, INVALID]} for t in labels}
    for t, p in zip(y_true, pred):
        confusion[t][p] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[t][label] for t in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    present = [c for c in per_class.values() if c["support"]]
    result = {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "balanced_accuracy": sum(c["recall"] for c in present) / len(present) if present else 0.0,
        "macro_f1": sum(c["f1"] for c in present) / len(present) if present else 0.0,
        "invalid_rate": pred.count(INVALID) / n if n else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }
    if sources is not None:
        tally = defaultdict(lambda: [0, 0])
        for t, p, s in zip(y_true, pred, sources):
            tally[s][0] += t == p
            tally[s][1] += 1
        result["per_source"] = {s: {"accuracy": c / total, "n": total} for s, (c, total) in sorted(tally.items())}
    return result


def format_report(m):
    out = [f"accuracy={m['accuracy']:.4f}  balanced_accuracy={m['balanced_accuracy']:.4f}  "
           f"macro_f1={m['macro_f1']:.4f}  invalid={m['invalid_rate']:.4f}  n={m['n']}"]
    for label, c in m["per_class"].items():
        out.append(f"  {label:<26} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} n={c['support']}")
    for source, s in m.get("per_source", {}).items():
        out.append(f"  source {source:<19} acc={s['accuracy']:.3f} n={s['n']}")
    return "\n".join(out)
