"""Tests for logic that needs no ML libraries. Run: python -m unittest discover -s tests -v"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sage_train.advisory import build_advisory_prompt, validate_advisory
from sage_train.catalog import CROPS, disease_names, resolve_crop
from sage_train.kaggle_env import prepare_processor_files
from sage_train.metrics import classification_metrics
from sage_train.prompts import build_messages, parse_prediction, user_prompt
from sage_train.selection import assign_splits, cluster_hashes, deduplicate, row_key
from sage_train.training_utils import FallbackPolicy, epoch_order, learning_rate, mask_labels, next_action


class Catalog(unittest.TestCase):
    def test_three_to_five_diseases_per_crop(self):
        for key in CROPS:
            self.assertTrue(3 <= len(disease_names(key)) <= 5, key)

    def test_maize_uses_corn_model(self):
        self.assertEqual(resolve_crop(" Maize "), "corn")
        with self.assertRaises(KeyError):
            resolve_crop("cotton")


class Prompts(unittest.TestCase):
    def test_context_never_reaches_prompt(self):
        for key in CROPS:
            text = user_prompt(key).lower()
            for word in ("soil", "weather", "stage", "pathogen"):
                self.assertNotIn(word, text)

    def test_every_disease_listed(self):
        for key in CROPS:
            for name in disease_names(key):
                self.assertIn(name, user_prompt(key))

    def test_answer_only_when_given(self):
        self.assertEqual(len(build_messages("rice")), 2)
        self.assertEqual(build_messages("rice", "Blast")[-1]["content"][0]["text"], "Disease: Blast")

    def test_parse(self):
        wheat, corn = disease_names("wheat"), disease_names("corn")
        self.assertEqual(parse_prediction("Disease: Rust", wheat), "Rust")
        self.assertEqual(parse_prediction("disease: southern rust.\nmore", corn), "Southern Rust")
        self.assertEqual(parse_prediction("Common Rust", corn), "Common Rust")
        self.assertIsNone(parse_prediction("Disease: Rust", corn))
        self.assertIsNone(parse_prediction("Disease: Smut or Common Rust", corn))
        self.assertIsNone(parse_prediction("", corn))


class Metrics(unittest.TestCase):
    def test_values(self):
        m = classification_metrics(["a", "a", "a", "b"], ["a", "a", None, "a"], ["a", "b"], ["x", "x", "y", "y"])
        self.assertEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["balanced_accuracy"], (2 / 3 + 0) / 2)
        self.assertEqual(m["invalid_rate"], 0.25)
        self.assertEqual(m["confusion"]["a"]["<invalid>"], 1)
        self.assertEqual(m["per_source"]["y"], {"accuracy": 0.0, "n": 2})


class Training(unittest.TestCase):
    P = FallbackPolicy(target=0.75, base_epochs=3, max_epochs=6, patience=2, max_fallback_rounds=1)

    def test_fallback_ladder(self):
        self.assertEqual(next_action(1, 0.99, 0, 0, self.P), "continue")
        self.assertEqual(next_action(3, 0.80, 0, 0, self.P), "stop_target_met")
        self.assertEqual(next_action(3, 0.60, 0, 0, self.P), "continue")
        self.assertEqual(next_action(4, 0.60, 2, 0, self.P), "reload_best")
        self.assertEqual(next_action(5, 0.60, 2, 1, self.P), "stop_plateau")
        self.assertEqual(next_action(6, 0.60, 0, 0, self.P), "stop_max_epochs")

    def test_order_deterministic_and_balanced(self):
        labels = ["a"] * 100 + ["b"] * 10
        order = epoch_order(labels, seed=1, oversample_cap=4)
        self.assertEqual(order, epoch_order(labels, seed=1, oversample_cap=4))
        self.assertNotEqual(order, epoch_order(labels, seed=2, oversample_cap=4))
        self.assertEqual(sum(labels[i] == "b" for i in order), 40)
        self.assertEqual(sum(labels[i] == "a" for i in order), 100)

    def test_lr_schedule(self):
        lr = lambda s, **k: learning_rate(s, 10, 100, 1.0, **k)
        self.assertAlmostEqual(lr(9), 1.0)
        self.assertLess(lr(99), lr(11))
        self.assertAlmostEqual(lr(100), 0.3)
        self.assertAlmostEqual(lr(5, fallback_round=1, past_base=True), 0.15)

    def test_mask_uses_last_header(self):
        self.assertEqual(mask_labels([7, 8, 1, 7, 8, 5, 6, 0, 0], [7, 8], 0), [-100] * 5 + [5, 6, -100, -100])
        with self.assertRaises(ValueError):
            mask_labels([1, 2], [7, 8], 0)


class Selection(unittest.TestCase):
    def test_key_survives_pandas(self):
        keys = pd.Series([row_key("Rice/Blast", f"x_{i}.jpg") for i in range(5)])
        self.assertEqual(keys.nunique(), 5)

    def test_clusters_near_and_rotated_copies(self):
        a, rotated_a, b = 0x123456789ABCDEF0, 0x0FEDCBA987654321, 0xF0F0F0F00F0F0F0F
        variants = [[a, rotated_a], [a ^ 0b111, a], [rotated_a, a], [b, b ^ 1], [a ^ 0xFF, a ^ 0xFF]]
        roots = cluster_hashes(variants)
        self.assertEqual(roots[0], roots[1])  # 3 bits apart
        self.assertEqual(roots[0], roots[2])  # rotated copy
        self.assertNotEqual(roots[0], roots[3])  # different image
        self.assertNotEqual(roots[0], roots[4])  # 5+ bits from everything else

    def _rows(self):
        return pd.DataFrame({
            "key": ["k1", "k2", "k3", "k4", "k5"], "label": ["A", "A", "A", "B", "B"],
            "source": ["big", "big", "small", "big", "big"],
            "phash": ["00000000000000ff", "00000000000000ff", "00000000000000ff",
                      "ffffffff00000000", "ffffffff00000000"],
        })

    def test_dedup_keeps_rarest_source(self):
        kept, duplicates, conflicts = deduplicate(self._rows())
        self.assertEqual((len(kept), duplicates, conflicts), (2, 3, 0))
        self.assertEqual(kept[kept.label == "A"].source.iloc[0], "small")

    def test_conflicting_labels_dropped(self):
        rows = self._rows()
        rows.loc[3, "phash"] = "00000000000000ff"
        kept, _, conflicts = deduplicate(rows)
        self.assertEqual(conflicts, 1)
        self.assertEqual(list(kept.label), ["B"])

    def test_splits_never_share_a_cluster(self):
        rows = pd.DataFrame({"key": [f"k{i}" for i in range(400)], "label": ["A", "B"] * 200,
                             "source": ["s"] * 400, "cluster": [f"c{i}" for i in range(400)]})
        plan = assign_splits(rows, 0.15, 0.15, 1000, 1000)
        self.assertEqual(plan.cluster.nunique(), 400)
        self.assertEqual(plan.split.value_counts().to_dict(), {"train": 280, "test": 60, "validation": 60})
        capped = assign_splits(rows, 0.15, 0.15, 50, 10)
        self.assertEqual(capped.groupby(["label", "split"]).size().max(), 50)
        self.assertEqual(int((capped.split == "test").sum()), 20)


class ProcessorFiles(unittest.TestCase):
    def test_corrects_config_and_skips_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model"
            source.mkdir()
            (source / "preprocessor_config.json").write_text(json.dumps(
                {"image_processor_type": "Qwen2_5_VLImageProcessor", "patch_size": 14}))
            (source / "tokenizer.json").write_text("{}")
            (source / "model.safetensors").write_text("weights")
            original = (source / "preprocessor_config.json").read_text()
            with contextlib.redirect_stdout(io.StringIO()):
                dest = Path(prepare_processor_files(source, Path(tmp) / "out", 100, 200))
            config = json.loads((dest / "preprocessor_config.json").read_text())
            self.assertEqual(config["image_processor_type"], "Qwen2VLImageProcessor")
            self.assertEqual(config["size"], {"shortest_edge": 100, "longest_edge": 200})
            self.assertEqual(config["patch_size"], 14)
            self.assertTrue((dest / "tokenizer.json").exists())
            self.assertFalse((dest / "model.safetensors").exists())
            self.assertEqual((source / "preprocessor_config.json").read_text(), original)


class Advisory(unittest.TestCase):
    CTX = {"soil": "clay", "weather": "humid", "leaf_stage": ""}

    def test_prompt_only_reported_fields(self):
        text = build_advisory_prompt("Rice", "Blast", self.CTX)
        self.assertIn("- soil: clay", text)
        self.assertNotIn("- leaf_stage", text)

    def test_established_needs_a_source(self):
        raw = json.dumps({"notes": [
            {"attribute": "weather", "effect": "x", "certainty": "established", "sources": ["https://a.org"]},
            {"attribute": "soil", "effect": "y", "certainty": "established", "sources": []},
            {"attribute": "leaf_stage", "effect": "not reported", "certainty": "possible"},
        ]})
        notes = validate_advisory(raw, self.CTX)["notes"]
        self.assertEqual([n["certainty"] for n in notes], ["established", "possible"])


if __name__ == "__main__":
    unittest.main()
