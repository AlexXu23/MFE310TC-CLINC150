"""Small boundary tests for the extended audit and error-analysis helpers."""
import unittest

import numpy as np
import pandas as pd

from controlled_audit import ece, paired_loss_intervals
from data_audit import duplicate_groups, normalize_text
from error_analysis import choose_cases, system_errors


class AnalysisTests(unittest.TestCase):
    def test_normalization_is_conservative(self):
        self.assertEqual(normalize_text("  Ａ B\tC "), "a b c")
        self.assertNotEqual(normalize_text("can't"), normalize_text("cant"))

    def test_duplicates_distinguish_scope_from_official_split(self):
        frame = pd.DataFrame({"text": ["x", "x", "y", "y"], "label": ["a", "b", "c", "c"],
                              "split": ["train", "train", "train", "test"],
                              "source_split": ["train", "oos_train", "train", "test"], "source_row": [0, 0, 1, 0]})
        table = duplicate_groups(frame, "text")
        self.assertFalse(table.loc[table.text == "x", "cross_official_split"].any())
        self.assertTrue(table.loc[table.text == "y", "cross_official_split"].all())
        self.assertTrue(table.loc[table.text == "x", "label_conflict"].all())

    def test_errors_form_disjoint_partition(self):
        frame = pd.DataFrame({"is_oos": [True, True, False, False, False],
                              "predicted_label_id": [0, 0, 0, 0, 1], "label_id": [-1, -1, 0, 1, 1],
                              "raw_msp_max_oos_f1_reject": [False, True, True, False, False]})
        self.assertEqual(system_errors(frame).tolist(), ["missed_oos", "correct", "false_oos", "accepted_intent_error", "correct"])

    def test_case_sampling_is_order_invariant(self):
        frame = pd.DataFrame({"row_id": np.arange(30)})
        first = choose_cases(frame, "errors", 20)["row_id"].tolist()
        second = choose_cases(frame.iloc[::-1], "errors", 20)["row_id"].tolist()
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)

    def test_ece_boundary_confidence_one(self):
        self.assertAlmostEqual(ece(np.array([0., 1.]), np.array([False, True])), 0)
        self.assertAlmostEqual(ece(np.array([1., 1.]), np.array([False, False])), 1)

    def test_bootstrap_identity_gives_zero_interval(self):
        values = {"nll": np.array([.1, .2]), "brier": np.array([.1, .2]),
                  "confidence": np.array([.7, .8]), "correct": np.array([True, False])}
        intervals = paired_loss_intervals(values, values, repeats=20)
        self.assertTrue(all(value == 0 for value in intervals.values()))


if __name__ == "__main__":
    unittest.main()
