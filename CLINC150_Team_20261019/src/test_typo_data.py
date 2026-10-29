"""Validate deterministic and nested perturbations, including character bounds."""

import random
import unittest

from typo_data import make_variants, mutate_token


class TypoTests(unittest.TestCase):
    def test_reproducible_nested_variants(self):
        text = "Please transfer funds between checking accounts!"
        first = make_variants(text, 7)
        second = make_variants(text, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["clean"]["text"], text)
        self.assertEqual(first["typo_light"]["modified_tokens"], 1)
        self.assertEqual(first["typo_medium"]["modified_tokens"], 2)
        self.assertEqual(first["typo_light"]["mutations"], first["typo_medium"]["mutations"][:1])

    def test_ineligible_inputs_unchanged(self):
        for text in ("yes", "no", "12345", "", "abcd a1234567b"):
            result = make_variants(text, 0)
            self.assertTrue(all(variant["text"] == text for variant in result.values()))

    def test_one_eligible_token_same_light_and_medium(self):
        result = make_variants("check my pin", 1)
        self.assertEqual(result["typo_light"]["text"], result["typo_medium"]["text"])
        self.assertEqual(result["typo_medium"]["modified_tokens"], 1)

    def test_operations_preserve_endpoints_and_follow_definition(self):
        for seed in range(200):
            for token in ("aaaaa", "checking", "transfer", "Please"):
                modified, operation, index = mutate_token(token, random.Random(seed))
                self.assertNotEqual(token, modified)
                self.assertEqual(token[0], modified[0])
                self.assertEqual(token[-1], modified[-1])
                if operation == "delete":
                    self.assertEqual(modified, token[:index] + token[index + 1:])
                    self.assertGreaterEqual(index, 1)
                    self.assertLess(index, len(token) - 1)
                else:
                    self.assertEqual(modified, token[:index] + token[index + 1] + token[index] + token[index + 2:])
                    self.assertLess(index + 1, len(token) - 1)


if __name__ == "__main__":
    unittest.main()
