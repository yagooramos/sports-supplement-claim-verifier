import unittest

from app import EXAMPLE_CLAIMS
from scripts.pipeline import Pipeline


EXPECTED_GROUP_VERDICTS = {
    "Backed claims": "backed",
    "Partially backed claims": "partially_backed",
    "Potentially misleading claims": "potentially_misleading",
    "Out-of-scope / unparseable": "not_evaluable",
}

EXPECTED_REASON_CODES = {
    "Caffeine improves endurance performance.": "dose_condition_not_met",
    "Electrolytes improve hydration performance.": "claim_outside_scope",
    "This supplement makes you unstoppable.": "claim_not_parseable",
}


class ExampleClaimsAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = Pipeline(use_llm=False)

    def test_example_groups_match_expected_verdicts(self):
        for group_name, expected_verdict in EXPECTED_GROUP_VERDICTS.items():
            claims = EXAMPLE_CLAIMS[group_name]
            self.assertTrue(claims, f"{group_name} should not be empty")
            for claim in claims:
                with self.subTest(group=group_name, claim=claim):
                    result = self.pipeline.run(claim_text=claim)
                    verdict = result["reasoning_result"].get("verdict")
                    self.assertEqual(verdict, expected_verdict)

    def test_demo_reason_codes_stay_stable(self):
        for claim, expected_reason_code in EXPECTED_REASON_CODES.items():
            with self.subTest(claim=claim):
                result = self.pipeline.run(claim_text=claim)
                reason_code = result["reasoning_result"].get("reason_code")
                self.assertEqual(reason_code, expected_reason_code)


if __name__ == "__main__":
    unittest.main()
