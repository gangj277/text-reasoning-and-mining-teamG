import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class ResearchGateTests(unittest.TestCase):
    def test_gate_passes_when_all_thresholds_met(self):
        from research_gate import DEFAULT_GATE, evaluate_gate

        metrics = {
            "accuracy_8way": DEFAULT_GATE["accuracy_8way"],
            "macro_f1_present_labels": DEFAULT_GATE["macro_f1_present_labels"],
            "risk_detection": {
                "precision": DEFAULT_GATE["risk_precision"],
                "recall": DEFAULT_GATE["risk_recall"],
                "f1": DEFAULT_GATE["risk_f1"],
                "average_precision": DEFAULT_GATE["risk_auprc"],
            },
            "risk_type_macro_f1_present_risk_labels": DEFAULT_GATE["risk_type_macro_f1_present"],
        }
        result = evaluate_gate(metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed"], {})

    def test_gate_reports_failed_thresholds(self):
        from research_gate import DEFAULT_GATE, evaluate_gate

        metrics = {
            "accuracy_8way": 0.1,
            "macro_f1_present_labels": DEFAULT_GATE["macro_f1_present_labels"],
            "risk_detection": {
                "precision": 0.2,
                "recall": DEFAULT_GATE["risk_recall"],
                "f1": DEFAULT_GATE["risk_f1"],
                "average_precision": DEFAULT_GATE["risk_auprc"],
            },
            "risk_type_macro_f1_present_risk_labels": DEFAULT_GATE["risk_type_macro_f1_present"],
        }
        result = evaluate_gate(metrics)
        self.assertFalse(result["passed"])
        self.assertIn("accuracy_8way", result["failed"])
        self.assertIn("risk_precision", result["failed"])


if __name__ == "__main__":
    unittest.main()
