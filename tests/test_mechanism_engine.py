import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class MechanismEngineTests(unittest.TestCase):
    def test_mechanism_features_fire_for_risk_mechanisms(self):
        from mechanism_engine import mechanism_feature_names, mechanism_feature_matrix

        rows = [
            {"title": "희토류 수출통제와 중동 전쟁이 공급망 리스크를 키운다", "lang": "ko"},
            {"title": "삼성전자 총파업으로 반도체 공급망 차질 우려", "lang": "ko"},
            {"title": "Supply chain webinar on digital transformation strategy", "lang": "en"},
        ]
        names = mechanism_feature_names()
        X = mechanism_feature_matrix(rows).toarray()

        self.assertGreater(X[0, names.index("lex_GEOPOLITICAL")], 0)
        self.assertGreater(X[1, names.index("lex_LABOR")], 0)
        self.assertGreater(X[2, names.index("lex_OTHER")], 0)

    def test_auto_cv_splits_respects_rare_classes(self):
        from mechanism_engine import choose_n_splits

        labels = ["OTHER"] * 10 + ["GEOPOLITICAL"] * 6 + ["NATURAL_DISASTER"] * 3
        self.assertEqual(choose_n_splits(labels, requested=5), 3)

    def test_metric_pack_reports_present_and_full_taxonomy(self):
        from mechanism_engine import metric_pack

        y_true = ["OTHER", "OTHER", "GEOPOLITICAL", "LABOR"]
        y_pred = ["OTHER", "GEOPOLITICAL", "GEOPOLITICAL", "OTHER"]
        scores = [0.1, 0.8, 0.9, 0.3]
        metrics = metric_pack(y_true, y_pred, risk_score=scores)

        self.assertIn("macro_f1_present_labels", metrics)
        self.assertIn("macro_f1_full_taxonomy", metrics)
        self.assertIn("risk_detection", metrics)
        self.assertGreater(metrics["macro_f1_present_labels"], metrics["macro_f1_full_taxonomy"])

    def test_best_threshold_by_risk_f1_uses_training_scores(self):
        from mechanism_engine import best_threshold_by_risk_f1

        y_bin = [0, 0, 1, 1]
        scores = [0.10, 0.40, 0.35, 0.90]
        threshold, score = best_threshold_by_risk_f1(y_bin, scores, grid=[0.2, 0.5, 0.8])

        self.assertEqual(threshold, 0.2)
        self.assertGreater(score, 0.79)

    def test_gate_aware_threshold_enforces_precision_floor(self):
        from mechanism_engine import best_threshold_for_research_gate

        y_bin = [0, 0, 0, 0, 1, 1, 1]
        scores = [0.44, 0.43, 0.20, 0.10, 0.42, 0.48, 0.90]
        threshold, metrics = best_threshold_for_research_gate(
            y_bin,
            scores,
            grid=[0.40, 0.45, 0.50],
            min_precision=0.80,
            min_recall=0.60,
        )

        self.assertEqual(threshold, 0.45)
        self.assertGreaterEqual(metrics["precision"], 0.80)
        self.assertGreaterEqual(metrics["recall"], 0.60)

    def test_gate_aware_threshold_fallback_prefers_precision_over_raw_f1(self):
        from mechanism_engine import best_threshold_for_research_gate

        y_bin = [0, 0, 0, 1, 1, 1]
        scores = [0.45, 0.10, 0.20, 0.40, 0.50, 0.90]
        threshold, metrics = best_threshold_for_research_gate(
            y_bin,
            scores,
            grid=[0.40, 0.50],
            min_precision=0.90,
            min_recall=0.90,
        )

        self.assertEqual(threshold, 0.50)
        self.assertGreaterEqual(metrics["precision"], 0.90)
        self.assertTrue(metrics["fallback"])


if __name__ == "__main__":
    unittest.main()
