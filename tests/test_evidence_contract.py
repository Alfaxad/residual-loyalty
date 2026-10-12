#!/usr/bin/env python3
"""Small release-contract checks for the frozen public evidence."""

import csv
import json
import unittest
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvidenceContract(unittest.TestCase):
    def test_vireo_raw_evidence_and_freezes(self):
        spec = importlib.util.spec_from_file_location("vireo_audit", ROOT / "scripts/audit_vireo_evidence.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.run()
        self.assertTrue(report["all_checks_pass"])
        self.assertEqual(report["condition_judge_checks"], 116)
        self.assertEqual([s["families"] for s in report["evaluations"]], [48, 32])

    def test_forecast_population(self):
        with (ROOT / "evidence/tables/checkpoint_roster.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 57)
        self.assertEqual(sum(int(r["forecast_queries"]) for r in rows), 14_592)
        self.assertEqual(sum(r["cohort"] == "old" for r in rows), 45)
        self.assertEqual(sum(r["cohort"] == "strong_remedy" for r in rows), 12)

    def test_core_joint_criterion(self):
        with (ROOT / "evidence/tables/core_remediation_by_checkpoint.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(r["bounded_joint_criterion"] == "True" for r in rows), 7)

    def test_recipe_gate(self):
        with (ROOT / "evidence/tables/remedy_recipe_gates.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["recipe"] for r in rows if r["passes_all"] == "True"], ["direct192"])

    def test_forecast_identity_audit(self):
        report = json.loads((ROOT / "evidence/tables/forecast_identity_audit.json").read_text())
        self.assertTrue(report["all_checks_pass"])
        self.assertEqual(len(report["models"]), 12)


if __name__ == "__main__":
    unittest.main()
