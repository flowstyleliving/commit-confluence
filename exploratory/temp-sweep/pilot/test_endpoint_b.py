"""Small A1 checks for Claude Code. Verification: not run by Codex."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sampling_endpoint_b import (HERE, REPO, MODELS, PLAN, TEMPS, K, cell_status, format_flags,
                                 greedy_audit, temperature_distribution, inverse_cdf, repo_relative_path)
from summarize_endpoint_b import (DERIVED_REPORTS, account_records, check_sampling, diagnostic_means,
                                  preserve_reports, reversal, summarize_cell)


class EndpointBChecks(unittest.TestCase):
    def test_budget_boundaries_and_no_cascade(self):
        self.assertEqual(cell_status(10, 0, 0, 990)[0], "PASS")
        self.assertEqual(cell_status(11, 0, 0, 989)[0], "FAILED")
        self.assertEqual(cell_status(0, 250, 0, 750)[0], "PASS")
        self.assertEqual(cell_status(0, 251, 0, 749)[0], "PASS")
        self.assertEqual(format_flags(250, 1000), [])
        self.assertEqual(format_flags(251, 1000), ["format_usability"])
        self.assertEqual(cell_status(0, 10, 0, 190, greedy=True)[0], "PASS")
        self.assertEqual(cell_status(0, 11, 0, 189, greedy=True)[0], "PASS")
        self.assertEqual(format_flags(10, 200, greedy=True), [])
        self.assertEqual(format_flags(11, 200, greedy=True), ["format_usability"])
        self.assertEqual(cell_status(0, 10, 0, 190, greedy=True, anli=True)[0], "PASS")
        self.assertEqual(cell_status(0, 11, 0, 189, greedy=True, anli=True)[0], "FAILED")
        self.assertEqual(format_flags(0, 0), [])
        self.assertEqual(cell_status(1000, 0, 0, 0)[0], "FAILED")
        self.assertEqual(cell_status(0, 0, 0, 1000)[0], "PASS")

    def test_historical_identity_is_not_an_infrastructure_allowance(self):
        self.assertEqual(cell_status(1, 0, 0, 999, historical_failures=1)[0],
                         "FAILED: historical_identity")
        self.assertEqual(cell_status(50, 0, 0, 4950, historical_failures=50)[0],
                         "FAILED: historical_identity")
        rs = [{"sample_idx": 0, "bucket": "a_harness", "historical_match": False}]
        rs += [{"sample_idx": i, "bucket": "d_valid_correct", "strict_non_yes": 0}
               for i in range(1, 1000)]
        cell = summarize_cell(rs)
        self.assertEqual(cell["status"], "FAILED: historical_identity")
        self.assertEqual(cell["historical_identity_failures"], 1)
        self.assertEqual(cell["infrastructure_failures"], 0)
        self.assertNotIn("harness_budget", cell["reasons"])
        rs[0]["infrastructure_failure"] = True
        self.assertEqual(summarize_cell(rs)["infrastructure_failures"], 1)
        self.assertEqual(summarize_cell(rs[1:])["status"], "PASS")

    def test_sampling_verification_is_zero_tolerance_and_cell_local(self):
        rs = [{"sample_idx": 0, "bucket": "a_harness", "sampling_verification_failed": True}]
        rs += [{"sample_idx": i, "bucket": "d_valid_correct", "strict_non_yes": 0}
               for i in range(1, 1000)]
        cell = summarize_cell(rs)
        self.assertEqual(cell["status"], "FAILED: sampling_verification")
        self.assertEqual(cell["reasons"], ["sampling_verification"])
        self.assertEqual(cell["sampling_verification_failures"], 1)
        self.assertEqual(cell["infrastructure_failures"], 0)
        self.assertEqual(cell["counts"]["a_harness"], 1)
        rs[0].update(infrastructure_failure=True, historical_identity_failed=True)
        cell = summarize_cell(rs)
        self.assertEqual(cell["status"], "FAILED: sampling_verification")
        self.assertEqual(cell["reasons"], ["historical_identity", "sampling_verification"])
        self.assertEqual(cell["historical_identity_failures"], 1)
        self.assertEqual(cell["infrastructure_failures"], 1)
        self.assertEqual(summarize_cell(rs[1:])["status"], "PASS")
        self.assertEqual(cell_status(1, 0, 0, 4999, sampling_verification_failures=1),
                         ("FAILED: sampling_verification", ["sampling_verification"]))

    def test_preserve_derived_reports_only(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            untouched = ("model.jsonl", "manifest.json", "uniforms.npy")
            for name in untouched:
                (run / name).write_bytes(b"raw fixture\x00")
            self.assertIsNone(preserve_reports(run))
            self.assertFalse((run / "superseded").exists())
            for name in DERIVED_REPORTS:
                (run / name).write_text(name)
            archive = preserve_reports(run)
            self.assertEqual(archive.parent, run / "superseded")
            self.assertEqual({p.name for p in archive.iterdir()}, set(DERIVED_REPORTS))
            for name in DERIVED_REPORTS:
                self.assertEqual((archive / name).read_text(), name)
                self.assertFalse((run / name).exists())
            (run / "SUMMARY.md").write_text("rebuilt")
            second = preserve_reports(run)
            self.assertNotEqual(second, archive)
            self.assertEqual((second / "SUMMARY.md").read_text(), "rebuilt")
            self.assertEqual((archive / "SUMMARY.md").read_text(), "SUMMARY.md")
            for name in untouched:
                self.assertEqual((run / name).read_bytes(), b"raw fixture\x00")

    def test_diagnostic_denominators_include_quarantined_distributions(self):
        rs = [{"sample_idx": 0, "bucket": "d_valid_correct", "p_valid": 0.8,
               "surprise": 1.0, "strict_non_yes": 0},
              {"sample_idx": 0, "bucket": "d_valid_correct", "p_valid": 0.8,
               "surprise": 3.0, "strict_non_yes": 0},
              {"sample_idx": 1, "bucket": "a_harness", "historical_match": False,
               "p_valid": 0.4, "surprise": 5.0, "strict_non_yes": 1},
              {"sample_idx": 1, "bucket": "a_harness"}]
        diagnostics = diagnostic_means(rs)
        for scope, expected in (("observed_only", (0.2, 1, 2.0, 2, 0.0)),
                                ("all_available_distribution", (0.4, 2, 3.0, 3, 1 / 3))):
            metrics = diagnostics[scope]
            for name, value, included, planned in (
                    ("mean_exact_p_invalid", expected[0], expected[1], 2),
                    ("mean_surprise", expected[2], expected[3], 4),
                    ("strict_non_yes", expected[4], expected[3], 4)):
                self.assertAlmostEqual(metrics[name]["value"], value)
                self.assertEqual(metrics[name]["included"], included)
                self.assertEqual(metrics[name]["planned"], planned)
        unavailable = diagnostic_means([{"sample_idx": 0, "bucket": "a_harness"}])
        for metrics in unavailable.values():
            for metric in metrics.values():
                self.assertIsNone(metric["value"])
                self.assertEqual(metric["included"], 0)
                self.assertEqual(metric["planned"], 1)

    def test_repo_relative_paths(self):
        relative = HERE.relative_to(REPO) / "sampling_endpoint_b.py"
        expected = relative.as_posix()
        self.assertEqual(repo_relative_path(relative), expected)
        self.assertEqual(repo_relative_path(REPO / relative), expected)
        self.assertEqual(repo_relative_path(relative.parent / ".." / relative.parent.name / relative.name), expected)
        with self.assertRaises(ValueError):
            repo_relative_path(REPO.parent / "outside-repository.py")

    def test_greedy_ties_and_margin(self):
        audit = greedy_audit(np.array([2.0, 2.0, 1.0]))
        self.assertEqual(audit["greedy_top_two_ids"], [0, 1])
        self.assertEqual(audit["greedy_logit_margin"], 0)
        self.assertTrue(audit["near_tie"])
        self.assertTrue(greedy_audit(np.array([0.15, 0.0]))["near_tie"])
        self.assertFalse(greedy_audit(np.array([0.151, 0.0]))["near_tie"])

    def test_missing_duplicate_and_corrupt_slots(self):
        rows = {"anli_r1": [{"sample_idx": 0, "label": 0, "prompt": "Answer:"}]}
        failure = {"model": MODELS[0][0], "revision": MODELS[0][1], "task": "anli_r1",
                   "sample_idx": 0, "label": 0, "stem_id": None, "T": 0.0, "sample_index": 0,
                   "seed": 20260826, "bucket": "a_harness", "error": "fixture"}
        with tempfile.TemporaryDirectory(dir=HERE / "runs") as directory:
            path = Path(directory) / "model.jsonl"
            recs, _, _ = account_records(path, 0, rows, {})
            self.assertEqual(len(recs), len(PLAN))
            self.assertTrue(all(r["bucket"] == "a_harness" for r in recs))
            path.write_text(json.dumps(failure) + "\n" + json.dumps(failure) + "\n" + "{partial\n")
            recs, _, issues = account_records(path, 0, rows, {})
            self.assertEqual(len(recs), len(PLAN))
            self.assertIn("duplicate", recs[0]["error"])
            self.assertEqual(len(issues), 1)

    def test_surprise_and_inverse_cdf(self):
        q, logq = temperature_distribution(np.array([0.0, 0.0]), 1.0)
        np.testing.assert_array_equal(q, [0.5, 0.5])
        np.testing.assert_allclose(-logq, [np.log(2), np.log(2)])
        self.assertEqual(int(np.searchsorted(np.cumsum(q), 0.5, side="right")), 1)
        q, logq = temperature_distribution(np.array([0.0, -1000.0]), 1.0)
        self.assertEqual(q[1], 0.0)
        self.assertEqual(-logq[1], 1000.0)

    def test_inverse_cdf_audit_boundaries_and_clip(self):
        cdf = np.array([0.0, 0.5, 0.5, 1.0])
        for uniform, token in ((0.0, 1), (0.5, 3), (0.999, 3)):
            selection = {**inverse_cdf(cdf, uniform), "uniform": uniform}
            self.assertEqual(selection["token_id"], token)
            check_sampling(selection, uniform)
        rounded = np.nextafter(1.0, 0.0)
        clipped = {**inverse_cdf(np.array([0.5, rounded]), rounded), "uniform": rounded}
        self.assertTrue(clipped["cdf_clipped"])
        check_sampling(clipped, rounded)
        for update in ({"cdf_clipped": False}, {"token_id": 0}, {"cdf_hi": 1.0}):
            with self.assertRaises(AssertionError):
                check_sampling({**clipped, **update}, rounded)
        selection = {**inverse_cdf(cdf, 0.5), "uniform": 0.5}
        for update in ({"cdf_lo": 0.6}, {"cdf_hi": 0.5}, {"uniform": 0.4}, {"cdf_hi": float("nan")}):
            with self.assertRaises(AssertionError):
                check_sampling({**selection, **update}, 0.5)

    def test_primary_conditional_bounds_and_anli_bar(self):
        rs = [{"bucket": "a_harness"}, {"bucket": "b_noncompliant", "strict_non_yes": 1},
              {"bucket": "c_valid_wrong", "strict_non_yes": 1},
              {"bucket": "d_valid_correct", "strict_non_yes": 0}]
        rs = [dict(r, sample_idx=i) for i, r in enumerate(rs)]
        cell = summarize_cell(rs)
        self.assertEqual(cell["primary_error_rate"], 2 / 3)
        self.assertEqual(cell["conditional_on_valid_error_rate"], 1 / 2)
        self.assertIsNone(cell["sample_error_rate"])
        self.assertEqual(cell["sample_error_bounds"], [0.5, 0.75])
        self.assertEqual(sum(cell["rates_planned"].values()), 1)
        no_valid = summarize_cell(rs[:2])
        self.assertIsNone(no_valid["conditional_on_valid_error_rate"])
        anli = [{"bucket": "a_harness"}] * 2 + [{"bucket": "b_noncompliant", "strict_non_yes": 1}] * 9
        anli += [{"bucket": "d_valid_correct", "strict_non_yes": 0}] * 189
        anli = [dict(r, sample_idx=i) for i, r in enumerate(anli)]
        self.assertIn("ANLI_canonical_below_190_of_200", summarize_cell(anli, True, True)["reasons"])

    def test_joint_temperature_stem_bootstrap_and_missingness(self):
        rows = [{"sample_idx": i, "stem_id": str(i // 2)} for i in range(4)]
        rs = [{**row, "T": t, "sample_index": d, "bucket": "d_valid_correct",
               "sample_error": int(row["stem_id"] == "0" and t == 1.0)}
              for row in rows for t in TEMPS for d in range(K)]
        result = reversal(rs, rows, "halueval_qa", 1)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["usable"], 2000)
        self.assertEqual(result["point_R"], -0.25)
        self.assertEqual(result["R_interval"], [-1.0, 0.0])
        rs[0]["bucket"] = "a_harness"
        incomplete = reversal(rs, rows, "halueval_qa", 1)
        self.assertEqual(incomplete["attempted"], 2000)
        self.assertEqual(incomplete["usable"], 0)
        self.assertIsNone(incomplete["point_R"])


if __name__ == "__main__":
    unittest.main()
