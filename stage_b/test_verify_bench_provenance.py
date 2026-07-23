"""Tests for verify_bench_provenance — it must PASS on the real sidecar and FAIL closed
on tampering, bad schema, unsafe paths, and a missing root.

Run: python3 -m unittest stage_b/test_verify_bench_provenance.py
"""
import json
import os
import tempfile
import unittest

import verify_bench_provenance as V

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "profiles_bench")
MANIFEST = os.path.join(BENCH, "PROVENANCE.json")


def _write_tmp(obj):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    return path


class VerifyProvenanceTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(MANIFEST):
            self.skipTest("PROVENANCE.json not present")
        self.manifest = json.load(open(MANIFEST))

    def test_real_sidecar_passes(self):
        code, _ = V.verify(MANIFEST, BENCH)
        self.assertEqual(code, 0)

    def test_tampered_digest_fails_1(self):
        m = json.loads(json.dumps(self.manifest))
        k = next(iter(m["matrix_sha256"]))
        m["matrix_sha256"][k] = "0" * 64
        path = _write_tmp(m)
        try:
            code, _ = V.verify(path, BENCH)
            self.assertEqual(code, 1)
        finally:
            os.unlink(path)

    def test_bad_schema_fails_2(self):
        m = json.loads(json.dumps(self.manifest))
        m["schema_version"] = "bogus/9.9"
        path = _write_tmp(m)
        try:
            code, _ = V.verify(path, BENCH)
            self.assertEqual(code, 2)
        finally:
            os.unlink(path)

    def test_unsafe_path_fails_2(self):
        m = json.loads(json.dumps(self.manifest))
        m["matrix_sha256"]["../evil.npz"] = "x"
        path = _write_tmp(m)
        try:
            code, _ = V.verify(path, BENCH)
            self.assertEqual(code, 2)
        finally:
            os.unlink(path)

    def test_missing_root_fails_1(self):
        with tempfile.TemporaryDirectory() as empty:
            code, _ = V.verify(MANIFEST, empty)
            self.assertEqual(code, 1)

    def test_counts_and_pairing_present(self):
        self.assertEqual(len(self.manifest["matrix_sha256"]), 53)
        self.assertEqual(len(self.manifest["profile_sha256"]), 53)
        self.assertEqual(self.manifest["expected_counts"]["matrices"], 53)


if __name__ == "__main__":
    unittest.main()
