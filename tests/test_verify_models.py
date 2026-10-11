import tempfile
import unittest
from pathlib import Path

from src.model.verify_models import MODELS, TRACKED, read_manifest, sha256, verify


class VerifyModelsTests(unittest.TestCase):
    def test_committed_models_match_the_manifest(self):
        self.assertEqual(verify(), [])
        self.assertEqual(set(read_manifest()), set(TRACKED))

    def test_a_changed_file_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "model.joblib").write_bytes(b"original")
            manifest = d / "SHA256SUMS"
            manifest.write_text(f"{sha256(d / 'model.joblib')}  model.joblib\n", encoding="utf-8")
            self.assertEqual(verify(d, manifest), [])
            (d / "model.joblib").write_bytes(b"tampered")
            self.assertEqual(len(verify(d, manifest)), 1)
            (d / "model.joblib").unlink()
            self.assertIn("missing", verify(d, manifest)[0])


if __name__ == "__main__":
    unittest.main()


class DeployedGapTests(unittest.TestCase):
    def test_saved_gap_table_matches_the_committed_model_and_challenge_set(self):
        """If this fails, run `python -m src.model.gap --save-deployed` and commit the result."""
        import json

        from src.model.gap import DEPLOYED_GAP, _inputs

        saved = json.loads(DEPLOYED_GAP.read_text(encoding="utf-8"))
        self.assertEqual({k: saved[k] for k in _inputs()}, _inputs())
        self.assertEqual([r["dataset"] for r in saved["rows"]], ["EMSCAD test", "Challenge set"])
