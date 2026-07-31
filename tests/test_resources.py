from pathlib import Path
import os
import tempfile
import unittest

from localasr.resources import (
    DEFAULT_MODEL,
    app_data_dir,
    default_model_cache_dir,
    list_cached_models,
    model_is_cached,
    onnx_model_cache_dir,
    onnx_models_ready,
    resolve_model,
)


class ResourceTests(unittest.TestCase):
    def test_list_cached_models_includes_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = list_cached_models(Path(tmp))

            self.assertIn(DEFAULT_MODEL, models)
            self.assertFalse(model_is_cached(DEFAULT_MODEL, Path(tmp)))

    def test_modelscope_cache_layout_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "hub" / "models" / "iic" / "SenseVoiceSmall"
            model_path.mkdir(parents=True)

            self.assertIn(DEFAULT_MODEL, list_cached_models(root))
            self.assertTrue(model_is_cached(DEFAULT_MODEL, root))
            self.assertEqual(resolve_model(DEFAULT_MODEL, [root]), str(model_path))

    def test_manual_cache_layout_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "iic" / "SenseVoiceSmall"
            model_path.mkdir(parents=True)

            self.assertTrue(model_is_cached(DEFAULT_MODEL, root))
            self.assertEqual(resolve_model(DEFAULT_MODEL, [root]), str(model_path))

    def test_custom_model_id_is_resolved_from_selected_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "hub" / "models" / "custom" / "InterviewModel"
            model_path.mkdir(parents=True)

            self.assertIn("custom/InterviewModel", list_cached_models(root))
            self.assertTrue(model_is_cached("custom/InterviewModel", root))
            self.assertEqual(resolve_model("custom/InterviewModel", [root]), str(model_path))

    def test_model_dir_can_be_overridden_by_environment(self) -> None:
        old_value = os.environ.get("LOCALASR_MODEL_DIR")
        try:
            os.environ["LOCALASR_MODEL_DIR"] = "/tmp/localasr-test-models"
            self.assertEqual(default_model_cache_dir(), Path("/tmp/localasr-test-models"))
        finally:
            if old_value is None:
                os.environ.pop("LOCALASR_MODEL_DIR", None)
            else:
                os.environ["LOCALASR_MODEL_DIR"] = old_value

    def test_default_model_dir_lives_under_app_data_dir(self) -> None:
        old_data = os.environ.get("LOCALASR_DATA_DIR")
        old_model = os.environ.get("LOCALASR_MODEL_DIR")
        try:
            os.environ["LOCALASR_DATA_DIR"] = "/tmp/localasr-test-data"
            os.environ.pop("LOCALASR_MODEL_DIR", None)
            self.assertEqual(app_data_dir(), Path("/tmp/localasr-test-data"))
            self.assertEqual(default_model_cache_dir(), Path("/tmp/localasr-test-data/models"))
        finally:
            if old_data is None:
                os.environ.pop("LOCALASR_DATA_DIR", None)
            else:
                os.environ["LOCALASR_DATA_DIR"] = old_data
            if old_model is None:
                os.environ.pop("LOCALASR_MODEL_DIR", None)
            else:
                os.environ["LOCALASR_MODEL_DIR"] = old_model

    def test_onnx_model_cache_dir_uses_model_root(self) -> None:
        self.assertEqual(
            onnx_model_cache_dir(Path("/tmp/models")),
            Path("/tmp/models/onnx/sherpa-sensevoice-2025-09-09"),
        )

    def test_onnx_models_ready_checks_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                "sensevoice/model.int8.onnx",
                "sensevoice/tokens.txt",
                "speaker/pyannote-segmentation.int8.onnx",
                "speaker/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                "vad/silero_vad.onnx",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            self.assertTrue(onnx_models_ready(root))


if __name__ == "__main__":
    unittest.main()
