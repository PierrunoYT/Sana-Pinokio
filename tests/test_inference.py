"""Fast regression checks without downloading model weights or requiring a GPU."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class InferenceTests(unittest.TestCase):
    def load_runtime(self, device="cpu", bf16=False):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: device == "cuda",
                                 is_bf16_supported=lambda: bf16),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: device == "mps")),
            bfloat16="bf16", float32="fp32", Generator=Mock(),
        )
        pipeline = Mock()
        pipeline.return_value.images = ["image"]
        factory = Mock(return_value=pipeline)
        spec = importlib.util.spec_from_file_location(
            "sana_inference_test", Path(__file__).parents[1] / "app" / "inference.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch, "diffusers": SimpleNamespace(
                SanaPipeline=SimpleNamespace(from_pretrained=factory))}), patch.dict(
                os.environ, {"SANA_DEVICE": "auto"}):
            spec.loader.exec_module(module)
        return module, torch, pipeline, factory

    def test_device_precision_and_cpu_noise(self):
        for device, bf16, dtype in [("cpu", False, "fp32"), ("mps", False, "fp32"),
                                     ("cuda", False, "fp32"), ("cuda", True, "bf16")]:
            with self.subTest(device=device, bf16=bf16):
                app, torch, pipe, factory = self.load_runtime(device, bf16)
                factory.assert_not_called()
                self.assertEqual(app.DTYPE, dtype)
                self.assertEqual(app.generate_image("test", seed=123, use_random_seed=False),
                                 ("image", 123))
                torch.Generator.assert_called_once_with(device="cpu")
                pipe.to.assert_called_once_with(device)

    def test_cpu_override_ignores_available_gpu(self):
        app, *_ = self.load_runtime("cuda", True)
        with patch.dict(os.environ, {"SANA_DEVICE": "cpu"}):
            self.assertEqual(app.get_device(), "cpu")
        with patch.dict(os.environ, {"SANA_DEVICE": " CPU "}):
            self.assertEqual(app.get_device(), "cpu")

    def test_saved_preference_is_applied_and_damage_is_tolerated(self):
        app, *_ = self.load_runtime("cuda", True)
        env = {k: v for k, v in os.environ.items() if k != "SANA_DEVICE"}
        for text, device in [('{"device": "cpu"}', "cpu"), ("", "cuda"),
                             ("null", "cuda"), ('{"other": 1}', "cuda")]:
            with self.subTest(text=text), patch.dict(os.environ, env, clear=True), \
                    patch.object(app.Path, "read_text", return_value=text), \
                    patch("builtins.print"):
                self.assertEqual(app.get_device(), device)

    def test_failed_transfer_can_retry(self):
        app, _, pipe, factory = self.load_runtime()
        pipe.to.side_effect = [RuntimeError("out of memory"), None]
        with self.assertRaises(RuntimeError):
            app.load_model()
        self.assertIsNone(app.pipe)
        self.assertIs(app.load_model(), pipe)
        self.assertEqual(factory.call_count, 2)
        app.load_model()
        self.assertEqual(factory.call_count, 2)

    def test_invalid_requests_fail_before_model_load(self):
        app, _, _, factory = self.load_runtime()
        for kwargs in [{"prompt": " "}, {"width": 257}, {"height": 2048},
                       {"steps": 5.5}, {"guidance": float("nan")},
                       {"seed": -1, "use_random_seed": False},
                       {"seed": 2**32, "use_random_seed": False},
                       {"use_random_seed": "false"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                app.generate_image(**({"prompt": "test"} | kwargs))
        factory.assert_not_called()

    def test_random_seed_is_returned_and_applied(self):
        app, torch, _, _ = self.load_runtime()
        with patch.object(app.random, "randint", return_value=987):
            self.assertEqual(app.generate_image("test", seed=None), ("image", 987))
        torch.Generator.return_value.manual_seed.assert_called_once_with(987)


if __name__ == "__main__":
    unittest.main()
