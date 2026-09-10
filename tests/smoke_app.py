"""Run with app dependencies installed; exercise real Gradio with stub model output."""
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["SANA_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import app
import inference
from gradio_client import Client
from PIL import Image

assert inference.pipe is None, "Importing the UI must not download/load model weights"
pipeline = Mock()
pipeline.return_value.images = [Image.new("RGB", (512, 512), "blue")]

try:
    with patch.object(inference.SanaPipeline, "from_pretrained", return_value=pipeline):
        _, url, _ = app.demo.launch(server_name="127.0.0.1", prevent_thread_lock=True,
                                    share=False, quiet=True)
        client = Client(url, verbose=False)
        image_path, seed = client.predict("A robot", 10, 4.5, 42, False, 512, 512,
                                          api_name="/generate")
        assert seed == 42, (image_path, seed)
        assert Path(image_path).is_file()
        assert pipeline.call_args.kwargs["generator"].initial_seed() == 42
        assert app.on_reuse_click(42)[0] is False
        assert app.on_random_click()[0] is True
        print("Gradio UI/API smoke passed: explicit seed, image output, seed controls")
finally:
    app.demo.close()
