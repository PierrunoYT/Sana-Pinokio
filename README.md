# Sana Pinokio Launcher

Run the SANA1.5 1.6B diffusion model locally with a Gradio text-to-image interface.

## Installation and use

1. Open this repository in Pinokio and click **Install**.
2. Choose **GPU** for automatic CUDA, ROCm, or Apple Silicon MPS detection, or **CPU Only**.
3. Wait for dependency verification and the model download to finish.
4. Click **Start**, then **Open Web UI**. The model loads on the first generation.
5. Enter a prompt and click **Generate**, or press Ctrl+Enter / Cmd+Enter.

The server binds to `127.0.0.1` and selects an available port. Use the URL shown by
Pinokio, which may differ from `http://127.0.0.1:7860`.

CPU mode is saved in `env/sana-device.json` and applies on subsequent starts,
including on Apple Silicon. To change modes, run **Reinstall**. For a manual
launch, `SANA_DEVICE=cpu` or `SANA_DEVICE=auto` overrides that saved preference.

An installation is ready only after all setup steps succeed. Existing installs
from before the readiness marker was introduced show **Install** once; run it
to verify dependencies and save the device preference. Model downloads reuse
the Hugging Face cache.

## Hardware and dependencies

- Pinokio with its AI bundle and Python 3.10–3.13.
- Windows/Linux x64, or macOS on Apple Silicon. Intel macOS is unsupported by
  the pinned PyTorch 2.7 release.
- NVIDIA on Windows/Linux: CUDA 12.8 wheels; requires a compatible NVIDIA driver.
- AMD on Linux: ROCm 6.3 wheels; requires compatible hardware and drivers.
- AMD on Windows and machines without a supported accelerator: CPU fallback.
  This app does not implement DirectML.
- Apple Silicon: MPS when available; CPU otherwise.
- Enough memory and disk space for the full pipeline, including the text encoder,
  VAE, weights, and Python environment. The 1.6B label describes the diffusion
  transformer, not the total download size. FP32 uses more memory than BF16.

The launcher pins PyTorch 2.7.0 and Diffusers 0.35.2, with Transformers 4.x to
match Hugging Face Hub 0.36.0. NVIDIA/ROCm devices use BF16 (emulated, and
slower, on NVIDIA cards older than Ampere); CPU and MPS use FP32. Sana's text
encoder and VAE require BF16 or FP32, so the MPS path uses FP32. See the [Sana precision documentation](https://huggingface.co/docs/diffusers/v0.35.0/en/api/pipelines/sana).

## Generation controls

| Control | CPU | GPU |
| --- | --- | --- |
| Width / height | 256–1024, default 512 | 256–2048, default 1024 |
| Inference steps | 5–30, default 10 | 5–50, default 20 |
| Guidance scale | 1–10, default 4.5 | 1–10, default 4.5 |

Dimensions use increments of 64. The model is trained for 1024px output; larger
sizes increase memory use and may not improve quality.

Random seed mode is enabled initially. **🎲** selects random mode and disables
the seed field. **♻️** selects manual mode and fills in the last successful seed
(initially 42). Manual seeds range from 0 to 4294967295. The field displays the
seed used after generation; reuse it with the same prompt and settings to
reproduce a result on the same software and hardware. Exact results can differ
across devices or dependency versions.

Example prompts:

- A futuristic city at sunset, cyberpunk style
- A serene mountain landscape with a lake, photorealistic
- A cute robot reading a book, digital art

## API

The named Gradio endpoint is `/generate`. It accepts seven arguments in this
order: prompt, steps, guidance, seed, random-seed mode, width, height. Set
random-seed mode to `false` to use the supplied seed. The endpoint returns an
image and the seed used. Calls share a serialized pipeline with the UI.

Replace the example base URL with the URL from Pinokio.

### Python

Install `gradio_client` in your client environment:

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
image_path, used_seed = client.predict(
    "A cute robot reading a book", 10, 4.5, 42, False, 512, 512,
    api_name="/generate",
)
print(image_path, used_seed)
```

### JavaScript

Install `@gradio/client` in your client project:

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const result = await client.predict("/generate", [
  "A cute robot reading a book", 10, 4.5, 42, false, 512, 512,
]);
console.log(result.data); // image metadata and seed used
```

### curl

Submit the call, then read its server-sent events using the returned `event_id`:

```sh
curl -X POST http://127.0.0.1:7860/gradio_api/call/generate \
  -H "Content-Type: application/json" \
  -d '{"data":["A cute robot reading a book",10,4.5,42,false,512,512]}'

curl -N http://127.0.0.1:7860/gradio_api/call/generate/EVENT_ID
```

The `complete` event contains image metadata (including its URL) and the seed.

## Maintenance

- **Update**: fast-forward the repository and refresh/verify Python dependencies.
  If local commits prevent a fast-forward, resolve the divergence before retrying.
- **Reinstall**: choose the device again and rebuild the selected PyTorch packages.
- **Save Disk Space**: use Pinokio's virtual-environment library deduplication.
- **Reset**: remove `env`, including dependencies, readiness, and device preference.
  Tracked application files and the shared Hugging Face model cache remain intact.

Model storage follows Hugging Face's cache configuration (`HF_HOME` /
`HF_HUB_CACHE`), normally `~/.cache/huggingface/hub` when not overridden by Pinokio.

## Troubleshooting

If installation fails, inspect Pinokio's terminal and `logs/api`. Retry **Install**
after fixing the reported download or dependency error.

### `ModuleNotFoundError: No module named 'torch'` on install or start

This means the Python interpreter running the app cannot import PyTorch. Older
launcher revisions used different working-directory/environment paths during
setup and launch. The current scripts install, verify, update, and launch using
one project-root `env`; installing PyTorch in your system Python does not repair
that environment.

1. Click **Update** to get the current launcher scripts, then run **Install** or
   **Reinstall** to install PyTorch for the selected device. Update alone does
   not rebuild the device-specific PyTorch installation.
2. Confirm dependency verification prints `PyTorch` and its version before the
   model download. Wait until installation finishes before clicking **Start**.
3. If reinstalling still fails, use **Reset**, then **Install** for a clean
   environment. Reset removes installed Python dependencies and the saved device
   choice, but keeps application files and the shared model cache.

If it still fails, include the first failing command and its full error from
Pinokio's terminal or `logs/api`, your OS, GPU, and CPU/GPU installation choice
in the issue. The final missing-module line alone cannot distinguish a failed
package download from an environment-path problem.

For generation failures, check the device displayed in the UI. Close other
memory-intensive applications, reduce dimensions, or reinstall in CPU mode.
CPU inference can be very slow. A failed model load can be retried with
**Generate**; the app only retains a pipeline after successful initialization.

## Development checks

```sh
python -m unittest discover -s tests -v
node --test tests/*.test.js
```

With the application dependencies installed, run `python tests/smoke_app.py`
to exercise the real localhost Gradio API using a stub model image.

The regression tests use model stubs and require no model download. Real image
quality, accelerator support, and memory consumption require a full inference
run on the target hardware.

App logic is in `app/app.py` (UI) and `app/inference.py` (device selection,
validation, and inference). Pinokio scripts live at the repository root.

## Credits and model license

- [SANA1.5 model and usage terms](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers)
- [Pinokio](https://pinokio.computer/)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Gradio](https://www.gradio.app/)
