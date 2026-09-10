module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Ask user to choose CPU or GPU installation
    {
      method: "input",
      params: {
        title: "Select Installation Type",
        description: "Choose GPU auto-detection or CPU only. Windows AMD uses CPU; Apple Silicon uses MPS.",
        form: [{
          type: "select",
          key: "device",
          title: "Device",
          default: "gpu",
          required: true,
          items: [{
            text: "GPU (Auto-detect CUDA/ROCm/MPS)",
            value: "gpu"
          }, {
            text: "CPU Only (No GPU required)",
            value: "cpu"
          }]
        }]
      }
    },
    // Save the user's selection to local variable (input only persists to next step)
    {
      method: "local.set",
      params: {
        device: "{{input.device}}"
      }
    },
    {
      when: "{{exists('env/.installed')}}",
      method: "fs.rm",
      params: { path: "env/.installed" }
    },
    // Install GPU PyTorch (auto-detect) - only if GPU was selected
    {
      when: "{{local.device === 'gpu'}}",
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "."
        }
      }
    },
    // Install CPU PyTorch - only if CPU was selected
    {
      when: "{{local.device === 'cpu'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install --reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu"
        ],
      }
    },
    // Install all dependencies from requirements.txt after torch
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt"
        ],
      }
    },
    // Fail the install early if PyTorch was not installed into the launch venv
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip check",
          "python -c \"import torch, gradio; from diffusers import SanaPipeline; print('PyTorch', torch.__version__)\""
        ],
      }
    },
    // Pre-download the Sana model
    {
      method: "hf.download",
      params: {
        "_": ["Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"]
      }
    },
    {
      method: "fs.write",
      params: {
        path: "env/sana-device.json",
        json: { device: "{{local.device}}" }
      }
    },
    {
      method: "fs.write",
      params: { path: "env/.installed", text: "ready" }
    }
  ]
}
