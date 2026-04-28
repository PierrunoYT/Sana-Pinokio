module.exports = {
  run: [
    // Ask user to choose CPU or GPU installation
    {
      method: "input",
      params: {
        title: "Select Installation Type",
        description: "Choose whether to install for GPU (auto-detect) or CPU only",
        form: [{
          type: "select",
          key: "device",
          title: "Device",
          items: [{
            text: "GPU (Auto-detect CUDA/ROCm)",
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
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu"
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
          "python -c \"import torch; print('PyTorch', torch.__version__)\""
        ],
      }
    },
    // Pre-download the Sana model
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "huggingface-cli download Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
        ],
      }
    }
  ]
}
