module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull --ff-only"
      }
    },
    {
      method: "local.set",
      params: { installed: "{{exists('env/.installed')}}" }
    },
    {
      when: "{{local.installed}}",
      method: "fs.rm",
      params: { path: "env/.installed" }
    },
    {
      when: "{{exists('env')}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt",
          "uv pip check",
          "python -c \"import torch, gradio; from diffusers import SanaPipeline; print('Dependencies verified')\""
        ]
      }
    },
    {
      when: "{{local.installed}}",
      method: "fs.write",
      params: { path: "env/.installed", text: "ready" }
    }
  ]
}
