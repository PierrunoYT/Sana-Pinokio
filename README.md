# Sana Pinokio Plugin

A Pinokio plugin for running the Sana diffusion model - a fast and efficient image generation model capable of creating high-quality 1024×1024 images from text prompts.

## Features

- 🎨 **Fast Image Generation**: Generate 1024×1024 images in seconds using the SANA1.5 1.6B model
- 🌐 **Web UI**: Beautiful Gradio interface for easy interaction
- 🚀 **Easy Installation**: Automated setup through Pinokio
- 🎛️ **Customizable Parameters**: Control inference steps, guidance scale, seed, and image dimensions
- 💻 **GPU Optimized**: Supports CUDA acceleration with bfloat16 precision

## Requirements

- Pinokio installed and running
- NVIDIA GPU with CUDA support (recommended) or CPU
- Python 3.10+ (managed by Pinokio)
- Sufficient disk space for the model (~3-4 GB)

## Installation

1. Install this plugin in Pinokio
2. Click **Install** in the Pinokio interface
3. Wait for dependencies and the model to download (first-time setup may take several minutes)

The installation process will:
- Create a Python virtual environment
- Install all required dependencies from `app/requirements.txt`
- Pre-download the Sana model from Hugging Face
- Install PyTorch with CUDA support (if NVIDIA GPU detected)

## Usage

1. Click **Start** in the Pinokio interface
2. Wait for the application to launch (the model will load into GPU memory)
3. Click **Open Web UI** when it appears in the menu
4. Enter your text prompt and adjust parameters as needed
5. Click **Generate** to create your image

### Parameters

- **Prompt**: Text description of the image you want to generate
- **Width/Height**: Image dimensions (512-2048px, default: 1024×1024)
- **Inference Steps**: Number of denoising steps (10-50, default: 20)
  - Higher values = better quality but slower generation
- **Guidance Scale**: How closely the image follows the prompt (1.0-10.0, default: 4.5)
  - Higher values = more adherence to prompt
- **Seed**: Random seed for reproducibility (default: 42)
  - Change for different variations

### Example Prompts

- `a futuristic city at sunset, cyberpunk style`
- `a serene mountain landscape with a lake, photorealistic`
- `a cute robot reading a book, digital art`
- `an abstract painting with vibrant colors`
- `a steampunk airship flying through clouds`

## Model Information

- **Model**: Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
- **Size**: ~1.6B parameters
- **Precision**: bfloat16 (BF16)
- **Resolution**: Up to 1024×1024 pixels
- **Speed**: ~1-2 seconds per image on RTX 4090

## Project Structure

```
Sana-Pinokio/
├── app/
│   ├── app.py          # Main Gradio web application
│   └── requirements.txt # Python dependencies
├── install.js          # Pinokio installation script
├── start.js            # Pinokio startup script
├── pinokio.js          # Pinokio UI and menu
├── pinokio.json        # Pinokio metadata (title, description, icon)
├── torch.js            # PyTorch installation script (platform-aware)
├── link.js             # Disk space optimization script
├── reset.js            # Reset installation script
├── update.js           # Update script (git pull + refresh deps)
└── icon.jpg            # Plugin icon
```

## Scripts

- **Install**: Sets up the environment and downloads the model
- **Start**: Launches the Gradio web interface
- **Update**: Pulls the latest repository changes and reinstalls Python dependencies from `app/requirements.txt`
- **Save Disk Space**: Deduplicates redundant library files
- **Reset**: Removes the installation and reverts to pre-install state

## Technical Details

### Architecture

The application uses:
- **Diffusers**: Hugging Face library for diffusion models
- **Gradio**: Web UI framework
- **PyTorch**: Deep learning framework with CUDA support
- **Transformers**: Model loading and text encoding

### Performance Optimizations

- Model loaded in bfloat16 precision for faster inference
- VAE and text encoder converted to bfloat16
- CUDA acceleration for GPU inference
- Optional xformers and triton support for additional speedups

### Model Storage

Models are cached in `~/.cache/huggingface/hub/` by default. The Sana model will be automatically downloaded on first use if not pre-downloaded during installation.

## Troubleshooting

### Model Loading Issues

If the model fails to load:
- Ensure you have sufficient GPU memory (recommended: 8GB+ VRAM)
- Check that CUDA is properly installed
- Verify internet connection for model download

### Slow Generation

- Ensure you're using GPU acceleration (check CUDA availability)
- Reduce inference steps for faster generation
- Lower image resolution if needed

### Out of Memory Errors

- Reduce image dimensions (e.g., 512×512 instead of 1024×1024)
- Close other GPU-intensive applications
- Consider using CPU mode (much slower)

## License

This plugin uses the Sana model from Efficient-Large-Model. Please refer to the original model's license for usage terms.

## Credits

- **Sana Model**: Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
- **Pinokio**: Pinokio automation platform
- **Gradio**: Web UI framework
- **Hugging Face**: Model hosting and diffusers library

