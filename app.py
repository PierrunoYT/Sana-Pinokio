import torch
import gradio as gr
from diffusers import SanaPipeline
import os

# Detect device - prioritize CUDA, then MPS (Apple Silicon), then CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        # Fallback safely if MPS availability check fails on some PyTorch versions
        print(f"Warning: MPS detection failed, falling back to CPU: {e}")
    return "cpu"

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Determine dtype based on device
# bfloat16 works on CUDA and newer MPS, but CPU needs float32
def get_dtype():
    if DEVICE == "cuda":
        return torch.bfloat16
    elif DEVICE == "mps":
        # MPS supports float16 but not always bfloat16
        return torch.float16
    else:
        # CPU - use float32 for compatibility
        return torch.float32

DTYPE = get_dtype()
print(f"Using dtype: {DTYPE}")

# Initialize pipeline
pipe = None

def load_model():
    """Load the Sana pipeline model"""
    global pipe
    if pipe is None:
        print(f"Loading Sana model on {DEVICE}...")
        
        # Load with appropriate dtype
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            torch_dtype=DTYPE,
        )
        
        # Move to device
        pipe.to(DEVICE)
        
        # Convert components to dtype for better performance
        if DEVICE != "cpu":
            pipe.vae.to(DTYPE)
            pipe.text_encoder.to(DTYPE)
        
        print(f"Model loaded successfully on {DEVICE}!")
    return pipe

def generate_image(prompt, steps=20, guidance=4.5, seed=42, width=1024, height=1024):
    """Generate an image from a text prompt"""
    try:
        if pipe is None:
            load_model()
        
        # Create generator on appropriate device
        if DEVICE == "cpu":
            generator = torch.Generator().manual_seed(int(seed))
        else:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        
        # Reduce image size for CPU to make it more manageable
        if DEVICE == "cpu":
            # Warn user if resolution is too high for CPU
            if width > 768 or height > 768:
                print(f"Note: High resolution ({width}x{height}) on CPU may be slow. Consider 512x512 or 768x768.")
        
        result = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        
        # Handle both return formats: [images] or .images
        if isinstance(result, list):
            image = result[0][0] if isinstance(result[0], list) else result[0]
        else:
            image = result.images[0]
        
        return image
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating image: {str(e)}"

# Load model on startup
print("Initializing Sana...")
load_model()

# Create Gradio interface
with gr.Blocks(title="Sana Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🎨 Sana Image Generator
        Fast image generation using the Sana diffusion model.
        
        **Device:** `{DEVICE}` | **Precision:** `{DTYPE}`
        
        {'⚠️ **Running on CPU** - Generation will be slower. Consider using 512x512 resolution.' if DEVICE == 'cpu' else '✅ GPU acceleration enabled'}
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="a futuristic city at sunset, cyberpunk style",
                lines=3,
            )
            
            with gr.Row():
                # Default to 512 for CPU, 1024 for GPU
                default_size = 512 if DEVICE == "cpu" else 1024
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1024 if DEVICE == "cpu" else 2048,
                    value=default_size,
                    step=64,
                    label="Width",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1024 if DEVICE == "cpu" else 2048,
                    value=default_size,
                    step=64,
                    label="Height",
                )
            
            # Fewer steps for CPU by default
            default_steps = 10 if DEVICE == "cpu" else 20
            steps_slider = gr.Slider(
                minimum=5,
                maximum=30 if DEVICE == "cpu" else 50,
                value=default_steps,
                step=1,
                label="Inference Steps",
            )
            
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=4.5,
                step=0.5,
                label="Guidance Scale",
            )
            
            seed_input = gr.Number(
                value=42,
                label="Seed",
                precision=0,
            )
            
            generate_btn = gr.Button("Generate", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
            )
    
    gr.Markdown(
        f"""
        ### Tips:
        - Higher inference steps = better quality but slower generation
        - Guidance scale controls how closely the image follows the prompt
        - Change the seed for different variations
        {'- **CPU Mode:** Use smaller resolutions (512x512) and fewer steps (10-15) for faster generation' if DEVICE == 'cpu' else ''}
        """
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, steps_slider, guidance_slider, seed_input, width_slider, height_slider],
        outputs=output_image,
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["a futuristic city at sunset, cyberpunk style"],
            ["a serene mountain landscape with a lake, photorealistic"],
            ["a cute robot reading a book, digital art"],
            ["an abstract painting with vibrant colors"],
            ["a steampunk airship flying through clouds"],
        ],
        inputs=prompt_input,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)
