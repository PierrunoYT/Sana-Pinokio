import os
import random

import torch
import gradio as gr
from diffusers import SanaPipeline

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

def generate_image(prompt, steps=20, guidance=4.5, seed=42, use_random_seed=True, width=1024, height=1024):
    """Generate an image from a text prompt.
    
    Returns (image, seed_used) so the UI can display the seed that was actually used.
    """
    try:
        if pipe is None:
            load_model()
        
        # Pick seed
        actual_seed = random.randint(0, 2**32 - 1) if use_random_seed else int(seed)
        
        # Create generator on appropriate device
        if DEVICE == "cpu":
            generator = torch.Generator().manual_seed(actual_seed)
        else:
            generator = torch.Generator(device=DEVICE).manual_seed(actual_seed)
        
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
        
        return image, actual_seed
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating image: {str(e)}") from e

# Load model on startup
print("Initializing Sana...")
load_model()

# Create Gradio interface
CTRL_ENTER_JS = """
function() {
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      var btn = document.getElementById('generate_btn');
      if (btn) btn.click();
    }
  });
}
"""

SEED_CSS = """
/* Seed row is the absolute-positioning context */
#seed_row {
    position: relative !important;
    overflow: visible  !important;
}

/* Pull buttons out of flex flow and overlay on right of input.
   'bottom' is tuned so they sit centred within the input field
   (below the "Seed" label that adds ~26 px at the top of the component). */
#random_seed_btn,
#reuse_seed_btn {
    position:   absolute !important;
    bottom:     12px     !important;
    z-index:    10       !important;
    width:      26px     !important;
    min-width:  26px     !important;
    max-width:  26px     !important;
    height:     26px     !important;
    min-height: 26px     !important;
    max-height: 26px     !important;
    padding:    0        !important;
    margin:     0        !important;
    font-size:  13px     !important;
    line-height:1        !important;
    flex:       none     !important;
    /* zero out the wrapper so it takes no flex width */
    --button-shadow: none;
}
/* Zero out the outer Gradio wrapper divs that wrap each button */
#random_seed_btn,
#reuse_seed_btn {
    /* wrapper also needs zero width in flex */
}
#reuse_seed_btn  { right: 6px  !important; }
#random_seed_btn { right: 36px !important; }

/* Pad right of input so typed text doesn't slide under the buttons */
#seed_number input {
    padding-right: 74px !important;
}

/* Gray-out the seed number when the field is disabled */
#seed_number input:disabled {
    opacity: 0.38;
    color:   #888 !important;
    -webkit-text-fill-color: #888 !important;
}
"""

with gr.Blocks(title="Sana Image Generator", theme=gr.themes.Soft(), js=CTRL_ENTER_JS, css=SEED_CSS) as demo:
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
            
            # --- Seed row ---
            # State: tracks whether random mode is active
            random_seed_state = gr.State(value=True)

            with gr.Row(equal_height=True, elem_id="seed_row"):
                seed_input = gr.Number(
                    value=42,
                    label="Seed",
                    precision=0,
                    interactive=False,   # disabled by default (random mode on)
                    scale=1,
                    elem_id="seed_number",
                )
                # Buttons are absolutely positioned over the seed input via CSS
                random_btn = gr.Button("🎲", variant="primary",  scale=0, elem_id="random_seed_btn")
                reuse_btn  = gr.Button("♻️", variant="secondary", scale=0, elem_id="reuse_seed_btn")

            generate_btn = gr.Button("Generate", variant="primary", size="lg", elem_id="generate_btn")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                format="png",
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
    
    # --- Seed button handlers ---

    def on_random_click():
        """Switch to random-seed mode: disable seed field, highlight 🎲 active."""
        return (
            True,
            gr.update(interactive=False),
            gr.update(variant="primary"),    # 🎲 → active
            gr.update(variant="secondary"),  # ♻️ → inactive
        )

    def on_reuse_click(last_seed):
        """Switch to manual mode and load the last used seed, highlight ♻️ active."""
        return (
            False,
            gr.update(value=last_seed, interactive=True),
            gr.update(variant="secondary"),  # 🎲 → inactive
            gr.update(variant="primary"),    # ♻️ → active
        )

    # State for last-used seed (starts at 42 so ♻️ has something on first press)
    last_seed_state = gr.State(value=42)

    random_btn.click(
        fn=on_random_click,
        inputs=[],
        outputs=[random_seed_state, seed_input, random_btn, reuse_btn],
    )

    reuse_btn.click(
        fn=on_reuse_click,
        inputs=[last_seed_state],
        outputs=[random_seed_state, seed_input, random_btn, reuse_btn],
    )

    def run_generate(prompt, steps, guidance, seed, use_random, width, height):
        image, used_seed = generate_image(prompt, steps, guidance, seed, use_random, width, height)
        # After generation, update seed field to show what was used
        return image, used_seed, gr.update(value=used_seed)

    generate_btn.click(
        fn=run_generate,
        inputs=[prompt_input, steps_slider, guidance_slider, seed_input, random_seed_state, width_slider, height_slider],
        outputs=[output_image, last_seed_state, seed_input],
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
