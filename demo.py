import os
import math
import cv2
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import color, img_as_float32, img_as_ubyte
from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything

import depth_pro

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEBLUR_LORA_PATH = "."
DEBLUR_WEIGHT_NAME = "deblurNet.safetensors"
BOKEH_LORA_DIR = "."
BOKEH_WEIGHT_NAME = "bokehNet.safetensors"

if not os.path.exists(os.path.join(DEBLUR_LORA_PATH, DEBLUR_WEIGHT_NAME)):
    print(f"❌ Warning: {DEBLUR_WEIGHT_NAME} not found.")
if not os.path.exists(os.path.join(BOKEH_LORA_DIR, BOKEH_WEIGHT_NAME)):
    print(f"❌ Warning: {BOKEH_WEIGHT_NAME} not found.")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"🚀 Device detected: {device}")

print("🔄 Loading FLUX pipeline...")
pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
current_adapter = None

if device == "cuda":
    print("🚀 Moving FLUX to CUDA...")
    pipe_flux.to("cuda")

print("🔄 Loading Depth Pro model...")
try:
    depth_model, depth_transform = depth_pro.create_model_and_transforms()
    if device == "cuda":
        depth_model.eval().to("cuda")
    else:
        depth_model.eval()
    print("✅ Depth Pro loaded.")
except Exception as e:
    print(f"❌ Failed to load Depth Pro: {e}")
    depth_model = None
    depth_transform = None


def resize_and_pad_image(img: Image.Image, target_long_side: int) -> Image.Image:
    w, h = img.size

    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)

        if w >= h:
            new_w = target_max
            scale = target_max / w
            new_h = int(h * scale)
        else:
            new_h = target_max
            scale = target_max / h
            new_w = int(w * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16

        final_w = max(final_w, 16)
        final_h = max(final_h, 16)

        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        right = left + final_w
        bottom = top + final_h

        return img.crop((left, top, right, bottom))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16

    if final_w == w and final_h == h:
        return img

    return img.resize((final_w, final_h), Image.LANCZOS)


def switch_lora(target_mode):
    global pipe_flux, current_adapter
    if current_adapter == target_mode:
        return
    print(f"🔄 Switching LoRA to [{target_mode}]...")
    pipe_flux.unload_lora_weights()
    if target_mode == "deblur":
        try:
            pipe_flux.load_lora_weights(DEBLUR_LORA_PATH, weight_name=DEBLUR_WEIGHT_NAME, adapter_name="deblurring")
            pipe_flux.set_adapters(["deblurring"])
            current_adapter = "deblur"
        except Exception as e:
            print(f"❌ Failed to load Deblur LoRA: {e}")
    elif target_mode == "bokeh":
        try:
            pipe_flux.load_lora_weights(BOKEH_LORA_DIR, weight_name=BOKEH_WEIGHT_NAME, adapter_name="bokeh")
            pipe_flux.set_adapters(["bokeh"])
            current_adapter = "bokeh"
        except Exception as e:
            print(f"❌ Failed to load Bokeh LoRA: {e}")


def preprocess_input_image(raw_img, target_long_side):
    if raw_img is None:
        return None, None, None, None, "Center (Default)"

    if target_long_side is None:
        target_long_side = 0

    try:
        target_long_side = int(target_long_side)
    except Exception:
        raise gr.Error("Resize target must be an integer (0 means keep original resolution).")

    mode_str = f"Resize Long Side={target_long_side}" if target_long_side > 0 else "Original Res (Align 16)"
    print(f"🔄 Preprocessing Input... Mode: {mode_str}")

    final_input = resize_and_pad_image(raw_img, target_long_side)
    return final_input, final_input, None, None, "Center (Default)"


def draw_red_dot_on_preview(clean_img, evt: gr.SelectData):
    if clean_img is None:
        return None, None

    img_copy = clean_img.copy()
    draw = ImageDraw.Draw(img_copy)
    x, y = evt.index
    r = 8
    draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
    draw.line((x - r, y, x + r, y), fill="red", width=2)
    draw.line((x, y - r, x, y + r), fill="red", width=2)

    return img_copy, evt.index


def run_genfocus_pipeline(clean_input_processed, click_coords, K_value, cached_latents, disable_tiling_tricks_for_accel, num_inference_steps):
    if clean_input_processed is None:
        raise gr.Error("Please complete Step 1 (Upload Image) first.")

    num_inference_steps = int(num_inference_steps)
    if num_inference_steps < 8 or num_inference_steps > 50:
        raise gr.Error("num_inference_steps must be within [8, 50].")



    w, h = clean_input_processed.size
    force_no_tile = min(w, h) < 512
    no_tiled_denoise = bool(disable_tiling_tricks_for_accel) or force_no_tile
    print(f"🚀 Starting Genfocus Pipeline... (Size: {w}x{h})")

    print("   ► Running Stage 1: DeblurNet")
    switch_lora("deblur")

    condition_0_img = Image.new("RGB", (w, h), (0, 0, 0))
    cond0 = Condition(condition_0_img, "deblurring", [0, 32], 1.0)
    cond1 = Condition(clean_input_processed, "deblurring", [0, 0], 1.0)

    seed_everything(42)
    deblurred_img = generate(
        pipe_flux,
        height=h,
        width=w,
        prompt="a sharp photo with everything in focus",
        num_inference_steps=num_inference_steps,
        conditions=[cond0, cond1],
        NO_TILED_DENOISE=no_tiled_denoise,
    ).images[0]

    if K_value == 0:
        print("✅ K=0, returning Deblur result.")
        return deblurred_img, cached_latents

    print(f"   ► Running Stage 2: BokehNet (K={K_value})")

    if click_coords is None:
        click_coords = [w // 2, h // 2]
        print("   ⚠️ No focus point selected. Defaulting to Center.")

    try:
        img_t = depth_transform(deblurred_img)
        if device == "cuda":
            img_t = img_t.to("cuda")
        with torch.no_grad():
            pred = depth_model.infer(img_t, f_px=None)
        depth_map = pred["depth"].cpu().numpy().squeeze()
        safe_depth = np.where(depth_map > 0.0, depth_map, np.finfo(np.float32).max)
        disp_orig = 1.0 / safe_depth
        disp = cv2.resize(disp_orig, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"❌ Depth Error: {e}")
        return deblurred_img, cached_latents

    tx, ty = click_coords
    tx = min(max(int(tx), 0), w - 1)
    ty = min(max(int(ty), 0), h - 1)

    disp_focus = float(disp[ty, tx])
    dmf = disp - np.float32(disp_focus)
    defocus_abs = np.abs(K_value * dmf)
    MAX_COC = 100.0
    defocus_t = torch.from_numpy(defocus_abs).unsqueeze(0).float()
    cond_map = (defocus_t / MAX_COC).clamp(0, 1).repeat(3, 1, 1).unsqueeze(0)

    if cached_latents is None:
        print("      Generating new fixed latents...")
        seed_everything(42)
        gen = torch.Generator(device=pipe_flux.device).manual_seed(1234)
        latents, _ = pipe_flux.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=h,
            width=w,
            dtype=pipe_flux.dtype,
            device=pipe_flux.device,
            generator=gen,
            latents=None,
        )
        current_latents = latents
    else:
        print("      Using cached latents...")
        current_latents = cached_latents

    switch_lora("bokeh")
    cond_img = Condition(deblurred_img, "bokeh")
    cond_dmf = Condition(cond_map, "bokeh", [0, 0], 1.0, No_preprocess=True)

    seed_everything(42)
    gen = torch.Generator(device=pipe_flux.device).manual_seed(1234)

    with torch.no_grad():
        res = generate(
            pipe_flux,
            height=h,
            width=w,
            prompt="an excellent photo with a large aperture",
            num_inference_steps=num_inference_steps,
            conditions=[cond_img, cond_dmf],
            guidance_scale=1.0,
            kv_cache=False,
            generator=gen,
            latents=current_latents,
            NO_TILED_DENOISE=no_tiled_denoise,
        )
    generated_bokeh = res.images[0]
    return generated_bokeh, current_latents


css = """
#col-container { margin: 0 auto; max-width: 1400px; }
"""

base_path = os.getcwd()
example_dir = os.path.join(base_path, "example")

valid_examples = []
allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if os.path.exists(example_dir):
    files = sorted(os.listdir(example_dir))
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in allowed_extensions:
            full_path = os.path.join(example_dir, filename)
            valid_examples.append([full_path])
    print(f"✅ Loaded {len(valid_examples)} examples from '{example_dir}'")
else:
    print(f"⚠️ Warning: Example directory '{example_dir}' not found.")

with gr.Blocks(css=css) as demo:
    clean_processed_state = gr.State(value=None)
    click_coords_state = gr.State(value=None)
    latents_state = gr.State(value=None)

    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 📷 Genfocus Pipeline: Interactive Refocusing")

        gr.Markdown("""
        ### 📖 User Guide
        **Generative Refocusing** supports two main applications:

        * **All-In-Focus (AIF) Estimation:** Set **K = 0**. The model will restore the AIF image from the blurry input.

        * **Refocusing:** 1. **Click** on the subject you want to bring into focus in the **Step 2** image preview.  
        2. Increase **K** (Blur Strength) to generate realistic bokeh effects based on the scene's depth.

        ⚠️ **Note (Advanced Settings / Acceleration):** You can override the default settings to speed up inference.
        1. **num_inference_steps:** Use fewer steps (e.g., **8**) for faster inference with a slight quality trade-off.
        2. **Resize longer side:** The resized result will be center-cropped to a multiple of **16**.
        3. **Disable tiling tricks (NO_TILED_DENOISE=True):** Not recommended when the longer side is around **1000px** or more; it may cause severe artifacts.
        """)



        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Upload Image")
                gr.Markdown("Click an example or upload your own image.")

                input_raw = gr.Image(label="Raw Input Image", type="pil")

                with gr.Accordion("Advanced Settings (Speed / Quality Trade-offs)", open=False):
                    steps_slider = gr.Slider(
                        minimum=8,
                        maximum=50,
                        value=28,
                        step=1,
                        label="num_inference_steps",
                    )
                    target_long_side_num = gr.Number(
                        label="Resize Long Side (px, 0 = keep original; recommended 512 ~ 2000)",
                        value=0,
                        precision=0,
                    )
                    disable_tiling_tricks_for_accel = gr.Checkbox(
                        label="Disable tiling tricks. Large-res images may cause artifacts.",
                        value=False,
                    ) 

                if valid_examples:
                    gr.Examples(examples=valid_examples, inputs=input_raw, label="Examples (Click to Load)", cache_examples=False)

            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Set Focus & K")
                gr.Markdown("The image below shows the actual input for the model. **Click on the image** to set the focus point.")

                focus_preview_img = gr.Image(label="Model Input (Processed) - Click Here", type="pil", interactive=False)

                with gr.Row():
                    click_status = gr.Textbox(label="Selected Coordinates", value="Center (Default)", interactive=False, scale=1)
                    k_slider = gr.Slider(minimum=0, maximum=50, value=20, step=1, label="Blur Strength (K)", scale=2)

                run_btn = gr.Button("✨ Run Genfocus", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Result")
                output_img = gr.Image(label="Final Output", type="pil", interactive=False, elem_id="output_image")

        input_raw.change(
            fn=preprocess_input_image,
            inputs=[input_raw, target_long_side_num],
            outputs=[focus_preview_img, clean_processed_state, latents_state, click_coords_state, click_status],
        )
        input_raw.upload(
            fn=preprocess_input_image,
            inputs=[input_raw, target_long_side_num],
            outputs=[focus_preview_img, clean_processed_state, latents_state, click_coords_state, click_status],
        )

        target_long_side_num.change(
            fn=preprocess_input_image,
            inputs=[input_raw, target_long_side_num],
            outputs=[focus_preview_img, clean_processed_state, latents_state, click_coords_state, click_status],
        )

        focus_preview_img.select(
            fn=draw_red_dot_on_preview,
            inputs=[clean_processed_state],
            outputs=[focus_preview_img, click_coords_state],
        ).then(
            fn=lambda x: f"x={x[0]}, y={x[1]}",
            inputs=[click_coords_state],
            outputs=[click_status],
        )

        run_btn.click(
            fn=run_genfocus_pipeline,
            inputs=[
                clean_processed_state,
                click_coords_state,
                k_slider,
                latents_state,
                disable_tiling_tricks_for_accel,
                steps_slider,
            ],
            outputs=[output_img, latents_state],
        )

if __name__ == "__main__":
    allowed_dir = os.path.join(base_path, "example")
    allowed_paths = [allowed_dir]
    demo.launch(server_name="0.0.0.0", share=True, allowed_paths=allowed_paths)
