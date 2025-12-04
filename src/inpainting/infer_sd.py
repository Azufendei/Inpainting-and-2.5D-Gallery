from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

image_path = ROOT / "data" / "inpaint_inputs" / "crumblingCherub.jpg"
mask_path  = ROOT / "data" / "masks" / "crumblingCherub_mask.png"

# NEW OUTPUT DIRECTORY
output_dir = ROOT / "Results" / "inpainted"
output_dir.mkdir(parents=True, exist_ok=True)

# SMART OUTPUT FILENAME
input_stem = image_path.stem  # e.g., 
output_path = output_dir / f"{input_stem}_inpainted.png"

counter = 1
while output_path.exists():
    output_path = output_dir / f"{input_stem}_inpainted_{counter}.png"
    counter += 1

# Load pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load inputs
image = Image.open(image_path).convert("RGB")
mask  = Image.open(mask_path).convert("RGB")

prompt = "Restore this cherub statue to its pristine form"

# Inpaint
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
    num_inference_steps=3
)

result.images[0].save(output_path)
print(f"✅ Saved Stable Diffusion result to {output_path}")
