
from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json
from pathlib import Path

# ------------------------------------------------------------
# Paths and parameters
# ------------------------------------------------------------
output_path = "Synthetic_Image/animals/"
json_path = "animals10_descriptions.json"  # <- use the file you generated
gray_scale = False
num_inference_steps = 20
thresh_default = 0.9  # default threshold used in generate_and_infer


# -------------------------------
# class names
# -------------------------------
'''
classes = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

classes = [ "T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" ]

classes = ["Normal", "Pneumonia"]

### imagenette
classes = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute"
]

### animals 10
classes = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]

### flowers 17
classes = [
    "daffodil",
    "snowdrop",
    "lilyvalley",
    "bluebell",
    "crocus",
    "iris",
    "tigerlily",
    "tulip",
    "fritillary",
    "sunflower",
    "daisy",
    "coltsfoot",
    "dandelion",
    "cowslip",
    "buttercup",
    "windflower",
    "pansy"
]
'''


### animals 10
classes = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]

cls_template_prompts = [f"a photo of a {cls}" for cls in classes]


# ------------------------------------------------------------
# Cache / offline settings
# ------------------------------------------------------------
cache_dir = "/home/shahab33/scratch/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["DIFFUSERS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_OFFLINE"] = "1"  # offline mode (no HTTP)

# ------------------------------------------------------------
# Stable Diffusion
# ------------------------------------------------------------
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    use_auth_token=True,  # optional if already cached
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()

# ------------------------------------------------------------
# CLIP (Transformers-only; offline-first)
# ------------------------------------------------------------
clip_repo_id = "openai/clip-vit-base-patch32"
clip_local_dir = "/home/shahab33/scratch/models/openai/clip-vit-base-patch32"

def _load_clip_from(source):
    """Load CLIP model+processor from source (repo id or local path) without any network calls."""
    clip_m = CLIPModel.from_pretrained(
        source, cache_dir=cache_dir, local_files_only=True
    ).to(device)
    clip_p = CLIPProcessor.from_pretrained(
        source, cache_dir=cache_dir, local_files_only=True
    )
    return clip_m, clip_p

try:
    clip_model, clip_processor = _load_clip_from(clip_repo_id)
except Exception as e_cache:
    if Path(clip_local_dir).exists():
        clip_model, clip_processor = _load_clip_from(clip_local_dir)
    else:
        raise RuntimeError(
            f"CLIP not found in cache and no local dir at: {clip_local_dir}\n"
            f"Stage the model on a login node and copy to compute node.\n"
            f"Cache error: {e_cache}"
        )


# ------------------------------------------------------------
# Load JSON descriptions
# ------------------------------------------------------------
with open(json_path, "r") as f:
    descriptions = json.load(f)

# ------------------------------------------------------------
# Reference text embeddings (normalized)
# ------------------------------------------------------------
with torch.no_grad():
    text_inputs = clip_processor(text=cls_template_prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features_ref = clip_model.get_text_features(**text_inputs)
    text_features_ref = text_features_ref / text_features_ref.norm(dim=-1, keepdim=True)


# ------------------------------------------------------------
# Output root
# ------------------------------------------------------------
os.makedirs(output_path, exist_ok=True)

# ------------------------------------------------------------
# Generate + Infer + Save
# ------------------------------------------------------------
def generate_and_infer(prompts_list, expected_class, thresh=thresh_default):
    results = []
    failed_descriptions = []
    failed_prompts = {}
    saved_count = 0

    for idx, prompt in enumerate(prompts_list):
        # -------- Image generation
        generator = torch.manual_seed(42 + idx)
        image = pipe(
            prompt, guidance_scale=7.5, num_inference_steps=num_inference_steps, generator=generator
        ).images[0]

        # -------- CLIP inference (Transformers-only)
        # Preprocess both text (class prompts) and the generated image
        inputs = clip_processor(
            text=cls_template_prompts, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            # logits_per_image shape: [1, num_classes]
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # -------- Top-1 prediction and confidence
        top_idx = int(probs.argmax())
        top_class = classes[top_idx]
        top_conf = float(probs[top_idx])
        aligned = (top_class == expected_class)
        high_conf = (top_conf >= thresh)

        if aligned and high_conf:
            class_folder = os.path.join(output_path, expected_class)
            os.makedirs(class_folder, exist_ok=True)
            img_path = os.path.join(class_folder, f"{expected_class}_{idx}.png")

            if gray_scale:
                image = image.convert("L")  # save grayscale

            image.save(img_path)
            saved_count += 1
            status = f"✅ Aligned & confident (confidence {top_conf:.2f}) - Saved to {img_path}"
        else:
            reason = "not aligned" if not aligned else f"low confidence {top_conf:.2f}"
            status = f"❌ Failed. Reason: {reason}. Prompt: {prompt} -> Predicted '{top_class}'"
            failed_descriptions.append({
                "idx": idx,
                "prompt": prompt,
                "expected_class": expected_class,
                "predicted_class": top_class,
                "confidence": top_conf,
                "soft_labels": {cls: float(pr) for cls, pr in zip(classes, probs)},
                "status": status,
            })
            failed_prompts[idx] = prompt

        print(f"{idx} - {status}")
        results.append({
            "idx": idx,
            "prompt": prompt,
            "expected_class": expected_class,
            "predicted_class": top_class,
            "confidence": top_conf,
            "soft_labels": {cls: float(pr) for cls, pr in zip(classes, probs)},
            "status": status,
        })

    return results, failed_descriptions, failed_prompts, saved_count



# -------------------------------
# Run generation for all classes
# -------------------------------
all_results = {}
all_failed = {}
all_failed_prompts = {}
saved_summary = {}

for cls in classes:
    print(f"\n--- Generating images for class: {cls} ---")
    prompts_list = descriptions[cls]
    results, failed, failed_prompts, saved_count = generate_and_infer(prompts_list, expected_class=cls, thresh=thresh_default)
    all_results[cls] = results
    all_failed[cls] = failed
    all_failed_prompts[cls] = failed_prompts
    saved_summary[cls] = saved_count

# -------------------------------
# Print final saved image counts
# -------------------------------
print("\nImage generation completed.\n")
print("Summary of saved images per class:")
for cls, count in saved_summary.items():
    print(f"- {cls}: {count} images saved")

