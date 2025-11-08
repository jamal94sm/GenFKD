from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json


'''
# -------------------------------
# Load Medical X-ray Stable Diffusion (X-rays, CTs, and MRIs)
# -------------------------------
from diffusers import DiffusionPipeline

# Set cache directory
cache_dir = "/home/shahab33/scratch/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["DIFFUSERS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

model_id = "CompVis/stable-diffusion-v1-4"
lora_local_path = os.path.join(cache_dir, "Osama03--Medical-X-ray-image-generation-stable-diffusion")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Load LoRA weights from local path
pipe.load_lora_weights(lora_local_path)




import open_clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model from specific file path
model = open_clip.create_model(
    model_name="ViT-L-14",
    pretrained="/home/shahab33/scratch/huggingface_cache/whyxrayclip/model.pt",
    precision="fp16" if device == "cuda" else "fp32"
).to(device)

# Load tokenizer and preprocessing
tokenizer = open_clip.get_tokenizer("ViT-L-14")
_, _, preprocess = open_clip.create_model_and_transforms(model_name="ViT-L-14")
'''


import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

# --------------------------------------------------------------------
# Define cache location (same as your diffusion setup)
# --------------------------------------------------------------------
cache_dir = "/home/shahab33/scratch/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["DIFFUSERS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

# Strongly recommended on compute nodes so no HTTP calls are made
os.environ["HF_HUB_OFFLINE"] = "1"  # offline mode (only cached/local files)  # <-- key for compute nodes

# --------------------------------------------------------------------
# Stable Diffusion (unchanged, uses cache_dir)
# --------------------------------------------------------------------
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,         # ✅ ensure caching in scratch
    use_auth_token=True,         # ✅ use your login token (optional if already cached)
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()

# --------------------------------------------------------------------
# CLIP (offline-first, cache + local fallback)
# --------------------------------------------------------------------
clip_repo_id = "openai/clip-vit-base-patch32"

# Optional: a pre-staged local directory you saved on a login node
# (see staging snippet below). Adjust to your path if you have one.
clip_local_dir = "/home/shahab33/scratch/models/openai/clip-vit-base-patch32"

def _load_clip_from(source):
    """Load CLIP model+processor from source (repo id or local path) without any network calls."""
    clip_m = CLIPModel.from_pretrained(
        source,
        cache_dir=cache_dir,       # align with your scratch cache
        local_files_only=True      # **critical**: use cache/local files only
    ).to(device)
    clip_p = CLIPProcessor.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=True
    )
    return clip_m, clip_p

try:
    # 1) Try cache for the Hub repo id (no internet; offline mode forces cache)
    clip_model, clip_processor = _load_clip_from(clip_repo_id)
except Exception as e_cache:
    # 2) Fallback: use your explicitly staged local directory (if present)
    if Path(clip_local_dir).exists():
        clip_model, clip_processor = _load_clip_from(clip_local_dir)
    else:
        raise RuntimeError(
            f"CLIP not found in cache and no local dir at: {clip_local_dir}\n"
            f"Stage the model on a login node (see snippet below) and copy to compute node.\n"
            f"Cache error: {e_cache}"
        )


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



output_path = "Synthetic_Image/flowers/"
json_path = "flowers17_descriptions.json"  # update path if needed
cls_template_prompts = [f"a photo of a {cls}" for cls in classes]
gray_scale = False
confident_value = 0.9
num_inference_steps = 20

# -------------------------------
# Load JSON descriptions
# -------------------------------
with open(json_path, "r") as f:
    descriptions = json.load(f)


# -------------------------------
# Prepare reference text embeddings
# -------------------------------
with torch.no_grad():
    text_inputs = clip_processor(text=cls_template_prompts, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)


# -------------------------------
# Output path
# -------------------------------
os.makedirs(output_path, exist_ok=True)

# -------------------------------
# Generate + Infer + Save function
# -------------------------------
def generate_and_infer(prompts_list, expected_class, thresh=0.7):
    results = []
    failed_descriptions = []
    failed_prompts = {}
    saved_count = 0  # track saved images

    for idx, prompt in enumerate(prompts_list):
        # Generate image
        generator = torch.manual_seed(42 + idx)
        image = pipe(
            prompt, guidance_scale=7.5, num_inference_steps=num_inference_steps, generator=generator
        ).images[0]


        
        
        # CLIP inference
        inputs = clip_processor(
            text=cls_template_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # [1, num_classes]
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        image_input = preprocess(image).unsqueeze(0).to(device)

        # Tokenize and move text input to device
        text_input = tokenizer(cls_template_prompts)
        if isinstance(text_input, dict):
            text_input = {k: v.to(device) for k, v in text_input.items()}
        else:
            text_input = text_input.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]



        
        # Get top-1 prediction
        top_class = classes[probs.argmax()]
        top_conf = float(probs.max())
        aligned = (top_class == expected_class)
        high_conf = (top_conf >= thresh)

        if aligned and high_conf:
            class_folder = os.path.join(output_path, expected_class)
            os.makedirs(class_folder, exist_ok=True)
            img_path = os.path.join(class_folder, f"{expected_class}_{idx}.png")
        
            if gray_scale:
                image = image.convert("L")  # convert to grayscale (L mode = 8-bit pixels)
            
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
                "status": status
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
            "status": status
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
    results, failed, failed_prompts, saved_count = generate_and_infer(prompts_list, expected_class=cls, thresh=confident_value)
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

