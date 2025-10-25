from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json

# -------------------------------
# Load Stable Diffusion
# -------------------------------
'''
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32 if device=="cpu" else torch.float16
)
pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()
'''


s

'''
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
import json

# -------------------------------
# Load Medical X-ray Stable Diffusion (X-rays, CTs, and MRIs)
# -------------------------------

model_id = "Osama03/Medical-X-ray-image-generation-stable-diffusion"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,         # ✅ ensure caching in scratch
    use_auth_token=True,         # ✅ use your login token
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()

'''

# Define cache location
cache_dir = "/home/shahab33/scratch/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["DIFFUSERS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,         # ✅ ensure caching in scratch
    use_auth_token=True,         # ✅ use your login token
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()
    
# -------------------------------
# Load Hugging Face CLIP model
# -------------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
'''


classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]


output_path = "Synthetic_Image/EuroSAT/"
json_path = "eurosat_descriptions.json"  # update path if needed
cls_template_prompts = [f"a photo of a {cls} area" for cls in classes]
gray_scale = False
confident_value = 0.5
num_inference_steps = 50

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

for cls in ["HerbaceousVegetation", "PermanentCrop"]:
#for cls in classes:
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

