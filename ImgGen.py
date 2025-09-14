from diffusers import StableDiffusionPipeline
import torch
import clip
from PIL import Image
import os
import json

# -------------------------------
# Load Stable Diffusion
# -------------------------------
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32 if device=="cpu" else torch.float16
)
pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()

# -------------------------------
# Load CLIP model
# -------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# CIFAR-10 class names
# -------------------------------
cifar_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -------------------------------
# Load JSON descriptions
# -------------------------------
json_path = "cifar10_descriptions.json"  # update path if needed
with open(json_path, "r") as f:
    descriptions = json.load(f)

# -------------------------------
# Reference text embeddings for CLIP
# -------------------------------
cls_template_prompts = [f"a photo of a {cls}" for cls in cifar_classes]
text_tokens = clip.tokenize(cls_template_prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# -------------------------------
# Output path
# -------------------------------
output_path = "Synthetic_Image/"
os.makedirs(output_path, exist_ok=True)

# -------------------------------
# Generate + Infer + Save function
# -------------------------------
def generate_and_infer(prompts_dict, expected_class, thresh=0.95):
    results = []
    failed_descriptions = []
    failed_prompts = {}

    for idx, p in enumerate(prompts_dict):
        # Generate image
        generator = torch.manual_seed(42 + idx)
        image = pipe(
            p, guidance_scale=7.5, num_inference_steps=20, generator=generator
        ).images[0]

        # CLIP inference
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_per_image = 100.0 * image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # Determine top-1 prediction
        top_class = cifar_classes[probs.argmax()]
        top_conf = float(probs.max())

        # Check alignment and confidence
        aligned = (top_class == expected_class)
        high_conf = (top_conf >= thresh)

        if aligned and high_conf:
            # Save image
            class_folder = os.path.join(output_path, expected_class)
            os.makedirs(class_folder, exist_ok=True)
            img_path = os.path.join(class_folder, f"{expected_class}_{idx}.png")
            image.save(img_path)
            status = f"✅ Aligned & confident (confidence {top_conf:.2f}) - Saved to {img_path}"
        else:
            # Determine reason
            if not aligned:
                reason = "not aligned"
            elif not high_conf:
                reason = f"low confidence {top_conf:.2f}"
            else:
                reason = "unknown"

            status = (
                f"❌ Failed. Reason: {reason}. Prompt: {p} -> Predicted '{top_class}'"
            )
            failed_descriptions.append({
                "idx": idx,
                "prompt": p,
                "expected_class": expected_class,
                "predicted_class": top_class,
                "confidence": top_conf,
                "soft_labels": {cls: float(pr) for cls, pr in zip(cifar_classes, probs)},
                "status": status
            })
            failed_prompts[idx] = p

        print(f"{idx} - {status}")
        results.append({
            "idx": idx,
            "prompt": p,
            "expected_class": expected_class,
            "predicted_class": top_class,
            "confidence": top_conf,
            "soft_labels": {cls: float(pr) for cls, pr in zip(cifar_classes, probs)},
            "status": status
        })

    return results, failed_descriptions, failed_prompts

# -------------------------------
# Run generation for all classes
# -------------------------------
all_results = {}
all_failed = {}
all_failed_prompts = {}

for cls in cifar_classes:
    print(f"\n--- Generating images for class: {cls} ---")
    prompts_list = descriptions[cls]
    results, failed, failed_prompts = generate_and_infer(prompts_list, expected_class=cls, thresh=0.95)
    all_results[cls] = results
    all_failed[cls] = failed
    all_failed_prompts[cls] = failed_prompts

print("\nAll generation completed.")
