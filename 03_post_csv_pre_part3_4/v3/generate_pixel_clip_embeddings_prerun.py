
"""
generate_pixel_clip_embeddings.py
Reproducible, provenance-explicit CLIP IMAGE encoding for KAN-Route.
Encodes real COCO image pixels via CLIP's image encoder (not text encoder).
"""
import torch
from PIL import Image
import numpy as np
import open_clip
from pathlib import Path

def generate_pixel_clip_embeddings(image_dir, image_ids, device="cuda", model_name="ViT-B-32", pretrained="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    embeddings = {}
    missing = []

    for img_id in image_ids:
        image_path = Path(image_dir) / f"{img_id}.jpg"
        if not image_path.exists():
            missing.append(img_id)
            continue
        image = Image.open(image_path).convert("RGB")
        pixel_values = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(pixel_values)  # IMAGE encoder, not text
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings[int(img_id)] = image_features.cpu().numpy()[0]

    report = {
        "requested": len(image_ids),
        "encoded": len(embeddings),
        "missing_ids": missing,
        "embedding_dim": next(iter(embeddings.values())).shape[0] if embeddings else None,
    }
    return embeddings, report

if __name__ == "__main__":
    import sys, json
    image_dir = sys.argv[1]
    ids = [int(p.stem) for p in Path(image_dir).glob("*.jpg")]
    embs, report = generate_pixel_clip_embeddings(image_dir, ids)
    print(json.dumps(report, indent=2))
    np.savez(f"{image_dir}_pixel_clip_embeddings.npz", **{str(k): v for k, v in embs.items()})
