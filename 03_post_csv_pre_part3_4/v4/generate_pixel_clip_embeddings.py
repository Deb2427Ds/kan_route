
"""generate_pixel_clip_embeddings.py — robust path discovery, fresh-kernel-safe."""
import torch, numpy as np, json
from pathlib import Path
from PIL import Image
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOLERANCE = 1e-4

def discover_image_dirs():
    """Find any directory literally named 'known' or 'new' anywhere under /kaggle/input."""
    dirs = [p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and p.name.lower() in ("known","new")]
    return dirs

def discover_clip_cache(keyword):
    matches = list(Path("/kaggle/input").rglob(f"*{keyword}*clip_features.npz"))
    return matches[0] if matches else None

def find_coco_image(img_id, image_dirs):
    for d in image_dirs:
        p = d / f"{img_id}.jpg"
        if p.exists(): return p
    return None

def load_clip_cache(path):
    d = np.load(path, allow_pickle=True)
    return {int(k): v for k, v in d[d.files[0]].item().items()}

def run_provenance_check(n_images=10):
    image_dirs = discover_image_dirs()
    if not image_dirs:
        return {"error": "No image directories found", "PROVENANCE_VERIFIED": False}

    qwen_npz = discover_clip_cache("qwen")
    llama_npz = discover_clip_cache("llama")
    caches = {}
    if qwen_npz: caches.update(load_clip_cache(qwen_npz))
    if llama_npz: caches.update(load_clip_cache(llama_npz))
    if not caches:
        return {"error": "No cached clip_features.npz found", "PROVENANCE_VERIFIED": False}

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    clip_model = clip_model.to(DEVICE).eval()

    candidates = [i for i in caches.keys() if find_coco_image(i, image_dirs) is not None][:n_images]
    diffs = []
    for img_id in candidates:
        img_path = find_coco_image(img_id, image_dirs)
        image = Image.open(img_path).convert("RGB")
        pixel_values = clip_preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fresh = clip_model.encode_image(pixel_values)
            fresh = (fresh / fresh.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        diff = float(np.abs(fresh - caches[img_id]).max())
        diffs.append(diff)
        print(f"img_id={img_id}: max_diff={diff:.6f}")

    result = {
        "images_tested": len(candidates), "images_matched": len(diffs),
        "maximum_absolute_difference": max(diffs) if diffs else None,
        "PROVENANCE_VERIFIED": bool(diffs) and len(diffs)==len(candidates) and max(diffs) < TOLERANCE,
    }
    return result

if __name__ == "__main__":
    result = run_provenance_check()
    print(json.dumps(result, indent=2))
