
"""
pixel_clip_provenance_check.py — fully self-contained.
Works after Restart Kernel -> Run All with zero dependency
on prior notebook state.
"""

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ACTUAL KAGGLE PATHS
# ============================================================

QWEN_CLIP_NPZ = Path(
    "/kaggle/input/datasets/deboparna9das/kan-qwen/qwen/kan_qwen/clip_features.npz"
)

LLAMA_CLIP_NPZ = Path(
    "/kaggle/input/datasets/deboparna9das/llama-kan/llama/kan_llama/clip_features.npz"
)

# ============================================================
# Verify cache files exist
# ============================================================

print("Checking CLIP cache files...")

print(f"Qwen cache:  {QWEN_CLIP_NPZ}")
print(f"Exists:      {QWEN_CLIP_NPZ.exists()}")

print(f"LLaMA cache: {LLAMA_CLIP_NPZ}")
print(f"Exists:      {LLAMA_CLIP_NPZ.exists()}")

if not QWEN_CLIP_NPZ.exists():
    raise FileNotFoundError(
        f"Qwen CLIP cache not found: {QWEN_CLIP_NPZ}"
    )

if not LLAMA_CLIP_NPZ.exists():
    raise FileNotFoundError(
        f"LLaMA CLIP cache not found: {LLAMA_CLIP_NPZ}"
    )

# ============================================================
# Image directories
# ============================================================

KNOWN_DIR = Path("/kaggle/input/images-folder/known")
NEW_DIR = Path("/kaggle/input/images-folder/new")

def find_coco_image(img_id):
    for d in [KNOWN_DIR, NEW_DIR]:
        p = d / f"{img_id}.jpg"
        if p.exists():
            return p
    return None

# ============================================================
# Load cached CLIP embeddings
# ============================================================

def load_clip_cache(path):
    d = np.load(path, allow_pickle=True)
    return {
        int(k): v
        for k, v in d[d.files[0]].item().items()
    }

qwen_clip = load_clip_cache(QWEN_CLIP_NPZ)
llama_clip = load_clip_cache(LLAMA_CLIP_NPZ)

print(f"Loaded Qwen CLIP embeddings:  {len(qwen_clip)}")
print(f"Loaded LLaMA CLIP embeddings: {len(llama_clip)}")

# ============================================================
# Load fresh OpenCLIP model
# ============================================================

print("Loading OpenCLIP ViT-B-32...")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    force_quick_gelu=True
)

clip_model = clip_model.to(DEVICE).eval()

print(f"Using device: {DEVICE}")

# ============================================================
# Select sample images
# ============================================================

all_ids = sorted(
    set(qwen_clip.keys()) | set(llama_clip.keys())
)

sample_ids = [
    i for i in all_ids
    if find_coco_image(i) is not None
][:10]

print(f"Testing {len(sample_ids)} images...")

# ============================================================
# Compare fresh pixel embeddings against cache
# ============================================================

TOLERANCE = 1e-4
diffs = []

for img_id in sample_ids:

    img_path = find_coco_image(img_id)

    image = Image.open(img_path).convert("RGB")

    pixel_values = (
        clip_preprocess(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        fresh_emb = clip_model.encode_image(pixel_values)

        fresh_emb = (
            fresh_emb /
            fresh_emb.norm(dim=-1, keepdim=True)
        ).cpu().numpy()[0]

    # Prefer Qwen cache, otherwise LLaMA cache
    cached = qwen_clip.get(img_id)

    if cached is None:
        cached = llama_clip.get(img_id)

    if cached is None:
        print(f"img_id={img_id}: NO CACHE FOUND")
        continue

    diff = float(
        np.abs(fresh_emb - cached).max()
    )

    diffs.append(diff)

    print(
        f"img_id={img_id}: "
        f"max_diff={diff:.6f}"
    )

# ============================================================
# Final result
# ============================================================

result = {
    "n_requested": 10,
    "n_tested": len(sample_ids),
    "n_compared": len(diffs),
    "max_diff": max(diffs) if diffs else None,
    "tolerance": TOLERANCE,
    "PROVENANCE_VERIFIED": (
        bool(diffs)
        and len(diffs) == len(sample_ids)
        and max(diffs) < TOLERANCE
    ),
}

print("\n" + "=" * 60)
print("PIXEL-CLIP PROVENANCE RESULT")
print("=" * 60)

print(json.dumps(result, indent=2))
