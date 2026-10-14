# ── Cell 0: Locate datasets, output scaffold ──────────────────────────────────
from pathlib import Path
import json, shutil, os, sys, re
import numpy as np, torch, torch.nn as nn, pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
print(f"Device: {DEVICE}")

def find_dir(*fragments):
    for frag in fragments:
        m = [p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and frag in p.name.lower()]
        if m: return m[0]
    return None

def find_file(*fragments):
    for frag in fragments:
        m = [p for p in Path("/kaggle/input").rglob("*") if p.is_file() and frag in p.name.lower()]
        if m: return m[0]
    return None

KAN_QWEN = find_dir("kan_qwen", "kan-qwen")
KAN_LLAMA = find_dir("kan_llama", "llama-kan")
PREV_V3 = find_dir("pre_run_v3", "pre-run_v3", "v3_final")
KNOWN_IMG_DIR = find_dir("known")
NEW_IMG_DIR = find_dir("new")
REVIEW_B = find_file("review_b_ambiguous_queries")

for name, p in [("KAN_QWEN",KAN_QWEN),("KAN_LLAMA",KAN_LLAMA),("PREV_V3",PREV_V3),
                ("KNOWN_IMG_DIR",KNOWN_IMG_DIR),("NEW_IMG_DIR",NEW_IMG_DIR),("REVIEW_B",REVIEW_B)]:
    print(f"{'✅' if p else '❌'} {name}: {p}")

OUT = Path("/kaggle/working/outputs")
for sub in ["class_mappings","splits","canonical","human_benchmark","sanity_checks",
            "leakage_audit","pixel_clip_provenance","validation","part4_eval_prep"]:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
print("✅ scaffold ready")

if not Path("/kaggle/working/fast-kan").exists():
    os.system("git clone https://github.com/ZiyaoLi/fast-kan.git /kaggle/working/fast-kan")
sys.path.append("/kaggle/working/fast-kan")


# ── Check if the v3 mapping files are actually inside the pre-run-v4 dataset ─
pre_run_v4_dir = find_dir("pre-run-v4", "pre_run_v4")
print(f"pre-run-v4 location: {pre_run_v4_dir}")

if pre_run_v4_dir:
    for pattern in ["*class_mapping*.json", "*35way*.pt", "*shared29*.pt", "CANONICAL_RESULTS.md"]:
        matches = list(pre_run_v4_dir.rglob(pattern))
        print(f"\n'{pattern}': {len(matches)} found")
        for m in matches[:5]:
            print(f"  {m}")


# ── Check what's actually inside pre-run-v4-1 ─────────────────────────────────
from pathlib import Path

candidate = Path("/kaggle/input/datasets/deboparna9das/pre-run-v4-1/kaggle/working")
print(f"Exists: {candidate.exists()}")

if candidate.exists():
    print("\nFull contents:")
    for p in sorted(candidate.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(candidate)}")

    print("\n--- Checking for the specific files we need ---")
    for pattern in ["*class_mapping*.json", "*35way*.pt", "kan_qwen_35way_best.pt",
                     "mlp_qwen_35way_best.pt", "kan_llama_35way_best.pt", "mlp_llama_35way_best.pt"]:
        matches = list(candidate.rglob(pattern))
        print(f"'{pattern}': {len(matches)} found")
        for m in matches:
            print(f"    {m}")
else:
    print("Path doesn't exist — check the exact dataset slug and folder structure")
    print("\nTrying broader search instead:")
    base = Path("/kaggle/input/datasets/deboparna9das/pre-run-v4-1")
    if base.exists():
        for p in sorted(base.rglob("*"))[:50]:
            print(f"  {p}")


# ── Fix: PREV_V3 points to pre-run-v4-1, not pre-run-v4 ──────────────────────
PREV_V3 = Path("/kaggle/input/datasets/deboparna9das/pre-run-v4-1/kaggle/working")
print(f"PREV_V3 set to: {PREV_V3}")
print(f"Exists: {PREV_V3.exists()}")

# Quick confirm all needed files resolve correctly
for pattern in ["qwen_35_class_mapping.json", "llama_35_class_mapping.json", "shared_29_class_mapping.json",
                "kan_qwen_35way_best.pt", "mlp_qwen_35way_best.pt", "kan_llama_35way_best.pt", "mlp_llama_35way_best.pt"]:
    found = list(PREV_V3.rglob(pattern))
    print(f"{'✅' if found else '❌'} {pattern}")


# ── Cell 1: Shared architecture + helpers ─────────────────────────────────────
from fastkan import FastKAN

class KANRoute(nn.Module):
    def __init__(self, input_dim=896, proj_dim=48, hidden_dims=[48], num_tools=35, num_grids=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.kan  = FastKAN(layers_hidden=[proj_dim]+hidden_dims+[num_tools], num_grids=num_grids)
    def forward(self, x): return self.kan(self.norm(self.proj(x.float())))
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=896, proj_dim=96, hidden_dims=[64,32], num_tools=35, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout)]
        prev = proj_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_tools))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.float())
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ToolRoutingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def evaluate_full(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for X_b,y_b in loader:
            X_b,y_b = X_b.float().to(DEVICE), y_b.to(DEVICE)
            p = model(X_b).argmax(-1)
            preds.extend(p.cpu().numpy()); labels.extend(y_b.cpu().numpy())
    return np.array(preds), np.array(labels)

def train_model(model, tr, va, name, epochs=60, lr=1e-3, wd=1e-4, is_kan=False, save_dir=None):
    if is_kan: lr, epochs = 5e-4, 80
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(tr), epochs=epochs, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, path = 0.0, (save_dir or OUT)/f"{name}_best.pt"
    for ep in range(1, epochs+1):
        model.train()
        for X_b,y_b in tr:
            X_b,y_b = X_b.float().to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad(); loss = crit(model(X_b), y_b); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        model.eval(); c,t=0,0
        with torch.no_grad():
            for X_b,y_b in va:
                X_b,y_b = X_b.float().to(DEVICE), y_b.to(DEVICE)
                c += (model(X_b).argmax(-1)==y_b).sum().item(); t += len(y_b)
        acc = c/t
        if acc > best: best = acc; torch.save(model.state_dict(), path)
        if ep%20==0 or ep==1: print(f"  [{name}] ep{ep}/{epochs} val={acc:.4f}")
    model.load_state_dict(torch.load(path))
    return best
print("✅ ready")


# ── Cell 2: Load raw datasets + class mappings from previous session ─────────
with open(KAN_QWEN / "augmented_dataset_final.json") as f:
    qwen_data = json.load(f)
qwen_text = np.load(KAN_QWEN / "aug_text_embeddings.npy")
qc = np.load(KAN_QWEN / "clip_features.npz", allow_pickle=True)
qwen_clip = {int(k):v for k,v in qc[qc.files[0]].item().items()}
qwen_clip_vecs = np.stack([qwen_clip[int(r["img_id"])] for r in qwen_data])
X_qwen_896 = np.concatenate([qwen_text, qwen_clip_vecs], axis=1).astype(np.float32)

with open(KAN_LLAMA / "raw_dataset_FINAL.json") as f:
    llama_data = json.load(f)
llama_text = np.load(KAN_LLAMA / "text_embeddings.npy")
lc = np.load(KAN_LLAMA / "clip_features.npz", allow_pickle=True)
llama_clip = {int(k):v for k,v in lc[lc.files[0]].item().items()}
llama_clip_vecs = np.stack([llama_clip[int(r["img_id"])] for r in llama_data])
X_llama_896 = np.concatenate([llama_text, llama_clip_vecs], axis=1).astype(np.float32)

with open(next(PREV_V3.rglob("qwen_35_class_mapping.json"))) as f:
    qwen_mapping = json.load(f)
with open(next(PREV_V3.rglob("llama_35_class_mapping.json"))) as f:
    llama_mapping = json.load(f)
with open(next(PREV_V3.rglob("shared_29_class_mapping.json"))) as f:
    shared_mapping = json.load(f)

qwen_classes = sorted(qwen_mapping["class_to_local_id"].keys())
llama_classes = sorted(llama_mapping["class_to_local_id"].keys())
shared_classes = sorted(shared_mapping["class_to_local_id"].keys())

y_qwen_local = np.array([qwen_mapping["class_to_local_id"][r["tool"]] for r in qwen_data], dtype=np.int64)
y_llama_local = np.array([llama_mapping["class_to_local_id"][r["tool"]] for r in llama_data], dtype=np.int64)

print(f"Qwen: {X_qwen_896.shape}, 35 classes | Llama: {X_llama_896.shape}, 35 classes | Shared: {len(shared_classes)}")


# ── Cell 3: Qwen-35 full class distribution audit ─────────────────────────────
qwen_class_counts = []
for tool_name in qwen_classes:
    rows = [r for r in qwen_data if r["tool"] == tool_name]
    total_samples = len(rows)
    unique_queries = len(set(r["query"] for r in rows))
    unique_images = len(set(r["img_id"] for r in rows))
    qwen_class_counts.append(dict(
        tool_name=tool_name, total_samples=total_samples,
        unique_queries=unique_queries, unique_images=unique_images
    ))

qwen_dist_df = pd.DataFrame(qwen_class_counts).sort_values("total_samples")
qwen_dist_df.to_csv(OUT/"class_mappings"/"qwen35_class_distribution_before.csv", index=False)
print(qwen_dist_df.to_string(index=False))

below_100 = qwen_dist_df[qwen_dist_df["total_samples"] < 100]
print(f"\nClasses below 100 samples: {len(below_100)}")
print(below_100[["tool_name","total_samples"]].to_string(index=False))


# ── Cell 4: Load Qwen2.5-7B, generate new queries for starved classes ────────
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    low_cpu_mem_usage=True, trust_remote_code=True,
)
teacher.eval()
print(f"✅ Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

TOOL_DESCRIPTIONS = {
    "image_colorization": "add realistic color to a grayscale or black-and-white image",
    "matting": "extract a foreground subject with soft, fine-detailed edges (hair, fur, transparency)",
    "pose_estimation": "determine body posture and joint/skeleton positions of a person",
}

def build_targeted_prompt(objects, tool_name):
    obj_str = ", ".join(objects) if objects else "various objects"
    messages = [
        {"role": "system", "content": (
            "You are a precise JSON generator for an AI benchmark dataset. "
            "You ALWAYS output valid JSON and nothing else. "
            "No markdown, no explanation, no code blocks."
        )},
        {"role": "user", "content": (
            f"Image contains: {obj_str}\n\n"
            f"Tool: {tool_name} — {TOOL_DESCRIPTIONS[tool_name]}\n\n"
            "Write ONE realistic, natural user query that would require this exact tool. "
            "Vary the phrasing style each time (direct request, casual question, or descriptive request).\n\n"
            'Output ONLY this JSON: {"query": "..."}'
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_for_class(tool_name, target_count, existing_records, batch_size=8, max_new_tokens=100):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    source_pool = [(r["img_id"], r.get("objects", [])) for r in existing_records]
    random.seed(SEED)
    random.shuffle(source_pool)

    new_records = []
    attempts = 0
    while len(new_records) < target_count and attempts < target_count * 3:
        start_idx = attempts % len(source_pool)
        batch = source_pool[start_idx: start_idx + batch_size]
        if not batch:
            attempts += batch_size; continue
        prompts = [build_targeted_prompt(objs, tool_name) for _, objs in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output_ids = teacher.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.8, top_p=0.9, repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for (img_id, objs), out_ids in zip(batch, output_ids):
            raw = tokenizer.decode(out_ids[input_len:], skip_special_tokens=True).strip()
            try:
                if "```" in raw:
                    raw = raw.split("```")[1].replace("json","",1).strip()
                parsed = json.loads(raw)
                query_text = parsed.get("query", "").strip()
                if query_text and len(new_records) < target_count:
                    new_records.append({
                        "img_id": img_id, "query": query_text, "tool": tool_name,
                        "objects": objs, "source": "item1_correction_qwen25_7b",
                    })
            except Exception:
                continue
        attempts += batch_size
        if attempts % 40 == 0:
            print(f"  [{tool_name}] {len(new_records)}/{target_count} generated...")
    return new_records

STARVED_CLASSES = below_100["tool_name"].tolist()
TARGET_PER_CLASS = 150

all_new_records = []
for tool_name in STARVED_CLASSES:
    current_n = int(below_100[below_100.tool_name==tool_name].total_samples.iloc[0])
    print(f"\nGenerating for '{tool_name}' (current: {current_n})...")
    new = generate_for_class(tool_name, TARGET_PER_CLASS, qwen_data)
    all_new_records.extend(new)
    print(f"  ✅ Generated {len(new)} new samples for {tool_name}")

print(f"\n✅ Total new records generated: {len(all_new_records)}")


# ── Cell 4b: Extend TOOL_DESCRIPTIONS to cover ALL classes in below_100 ───────
print("Full list of classes below 100:")
print(below_100[["tool_name","total_samples"]].to_string(index=False))

# Generic fallback description generator for any class not already covered
GENERIC_DESCRIPTIONS = {
    "face_detection": "locate and identify human faces in an image",
    "text_detection_ocr": "detect and read visible text in an image",
    "logo_detection": "identify brand logos visible in an image",
    "vehicle_detection": "locate vehicles such as cars, trucks, or motorcycles in an image",
    "animal_detection": "locate animals present in an image",
    "gaze_estimation": "determine where a person is looking in an image",
    "age_gender_estimation": "estimate the age and gender of people in an image",
    "facial_expression_recognition": "identify the emotional expression on a person's face",
    "super_resolution": "increase the resolution and sharpness of an image",
    "image_deblurring": "remove blur and sharpen a blurry image",
    "image_denoising": "remove visual noise or grain from an image",
    "anomaly_detection": "identify something unusual or out of place in an image",
    "point_cloud_generation": "generate 3D point-cloud data from an image",
    # add more as needed based on what below_100 actually shows
}

for tool_name in below_100["tool_name"].tolist():
    if tool_name not in TOOL_DESCRIPTIONS:
        if tool_name in GENERIC_DESCRIPTIONS:
            TOOL_DESCRIPTIONS[tool_name] = GENERIC_DESCRIPTIONS[tool_name]
        else:
            # Last-resort fallback: derive a readable description from the tool name itself
            TOOL_DESCRIPTIONS[tool_name] = tool_name.replace("_", " ")
            print(f"⚠️ No hand-written description for '{tool_name}' — using auto-derived fallback: "
                  f"'{TOOL_DESCRIPTIONS[tool_name]}'. Consider writing a better one before continuing.")

print(f"\nTOOL_DESCRIPTIONS now covers: {sorted(TOOL_DESCRIPTIONS.keys())}")


# ── Cell 4c: Resume generation for remaining classes only ────────────────────
already_done = {"image_colorization", "matting", "pose_estimation"}
remaining_classes = [t for t in below_100["tool_name"].tolist() if t not in already_done]
print(f"Remaining classes to generate: {remaining_classes}")

for tool_name in remaining_classes:
    current_n = int(below_100[below_100.tool_name==tool_name].total_samples.iloc[0])
    print(f"\nGenerating for '{tool_name}' (current: {current_n})...")
    new = generate_for_class(tool_name, TARGET_PER_CLASS, qwen_data)
    all_new_records.extend(new)
    print(f"  ✅ Generated {len(new)} new samples for {tool_name}")

print(f"\n✅ Total new records generated (all classes): {len(all_new_records)}")


# ── Cell 5: Dedup new queries, assign tool_ids, merge, save before/after ─────
def normalize_q(q): return re.sub(r"[^\w\s]", "", str(q).lower()).strip()

existing_norms = set(normalize_q(r["query"]) for r in qwen_data)
seen_new_norms = set()
deduped_new = []
for r in all_new_records:
    n = normalize_q(r["query"])
    if n in existing_norms or n in seen_new_norms:
        continue
    seen_new_norms.add(n)
    deduped_new.append(r)

print(f"New records after dedup: {len(deduped_new)} (removed {len(all_new_records)-len(deduped_new)} duplicates)")

for r in deduped_new:
    r["tool_id"] = qwen_mapping["class_to_local_id"][r["tool"]]

qwen35_balanced = qwen_data + deduped_new
pd.DataFrame(qwen35_balanced).to_csv(OUT/"class_mappings"/"qwen35_balanced_final.csv", index=False)

after_counts = []
for tool_name in qwen_classes:
    rows = [r for r in qwen35_balanced if r["tool"]==tool_name]
    after_counts.append(dict(tool_name=tool_name, total_samples=len(rows),
                              unique_queries=len(set(r["query"] for r in rows)),
                              unique_images=len(set(r["img_id"] for r in rows))))
after_df = pd.DataFrame(after_counts).sort_values("total_samples")
after_df.to_csv(OUT/"class_mappings"/"qwen35_class_distribution_after.csv", index=False)

print(f"\n=== Before -> After for corrected classes ===")
for tool_name in STARVED_CLASSES:
    before_n = int(qwen_dist_df[qwen_dist_df.tool_name==tool_name].total_samples.iloc[0])
    after_n = int(after_df[after_df.tool_name==tool_name].total_samples.iloc[0])
    print(f"  {tool_name}: {before_n} -> {after_n}")

print(f"\nAny class still below 100: {(after_df['total_samples']<100).sum()}")


# ── Verify what's actually on disk right now ──────────────────────────────────
print("Files saved so far:")
for f in OUT.rglob("*"):
    if f.is_file():
        print(f"  {f.relative_to(OUT)}  ({f.stat().st_size/1e3:.1f} KB)")


# ── Cell 5b: Safety checkpoint — save intermediate generation results ────────
with open(OUT/"class_mappings"/"all_new_records_raw.json", "w") as f:
    json.dump(all_new_records, f, indent=2)
with open(OUT/"class_mappings"/"deduped_new_records.json", "w") as f:
    json.dump(deduped_new, f, indent=2)

print(f"✅ Checkpointed {len(all_new_records)} raw + {len(deduped_new)} deduped new records to disk")
print("Files now saved:")
for f in OUT.rglob("*"):
    if f.is_file():
        print(f"  {f.relative_to(OUT)}")


# ── Cell 6: MiniLM + Pixel-CLIP embeddings for new samples only ──────────────
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('all-MiniLM-L6-v2', device=str(DEVICE))
for p in sbert.parameters(): p.requires_grad = False

new_queries = [r["query"] for r in deduped_new]
new_text_emb = sbert.encode(new_queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
print(f"✅ New text embeddings: {new_text_emb.shape}")

# Pixel-CLIP: reuse existing cached vectors where the img_id already has one
# (new samples reuse REAL existing COCO images from the corpus, so almost all
# should already be cached — only genuinely new img_ids need fresh encoding)
missing_clip_ids = [r["img_id"] for r in deduped_new if int(r["img_id"]) not in qwen_clip]
print(f"New samples needing fresh CLIP encoding: {len(missing_clip_ids)} / {len(deduped_new)}")

if missing_clip_ids:
    import open_clip
    from PIL import Image
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', force_quick_gelu=True
    )
    clip_model = clip_model.to(DEVICE).eval()

    def find_coco_image(img_id):
        for d in [KNOWN_IMG_DIR, NEW_IMG_DIR]:
            if d is None: continue
            p = Path(d) / f"{img_id}.jpg"
            if p.exists(): return p
        return None

    newly_encoded = 0
    for img_id in set(missing_clip_ids):
        img_path = find_coco_image(img_id)
        if img_path is None:
            continue
        image = Image.open(img_path).convert("RGB")
        pixel_values = clip_preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_image(pixel_values)
            feat = (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        qwen_clip[int(img_id)] = feat
        newly_encoded += 1
    print(f"✅ Freshly encoded {newly_encoded} new images")

new_clip_vecs = np.stack([qwen_clip[int(r["img_id"])] for r in deduped_new])
X_new = np.concatenate([new_text_emb, new_clip_vecs], axis=1).astype(np.float32)
print(f"\n✅ New sample feature matrix: {X_new.shape}")

# Merge with existing canonical features
X_qwen_balanced = np.concatenate([X_qwen_896, X_new], axis=0)
y_qwen_balanced_local = np.concatenate([
    y_qwen_local,
    np.array([r["tool_id"] for r in deduped_new], dtype=np.int64)
])

print(f"\n✅ Combined balanced dataset: X={X_qwen_balanced.shape}  y={y_qwen_balanced_local.shape}")
assert X_qwen_balanced.shape[0] == y_qwen_balanced_local.shape[0] == len(qwen35_balanced)
print("✅ Row/label/feature alignment verified — no missing IDs")

np.save(OUT/"class_mappings"/"X_qwen_balanced.npy", X_qwen_balanced)
np.save(OUT/"class_mappings"/"y_qwen_balanced_local.npy", y_qwen_balanced_local)
print("✅ Saved balanced feature matrix and labels")


import shutil

shutil.make_archive(
    "/kaggle/working/kaggle_working_backup",
    "zip",
    "/kaggle/working"
)

print("/kaggle/working/kaggle_working_backup.zip")
