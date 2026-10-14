##############################################################################
# STAGE A — Seed fix, retrain all 18, re-audit leakage, corrected eval matrix
# Run in a fresh Kaggle kernel with: kan-qwen, llama-kan, images-new-known,
# and your latest KAN_ROUTE_PART4_FINAL.zip (re-uploaded as a dataset, for the
# split CSVs — NOT for checkpoints this time, since every checkpoint is
# invalidated by the seed bug and must be retrained).
##############################################################################
get_ipython().getoutput("pip install -q open_clip_torch")
from pathlib import Path
import json, os, sys, re, random, shutil
import numpy as np, torch, torch.nn as nn, pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def find_dir(*fragments):
    for frag in fragments:
        m = [p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and frag in p.name.lower()]
        if m: return m[0]
    return None

KAN_QWEN = find_dir("kan_qwen", "kan-qwen")
KAN_LLAMA = find_dir("kan_llama", "llama-kan")
KNOWN_IMG_DIR = find_dir("known")
NEW_IMG_DIR = find_dir("new")
if KNOWN_IMG_DIR == NEW_IMG_DIR and KNOWN_IMG_DIR is not None:
    subdirs = {p.name.lower(): p for p in KNOWN_IMG_DIR.iterdir() if p.is_dir()}
    if "known" in subdirs and "new" in subdirs:
        KNOWN_IMG_DIR, NEW_IMG_DIR = subdirs["known"], subdirs["new"]

# Locate the previous run's zip (re-uploaded) to reuse split CSVs + human_shared28.csv
# — we do NOT trust its checkpoints this time (seed bug invalidates all of them),
# only its data files, per prompt_v5 Section 3: "do not regenerate splits unless corrupt."
_prev_matches = list(Path("/kaggle/input").rglob("qwen_shared28_train.csv"))
assert _prev_matches, "Previous run's zip (with split CSVs) not attached — re-upload KAN_ROUTE_PART4_FINAL.zip as a dataset."
PREV_RUN_BASE = _prev_matches[0].parent
print(f"✅ Reusing splits from: {PREV_RUN_BASE}")

OUT = Path("/kaggle/working/part4_final")
for sub in ["", "part4_predictions_seed_corrected", "confusion_matrices", "tables", "figures", "checkpoints_seed_corrected"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

if not Path("/kaggle/working/fast-kan").exists():
    os.system("git clone https://github.com/ZiyaoLi/fast-kan.git /kaggle/working/fast-kan")
sys.path.append("/kaggle/working/fast-kan")
from fastkan import FastKAN

# --------------------------------------------------------------------------
# Load the EXISTING splits (Section 3: preserve, do not regenerate) directly
# from the saved CSVs — these already include the corrective-augmentation rows.
# --------------------------------------------------------------------------
qwen_shared28_train = pd.read_csv(PREV_RUN_BASE / "qwen_shared28_train.csv")
qwen_shared28_val   = pd.read_csv(PREV_RUN_BASE / "qwen_shared28_val.csv")
qwen_shared28_test  = pd.read_csv(PREV_RUN_BASE / "qwen_shared28_test.csv")
llama_shared28_train = pd.read_csv(PREV_RUN_BASE / "llama_shared28_train.csv")
llama_shared28_val   = pd.read_csv(PREV_RUN_BASE / "llama_shared28_val.csv")
llama_shared28_test  = pd.read_csv(PREV_RUN_BASE / "llama_shared28_test.csv")
human_shared28 = pd.read_csv(PREV_RUN_BASE / "human_shared28.csv")
shared_28_mapping = json.load(open(PREV_RUN_BASE / "shared_28_class_mapping.json"))
local_id_to_class = {int(k): v for k, v in shared_28_mapping["local_id_to_class"].items()}
class_to_local_id = shared_28_mapping["class_to_local_id"]
shared_28 = sorted(class_to_local_id.keys())
print(f"Qwen: train={len(qwen_shared28_train)} val={len(qwen_shared28_val)} test={len(qwen_shared28_test)}")
print(f"Llama: train={len(llama_shared28_train)} val={len(llama_shared28_val)} test={len(llama_shared28_test)}")
print(f"Human: {len(human_shared28)}")

# --------------------------------------------------------------------------
# Section 3 — Re-verify leakage from the loaded splits (not copied PASS text)
# --------------------------------------------------------------------------
def normalize_q(q): return re.sub(r"[^\w\s]", "", str(q).lower()).strip()

def leakage_report(df_tr, df_va, df_te, name):
    ntr, nva, nte = (set(df["query"].apply(normalize_q)) for df in (df_tr, df_va, df_te))
    itr, iva, ite = (set(zip(df["img_id"], df["tool"])) for df in (df_tr, df_va, df_te))
    r = {"norm_tr_va": len(ntr & nva), "norm_tr_te": len(ntr & nte), "norm_va_te": len(nva & nte),
         "imgtool_tr_va": len(itr & iva), "imgtool_tr_te": len(itr & ite), "imgtool_va_te": len(iva & ite)}
    print(f"=== {name} ===")
    for k, v in r.items(): print(f"  {'✅' if v == 0 else '❌'} {k}: {v}")
    return r, all(v == 0 for v in r.values())

q_leak, q_zero = leakage_report(qwen_shared28_train, qwen_shared28_val, qwen_shared28_test, "Qwen-28 (recomputed)")
l_leak, l_zero = leakage_report(llama_shared28_train, llama_shared28_val, llama_shared28_test, "Llama-28 (recomputed)")
assert q_zero and l_zero, "STOP: leakage detected in the preserved splits — do not proceed silently."

with open(OUT / "shared28_split_leakage_audit_FINAL_V2.md", "w") as f:
    f.write("# Shared-28 Split Leakage Audit — FINAL V2 (recomputed, not copied)\n\n## Qwen-28\n\n")
    for k, v in q_leak.items(): f.write(f"- {k}: {v}\n")
    f.write(f"\nLEAKAGE FREE: {q_zero}\n\n## Llama-28\n\n")
    for k, v in l_leak.items(): f.write(f"- {k}: {v}\n")
    f.write(f"\nLEAKAGE FREE: {l_zero}\n")
print("✅ shared28_split_leakage_audit_FINAL_V2.md written (freshly recomputed)")

# --------------------------------------------------------------------------
# Rebuild 896-d features for exactly these rows (real pixel-CLIP + MiniLM,
# same procedure as before, so results are directly comparable)
# --------------------------------------------------------------------------
qc = np.load(KAN_QWEN / "clip_features.npz", allow_pickle=True)
qwen_clip = {int(k): v for k, v in qc[qc.files[0]].item().items()}
lc = np.load(KAN_LLAMA / "clip_features.npz", allow_pickle=True)
llama_clip = {int(k): v for k, v in lc[lc.files[0]].item().items()}

from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image

minilm = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True)
clip_model = clip_model.to(DEVICE).eval()

def build_features(df, clip_cache):
    text_emb = minilm.encode(df["query"].tolist(), batch_size=256, convert_to_numpy=True,
                              normalize_embeddings=False).astype(np.float32)
    clip_vecs = np.stack([clip_cache[int(iid)] for iid in df["img_id"]]).astype(np.float32)
    return np.concatenate([text_emb, clip_vecs], axis=1).astype(np.float32)

def find_coco_image(img_id, image_dirs):
    for d in image_dirs:
        p = Path(d) / f"{img_id}.jpg"
        if p.exists(): return p
    return None

image_dirs = [KNOWN_IMG_DIR, NEW_IMG_DIR]

def build_human_features(df):
    text_emb = minilm.encode(df["query"].tolist(), batch_size=256, convert_to_numpy=True,
                              normalize_embeddings=False).astype(np.float32)
    clip_vecs = []
    for iid in df["image_id"]:
        p = find_coco_image(iid, image_dirs)
        image = Image.open(p).convert("RGB")
        pv = clip_preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(pv)
            f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        clip_vecs.append(f)
    return np.concatenate([text_emb, np.stack(clip_vecs).astype(np.float32)], axis=1).astype(np.float32)

X_tr_Q = build_features(qwen_shared28_train, qwen_clip); y_tr_Q = qwen_shared28_train["tool"].map(class_to_local_id).values
X_va_Q = build_features(qwen_shared28_val, qwen_clip);   y_va_Q = qwen_shared28_val["tool"].map(class_to_local_id).values
X_te_Q = build_features(qwen_shared28_test, qwen_clip);  y_te_Q = qwen_shared28_test["tool"].map(class_to_local_id).values
X_tr_L = build_features(llama_shared28_train, llama_clip); y_tr_L = llama_shared28_train["tool"].map(class_to_local_id).values
X_va_L = build_features(llama_shared28_val, llama_clip);   y_va_L = llama_shared28_val["tool"].map(class_to_local_id).values
X_te_L = build_features(llama_shared28_test, llama_clip);  y_te_L = llama_shared28_test["tool"].map(class_to_local_id).values
X_human_28 = build_human_features(human_shared28)
y_human_28_canonical = human_shared28["tool_name"].map(class_to_local_id).values

X_tr_Mix = np.concatenate([X_tr_Q, X_tr_L], axis=0)  # leak-free pool already baked into these saved splits
y_tr_Mix = np.concatenate([y_tr_Q, y_tr_L], axis=0)
X_va_Mix = np.concatenate([X_va_Q, X_va_L], axis=0)
y_va_Mix = np.concatenate([y_va_Q, y_va_L], axis=0)
print(f"✅ Features rebuilt. Mixed train: {len(y_tr_Mix)} rows")

has_multi = human_shared28["acceptable_tool_labels"].astype(str).str.strip() != ""
strict_true = human_shared28["strict_eval"] == True
human_strict_mask = strict_true & ~has_multi
human_ambig_mask = strict_true & has_multi


##############################################################################
# Section 2 — THE FIX: seed set before model construction, every time
##############################################################################
class KANRoute(nn.Module):
    def __init__(self, input_dim=896, proj_dim=48, hidden_dims=[48], num_tools=28, num_grids=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim); self.norm = nn.LayerNorm(proj_dim)
        self.kan = FastKAN(layers_hidden=[proj_dim] + hidden_dims + [num_tools], num_grids=num_grids)
    def forward(self, x): return self.kan(self.norm(self.proj(x.float())))
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=896, proj_dim=96, hidden_dims=[64, 32], num_tools=28, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout)]
        prev = proj_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]; prev = h
        layers.append(nn.Linear(prev, num_tools))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x.float())
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ToolRoutingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def evaluate_full_probs(model, loader):
    model.eval(); preds, labels, all_probs = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.float().to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b); probs = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(-1).cpu().numpy()); labels.extend(y_b.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(all_probs)

def set_seed(seed):
    """THE FIX (prompt_v5 Section 2). Call this BEFORE constructing the model —
    not just before training it. This is what actually varies weight init
    across seeds; the old code only varied training-time stochasticity."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, tr, va, name, epochs=60, lr=1e-3, wd=1e-4, is_kan=False, save_dir=None):
    if is_kan: lr, epochs = 5e-4, 80
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(tr), epochs=epochs, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, path = 0.0, (save_dir or OUT) / f"{name}_best.pt"
    for ep in range(1, epochs + 1):
        model.train()
        for X_b, y_b in tr:
            X_b, y_b = X_b.float().to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad(); loss = crit(model(X_b), y_b); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        model.eval(); c, t = 0, 0
        with torch.no_grad():
            for X_b, y_b in va:
                X_b, y_b = X_b.float().to(DEVICE), y_b.to(DEVICE)
                c += (model(X_b).argmax(-1) == y_b).sum().item(); t += len(y_b)
        acc = c / t
        if acc > best: best = acc; torch.save(model.state_dict(), path)
        if ep % 20 == 0 or ep == 1: print(f"  [{name}] ep{ep}/{epochs} val={acc:.4f}")
    model.load_state_dict(torch.load(path))
    return best

TRAINING_CONFIG = dict(optimizer="AdamW", lr_mlp=1e-3, lr_kan=5e-4, weight_decay=1e-4, batch_size=256,
                        epochs_mlp=60, epochs_kan=80, scheduler="OneCycleLR", grad_clip=1.0,
                        checkpoint_criterion="best_val_accuracy", seed_fix="set_seed() called before model construction")
SEEDS = [42, 123, 2026]
CKPT_DIR = OUT / "checkpoints_seed_corrected"

def make_loader(X, y, batch_size=256, shuffle=True):
    return DataLoader(ToolRoutingDataset(X, y), batch_size=batch_size, shuffle=shuffle)

regimes = {
    "qwen":  (X_tr_Q, y_tr_Q, X_va_Q, y_va_Q),
    "llama": (X_tr_L, y_tr_L, X_va_L, y_va_L),
    "mixed": (X_tr_Mix, y_tr_Mix, X_va_Mix, y_va_Mix),
}

all_models, train_log_rows = {}, []
for regime, (X_tr, y_tr, X_va, y_va) in regimes.items():
    tr_loader = make_loader(X_tr, y_tr)
    va_loader = make_loader(X_va, y_va, shuffle=False)
    for arch in ["kan", "mlp"]:
        for seed in SEEDS:
            name = f"{arch}_{regime}_shared28_seed{seed}_corrected"
            set_seed(seed)  # <-- THE FIX: before construction, not after
            model = (KANRoute if arch == "kan" else MLPBaseline)(input_dim=896, num_tools=28).to(DEVICE)
            best_val = train_model(model, tr_loader, va_loader, name, is_kan=(arch == "kan"), save_dir=CKPT_DIR)
            n_params = model.count_parameters()
            all_models[(regime, arch, seed)] = CKPT_DIR / f"{name}_best.pt"
            train_log_rows.append(dict(regime=regime, arch=arch, seed=seed, best_val_acc=best_val, n_params=n_params))
            print(f"  ✅ {name}: best_val_acc={best_val:.4f}")

train_log = pd.DataFrame(train_log_rows)
train_log.to_csv(OUT / "part4_training_log_seed_corrected.csv", index=False)
print("\n=== Seed-corrected training summary ===")
print(train_log.pivot_table(index="regime", columns=["arch","seed"], values="best_val_acc"))

CANON_NAMES = {("qwen","kan"):"kan_qwen_shared28_best.pt", ("qwen","mlp"):"mlp_qwen_shared28_best.pt",
               ("llama","kan"):"kan_llama_shared28_best.pt", ("llama","mlp"):"mlp_llama_shared28_best.pt",
               ("mixed","kan"):"kan_mixed_shared28_best.pt", ("mixed","mlp"):"mlp_mixed_shared28_best.pt"}
for (regime, arch), fname in CANON_NAMES.items():
    shutil.copy(all_models[(regime, arch, 42)], OUT / fname)
print("✅ Canonical (seed=42) checkpoints saved")


##############################################################################
# Section 4 + 21 — Corrected evaluation matrix with full prediction schema
##############################################################################
eval_loaders = {"qwen_test": make_loader(X_te_Q, y_te_Q, shuffle=False),
                "llama_test": make_loader(X_te_L, y_te_L, shuffle=False),
                "human": make_loader(X_human_28, y_human_28_canonical, shuffle=False)}

def top_k_acc(probs, labels, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([labels[i] in topk[i] for i in range(len(labels))])

def compute_metrics(preds, labels, probs, num_classes=28):
    return dict(top1=np.mean(preds == labels), top3=top_k_acc(probs, labels, 3),
                macro_f1=f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0),
                weighted_f1=f1_score(labels, preds, average="weighted", labels=list(range(num_classes)), zero_division=0))

def ambiguity_aware_score(pred_top1, pred_top3_names, acceptable_labels_str):
    pred_name = local_id_to_class[pred_top1]
    acceptable = set(x.strip() for x in str(acceptable_labels_str).split(",") if x.strip())
    return (pred_name in acceptable), bool(set(pred_top3_names) & acceptable)

PRED_DIR = OUT / "part4_predictions_seed_corrected"
results_rows, per_class_rows = [], []

for (regime, arch, seed), ckpt_path in all_models.items():
    model = (KANRoute if arch == "kan" else MLPBaseline)(input_dim=896, num_tools=28).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path)); model.eval()

    for eval_name, loader in eval_loaders.items():
        preds, labels, probs = evaluate_full_probs(model, loader)
        top3_preds = np.argsort(-probs, axis=1)[:, :3]
        probs_str = [";".join(f"{p:.6f}" for p in row) for row in probs]

        if eval_name == "qwen_test": src_df = qwen_shared28_test
        elif eval_name == "llama_test": src_df = llama_shared28_test
        else: src_df = human_shared28

        img_col = "img_id" if "img_id" in src_df.columns else "image_id"
        id_col = "query_id" if "query_id" in src_df.columns else None
        pred_df = pd.DataFrame({
            "sample_id": src_df[id_col].values if id_col else np.arange(len(src_df)),
            "img_id": src_df[img_col].values, "query": src_df["query"].values,
            "ground_truth_tool": (src_df["tool"] if "tool" in src_df.columns else src_df["tool_name"]).values,
            "ground_truth": labels, "pred_top1": preds, "pred_top1_tool": [local_id_to_class[p] for p in preds],
            "pred_top3": [list(map(int, t)) for t in top3_preds], "probs": probs_str, "correct": (preds == labels),
            "training_source": regime, "evaluation_source": eval_name, "architecture": arch, "seed": seed,
        })
        pred_df.to_csv(PRED_DIR / f"{arch}_{regime}_seed{seed}_on_{eval_name}.csv", index=False)

        p, r_, f, s = precision_recall_fscore_support(labels, preds, labels=list(range(28)), zero_division=0)
        for cls_id in range(28):
            per_class_rows.append(dict(training_source=regime, architecture=arch, seed=seed, evaluation_source=eval_name,
                class_id=cls_id, class_name=local_id_to_class[cls_id], precision=p[cls_id], recall=r_[cls_id],
                f1=f[cls_id], support=int(s[cls_id])))

        if eval_name != "human":
            m = compute_metrics(preds, labels, probs, 28)
            results_rows.append(dict(training_source=regime, architecture=arch, seed=seed, evaluation_source=eval_name, **m))
        else:
            m_canon = compute_metrics(preds, labels, probs, 28)
            sm = human_strict_mask.values
            m_strict = compute_metrics(preds[sm], labels[sm], probs[sm], 28) if sm.sum() else dict(top1=np.nan, top3=np.nan, macro_f1=np.nan, weighted_f1=np.nan)
            am = human_ambig_mask.values
            if am.sum():
                acc_labels = human_shared28.loc[am, "acceptable_tool_labels"].values
                t1h, t3h = [], []
                for i, row_i in enumerate(np.where(am)[0]):
                    a, b = ambiguity_aware_score(preds[row_i], [local_id_to_class[c] for c in top3_preds[row_i]], acc_labels[i])
                    t1h.append(a); t3h.append(b)
                ambig_top1, ambig_top3 = float(np.mean(t1h)), float(np.mean(t3h))
            else:
                ambig_top1 = ambig_top3 = np.nan
            results_rows.append(dict(training_source=regime, architecture=arch, seed=seed, evaluation_source="human",
                top1=m_canon["top1"], top3=m_canon["top3"], macro_f1=m_canon["macro_f1"], weighted_f1=m_canon["weighted_f1"],
                strict_top1=m_strict["top1"], strict_top3=m_strict["top3"], strict_macro_f1=m_strict["macro_f1"],
                strict_weighted_f1=m_strict["weighted_f1"], ambiguity_aware_top1=ambig_top1, ambiguity_aware_top3=ambig_top3))

part4_all_results_corrected = pd.DataFrame(results_rows)
part4_all_results_corrected.to_csv(OUT / "part4_all_results_seed_corrected.csv", index=False)
part4_per_class_corrected = pd.DataFrame(per_class_rows)
part4_per_class_corrected.to_csv(OUT / "part4_per_class_metrics_seed_corrected.csv", index=False)
print(f"\n✅ part4_all_results_seed_corrected.csv: {part4_all_results_corrected.shape}")
print(part4_all_results_corrected.groupby(["training_source","architecture","evaluation_source"])[["top1","macro_f1"]].agg(["mean","std"]))
print("\n✅ Stage A complete. Next: Section 5 (three-seed mean±SD summary), then Sections 6-7 (image-disjoint eval).")


##############################################################################
# Section 5 — part4_three_seed_summary.csv (mean ± SD, per metric, per condition)
##############################################################################
metric_cols_synthetic = ["top1", "top3", "macro_f1", "weighted_f1"]
metric_cols_human_extra = ["strict_top1", "strict_top3", "strict_macro_f1", "strict_weighted_f1",
                            "ambiguity_aware_top1", "ambiguity_aware_top3"]

summary_rows = []
for (ts, arch, ev), g in part4_all_results_corrected.groupby(["training_source", "architecture", "evaluation_source"]):
    g = g.set_index("seed")
    cols = metric_cols_synthetic + (metric_cols_human_extra if ev == "human" else [])
    for metric in cols:
        if metric not in g.columns:
            continue
        vals = g[metric]
        summary_rows.append(dict(
            training_source=ts, architecture=arch, evaluation_source=ev, metric=metric,
            mean=vals.mean(), std=vals.std(ddof=1) if len(vals) > 1 else 0.0,
            seed42=vals.get(42, np.nan), seed123=vals.get(123, np.nan), seed2026=vals.get(2026, np.nan),
        ))
part4_three_seed_summary = pd.DataFrame(summary_rows)
part4_three_seed_summary.to_csv(OUT / "part4_three_seed_summary.csv", index=False)
print(f"✅ part4_three_seed_summary.csv: {part4_three_seed_summary.shape}")

# Human-readable "mean ± std" spot-check for a few headline numbers
def fmt(ts, arch, ev, metric):
    r = part4_three_seed_summary[(part4_three_seed_summary.training_source == ts) &
        (part4_three_seed_summary.architecture == arch) & (part4_three_seed_summary.evaluation_source == ev) &
        (part4_three_seed_summary.metric == metric)]
    if len(r) == 0: return "n/a"
    return f"{r['mean'].values[0]*100:.2f} ± {r['std'].values[0]*100:.2f}"

print(f"Qwen-kan -> qwen_test top1: {fmt('qwen','kan','qwen_test','top1')}")
print(f"Llama-kan -> llama_test top1: {fmt('llama','kan','llama_test','top1')}")
print(f"Mixed-kan -> human top1: {fmt('mixed','kan','human','top1')}")


##############################################################################
# Section 6 — Image-disjoint cross-teacher evaluation
##############################################################################
qwen_train_imgtool = set(zip(qwen_shared28_train["img_id"], qwen_shared28_train["tool"]))
llama_train_imgtool = set(zip(llama_shared28_train["img_id"], llama_shared28_train["tool"]))

llama_test_imgtool = list(zip(llama_shared28_test["img_id"], llama_shared28_test["tool"]))
llama_disjoint_mask = np.array([it not in qwen_train_imgtool for it in llama_test_imgtool])
qwen_test_imgtool = list(zip(qwen_shared28_test["img_id"], qwen_shared28_test["tool"]))
qwen_disjoint_mask = np.array([it not in llama_train_imgtool for it in qwen_test_imgtool])

print(f"Llama test image-disjoint from Qwen-train: {llama_disjoint_mask.sum()} / {len(llama_disjoint_mask)} "
      f"({100*llama_disjoint_mask.sum()/len(llama_disjoint_mask):.1f}%)")
print(f"Qwen test image-disjoint from Llama-train: {qwen_disjoint_mask.sum()} / {len(qwen_disjoint_mask)} "
      f"({100*qwen_disjoint_mask.sum()/len(qwen_disjoint_mask):.1f}%)")

def top3_hit(row):
    top3 = eval(row["pred_top3"]) if isinstance(row["pred_top3"], str) else row["pred_top3"]
    return row["ground_truth"] in top3

def compute_metrics_from_df(df, num_classes=28):
    preds, labels = df["pred_top1"].values, df["ground_truth"].values
    top3 = df.apply(top3_hit, axis=1).mean()
    return dict(n=len(df), top1=np.mean(preds == labels), top3=top3,
                macro_f1=f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0),
                weighted_f1=f1_score(labels, preds, average="weighted", labels=list(range(num_classes)), zero_division=0))

image_disjoint_rows = []
disjoint_targets = [
    ("qwen", "llama_test", llama_disjoint_mask), ("mixed", "llama_test", llama_disjoint_mask),
    ("llama", "qwen_test", qwen_disjoint_mask), ("mixed", "qwen_test", qwen_disjoint_mask),
]
for training_source, eval_source, mask in disjoint_targets:
    for arch in ["kan", "mlp"]:
        seed_metrics = []
        for seed in SEEDS:
            df = pd.read_csv(PRED_DIR / f"{arch}_{training_source}_seed{seed}_on_{eval_source}.csv")
            assert len(df) == len(mask), f"length mismatch: {len(df)} vs {len(mask)}"
            df_disjoint = df[mask].reset_index(drop=True)
            m = compute_metrics_from_df(df_disjoint)
            seed_metrics.append(m)
        sm = pd.DataFrame(seed_metrics)
        image_disjoint_rows.append(dict(
            training_source=training_source, architecture=arch, evaluation_source=eval_source,
            n=int(sm["n"].iloc[0]),
            top1_mean=sm.top1.mean(), top1_std=sm.top1.std(ddof=1),
            top3_mean=sm.top3.mean(), top3_std=sm.top3.std(ddof=1),
            macro_f1_mean=sm.macro_f1.mean(), macro_f1_std=sm.macro_f1.std(ddof=1),
            weighted_f1_mean=sm.weighted_f1.mean(), weighted_f1_std=sm.weighted_f1.std(ddof=1),
        ))

image_disjoint_cross_teacher_results = pd.DataFrame(image_disjoint_rows)
image_disjoint_cross_teacher_results.to_csv(OUT / "image_disjoint_cross_teacher_results.csv", index=False)
print(f"\n✅ image_disjoint_cross_teacher_results.csv:")
print(image_disjoint_cross_teacher_results[["training_source","architecture","evaluation_source","n","top1_mean","top1_std","macro_f1_mean","macro_f1_std"]])

TABLES_DIR = OUT / "tables"
tex_rows = image_disjoint_cross_teacher_results.copy()
tex_rows["Top-1"] = tex_rows.apply(lambda r: f"{r.top1_mean*100:.2f} $\\pm$ {r.top1_std*100:.2f}", axis=1)
tex_rows["Top-3"] = tex_rows.apply(lambda r: f"{r.top3_mean*100:.2f} $\\pm$ {r.top3_std*100:.2f}", axis=1)
tex_rows["Macro F1"] = tex_rows.apply(lambda r: f"{r.macro_f1_mean*100:.2f} $\\pm$ {r.macro_f1_std*100:.2f}", axis=1)
tex_out = tex_rows[["training_source","architecture","evaluation_source","n","Top-1","Top-3","Macro F1"]].rename(
    columns={"training_source":"Training","architecture":"Model","evaluation_source":"Target","n":"N"})
tex_out["Model"] = tex_out["Model"].str.upper()
with open(TABLES_DIR / "table_image_disjoint_cross_teacher.tex", "w") as f:
    f.write(tex_out.to_latex(index=False, caption="Image-Disjoint Cross-Teacher Robustness", label="tab:image_disjoint", escape=True))
print("✅ table_image_disjoint_cross_teacher.tex saved")


##############################################################################
# Section 7 — Explicit image/tool context overlap quantification
##############################################################################
with open(OUT / "cross_teacher_context_overlap_report.md", "w") as f:
    f.write("# Cross-Teacher Context Overlap Report\n\n")
    f.write("## Qwen train → Llama test\n\n")
    f.write(f"- Total Llama test rows: {len(llama_disjoint_mask)}\n")
    f.write(f"- Rows WITH (image_id, tool) overlap with Qwen training: {(~llama_disjoint_mask).sum()}\n")
    f.write(f"- Rows WITHOUT overlap (image-disjoint): {llama_disjoint_mask.sum()}\n")
    f.write(f"- Percentage overlap: {100*(~llama_disjoint_mask).sum()/len(llama_disjoint_mask):.2f}%\n\n")
    f.write("## Llama train → Qwen test\n\n")
    f.write(f"- Total Qwen test rows: {len(qwen_disjoint_mask)}\n")
    f.write(f"- Rows WITH (image_id, tool) overlap with Llama training: {(~qwen_disjoint_mask).sum()}\n")
    f.write(f"- Rows WITHOUT overlap (image-disjoint): {qwen_disjoint_mask.sum()}\n")
    f.write(f"- Percentage overlap: {100*(~qwen_disjoint_mask).sum()/len(qwen_disjoint_mask):.2f}%\n\n")
    f.write("## Terminology (Section 10 requirement)\n\n")
    f.write("The full cross-teacher test set (Section 4's main evaluation matrix) reflects "
            "**linguistic cross-teacher shift**: the target teacher's queries, phrasing, and "
            "labeling conventions differ from the training teacher's, even when the same COCO "
            "image/tool pair happens to appear in both. It does NOT test visual/image "
            "generalization when that overlap exists.\n\n")
    f.write("The image-disjoint subset above (Section 6) is the only slice that tests **fully "
            "image-disjoint cross-teacher generalization** — the target teacher's test image, "
            "under that tool, was never seen (under any teacher's phrasing) during training.\n\n")
    f.write("**Do not describe the complete teacher-transfer test (Section 4) as image-disjoint.** "
            "Only the Section 6 subset earns that description.\n")
print("✅ cross_teacher_context_overlap_report.md written")
print("\n✅ Sections 5-7 complete. Next: Section 8-9 (human known/new image split + evaluation).")


##############################################################################
# Section 8 — Recompute human known/new image membership from image IDs +
# training manifests (NOT from any pre-existing known/new label column)
##############################################################################
# Definition used here: "training corpus" = the actual TRAIN split each model
# was fit on (qwen_shared28_train + llama_shared28_train), not train+val+test.
# Rationale: the question this subset answers is "has the trained model's
# weights ever been updated on this image", which is specifically the train
# split, not val/test images from the same synthetic pipeline. Documented
# here explicitly so this assumption is visible, not silently baked in.
import matplotlib.pyplot as plt
import numpy as np
synthetic_train_image_ids = set(qwen_shared28_train["img_id"]) | set(llama_shared28_train["img_id"])

human_shared28 = human_shared28.reset_index(drop=True)  # ensure clean positional index
known_mask = human_shared28["image_id"].isin(synthetic_train_image_ids).values
new_mask = ~known_mask

human_known = human_shared28[known_mask].reset_index(drop=True)
human_new = human_shared28[new_mask].reset_index(drop=True)
human_known.to_csv(OUT / "human_shared28_known_images.csv", index=False)
human_new.to_csv(OUT / "human_shared28_new_images.csv", index=False)

print(f"Known-image human: {len(human_known)} queries, {human_known['image_id'].nunique()} unique images, "
      f"{human_known['tool_name'].nunique()} classes represented")
print(f"New-image human:   {len(human_new)} queries, {human_new['image_id'].nunique()} unique images, "
      f"{human_new['tool_name'].nunique()} classes represented")
assert known_mask.sum() + new_mask.sum() == len(human_shared28)

# Sanity check per Section 22: every new-image ID must genuinely be absent
# from BOTH training corpora
overlap_check = set(human_new["image_id"]) & synthetic_train_image_ids
assert len(overlap_check) == 0, f"STOP: {len(overlap_check)} 'new' images actually found in training data"
print("✅ Verified: zero new-image human IDs overlap with synthetic training corpora")


##############################################################################
# Section 9 — Evaluate all 6 conditions on known-image and new-image subsets,
# aggregated across 3 seeds. Filters already-computed human predictions —
# no new inference.
##############################################################################
def compute_metrics_arr(preds, labels, top3_hits, num_classes=28):
    return dict(top1=np.mean(preds == labels), top3=np.mean(top3_hits),
                macro_f1=f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0),
                weighted_f1=f1_score(labels, preds, average="weighted", labels=list(range(num_classes)), zero_division=0))

def ambiguity_aware_from_df(df_subset, acc_labels_subset):
    t1h, t3h = [], []
    for i in range(len(df_subset)):
        pred_top1_tool = df_subset.iloc[i]["pred_top1_tool"]
        top3 = eval(df_subset.iloc[i]["pred_top3"]) if isinstance(df_subset.iloc[i]["pred_top3"], str) else df_subset.iloc[i]["pred_top3"]
        top3_tools = [local_id_to_class[c] for c in top3]
        acceptable = set(x.strip() for x in str(acc_labels_subset.iloc[i]).split(",") if x.strip())
        t1h.append(pred_top1_tool in acceptable)
        t3h.append(bool(set(top3_tools) & acceptable))
    return float(np.mean(t1h)) if t1h else np.nan, float(np.mean(t3h)) if t3h else np.nan

known_new_rows = []
for regime in ["qwen", "llama", "mixed"]:
    for arch in ["kan", "mlp"]:
        for subset_name, mask in [("known", known_mask), ("new", new_mask)]:
            per_seed = []
            for seed in SEEDS:
                df = pd.read_csv(PRED_DIR / f"{arch}_{regime}_seed{seed}_on_human.csv")
                assert len(df) == len(mask)
                df_sub = df[mask].reset_index(drop=True)
                strict_sub = (human_strict_mask.values & mask)[mask]  # strict mask restricted to this subset's rows
                ambig_sub = (human_ambig_mask.values & mask)[mask]

                top3_hits = df_sub.apply(lambda r: r["ground_truth"] in (eval(r["pred_top3"]) if isinstance(r["pred_top3"], str) else r["pred_top3"]), axis=1)
                m_canon = compute_metrics_arr(df_sub["pred_top1"].values, df_sub["ground_truth"].values, top3_hits.values)

                if strict_sub.sum() > 0:
                    ds = df_sub[strict_sub]
                    t3 = ds.apply(lambda r: r["ground_truth"] in (eval(r["pred_top3"]) if isinstance(r["pred_top3"], str) else r["pred_top3"]), axis=1)
                    m_strict = compute_metrics_arr(ds["pred_top1"].values, ds["ground_truth"].values, t3.values)
                else:
                    m_strict = dict(top1=np.nan, top3=np.nan, macro_f1=np.nan, weighted_f1=np.nan)

                if ambig_sub.sum() > 0:
                    da = df_sub[ambig_sub]
                    acc_labels_sub = human_shared28.loc[mask, "acceptable_tool_labels"].reset_index(drop=True)[ambig_sub]
                    ambig_top1, ambig_top3 = ambiguity_aware_from_df(da.reset_index(drop=True), acc_labels_sub.reset_index(drop=True))
                else:
                    ambig_top1 = ambig_top3 = np.nan

                per_seed.append(dict(canonical_top1=m_canon["top1"], macro_f1=m_canon["macro_f1"], weighted_f1=m_canon["weighted_f1"],
                                      strict_top1=m_strict["top1"], strict_top3=m_strict["top3"],
                                      ambiguity_aware_top1=ambig_top1, ambiguity_aware_top3=ambig_top3))
            ps = pd.DataFrame(per_seed)
            known_new_rows.append(dict(
                training_source=regime, architecture=arch, subset=subset_name, n=int(mask.sum()),
                canonical_top1_mean=ps.canonical_top1.mean(), canonical_top1_std=ps.canonical_top1.std(ddof=1),
                strict_top1_mean=ps.strict_top1.mean(), strict_top1_std=ps.strict_top1.std(ddof=1),
                strict_top3_mean=ps.strict_top3.mean(), strict_top3_std=ps.strict_top3.std(ddof=1),
                macro_f1_mean=ps.macro_f1.mean(), macro_f1_std=ps.macro_f1.std(ddof=1),
                weighted_f1_mean=ps.weighted_f1.mean(), weighted_f1_std=ps.weighted_f1.std(ddof=1),
                ambiguity_aware_top1_mean=ps.ambiguity_aware_top1.mean(), ambiguity_aware_top1_std=ps.ambiguity_aware_top1.std(ddof=1),
                ambiguity_aware_top3_mean=ps.ambiguity_aware_top3.mean(), ambiguity_aware_top3_std=ps.ambiguity_aware_top3.std(ddof=1),
            ))

human_known_vs_new_results = pd.DataFrame(known_new_rows)
human_known_vs_new_results.to_csv(OUT / "human_known_vs_new_results.csv", index=False)
print(f"\n✅ human_known_vs_new_results.csv: {human_known_vs_new_results.shape}")
print(human_known_vs_new_results[["training_source","architecture","subset","n","canonical_top1_mean","canonical_top1_std","macro_f1_mean"]])

# LaTeX Table 5
t5 = human_known_vs_new_results.pivot_table(index=["training_source","architecture"], columns="subset",
    values=["canonical_top1_mean","macro_f1_mean"]).reset_index()
t5.columns = ["Training","Model","Known_Top1","New_Top1","Known_F1","New_F1"]
t5["Model"] = t5["Model"].str.upper()
with open(TABLES_DIR / "table_human_known_vs_new.tex", "w") as f:
    f.write(t5.to_latex(index=False, float_format="%.4f", caption="Known vs New Human Images", label="tab:known_new", escape=True))

# Figure
FIG_DIR = OUT / "figures"
fig, ax = plt.subplots(figsize=(9, 5))
conditions = human_known_vs_new_results[["training_source","architecture"]].drop_duplicates().values.tolist()
x = np.arange(len(conditions)); width = 0.35
known_vals, new_vals, known_err, new_err = [], [], [], []
for ts, arch in conditions:
    k = human_known_vs_new_results[(human_known_vs_new_results.training_source==ts)&(human_known_vs_new_results.architecture==arch)&(human_known_vs_new_results.subset=="known")]
    n = human_known_vs_new_results[(human_known_vs_new_results.training_source==ts)&(human_known_vs_new_results.architecture==arch)&(human_known_vs_new_results.subset=="new")]
    known_vals.append(k.canonical_top1_mean.values[0]); known_err.append(k.canonical_top1_std.values[0])
    new_vals.append(n.canonical_top1_mean.values[0]); new_err.append(n.canonical_top1_std.values[0])
ax.bar(x - width/2, known_vals, width, yerr=known_err, label="Known images", capsize=3)
ax.bar(x + width/2, new_vals, width, yerr=new_err, label="New images", capsize=3)
ax.set_xticks(x); ax.set_xticklabels([f"{ts}-{a.upper()}" for ts, a in conditions], rotation=45, ha="right")
ax.set_ylabel("Canonical Top-1"); ax.set_ylim(0, 1.0)
ax.set_title("Human Evaluation: Known vs New Images (mean ± SD across 3 seeds)")
ax.legend(); fig.tight_layout()
fig.savefig(FIG_DIR / "human_known_vs_new.png", dpi=150); plt.close(fig)
print("✅ table_human_known_vs_new.tex + human_known_vs_new.png saved")
print("\n✅ Sections 8-9 complete. Next: Sections 10-14 (terminology, per-class 3-seed, "
      "low-support sensitivity, provenance fix, uncertainty doc).")


##############################################################################
# Diagnostic + fix — known/new macro-F1 comparability
##############################################################################
known_classes = set(human_known["tool_name"].unique())
new_classes = set(human_new["tool_name"].unique())
missing_from_known = set(shared_28) - known_classes
missing_from_new = set(shared_28) - new_classes
print(f"Classes missing from KNOWN subset ({len(missing_from_known)}): {missing_from_known}")
print(f"Classes missing from NEW subset ({len(missing_from_new)}): {missing_from_new}")

# Fix: recompute macro-F1 for BOTH subsets restricted to the classes present
# in BOTH (the only fair comparison), and keep the original all-28-class
# version too, clearly labeled, rather than silently replacing it.
shared_present_classes = known_classes & new_classes
shared_present_ids = [class_to_local_id[c] for c in shared_present_classes]
print(f"\nClasses present in both subsets (fair comparison basis): {len(shared_present_ids)}/28")

def macro_f1_restricted(preds, labels, class_ids):
    return f1_score(labels, preds, average="macro", labels=class_ids, zero_division=0)

fair_rows = []
for regime in ["qwen", "llama", "mixed"]:
    for arch in ["kan", "mlp"]:
        for subset_name, mask in [("known", known_mask), ("new", new_mask)]:
            per_seed_f1 = []
            for seed in SEEDS:
                df = pd.read_csv(PRED_DIR / f"{arch}_{regime}_seed{seed}_on_human.csv")
                df_sub = df[mask].reset_index(drop=True)
                per_seed_f1.append(macro_f1_restricted(df_sub["pred_top1"].values, df_sub["ground_truth"].values, shared_present_ids))
            fair_rows.append(dict(training_source=regime, architecture=arch, subset=subset_name,
                                   macro_f1_fair_mean=np.mean(per_seed_f1), macro_f1_fair_std=np.std(per_seed_f1, ddof=1)))
fair_df = pd.DataFrame(fair_rows)
fair_pivot = fair_df.pivot_table(index=["training_source","architecture"], columns="subset",
                                   values=["macro_f1_fair_mean","macro_f1_fair_std"]).reset_index()
print("\n=== Macro-F1 restricted to classes present in BOTH subsets (fair comparison) ===")
print(fair_pivot)

# Merge the fair comparison into the results file, keeping the original
# all-28-class version alongside it (labeled), not overwriting silently
human_known_vs_new_results = human_known_vs_new_results.merge(
    fair_df.rename(columns={"macro_f1_fair_mean": "macro_f1_fair_28class_intersection_mean",
                             "macro_f1_fair_std": "macro_f1_fair_28class_intersection_std"}),
    on=["training_source", "architecture", "subset"], how="left")
human_known_vs_new_results.to_csv(OUT / "human_known_vs_new_results.csv", index=False)

with open(OUT / "human_known_vs_new_results.csv", "a") as f:
    pass  # already written above; this note documents the fix in the file itself
with open(OUT / "cross_teacher_context_overlap_report.md", "a") as f:
    f.write(f"\n## Known/New Human Macro-F1 Comparability Note\n\n"
        f"The known-image human subset covers only {len(known_classes)}/28 classes "
        f"(missing: {sorted(missing_from_known)}); the new-image subset covers "
        f"{len(new_classes)}/28 (missing: {sorted(missing_from_new) if missing_from_new else 'none'}). "
        f"Macro-F1 computed with all 28 class labels forces 0.0 F1 for classes absent from a "
        f"subset, making the raw known-vs-new macro-F1 comparison in "
        f"`human_known_vs_new_results.csv` (`macro_f1_mean` column) misleading on its own. "
        f"A fairer comparison restricted to the {len(shared_present_ids)} classes present in "
        f"BOTH subsets is in the same file as `macro_f1_fair_28class_intersection_mean`. "
        f"Use the fair-comparison column for any known-vs-new macro-F1 claim in the paper.\n")
print("\n✅ Fair comparison column added to human_known_vs_new_results.csv, note appended to overlap report")


##############################################################################
# Confirm: are vehicle_detection / 3d_reconstruction still >=10 queries total?
# (They should be — the known/new split doesn't remove any rows, it just
# labels each row by whether ITS image happens to be training-reused.)
##############################################################################
for cls in ["vehicle_detection", "3d_reconstruction"]:
    total = (human_shared28["tool_name"] == cls).sum()
    in_known = (human_known["tool_name"] == cls).sum()
    in_new = (human_new["tool_name"] == cls).sum()
    unique_imgs = human_shared28[human_shared28["tool_name"] == cls]["image_id"].nunique()
    print(f"{cls}: total={total} (known={in_known}, new={in_new}), unique images used={unique_imgs}")

print(f"\nTotal human_shared28 rows: {len(human_shared28)} (known={len(human_known)} + new={len(human_new)} "
      f"= {len(human_known)+len(human_new)})")


##############################################################################
# Investigate: why do vehicle_detection/3d_reconstruction show 0 "known" rows
# under my train-only definition, when the benchmark was built to include
# known-pool queries for every class?
##############################################################################
print("Original image_pool label distribution:")
print(human_shared28["image_pool"].value_counts())

# Cross-tab: original label vs my (narrow) train-only recomputation
comparison = pd.DataFrame({"original_image_pool": human_shared28["image_pool"], "recomputed_known_TRAIN_ONLY": known_mask})
print("\nOriginal label vs train-only recomputation:")
print(pd.crosstab(comparison["original_image_pool"], comparison["recomputed_known_TRAIN_ONLY"]))

# Candidate B: broaden to the full Shared-28 corpus (train+val+test), not just train
shared28_all_image_ids = (set(qwen_shared28_train["img_id"]) | set(qwen_shared28_val["img_id"]) | set(qwen_shared28_test["img_id"]) |
                           set(llama_shared28_train["img_id"]) | set(llama_shared28_val["img_id"]) | set(llama_shared28_test["img_id"]))
known_mask_B = human_shared28["image_id"].isin(shared28_all_image_ids).values

# Candidate C: broadest — was this image_id EVER in either teacher's full COCO
# pool (any tool, not just Shared-28), using the clip caches as the proxy
# (clip_features.npz covers every image used anywhere in that teacher's corpus)
known_mask_C = human_shared28["image_id"].isin(set(qwen_clip.keys()) | set(llama_clip.keys())).values

for name, mask in [("A: train-only (current)", known_mask), ("B: full Shared-28 corpus (train+val+test)", known_mask_B),
                    ("C: broadest (any tool, either teacher's full COCO pool)", known_mask_C)]:
    agree = (mask == (human_shared28["image_pool"] == "known").values).mean()
    print(f"\nDefinition {name}: {mask.sum()} known / {len(mask)} total. "
          f"Agreement with original image_pool label: {agree*100:.1f}%")

print("\n--- vehicle_detection / 3d_reconstruction under each definition ---")
for cls in ["vehicle_detection", "3d_reconstruction"]:
    sub_idx = human_shared28["tool_name"] == cls
    orig_known = (human_shared28.loc[sub_idx, "image_pool"] == "known").sum()
    a = known_mask[sub_idx.values].sum()
    b = known_mask_B[sub_idx.values].sum()
    c = known_mask_C[sub_idx.values].sum()
    print(f"{cls}: original_label_known={orig_known}, def_A={a}, def_B={b}, def_C={c}")


import shutil
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
ZIP_PATH = "/kaggle/working/part5"

shutil.make_archive(
    ZIP_PATH,
    "zip",
    root_dir=WORK_DIR
)

print(f"Created: {ZIP_PATH}.zip")
