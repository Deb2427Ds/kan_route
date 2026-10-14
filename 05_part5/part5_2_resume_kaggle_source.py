##############################################################################
# RESUME — new notebook, no retraining needed. Just reloads state from your
# two zips and finishes the known/new investigation (saved to a file this time).
##############################################################################
from pathlib import Path
import json, re, shutil
import numpy as np, torch, torch.nn as nn, pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 2026]

def find_file(fname):
    m = list(Path("/kaggle/input").rglob(fname))
    return m[0] if m else None

KAN_QWEN_CLIP = find_file("clip_features.npz")  # first match; see note below if ambiguous
# If both Qwen and Llama clip_features.npz share a filename, resolve explicitly:
_clip_matches = list(Path("/kaggle/input").rglob("clip_features.npz"))
print(f"Found {len(_clip_matches)} clip_features.npz files:")
for p in _clip_matches: print(" ", p)
# Expect exactly 2 — one under a kan_qwen-ish path, one under llama_kan-ish path
qwen_clip_path = next(p for p in _clip_matches if "qwen" in str(p).lower())
llama_clip_path = next(p for p in _clip_matches if "llama" in str(p).lower())
qc = np.load(qwen_clip_path, allow_pickle=True)
qwen_clip = {int(k): v for k, v in qc[qc.files[0]].item().items()}
lc = np.load(llama_clip_path, allow_pickle=True)
llama_clip = {int(k): v for k, v in lc[lc.files[0]].item().items()}
print(f"✅ Qwen clip cache: {len(qwen_clip)} images, Llama clip cache: {len(llama_clip)} images")

# Earlier full zip — has the split CSVs, human_shared28.csv, mapping
_split_match = find_file("qwen_shared28_train.csv")
assert _split_match, "Earlier full zip (with split CSVs) not attached."
EARLIER_BASE = _split_match.parent
print(f"✅ Earlier zip base: {EARLIER_BASE}")

qwen_shared28_train = pd.read_csv(EARLIER_BASE / "qwen_shared28_train.csv")
qwen_shared28_val   = pd.read_csv(EARLIER_BASE / "qwen_shared28_val.csv")
qwen_shared28_test  = pd.read_csv(EARLIER_BASE / "qwen_shared28_test.csv")
llama_shared28_train = pd.read_csv(EARLIER_BASE / "llama_shared28_train.csv")
llama_shared28_val   = pd.read_csv(EARLIER_BASE / "llama_shared28_val.csv")
llama_shared28_test  = pd.read_csv(EARLIER_BASE / "llama_shared28_test.csv")
shared_28_mapping = json.load(open(EARLIER_BASE / "shared_28_class_mapping.json"))
local_id_to_class = {int(k): v for k, v in shared_28_mapping["local_id_to_class"].items()}
class_to_local_id = shared_28_mapping["class_to_local_id"]
shared_28 = sorted(class_to_local_id.keys())

# part5.zip — has human_shared28 with the ORIGINAL image_pool label, plus every
# Stage A-C output, checkpoints, and predictions. Prefer this copy of
# human_shared28.csv if present there (it may have been carried through);
# otherwise fall back to the earlier zip's copy.
_part5_match = find_file("human_shared28_known_images.csv")
assert _part5_match, "part5.zip not attached."
PART5_BASE = _part5_match.parent
print(f"✅ part5.zip base: {PART5_BASE}")

_human_full = find_file("human_shared28.csv")
human_shared28 = pd.read_csv(_human_full).reset_index(drop=True)
print(f"✅ human_shared28: {len(human_shared28)} rows, image_pool values: {human_shared28['image_pool'].unique().tolist()}")

has_multi = human_shared28["acceptable_tool_labels"].astype(str).str.strip() != ""
strict_true = human_shared28["strict_eval"] == True
human_strict_mask = strict_true & ~has_multi
human_ambig_mask = strict_true & has_multi

# Reload the already-computed Stage A-C results (no retraining, no re-inference)
part4_all_results_corrected = pd.read_csv(PART5_BASE / "part4_all_results_seed_corrected.csv")
part4_per_class_corrected = pd.read_csv(PART5_BASE / "part4_per_class_metrics_seed_corrected.csv")
part4_three_seed_summary = pd.read_csv(PART5_BASE / "part4_three_seed_summary.csv")
image_disjoint_cross_teacher_results = pd.read_csv(PART5_BASE / "image_disjoint_cross_teacher_results.csv")
human_known_vs_new_results = pd.read_csv(PART5_BASE / "human_known_vs_new_results.csv")
PRED_DIR = PART5_BASE / "part4_predictions_seed_corrected"
print(f"✅ Reloaded all Stage A-C results — {len(part4_all_results_corrected)} result rows, "
      f"predictions dir: {PRED_DIR} ({len(list(PRED_DIR.glob('*.csv')))} files)")

OUT = Path("/kaggle/working/part4_final")
for sub in ["", "tables", "figures"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
# Copy forward everything already computed, so this session's OUT is complete
# and re-packaging at the end produces one coherent zip, not a partial one
for f in PART5_BASE.rglob("*"):
    if f.is_file():
        rel = f.relative_to(PART5_BASE)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(f, dst)
print(f"✅ Copied forward {sum(1 for _ in OUT.rglob('*') if _.is_file())} files into this session's OUT")


##############################################################################
# Finish Section 8 — the known/new investigation, THIS TIME SAVED TO A FILE
##############################################################################
known_mask = human_shared28["image_id"].isin(
    set(qwen_shared28_train["img_id"]) | set(llama_shared28_train["img_id"])).values
new_mask = ~known_mask

shared28_all_image_ids = (set(qwen_shared28_train["img_id"]) | set(qwen_shared28_val["img_id"]) | set(qwen_shared28_test["img_id"]) |
                           set(llama_shared28_train["img_id"]) | set(llama_shared28_val["img_id"]) | set(llama_shared28_test["img_id"]))
known_mask_B = human_shared28["image_id"].isin(shared28_all_image_ids).values
known_mask_C = human_shared28["image_id"].isin(set(qwen_clip.keys()) | set(llama_clip.keys())).values

investigation_lines = ["# Known/New Definition Investigation\n"]
investigation_lines.append(f"Original image_pool label distribution:\n{human_shared28['image_pool'].value_counts().to_string()}\n")

for name, mask in [("A: train-only", known_mask), ("B: full Shared-28 corpus (train+val+test)", known_mask_B),
                    ("C: broadest (any tool, either teacher's full COCO pool)", known_mask_C)]:
    agree = (mask == (human_shared28["image_pool"] == "known").values).mean()
    line = f"Definition {name}: {mask.sum()} known / {len(mask)} total. Agreement with original label: {agree*100:.1f}%"
    print(line)
    investigation_lines.append(line + "\n")

investigation_lines.append("\n## vehicle_detection / 3d_reconstruction breakdown\n")
for cls in ["vehicle_detection", "3d_reconstruction"]:
    sub_idx = human_shared28["tool_name"] == cls
    orig_known = (human_shared28.loc[sub_idx, "image_pool"] == "known").sum()
    a, b, c = known_mask[sub_idx.values].sum(), known_mask_B[sub_idx.values].sum(), known_mask_C[sub_idx.values].sum()
    line = f"{cls}: original_label_known={orig_known}, def_A={a}, def_B={b}, def_C={c}"
    print(line)
    investigation_lines.append(line + "\n")

with open(OUT / "known_new_definition_investigation.md", "w") as f:
    f.write("\n".join(investigation_lines))
print("\n✅ known_new_definition_investigation.md saved — will not be lost again")


##############################################################################
# Section 10 — Correct Human-OOD Terminology (documentation, not computation)
##############################################################################
with open(OUT / "HUMAN_EVAL_TERMINOLOGY.md", "w") as f:
    f.write("""# Human Evaluation Terminology — Required Usage

Use these terms precisely in all reports/tables/figures/manuscript text:

- **"human-written evaluation"** — the full 372-row Shared-28 human benchmark.
  Do NOT call this "human OOD" — most of it (162/372, 43.5%) uses images the
  models were actually trained on; only the phrasing is human-authored, not
  the images.

- **"human new-image evaluation"** or **"human image-OOD evaluation"** —
  ONLY the 210-row subset (`human_shared28_new_images.csv`) whose image IDs
  never occur in either teacher's Shared-28 training split. This is the only
  slice that tests genuine visual/image OOD.

- **"linguistic OOD"** — applies to the FULL 372-row set relative to the
  synthetic training queries: human phrasing differs from teacher-generated
  phrasing even when the underlying image is one the model has seen.

- **"visual/image OOD"** — applies ONLY to the new-image subset.

## Why this distinction matters here specifically

A model can score well on "linguistic OOD" (novel phrasing, familiar image)
while scoring worse on "visual/image OOD" (novel image), or vice versa — they
are different generalization axes. Collapsing both into "human OOD" without
qualification overstates what the full-set number actually demonstrates.

Per Section 9's results, canonical Top-1 differs only modestly between known
and new images (1-4pp, mostly within 1 SD) — meaning most of the difficulty
in the full human-written evaluation comes from linguistic OOD, not visual
OOD. State this explicitly wherever the human results are discussed.
""")
print("✅ HUMAN_EVAL_TERMINOLOGY.md written")


##############################################################################
# Section 11 — Three-seed per-class aggregation (replaces single-seed file)
##############################################################################
agg_cols = part4_per_class_corrected.groupby(
    ["training_source", "architecture", "evaluation_source", "class_id", "class_name"]
).agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), precision_mean=("precision", "mean"),
      recall_mean=("recall", "mean"), support=("support", "first")).reset_index()
agg_cols["f1_std"] = agg_cols["f1_std"].fillna(0.0)  # std of 1 value (shouldn't happen with 3 seeds, but guard anyway)

shared28_per_class_transfer_3seed = agg_cols
shared28_per_class_transfer_3seed.to_csv(OUT / "shared28_per_class_transfer_3seed.csv", index=False)
print(f"✅ shared28_per_class_transfer_3seed.csv: {shared28_per_class_transfer_3seed.shape} "
      f"({shared28_per_class_transfer_3seed[['training_source','architecture','evaluation_source']].drop_duplicates().shape[0]} conditions × 28 classes)")

# Confirm every group actually had 3 seeds contributing (catches silent gaps)
seed_counts = part4_per_class_corrected.groupby(
    ["training_source", "architecture", "evaluation_source", "class_id"]).size()
assert (seed_counts == 3).all(), f"STOP: some (condition, class) combos don't have exactly 3 seeds: {seed_counts[seed_counts != 3]}"
print("✅ Verified: every (condition, class) combination has exactly 3 seeds contributing")


##############################################################################
# Section 12 — Low-support sensitivity analysis
##############################################################################
support_qwen = part4_per_class_corrected[(part4_per_class_corrected.evaluation_source == "qwen_test") &
    (part4_per_class_corrected.architecture == "kan") & (part4_per_class_corrected.seed == 42)][["class_name", "support"]].set_index("class_name")["support"]
support_llama = part4_per_class_corrected[(part4_per_class_corrected.evaluation_source == "llama_test") &
    (part4_per_class_corrected.architecture == "kan") & (part4_per_class_corrected.seed == 42)][["class_name", "support"]].set_index("class_name")["support"]

stable_support_classes = sorted([c for c in shared_28 if support_qwen.get(c, 0) >= 5 and support_llama.get(c, 0) >= 5])
print(f"Stable-support classes (test support >=5 in BOTH Qwen and Llama test): {len(stable_support_classes)}/28")
print(stable_support_classes)
excluded = sorted(set(shared_28) - set(stable_support_classes))
print(f"Excluded (low support in at least one teacher's test set): {excluded}")
for c in excluded:
    print(f"  {c}: qwen_test support={support_qwen.get(c,0)}, llama_test support={support_llama.get(c,0)}")

with open(OUT / "stable_support_classes.txt", "w") as f:
    f.write("\n".join(stable_support_classes))

stable_ids = [class_to_local_id[c] for c in stable_support_classes]

def top3_hit_row(row):
    top3 = eval(row["pred_top3"]) if isinstance(row["pred_top3"], str) else row["pred_top3"]
    return row["ground_truth"] in top3

support_qualified_rows = []
for regime in ["qwen", "llama", "mixed"]:
    for arch in ["kan", "mlp"]:
        for ev in ["qwen_test", "llama_test", "human"]:
            per_seed = []
            for seed in SEEDS:
                df = pd.read_csv(PRED_DIR / f"{arch}_{regime}_seed{seed}_on_{ev}.csv")
                df_sub = df[df["ground_truth"].isin(stable_ids)].reset_index(drop=True)
                if len(df_sub) == 0:
                    continue
                top1 = (df_sub["pred_top1"] == df_sub["ground_truth"]).mean()
                top3 = df_sub.apply(top3_hit_row, axis=1).mean()
                macro_f1 = f1_score(df_sub["ground_truth"], df_sub["pred_top1"], average="macro", labels=stable_ids, zero_division=0)
                per_seed.append(dict(top1=top1, top3=top3, macro_f1=macro_f1, n=len(df_sub)))
            if not per_seed:
                continue
            ps = pd.DataFrame(per_seed)
            support_qualified_rows.append(dict(
                training_source=regime, architecture=arch, evaluation_source=ev, n=int(ps.n.iloc[0]),
                top1_mean=ps.top1.mean(), top1_std=ps.top1.std(ddof=1),
                top3_mean=ps.top3.mean(), top3_std=ps.top3.std(ddof=1),
                macro_f1_mean=ps.macro_f1.mean(), macro_f1_std=ps.macro_f1.std(ddof=1),
            ))
support_qualified_results = pd.DataFrame(support_qualified_rows)
support_qualified_results.to_csv(OUT / "support_qualified_results.csv", index=False)
print(f"\n✅ support_qualified_results.csv: {support_qualified_results.shape}")
print(support_qualified_results[["training_source","architecture","evaluation_source","n","top1_mean","macro_f1_mean"]])

sq_tex = support_qualified_results.copy()
sq_tex["Top-1"] = sq_tex.apply(lambda r: f"{r.top1_mean*100:.2f} $\\pm$ {r.top1_std*100:.2f}", axis=1)
sq_tex["Macro F1"] = sq_tex.apply(lambda r: f"{r.macro_f1_mean*100:.2f} $\\pm$ {r.macro_f1_std*100:.2f}", axis=1)
sq_out = sq_tex[["training_source","architecture","evaluation_source","n","Top-1","Macro F1"]].rename(
    columns={"training_source":"Training","architecture":"Model","evaluation_source":"Eval"})
sq_out["Model"] = sq_out["Model"].str.upper()
with open(TABLES_DIR / "table_support_sensitivity.tex", "w") as f:
    f.write(sq_out.to_latex(index=False, caption=f"Stable-Support Sensitivity ({len(stable_support_classes)}/28 classes, "
                             "full Shared-28 remains primary)", label="tab:support_sensitivity", escape=True))
print("✅ table_support_sensitivity.tex saved")


##############################################################################
# Section 13 — Correct the Qwen corrective-augmentation provenance (audited,
# not asserted from memory)
##############################################################################
aug_rows = qwen_shared28_train[qwen_shared28_train["source"] == "manual_corrective_augmentation"]
aug_counts = aug_rows["tool"].value_counts()
all_shared28_raw_counts = qwen_shared28_train["tool"].value_counts()  # post-split counts, for context only

# Classes that were BELOW the 100-sample threshold pre-augmentation but got
# NO rows added (i.e., considered but not actually augmented)
below_100_originally = ["pose_estimation", "face_detection", "facial_expression_recognition", "logo_detection"]
actually_augmented = sorted(aug_counts.index.tolist())
not_augmented = sorted(set(below_100_originally) - set(actually_augmented))

with open(OUT / "qwen_corrective_augmentation_provenance_FINAL.md", "w") as f:
    f.write("# Qwen Corrective Augmentation — Provenance (FINAL, audited)\n\n")
    f.write("Audited directly from `qwen_shared28_train.csv`'s `source` column — not asserted "
            "from memory or from an earlier draft.\n\n")
    f.write("## Classes targeted (rows actually added)\n\n")
    f.write("| Class | Rows added |\n|---|---|\n")
    for cls in actually_augmented:
        f.write(f"| {cls} | {int(aug_counts[cls])} |\n")
    f.write(f"\n**Total rows added: {int(aug_counts.sum())}**\n\n")
    f.write("## Classes considered but NOT augmented\n\n")
    for cls in not_augmented:
        f.write(f"- **{cls}**: flagged as under the 100-raw-sample threshold, but no query "
                f"templates were authored for this class in the correction script (an oversight, "
                f"not a deliberate exclusion). Its original raw sample count nonetheless proved "
                f"sufficient to achieve full 28/28 train/val/test coverage after leakage-safe "
                f"splitting (verified in `shared28_split_leakage_audit_FINAL_V2.md`), so no "
                f"retroactive fix was applied.\n")
    f.write("\n**Correction from the earlier draft:** a previous version of this provenance file "
            "incorrectly implied `logo_detection` received the same targeted augmentation as the "
            "other three classes. It did not — zero rows were added for `logo_detection`. This "
            "file corrects that.\n\n")
    f.write("## Method (unchanged from original)\n\n")
    f.write("Per-image phrasing varied (opener + noun choice) so each image's rows form an "
            "independent connected component under leakage-safe grouping, rather than merging "
            "into one atomic block. Images reused from existing corpus images already labeled "
            "with a person-indicating tool. CLIP image embeddings pulled from the existing cache; "
            "only MiniLM text embeddings freshly computed.\n")
print(f"✅ qwen_corrective_augmentation_provenance_FINAL.md written")
print(f"   Actually augmented: {actually_augmented} (total {int(aug_counts.sum())} rows)")
print(f"   Considered but NOT augmented: {not_augmented}")


##############################################################################
# Section 14 — Fix bootstrap-uncertainty reporting
##############################################################################
def bootstrap_ci(preds, labels, metric_fn, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed); n = len(labels); stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n); stats.append(metric_fn(preds[idx], labels[idx]))
    stats = np.array(stats)
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def top1_fn(p, l): return np.mean(p == l)
def macrof1_fn(p, l): return f1_score(l, p, average="macro", labels=list(range(28)), zero_division=0)

# Within-test-set bootstrap CI, computed SEPARATELY for each seed, clearly
# labeled as such — NOT presented as uncertainty around the 3-seed mean.
within_seed_ci_rows = []
for arch in ["kan", "mlp"]:
    for ts in ["qwen", "llama", "mixed"]:
        for ev in ["qwen_test", "llama_test", "human"]:
            for seed in SEEDS:
                path = PRED_DIR / f"{arch}_{ts}_seed{seed}_on_{ev}.csv"
                if not path.exists(): continue
                df = pd.read_csv(path)
                preds, labels = df["pred_top1"].values, df["ground_truth"].values
                t1m, t1l, t1h = bootstrap_ci(preds, labels, top1_fn)
                fm, fl, fh = bootstrap_ci(preds, labels, macrof1_fn)
                within_seed_ci_rows.append(dict(architecture=arch, training_source=ts, evaluation_source=ev, seed=seed,
                    top1_estimate=t1m, top1_ci_low=t1l, top1_ci_high=t1h,
                    macro_f1_estimate=fm, macro_f1_ci_low=fl, macro_f1_ci_high=fh))
within_seed_bootstrap_ci = pd.DataFrame(within_seed_ci_rows)
within_seed_bootstrap_ci.to_csv(OUT / "within_seed_bootstrap_ci.csv", index=False)
print(f"✅ within_seed_bootstrap_ci.csv: {within_seed_bootstrap_ci.shape}")

with open(OUT / "UNCERTAINTY_REPORTING.md", "w") as f:
    f.write("""# Uncertainty Reporting — What Each Measure Represents

This project reports two genuinely different sources of uncertainty. Do not
conflate them, and do not present the seed-42-only bootstrap interval from an
earlier draft as if it captured seed-to-seed variance — it did not.

## Primary: mean ± SD across 3 seeds

File: `part4_three_seed_summary.csv`, `shared28_per_class_transfer_3seed.csv`,
`support_qualified_results.csv`, `image_disjoint_cross_teacher_results.csv`,
`human_known_vs_new_results.csv`.

This captures **training-run-to-training-run variance**: how much the result
changes due to random weight initialization and batch ordering, holding the
train/val/test split and test set fixed. This is the uncertainty measure to
lead with in the paper. All of it comes from the seed-corrected runs (the
seed bug identified in Section 2 is fixed in every number here).

## Secondary (optional): within-test-set bootstrap CI, per seed

File: `within_seed_bootstrap_ci.csv`.

For a SINGLE fixed trained model (one specific seed), this resamples the test
set itself (1000 resamples, with replacement) to ask: "how much would this
model's reported metric wobble if we happened to draw a slightly different
test set from the same distribution?" This is **test-set-sampling variance**,
a completely different question from seed-to-seed training variance.

Label any use of this as, e.g., "within-test-set bootstrap CI for seed 42" —
never as "the confidence interval for this condition" unqualified, since that
would imply it captures training variance too, which it does not.

## What was NOT done

No hierarchical bootstrap (resampling both seeds AND test examples jointly)
was implemented. Doing this correctly requires nested resampling that
respects the two-level structure (seed within condition, example within
test set) — approximating it by flattening both sources into one bootstrap
would understate or misstate the true combined uncertainty. Rather than
fabricate this, it is left undone; the 3-seed mean±SD and the per-seed
within-test-set CI are reported as the two distinct, honestly-computed
measures available.
""")
print("✅ UNCERTAINTY_REPORTING.md written")
print("\n✅ Sections 10-14 complete.")


##############################################################################
# Section 12 — Low-support sensitivity analysis
##############################################################################
support_check = part4_per_class_corrected[
    (part4_per_class_corrected.evaluation_source == "qwen_test") &
    (part4_per_class_corrected.architecture == "kan") & (part4_per_class_corrected.seed == 42)
].groupby("class_name")["support"].nunique()
assert (support_check == 1).all(), "STOP: support varies by training_source — investigate before trusting Section 12"
print("✅ Confirmed: support is identical across training_source for a fixed eval set")

support_qwen = part4_per_class_corrected[(part4_per_class_corrected.evaluation_source == "qwen_test") &
    (part4_per_class_corrected.training_source == "qwen") &
    (part4_per_class_corrected.architecture == "kan") & (part4_per_class_corrected.seed == 42)][["class_name", "support"]].set_index("class_name")["support"]
support_llama = part4_per_class_corrected[(part4_per_class_corrected.evaluation_source == "llama_test") &
    (part4_per_class_corrected.training_source == "llama") &
    (part4_per_class_corrected.architecture == "kan") & (part4_per_class_corrected.seed == 42)][["class_name", "support"]].set_index("class_name")["support"]

stable_support_classes = sorted([c for c in shared_28 if support_qwen.get(c, 0) >= 5 and support_llama.get(c, 0) >= 5])
print(f"Stable-support classes (test support >=5 in BOTH Qwen and Llama test): {len(stable_support_classes)}/28")
print(stable_support_classes)
excluded = sorted(set(shared_28) - set(stable_support_classes))
print(f"Excluded (low support in at least one teacher's test set): {excluded}")
for c in excluded:
    print(f"  {c}: qwen_test support={support_qwen.get(c,0)}, llama_test support={support_llama.get(c,0)}")

with open(OUT / "stable_support_classes.txt", "w") as f:
    f.write("\n".join(stable_support_classes))

stable_ids = [class_to_local_id[c] for c in stable_support_classes]

def top3_hit_row(row):
    top3 = eval(row["pred_top3"]) if isinstance(row["pred_top3"], str) else row["pred_top3"]
    return row["ground_truth"] in top3

support_qualified_rows = []
for regime in ["qwen", "llama", "mixed"]:
    for arch in ["kan", "mlp"]:
        for ev in ["qwen_test", "llama_test", "human"]:
            per_seed = []
            for seed in SEEDS:
                df = pd.read_csv(PRED_DIR / f"{arch}_{regime}_seed{seed}_on_{ev}.csv")
                df_sub = df[df["ground_truth"].isin(stable_ids)].reset_index(drop=True)
                if len(df_sub) == 0:
                    continue
                top1 = (df_sub["pred_top1"] == df_sub["ground_truth"]).mean()
                top3 = df_sub.apply(top3_hit_row, axis=1).mean()
                macro_f1 = f1_score(df_sub["ground_truth"], df_sub["pred_top1"], average="macro", labels=stable_ids, zero_division=0)
                per_seed.append(dict(top1=top1, top3=top3, macro_f1=macro_f1, n=len(df_sub)))
            if not per_seed:
                continue
            ps = pd.DataFrame(per_seed)
            support_qualified_rows.append(dict(
                training_source=regime, architecture=arch, evaluation_source=ev, n=int(ps.n.iloc[0]),
                top1_mean=ps.top1.mean(), top1_std=ps.top1.std(ddof=1),
                top3_mean=ps.top3.mean(), top3_std=ps.top3.std(ddof=1),
                macro_f1_mean=ps.macro_f1.mean(), macro_f1_std=ps.macro_f1.std(ddof=1),
            ))
support_qualified_results = pd.DataFrame(support_qualified_rows)
support_qualified_results.to_csv(OUT / "support_qualified_results.csv", index=False)
print(f"\n✅ support_qualified_results.csv: {support_qualified_results.shape}")
print(support_qualified_results[["training_source","architecture","evaluation_source","n","top1_mean","macro_f1_mean"]])

sq_tex = support_qualified_results.copy()
sq_tex["Top-1"] = sq_tex.apply(lambda r: f"{r.top1_mean*100:.2f} $\\pm$ {r.top1_std*100:.2f}", axis=1)
sq_tex["Macro F1"] = sq_tex.apply(lambda r: f"{r.macro_f1_mean*100:.2f} $\\pm$ {r.macro_f1_std*100:.2f}", axis=1)
sq_out = sq_tex[["training_source","architecture","evaluation_source","n","Top-1","Macro F1"]].rename(
    columns={"training_source":"Training","architecture":"Model","evaluation_source":"Eval"})
sq_out["Model"] = sq_out["Model"].str.upper()


##############################################################################
# THE FIX — one missing line, then finish writing Section 12's table
##############################################################################
TABLES_DIR = OUT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

with open(TABLES_DIR / "table_support_sensitivity.tex", "w") as f:
    f.write(sq_out.to_latex(index=False, caption=f"Stable-Support Sensitivity ({len(stable_support_classes)}/28 classes, "
                             "full Shared-28 remains primary)", label="tab:support_sensitivity", escape=True))
print("✅ table_support_sensitivity.tex saved")


##############################################################################
# Section 13 — Correct the Qwen corrective-augmentation provenance (audited,
# not asserted from memory)
##############################################################################
aug_rows = qwen_shared28_train[qwen_shared28_train["source"] == "manual_corrective_augmentation"]
aug_counts = aug_rows["tool"].value_counts()

below_100_originally = ["pose_estimation", "face_detection", "facial_expression_recognition", "logo_detection"]
actually_augmented = sorted(aug_counts.index.tolist())
not_augmented = sorted(set(below_100_originally) - set(actually_augmented))

with open(OUT / "qwen_corrective_augmentation_provenance_FINAL.md", "w") as f:
    f.write("# Qwen Corrective Augmentation — Provenance (FINAL, audited)\n\n")
    f.write("Audited directly from `qwen_shared28_train.csv`'s `source` column — not asserted "
            "from memory or from an earlier draft.\n\n")
    f.write("## Classes targeted (rows actually added)\n\n")
    f.write("| Class | Rows added |\n|---|---|\n")
    for cls in actually_augmented:
        f.write(f"| {cls} | {int(aug_counts[cls])} |\n")
    f.write(f"\n**Total rows added: {int(aug_counts.sum())}**\n\n")
    f.write("## Classes considered but NOT augmented\n\n")
    for cls in not_augmented:
        f.write(f"- **{cls}**: flagged as under the 100-raw-sample threshold, but no query "
                f"templates were authored for this class in the correction script (an oversight, "
                f"not a deliberate exclusion). Its original raw sample count nonetheless proved "
                f"sufficient to achieve full 28/28 train/val/test coverage after leakage-safe "
                f"splitting (verified in `shared28_split_leakage_audit_FINAL_V2.md`), so no "
                f"retroactive fix was applied.\n")
    f.write("\n**Correction from the earlier draft:** a previous version of this provenance file "
            "incorrectly implied `logo_detection` received the same targeted augmentation as the "
            "other three classes. It did not — zero rows were added for `logo_detection`. This "
            "file corrects that.\n\n")
    f.write("## Method (unchanged from original)\n\n")
    f.write("Per-image phrasing varied (opener + noun choice) so each image's rows form an "
            "independent connected component under leakage-safe grouping, rather than merging "
            "into one atomic block. Images reused from existing corpus images already labeled "
            "with a person-indicating tool. CLIP image embeddings pulled from the existing cache; "
            "only MiniLM text embeddings freshly computed.\n")
print(f"✅ qwen_corrective_augmentation_provenance_FINAL.md written")
print(f"   Actually augmented: {actually_augmented} (total {int(aug_counts.sum())} rows)")
print(f"   Considered but NOT augmented: {not_augmented}")


##############################################################################
# Section 14 — Fix bootstrap-uncertainty reporting
##############################################################################
def bootstrap_ci(preds, labels, metric_fn, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed); n = len(labels); stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n); stats.append(metric_fn(preds[idx], labels[idx]))
    stats = np.array(stats)
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def top1_fn(p, l): return np.mean(p == l)
def macrof1_fn(p, l): return f1_score(l, p, average="macro", labels=list(range(28)), zero_division=0)

within_seed_ci_rows = []
for arch in ["kan", "mlp"]:
    for ts in ["qwen", "llama", "mixed"]:
        for ev in ["qwen_test", "llama_test", "human"]:
            for seed in SEEDS:
                path = PRED_DIR / f"{arch}_{ts}_seed{seed}_on_{ev}.csv"
                if not path.exists(): continue
                df = pd.read_csv(path)
                preds, labels = df["pred_top1"].values, df["ground_truth"].values
                t1m, t1l, t1h = bootstrap_ci(preds, labels, top1_fn)
                fm, fl, fh = bootstrap_ci(preds, labels, macrof1_fn)
                within_seed_ci_rows.append(dict(architecture=arch, training_source=ts, evaluation_source=ev, seed=seed,
                    top1_estimate=t1m, top1_ci_low=t1l, top1_ci_high=t1h,
                    macro_f1_estimate=fm, macro_f1_ci_low=fl, macro_f1_ci_high=fh))
within_seed_bootstrap_ci = pd.DataFrame(within_seed_ci_rows)
within_seed_bootstrap_ci.to_csv(OUT / "within_seed_bootstrap_ci.csv", index=False)
print(f"✅ within_seed_bootstrap_ci.csv: {within_seed_bootstrap_ci.shape}")

with open(OUT / "UNCERTAINTY_REPORTING.md", "w") as f:
    f.write("""# Uncertainty Reporting — What Each Measure Represents

This project reports two genuinely different sources of uncertainty. Do not
conflate them, and do not present the seed-42-only bootstrap interval from an
earlier draft as if it captured seed-to-seed variance — it did not.

## Primary: mean ± SD across 3 seeds

File: `part4_three_seed_summary.csv`, `shared28_per_class_transfer_3seed.csv`,
`support_qualified_results.csv`, `image_disjoint_cross_teacher_results.csv`,
`human_known_vs_new_results.csv`.

This captures **training-run-to-training-run variance**: how much the result
changes due to random weight initialization and batch ordering, holding the
train/val/test split and test set fixed. This is the uncertainty measure to
lead with in the paper. All of it comes from the seed-corrected runs (the
seed bug identified in Section 2 is fixed in every number here).

## Secondary (optional): within-test-set bootstrap CI, per seed

File: `within_seed_bootstrap_ci.csv`.

For a SINGLE fixed trained model (one specific seed), this resamples the test
set itself (1000 resamples, with replacement) to ask: "how much would this
model's reported metric wobble if we happened to draw a slightly different
test set from the same distribution?" This is **test-set-sampling variance**,
a completely different question from seed-to-seed training variance.

Label any use of this as, e.g., "within-test-set bootstrap CI for seed 42" —
never as "the confidence interval for this condition" unqualified, since that
would imply it captures training variance too, which it does not.

## What was NOT done

No hierarchical bootstrap (resampling both seeds AND test examples jointly)
was implemented. Doing this correctly requires nested resampling that
respects the two-level structure (seed within condition, example within
test set) — approximating it by flattening both sources into one bootstrap
would understate or misstate the true combined uncertainty. Rather than
fabricate this, it is left undone; the 3-seed mean±SD and the per-seed
within-test-set CI are reported as the two distinct, honestly-computed
measures available.
""")
print("✅ UNCERTAINTY_REPORTING.md written")
print("\n✅ Sections 10-14 complete.")


##############################################################################
# Diagnostic — why 85 rows in TRAIN, not 135 total?
##############################################################################
print("Per-class breakdown in qwen_shared28_train.csv only:")
print(aug_counts)

# Check val and test splits too — the augmentation added rows to the FULL
# corpus before splitting; some of those 9 independent image-groups per class
# may have been allocated to val/test by the leakage-safe splitter to satisfy
# per-class coverage there, not just train. That would fully explain a lower
# train-only count without anything being wrong.
aug_val = qwen_shared28_val[qwen_shared28_val["source"] == "manual_corrective_augmentation"]["tool"].value_counts()
aug_test = qwen_shared28_test[qwen_shared28_test["source"] == "manual_corrective_augmentation"]["tool"].value_counts()

print("\nPer-class breakdown in qwen_shared28_val.csv:")
print(aug_val)
print("\nPer-class breakdown in qwen_shared28_test.csv:")
print(aug_test)

total_per_class = (aug_counts.add(aug_val, fill_value=0).add(aug_test, fill_value=0))
print("\nTotal across train+val+test (should be 45 per class, 135 total):")
print(total_per_class)
print(f"Grand total: {int(total_per_class.sum())}")


import zipfile
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
ZIP_PATH = WORK_DIR / "part5_2.zip"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in WORK_DIR.rglob("*"):
        if path.is_file() and path != ZIP_PATH:
            zf.write(path, path.relative_to(WORK_DIR))

print(f"Created: {ZIP_PATH}")
print(f"Size: {ZIP_PATH.stat().st_size / (1024**2):.2f} MB")
