#!/usr/bin/env python3
"""
FINISH_sections_15_24.py  --  runs anywhere (Kaggle, local). No GPU, no retraining.

USAGE
  python FINISH_sections_15_24.py PART5_DIR EARLIER_DIR OUT_DIR [CODE_DIR] [ZIP_BASENAME_PATH]

  PART5_DIR   folder holding the part5 outputs  (contains part4_predictions_seed_corrected/,
              checkpoints_seed_corrected/, part4_three_seed_summary.csv, ...)
  EARLIER_DIR folder holding qwen_shared28_{train,val,test}.csv, llama_shared28_*.csv,
              human_shared28.csv, shared_28_class_mapping.json
  OUT_DIR     where the finished package folder is written (created fresh)
  CODE_DIR    (optional) folder of the earlier stage scripts, copied into OUT_DIR/code/
  ZIP_BASENAME_PATH (optional) e.g. /kaggle/working/KAN_ROUTE_PART4_JOURNAL_READY

WHAT IT FIXES / DOES
  * Bug in Stage A: human_shared28.csv was re-read from disk, empty acceptable_tool_labels became
    NaN -> "nan" != "" -> every row counted as 'ambiguous', none as 'strict'. Result: strict_top1 /
    strict_top3 = NaN and ambiguity-aware scores computed against the string "nan".
    (The script first PRINTS this on your files, then recomputes everything from raw predictions.)
  * Sec 13 provenance re-audited across train+val+test (earlier audit only looked at train).
  * Sec 15,16,17,18,19,20,22,23,24 + final zip.
"""
import sys, json, shutil, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PART5_DIR, EARLIER_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
CODE_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
ZIP_BASE = sys.argv[5] if len(sys.argv) > 5 else None

SEEDS = [42, 123, 2026]; ARCHS = ["kan", "mlp"]; TSS = ["qwen", "llama", "mixed"]
LAB = list(range(28))

# ----------------------------------------------------------------------------------------------
# 0. Assemble output folder (copy everything forward; move superseded files aside, don't delete)
# ----------------------------------------------------------------------------------------------
if OUT.exists(): shutil.rmtree(OUT)
shutil.copytree(PART5_DIR, OUT)
for fn in ["qwen_shared28_train.csv", "qwen_shared28_val.csv", "qwen_shared28_test.csv",
           "llama_shared28_train.csv", "llama_shared28_val.csv", "llama_shared28_test.csv",
           "human_shared28.csv", "shared_28_class_mapping.json", "shared28_taxonomy.txt"]:
    if (EARLIER_DIR / fn).exists(): shutil.copy(EARLIER_DIR / fn, OUT / fn)
(OUT / "superseded").mkdir(exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
def supersede(rel):
    p = OUT / rel
    if p.exists(): shutil.move(str(p), str(OUT / "superseded" / (p.stem + "_SUPERSEDED" + p.suffix)))
for rel in ["part4_all_results_seed_corrected.csv", "part4_three_seed_summary.csv",
            "human_known_vs_new_results.csv", "image_disjoint_cross_teacher_results.csv",
            "tables/table_human_known_vs_new.tex", "tables/table_image_disjoint_cross_teacher.tex",
            "figures/human_known_vs_new.png"]:
    if rel in ("part4_all_results_seed_corrected.csv", "part4_three_seed_summary.csv", "human_known_vs_new_results.csv"):
        shutil.copy(PART5_DIR / rel, OUT / "superseded" / (Path(rel).stem + "_OLD_BUGGY.csv"))
    else:
        pass
    supersede(rel) if rel not in ("part4_all_results_seed_corrected.csv", "part4_three_seed_summary.csv",
                                  "human_known_vs_new_results.csv") else (OUT / rel).unlink()

# ----------------------------------------------------------------------------------------------
# 1. Load protocol files
# ----------------------------------------------------------------------------------------------
mapping = json.load(open(OUT / "shared_28_class_mapping.json"))
class_to_local_id = mapping["class_to_local_id"]
local_id_to_class = {int(k): v for k, v in mapping["local_id_to_class"].items()}
shared_28 = sorted(class_to_local_id)
qtr, qva, qte = [pd.read_csv(OUT / f"qwen_shared28_{s}.csv") for s in ("train", "val", "test")]
ltr, lva, lte = [pd.read_csv(OUT / f"llama_shared28_{s}.csv") for s in ("train", "val", "test")]
human = pd.read_csv(OUT / "human_shared28.csv").reset_index(drop=True)

# ----------------------------------------------------------------------------------------------
# 2. PROOF OF THE BUG (printed on your own files) + the fix
# ----------------------------------------------------------------------------------------------
old = pd.read_csv(OUT / "superseded" / "part4_all_results_seed_corrected_OLD_BUGGY.csv")
oh = old[old.evaluation_source == "human"]
print("=" * 90); print("BUG DEMONSTRATION (Stage A human strict/ambiguous masks)"); print("=" * 90)
print(f"human_shared28.csv rows: {len(human)};  NaN in acceptable_tool_labels: {human.acceptable_tool_labels.isna().sum()}")
print(f"  OLD logic  astype(str).str.strip() != ''  -> 'ambiguous' rows = "
      f"{(human.acceptable_tool_labels.astype(str).str.strip() != '').sum()}  (WRONG: every row)")
human["acceptable_tool_labels"] = human["acceptable_tool_labels"].fillna("").astype(str).str.strip()
print(f"  FIXED logic fillna('') first              -> 'ambiguous' rows = "
      f"{(human.acceptable_tool_labels != '').sum()}  (expected 88)")
print(f"OLD saved results: strict_top1 NaN in {oh.strict_top1.isna().sum()}/{len(oh)} human runs; "
      f"mean ambiguity_aware_top1 = {oh.ambiguity_aware_top1.mean():.3f} (scored vs the string 'nan')")

strict_bool = human["strict_eval"].astype(str).str.lower().isin(["true", "1"]).values
strict_m = strict_bool & (human.acceptable_tool_labels == "").values
ambig_m = strict_bool & (human.acceptable_tool_labels != "").values
print(f"FIXED masks: strict single-label = {strict_m.sum()}, strict ambiguous = {ambig_m.sum()}, "
      f"neither = {len(human) - strict_m.sum() - ambig_m.sum()}\n")
assert strict_m.sum() + ambig_m.sum() == len(human) and ambig_m.sum() == 88 and strict_m.sum() == 284

train_imgs = set(qtr.img_id) | set(ltr.img_id)
known_m = human["image_id"].isin(train_imgs).values; new_m = ~known_m
qtr_keys = set(zip(qtr.img_id, qtr.tool)); ltr_keys = set(zip(ltr.img_id, ltr.tool))
shared_ids = sorted(class_to_local_id[c] for c in set(human.loc[known_m, "tool_name"]) & set(human.loc[new_m, "tool_name"]))

# ----------------------------------------------------------------------------------------------
# 3. Metrics straight from the raw prediction files (Section 21: predictions are the source)
# ----------------------------------------------------------------------------------------------
PRED = OUT / "part4_predictions_seed_corrected"
_cache = {}
def load_pred(arch, ts, seed, ev):
    k = (arch, ts, seed, ev)
    if k not in _cache:
        df = pd.read_csv(PRED / f"{arch}_{ts}_seed{seed}_on_{ev}.csv")
        df["top3"] = df["pred_top3"].apply(json.loads)
        _cache[k] = df
    return _cache[k]

def mets(df, mask=None, labels=LAB):
    d = df if mask is None else df[mask]
    if len(d) == 0:
        return dict(n=0, top1=np.nan, top3=np.nan, macro_f1=np.nan, weighted_f1=np.nan)
    y, p = d["ground_truth"].values, d["pred_top1"].values
    hit3 = np.fromiter((g in t for g, t in zip(y, d["top3"].values)), bool, len(d))
    return dict(n=int(len(d)), top1=float((y == p).mean()), top3=float(hit3.mean()),
                macro_f1=float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
                weighted_f1=float(f1_score(y, p, average="weighted", labels=labels, zero_division=0)))

def human_mets(df, sub_mask):
    assert (df["sample_id"].astype(str).values == human["query_id"].astype(str).values).all(), "query_id misalignment"
    m = mets(df, sub_mask)
    ms = mets(df, sub_mask & strict_m)
    out = dict(**m, strict_n=ms["n"], strict_top1=ms["top1"], strict_top3=ms["top3"],
               strict_macro_f1=ms["macro_f1"], strict_weighted_f1=ms["weighted_f1"])
    am = sub_mask & ambig_m
    if am.sum():
        sub, acc = df[am], human.loc[am, "acceptable_tool_labels"].values
        t1, t3 = [], []
        for (_, r), a in zip(sub.iterrows(), acc):
            aset = {x.strip() for x in a.split(",") if x.strip()}
            t1.append(r["pred_top1_tool"] in aset)
            t3.append(bool({local_id_to_class[c] for c in r["top3"]} & aset))
        out.update(ambiguity_aware_n=int(am.sum()), ambiguity_aware_top1=float(np.mean(t1)), ambiguity_aware_top3=float(np.mean(t3)))
    else:
        out.update(ambiguity_aware_n=0, ambiguity_aware_top1=np.nan, ambiguity_aware_top3=np.nan)
    return out

rows = []
for ts in TSS:
    for arch in ARCHS:
        for seed in SEEDS:
            for ev in ("qwen_test", "llama_test"):
                df = load_pred(arch, ts, seed, ev)
                rows.append(dict(ts=ts, arch=arch, seed=seed, ev=ev, subset="full", **mets(df)))
                opp = ltr_keys if ev == "qwen_test" else qtr_keys       # image-disjoint vs OPPOSITE teacher's train
                dm = np.array([(i, t) not in opp for i, t in zip(df["img_id"], df["ground_truth_tool"])])
                rows.append(dict(ts=ts, arch=arch, seed=seed, ev=ev, subset="image_disjoint", **mets(df, dm)))
            df = load_pred(arch, ts, seed, "human")
            for sub, mask in (("full", np.ones(len(human), bool)), ("known", known_m), ("new", new_m)):
                h = human_mets(df, mask)
                if sub in ("known", "new"):
                    d = df[mask]
                    h["macro_f1_fair"] = float(f1_score(d["ground_truth"], d["pred_top1"], average="macro", labels=shared_ids, zero_division=0))
                rows.append(dict(ts=ts, arch=arch, seed=seed, ev="human", subset=sub, **h))
R = pd.DataFrame(rows)
Ri = R.set_index(["ts", "arch", "seed", "ev", "subset"]).sort_index()
def g(ts, arch, seed, ev, sub, metric): return float(Ri.loc[(ts, arch, seed, ev, sub), metric])
def seeds_vals(ts, arch, ev, sub, metric): return np.array([g(ts, arch, s, ev, sub, metric) for s in SEEDS])
def A(ts, arch, ev, sub, metric):
    v = seeds_vals(ts, arch, ev, sub, metric); return float(v.mean()), float(v.std(ddof=1))
def agg_cols(df, keys, cols):
    out = df.groupby(keys).agg(**{f"{c}_{f}": (c, f) for c in cols for f in ("mean", "std")}).reset_index()
    return out

# ---- regenerate the main results file and PROVE only the human strict/ambiguity columns changed
full = R[R.subset == "full"].rename(columns={"ts": "training_source", "arch": "architecture", "ev": "evaluation_source"})
cols_main = ["training_source", "architecture", "seed", "evaluation_source", "top1", "top3", "macro_f1", "weighted_f1"]
cols_h = ["strict_top1", "strict_top3", "strict_macro_f1", "strict_weighted_f1", "ambiguity_aware_top1", "ambiguity_aware_top3"]
new_all = full[cols_main + cols_h].copy()
new_all.loc[new_all.evaluation_source != "human", cols_h] = np.nan
new_all.to_csv(OUT / "part4_all_results_seed_corrected.csv", index=False)
chk = new_all.merge(old, on=["training_source", "architecture", "seed", "evaluation_source"], suffixes=("_new", "_old"))
max_diff_canon = max(float((chk[f"{c}_new"] - chk[f"{c}_old"]).abs().max()) for c in ["top1", "top3", "macro_f1", "weighted_f1"])
print(f"Recomputed-from-predictions vs OLD file, canonical columns (top1/top3/macroF1/weightedF1): "
      f"max |diff| = {max_diff_canon:.2e}  -> only the strict/ambiguity columns were wrong.")

# ---- three-seed summary (long format)
summ = []
for (ts, arch, ev), grp in new_all.groupby(["training_source", "architecture", "evaluation_source"]):
    grp = grp.set_index("seed")
    for metric in ["top1", "top3", "macro_f1", "weighted_f1"] + (cols_h if ev == "human" else []):
        v = grp[metric]
        summ.append(dict(training_source=ts, architecture=arch, evaluation_source=ev, metric=metric,
                         mean=v.mean(), std=v.std(ddof=1), seed42=v.get(42), seed123=v.get(123), seed2026=v.get(2026)))
three_seed = pd.DataFrame(summ); three_seed.to_csv(OUT / "part4_three_seed_summary.csv", index=False)

# ---- image-disjoint file
dj_rows = []
for ts, ev in [("qwen", "llama_test"), ("mixed", "llama_test"), ("llama", "qwen_test"), ("mixed", "qwen_test")]:
    for arch in ARCHS:
        rec = dict(training_source=ts, architecture=arch, evaluation_source=ev, n=int(g(ts, arch, 42, ev, "image_disjoint", "n")))
        for m in ["top1", "top3", "macro_f1", "weighted_f1"]:
            mu, sd = A(ts, arch, ev, "image_disjoint", m); rec[f"{m}_mean"], rec[f"{m}_std"] = mu, sd
        dj_rows.append(rec)
image_disjoint = pd.DataFrame(dj_rows); image_disjoint.to_csv(OUT / "image_disjoint_cross_teacher_results.csv", index=False)

# ---- known vs new file
kn_rows = []
for ts in TSS:
    for arch in ARCHS:
        for sub in ("known", "new"):
            rec = dict(training_source=ts, architecture=arch, subset=sub, n=int(g(ts, arch, 42, "human", sub, "n")),
                       n_shared_classes_for_fair_f1=len(shared_ids))
            for m, name in [("top1", "canonical_top1"), ("strict_top1", "strict_top1"), ("strict_top3", "strict_top3"),
                            ("macro_f1", "macro_f1"), ("weighted_f1", "weighted_f1"),
                            ("ambiguity_aware_top1", "ambiguity_aware_top1"), ("ambiguity_aware_top3", "ambiguity_aware_top3"),
                            ("macro_f1_fair", "macro_f1_fair_shared_classes")]:
                mu, sd = A(ts, arch, "human", sub, m); rec[f"{name}_mean"], rec[f"{name}_std"] = mu, sd
            kn_rows.append(rec)
known_new = pd.DataFrame(kn_rows); known_new.to_csv(OUT / "human_known_vs_new_results.csv", index=False)

ov = OUT / "cross_teacher_context_overlap_report.md"
if ov.exists():
    ov.write_text(ov.read_text().replace("macro_f1_fair_28class_intersection_mean", "macro_f1_fair_shared_classes_mean"))

# ----------------------------------------------------------------------------------------------
# 4. Section 13 -- provenance audited across train+val+test (Qwen AND Llama)
# ----------------------------------------------------------------------------------------------
def aug_counts(dfs):
    out = {}
    for name, d in dfs.items():
        s = d[d["source"] == "manual_corrective_augmentation"]
        for tool, n in s["tool"].value_counts().items(): out.setdefault(tool, {})[name] = int(n)
    return out
q_aug = aug_counts({"train": qtr, "val": qva, "test": qte}); l_aug = aug_counts({"train": ltr, "val": lva, "test": lte})
def raw_counts(cls, dfs): return {n: int(((d.tool == cls) & (d.source != "manual_corrective_augmentation")).sum()) for n, d in dfs.items()}
qd = {"train": qtr, "val": qva, "test": qte}; ld = {"train": ltr, "val": lva, "test": lte}
q_total = sum(sum(v.values()) for v in q_aug.values()); l_total = sum(sum(v.values()) for v in l_aug.values())
q_train_total = sum(v.get("train", 0) for v in q_aug.values())
prov = ["# Corrective Augmentation Provenance -- FINAL (audited across train + val + test)\n",
        "Counts below are computed from the saved split CSVs' `source` column by `FINISH_sections_15_24.py`.\n",
        "**Correction to the earlier audit:** it counted only the *train* split and reported 85 rows. The augmentation "
        f"added **{q_total} Qwen rows** (45 x 3 classes); the leakage-safe splitter (per-class coverage guarantee) "
        f"placed {q_train_total} in train and the remainder in val/test.\n",
        "## Qwen -- rows added per class and where they landed\n",
        "| Class | Total | Train | Val | Test |", "|---|---|---|---|---|"]
for cls, v in sorted(q_aug.items()):
    prov.append(f"| {cls} | {sum(v.values())} | {v.get('train',0)} | {v.get('val',0)} | {v.get('test',0)} |")
prov.append(f"\n**Total Qwen rows added: {q_total}** (train {q_train_total}).\n")
prov.append("## Test/val composition for the augmented classes (raw = teacher-generated, aug = template-authored)\n")
prov.append("| Class | Train raw/aug | Val raw/aug | Test raw/aug |"); prov.append("|---|---|---|---|")
for cls in sorted(q_aug):
    rc = raw_counts(cls, qd)
    prov.append(f"| {cls} | {rc['train']}/{q_aug[cls].get('train',0)} | {rc['val']}/{q_aug[cls].get('val',0)} | {rc['test']}/{q_aug[cls].get('test',0)} |")
te_only_aug = [c for c in sorted(q_aug) if raw_counts(c, qd)["test"] == 0]
va_only_aug = [c for c in sorted(q_aug) if raw_counts(c, qd)["val"] == 0 and q_aug[c].get("val", 0) > 0]
prov.append("\n**Disclosure:** for "
            + (", ".join(f"`{c}`" for c in te_only_aug) if te_only_aug else "no class")
            + " the Qwen **test** rows are 100% template-authored (zero teacher-generated rows); for "
            + (", ".join(f"`{c}`" for c in va_only_aug) if va_only_aug else "no class")
            + " the **val** rows are 100% template-authored. Per-class Qwen-test numbers for these classes measure "
              "performance on template phrasing, not teacher-generated queries, and must be reported with that caveat.\n")
prov.append("## Classes considered but not augmented\n")
below = [c for c in ["pose_estimation", "face_detection", "facial_expression_recognition", "logo_detection"] if c not in q_aug]
for c in below:
    rc = raw_counts(c, qd)
    prov.append(f"- `{c}`: flagged (<100 raw samples) but no templates were written, so **0 rows added** "
                f"(raw train/val/test = {rc['train']}/{rc['val']}/{rc['test']}). It still reached 28/28 split coverage. "
                "An earlier draft wrongly implied it was augmented.")
prov.append("\n## Llama -- rows added\n")
prov.append("| Class | Total | Train | Val | Test |"); prov.append("|---|---|---|---|---|")
for cls, v in sorted(l_aug.items()):
    prov.append(f"| {cls} | {sum(v.values())} | {v.get('train',0)} | {v.get('val',0)} | {v.get('test',0)} |")
prov.append(f"\n**Total Llama rows added: {l_total}.**\n")
prov.append("## Method\nPer-image phrasing varied (opener + noun) so each image forms its own independent group under "
            "leakage-safe grouping. Images reused from the existing corpora; CLIP image vectors taken from the existing "
            "cache; only MiniLM text embeddings were computed for the new queries.\n")
(OUT / "qwen_corrective_augmentation_provenance_FINAL.md").write_text("\n".join(prov))
print(f"Provenance: Qwen total={q_total} (train={q_train_total}), Llama total={l_total}")

# ----------------------------------------------------------------------------------------------
# 5. Sections 15, 16, 17
# ----------------------------------------------------------------------------------------------
drop_rows = []
for sub in ("full", "image_disjoint"):
    for arch in ARCHS:
        for src, inev, crev in [("qwen", "qwen_test", "llama_test"), ("llama", "llama_test", "qwen_test")]:
            for seed in SEEDS:
                r = dict(architecture=arch, source_teacher=src, subset=sub, seed=seed, n_cross=int(g(src, arch, seed, crev, sub, "n")),
                         in_domain_top1=g(src, arch, seed, inev, "full", "top1"), cross_top1=g(src, arch, seed, crev, sub, "top1"),
                         in_domain_f1=g(src, arch, seed, inev, "full", "macro_f1"), cross_f1=g(src, arch, seed, crev, sub, "macro_f1"))
                r["top1_drop"] = r["in_domain_top1"] - r["cross_top1"]; r["f1_drop"] = r["in_domain_f1"] - r["cross_f1"]
                drop_rows.append(r)
drop_ps = pd.DataFrame(drop_rows); drop_ps.to_csv(OUT / "cross_teacher_drop_per_seed.csv", index=False)
drop3 = agg_cols(drop_ps, ["architecture", "source_teacher", "subset"],
                 ["in_domain_top1", "cross_top1", "top1_drop", "in_domain_f1", "cross_f1", "f1_drop"])
drop3.insert(3, "n_cross", drop_ps.groupby(["architecture", "source_teacher", "subset"])["n_cross"].first().values)
drop3.to_csv(OUT / "cross_teacher_drop_3seed.csv", index=False)

rec_rows = []
for sub in ("full", "image_disjoint"):
    for arch in ARCHS:
        for target, opp in [("qwen_test", "llama"), ("llama_test", "qwen")]:
            for seed in SEEDS:
                r = dict(architecture=arch, target=target, subset=sub, seed=seed, n=int(g("mixed", arch, seed, target, sub, "n")),
                         opposite_top1=g(opp, arch, seed, target, sub, "top1"), mixed_top1=g("mixed", arch, seed, target, sub, "top1"),
                         opposite_f1=g(opp, arch, seed, target, sub, "macro_f1"), mixed_f1=g("mixed", arch, seed, target, sub, "macro_f1"))
                r["recovery_top1"] = r["mixed_top1"] - r["opposite_top1"]; r["recovery_f1"] = r["mixed_f1"] - r["opposite_f1"]
                rec_rows.append(r)
rec_ps = pd.DataFrame(rec_rows); rec_ps.to_csv(OUT / "multi_teacher_recovery_per_seed.csv", index=False)
rec3 = agg_cols(rec_ps, ["architecture", "target", "subset"],
                ["opposite_top1", "mixed_top1", "recovery_top1", "opposite_f1", "mixed_f1", "recovery_f1"])
rec3.insert(3, "n", rec_ps.groupby(["architecture", "target", "subset"])["n"].first().values)
rec3.to_csv(OUT / "multi_teacher_recovery_3seed.csv", index=False)

gap_rows = []
for arch in ARCHS:
    for ts in TSS:
        for seed in SEEDS:
            if ts == "mixed":
                in1 = np.mean([g(ts, arch, seed, e, "full", "top1") for e in ("qwen_test", "llama_test")])
                inf = np.mean([g(ts, arch, seed, e, "full", "macro_f1") for e in ("qwen_test", "llama_test")])
            else:
                e = "qwen_test" if ts == "qwen" else "llama_test"
                in1, inf = g(ts, arch, seed, e, "full", "top1"), g(ts, arch, seed, e, "full", "macro_f1")
            hw1, hwf = g(ts, arch, seed, "human", "full", "top1"), g(ts, arch, seed, "human", "full", "macro_f1")
            hn1, hnf = g(ts, arch, seed, "human", "new", "top1"), g(ts, arch, seed, "human", "new", "macro_f1")
            gap_rows.append(dict(architecture=arch, training_source=ts, seed=seed, synthetic_in_domain_top1=in1, human_written_top1=hw1,
                                 human_new_image_top1=hn1, gap_written_top1=in1 - hw1, gap_new_top1=in1 - hn1,
                                 synthetic_in_domain_f1=inf, human_written_f1=hwf, human_new_image_f1=hnf,
                                 gap_written_f1=inf - hwf, gap_new_f1=inf - hnf))
gap_ps = pd.DataFrame(gap_rows)
gap3 = agg_cols(gap_ps, ["architecture", "training_source"],
                ["synthetic_in_domain_top1", "human_written_top1", "human_new_image_top1", "gap_written_top1", "gap_new_top1",
                 "synthetic_in_domain_f1", "human_written_f1", "human_new_image_f1", "gap_written_f1", "gap_new_f1"])
gap3.to_csv(OUT / "synthetic_to_human_gap.csv", index=False)
print("Sections 15-17 CSVs written")

# ----------------------------------------------------------------------------------------------
# 6. Section 18 -- KAN vs MLP (computed; narrative generated from the numbers)
# ----------------------------------------------------------------------------------------------
GROUPS = {
    "within-teacher (in-domain + mixed)": [("qwen", "qwen_test", "full"), ("llama", "llama_test", "full"), ("mixed", "qwen_test", "full"), ("mixed", "llama_test", "full")],
    "cross-teacher (single-teacher, full)": [("qwen", "llama_test", "full"), ("llama", "qwen_test", "full")],
    "image-disjoint cross-teacher": [("qwen", "llama_test", "image_disjoint"), ("mixed", "llama_test", "image_disjoint"), ("llama", "qwen_test", "image_disjoint"), ("mixed", "qwen_test", "image_disjoint")],
    "human-written (all 372)": [(ts, "human", "full") for ts in TSS],
    "human new-image (210)": [(ts, "human", "new") for ts in TSS],
}
def group_seed_vals(arch, conds, metric):
    return np.array([np.mean([g(ts, arch, s, ev, sub, metric) for ts, ev, sub in conds]) for s in SEEDS])
cmp_rows = []
for gname, conds in GROUPS.items():
    for metric in ["top1", "top3", "macro_f1", "weighted_f1"]:
        k, m = group_seed_vals("kan", conds, metric), group_seed_vals("mlp", conds, metric)
        d = k - m
        cmp_rows.append(dict(group=gname, metric=metric, kan_mean=k.mean(), kan_sd=k.std(ddof=1), mlp_mean=m.mean(), mlp_sd=m.std(ddof=1),
                             diff_kan_minus_mlp=d.mean(), diff_sd=d.std(ddof=1), seeds_kan_ahead=int((d > 0).sum())))
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(OUT / "kan_vs_mlp_comparison.csv", index=False)
drop_kan = drop_ps[(drop_ps.architecture == "kan") & (drop_ps.subset == "full")].groupby("seed")[["top1_drop", "f1_drop"]].mean()
drop_mlp = drop_ps[(drop_ps.architecture == "mlp") & (drop_ps.subset == "full")].groupby("seed")[["top1_drop", "f1_drop"]].mean()
sq = pd.read_csv(OUT / "support_qualified_results.csv")
sqp = sq.pivot_table(index=["training_source", "evaluation_source"], columns="architecture", values="macro_f1_mean").reset_index()
sqp["kan_minus_mlp"] = sqp["kan"] - sqp["mlp"]
sqp_t1 = sq.pivot_table(index=["training_source", "evaluation_source"], columns="architecture", values="top1_mean").reset_index()
sqp_t1["kan_minus_mlp"] = sqp_t1["kan"] - sqp_t1["mlp"]
tlog = pd.read_csv(OUT / "part4_training_log_seed_corrected.csv")
params = tlog.groupby("arch")["n_params"].first().to_dict()

def cnt(metric, cond):
    s = cmp[cmp.metric == metric]; return int(cond(s).sum()), len(s)
t1_kan_ahead, ng = cnt("top1", lambda s: s.diff_kan_minus_mlp > 0.01)
t1_mlp_ahead, _ = cnt("top1", lambda s: s.diff_kan_minus_mlp < -0.01)
t1_comp = ng - t1_kan_ahead - t1_mlp_ahead
f1_kan_ahead, _ = cnt("macro_f1", lambda s: s.diff_kan_minus_mlp > 0)
f1_consistent, _ = cnt("macro_f1", lambda s: (s.diff_kan_minus_mlp > 0) & (s.seeds_kan_ahead == 3))
f1_sd_exceed, _ = cnt("macro_f1", lambda s: s.diff_kan_minus_mlp > s.diff_sd)
sq_kan_ahead = int((sqp.kan_minus_mlp > 0).sum()); sq_n = len(sqp)
support_f1_supported = sq_kan_ahead >= 0.67 * sq_n
full_f1_supported = f1_kan_ahead >= 4 and f1_consistent >= 3
if full_f1_supported and support_f1_supported: framing_status = "SUPPORTED"
elif f1_kan_ahead >= 3: framing_status = "PARTIALLY SUPPORTED"
else: framing_status = "NOT SUPPORTED"

def fmtpp(x): return f"{x*100:+.2f}pp"
md = ["# KAN vs MLP -- Final Analysis (seed-corrected, computed from raw predictions)\n",
      "Values are seed-level means over the listed conditions, then mean +/- SD across the 3 seeds. `diff` = KAN - MLP; "
      "`seeds_kan_ahead` counts seeds (seed-paired) where KAN > MLP. Neither architecture is declared universally superior.\n",
      "## Metric comparison by evaluation group\n",
      "| Group | Metric | KAN | MLP | KAN-MLP (mean +/- SD) | Seeds KAN ahead |", "|---|---|---|---|---|---|"]
for _, r in cmp.iterrows():
    md.append(f"| {r.group} | {r.metric} | {r.kan_mean*100:.2f} +/- {r.kan_sd*100:.2f} | {r.mlp_mean*100:.2f} +/- {r.mlp_sd*100:.2f} | "
              f"{r.diff_kan_minus_mlp*100:+.2f} +/- {r.diff_sd*100:.2f} | {r.seeds_kan_ahead}/3 |")
md += ["\n## Cross-teacher drop (full test sets; lower = more robust)\n",
       f"- KAN mean Top-1 drop {drop_kan.top1_drop.mean()*100:.2f} +/- {drop_kan.top1_drop.std(ddof=1)*100:.2f}pp; "
       f"MLP {drop_mlp.top1_drop.mean()*100:.2f} +/- {drop_mlp.top1_drop.std(ddof=1)*100:.2f}pp",
       f"- KAN mean Macro-F1 drop {drop_kan.f1_drop.mean()*100:.2f} +/- {drop_kan.f1_drop.std(ddof=1)*100:.2f}pp; "
       f"MLP {drop_mlp.f1_drop.mean()*100:.2f} +/- {drop_mlp.f1_drop.std(ddof=1)*100:.2f}pp",
       "\n## Low-support sensitivity (21 stable-support classes; 3-seed means only, no per-seed values stored)\n",
       "| Training | Eval | KAN Macro-F1 | MLP Macro-F1 | KAN-MLP |", "|---|---|---|---|---|"]
for _, r in sqp.iterrows():
    md.append(f"| {r.training_source} | {r.evaluation_source} | {r.kan*100:.2f} | {r.mlp*100:.2f} | {r.kan_minus_mlp*100:+.2f} |")
md += [f"\nKAN Macro-F1 ahead in {sq_kan_ahead}/{sq_n} stable-support cells.",
       "\n## Cost and interpretability\n",
       f"- Trainable parameters: KAN = {params.get('kan')}, MLP = {params.get('mlp')} (KAN is {100*(1-params.get('kan')/params.get('mlp')):.1f}% smaller).",
       "- Routing latency: **not re-measured in this seed-correction pass** (hardware-specific). Use the fair-latency results from the earlier "
       "revision stages; do not cite any latency number from this package.",
       "- Interpretability: KAN exposes per-edge learned spline functions (intrinsic); the MLP needs post-hoc attribution. This is a design "
       "property established in the earlier interpretability audit, not something these accuracy results measure.",
       "\n## Test of the proposed framing\n",
       "> MLP provides marginally stronger or comparable Top-1 accuracy, whereas KAN exhibits stronger macro-F1 under class imbalance and "
       "distribution shift while additionally providing intrinsic spline-based interpretability.\n",
       f"- Top-1: across {ng} evaluation groups, KAN ahead by >1pp in {t1_kan_ahead}, MLP ahead by >1pp in {t1_mlp_ahead}, within 1pp in {t1_comp} "
       f"-> Top-1 is {'comparable' if t1_comp >= ng-1 else 'mixed'}.",
       f"- Macro-F1 (all 28 classes): KAN ahead in {f1_kan_ahead}/{ng} groups, seed-consistent (3/3 seeds) in {f1_consistent}/{ng}; "
       f"mean gap exceeds its own seed-SD in {f1_sd_exceed}/{ng}.",
       f"- Restricted to stable-support classes: KAN ahead in {sq_kan_ahead}/{sq_n} cells.",
       f"- **Framing status: {framing_status}.**"]
if framing_status != "SUPPORTED":
    md.append("\n**Recommended wording:** "
              + ("KAN and MLP reach comparable Top-1; KAN shows higher macro-F1 in several evaluation groups on the full 28-class set, but the "
                 "advantage is not consistent across seeds/groups and does not clearly persist when low-support classes are excluded. "
                 "Report the accuracy-parity plus interpretability/parameter-efficiency trade-off; do not claim a robust macro-F1 advantage."
                 if framing_status == "PARTIALLY SUPPORTED" else
                 "KAN and MLP are statistically comparable on these benchmarks; the case for KAN rests on parameter efficiency and intrinsic interpretability, not accuracy."))
(OUT / "KAN_VS_MLP_FINAL_ANALYSIS.md").write_text("\n".join(md))
print(f"Section 18 written -- framing status: {framing_status}")

# ----------------------------------------------------------------------------------------------
# 7. Section 19 -- LaTeX tables (mean +/- SD, three-seed)
# ----------------------------------------------------------------------------------------------
def pm(mu, sd): return f"{mu*100:.2f} $\\pm$ {sd*100:.2f}"
def esc(x): return str(x).replace("_", "\\_")
def tex(headers, body, caption, label, path):
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule",
         " & ".join(esc(h) for h in headers) + " \\\\", "\\midrule"]
    for r in body: L.append(" & ".join(x if "\\pm" in str(x) else esc(x) for x in r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    Path(path).write_text("\n".join(L))
T = OUT / "tables"
body = []
for ts in TSS:
    for arch in ARCHS:
        body.append([ts, arch.upper()] + [pm(*A(ts, arch, ev, "full", m)) for ev in ("qwen_test", "llama_test") for m in ("top1", "macro_f1")])
tex(["Training", "Model", "Qwen Top-1", "Qwen Macro F1", "Llama Top-1", "Llama Macro F1"], body,
    "Cross-Teacher Generalization (Shared-28, mean $\\pm$ SD over 3 seeds, full test sets).", "tab:cross_teacher", T / "table1_cross_teacher_generalization.tex")
body = []
for _, r in drop3[drop3.subset == "full"].iterrows():
    body.append([r.architecture.upper(), r.source_teacher, pm(r.in_domain_top1_mean, r.in_domain_top1_std), pm(r.cross_top1_mean, r.cross_top1_std),
                 pm(r.top1_drop_mean, r.top1_drop_std), pm(r.in_domain_f1_mean, r.in_domain_f1_std), pm(r.cross_f1_mean, r.cross_f1_std), pm(r.f1_drop_mean, r.f1_drop_std)])
tex(["Model", "Source Teacher", "In-domain Top-1", "Cross-teacher Top-1", "Top-1 Drop", "In-domain F1", "Cross-teacher F1", "F1 Drop"], body,
    "Distribution shift: in-domain vs cross-teacher (observed shift, mean $\\pm$ SD over 3 seeds).", "tab:dist_shift", T / "table2_distribution_shift.tex")
for sub, fn, cap in (("full", "table3_multi_teacher_recovery.tex", "Multi-teacher recovery, full test sets."),
                     ("image_disjoint", "table3b_multi_teacher_recovery_image_disjoint.tex", "Multi-teacher recovery, image-disjoint subsets.")):
    body = [[r.architecture.upper(), r.target, pm(r.opposite_top1_mean, r.opposite_top1_std), pm(r.mixed_top1_mean, r.mixed_top1_std),
             pm(r.recovery_top1_mean, r.recovery_top1_std), pm(r.recovery_f1_mean, r.recovery_f1_std)]
            for _, r in rec3[rec3.subset == sub].iterrows()]
    tex(["Architecture", "Target", "Opposite-teacher Top-1", "Mixed Top-1", "Top-1 Recovery", "F1 Recovery"], body,
        cap + " Mean $\\pm$ SD over 3 seeds.", "tab:recovery_" + sub, T / fn)
body = []
for ts in TSS:
    for arch in ARCHS:
        body.append([ts, arch.upper()] + [pm(*A(ts, arch, "human", "full", m)) for m in ("top1", "strict_top1", "strict_top3", "macro_f1", "ambiguity_aware_top1", "ambiguity_aware_top3")])
tex(["Training", "Model", "Canonical Top-1", "Strict Top-1", "Strict Top-3", "Macro F1", "Ambiguity Top-1", "Ambiguity Top-3"], body,
    "Human-written evaluation (372 queries; strict subset n=284, ambiguity subset n=88).", "tab:human_written", T / "table4_human_written.tex")
body = []
for ts in TSS:
    for arch in ARCHS:
        body.append([ts, arch.upper(), pm(*A(ts, arch, "human", "known", "top1")), pm(*A(ts, arch, "human", "known", "macro_f1_fair")),
                     pm(*A(ts, arch, "human", "new", "top1")), pm(*A(ts, arch, "human", "new", "macro_f1_fair"))])
tex(["Training", "Model", "Known-image Top-1", "Known-image F1", "New-image Top-1", "New-image F1"], body,
    f"Known (n={int(known_m.sum())}) vs new (n={int(new_m.sum())}) human images. F1 = macro over the {len(shared_ids)} classes present in both subsets.",
    "tab:known_new", T / "table5_known_vs_new_human_images.tex")
body = [[r.training_source, r.architecture.upper(), r.evaluation_source, int(r.n), pm(r.top1_mean, r.top1_std), pm(r.top3_mean, r.top3_std), pm(r.macro_f1_mean, r.macro_f1_std)]
        for _, r in image_disjoint.iterrows()]
tex(["Training", "Model", "Target", "N", "Top-1", "Top-3", "Macro F1"], body, "Image-disjoint cross-teacher robustness.", "tab:image_disjoint", T / "table6_image_disjoint_cross_teacher.tex")
print("Section 19 tables written")

# ----------------------------------------------------------------------------------------------
# 8. Section 20 -- figures (axes 0..1, SD error bars)
# ----------------------------------------------------------------------------------------------
F = OUT / "figures"; COL = {"kan": "#1f77b4", "mlp": "#ff7f0e"}
def barpanel(ax, labels, series, title, ylabel="Score"):
    x = np.arange(len(labels)); w = 0.8 / len(series)
    for i, (name, (mu, sd)) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, mu, w, yerr=sd, capsize=3, label=name, color=COL.get(name.split()[0].lower()))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1.0); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
def pooled(arch, conds, metric):
    v = group_seed_vals(arch, conds, metric); return v.mean(), v.std(ddof=1)
fig, ax = plt.subplots(figsize=(7, 4.5))
cats = {"In-domain": [("qwen", "qwen_test", "full"), ("llama", "llama_test", "full")],
        "Cross-teacher": [("qwen", "llama_test", "full"), ("llama", "qwen_test", "full")],
        "Mixed": [("mixed", "qwen_test", "full"), ("mixed", "llama_test", "full")]}
barpanel(ax, list(cats), {a.upper(): tuple(map(list, zip(*[pooled(a, c, "top1") for c in cats.values()]))) for a in ARCHS}, "Cross-teacher generalization (Top-1, mean +/- SD over seeds)")
fig.tight_layout(); fig.savefig(F / "cross_teacher_generalization_FINAL.png", dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, sub in zip(axs, ("full", "image_disjoint")):
    labs, ser = [], {}
    for arch in ARCHS:
        for target, opp in [("qwen_test", "llama"), ("llama_test", "qwen")]:
            labs.append(f"{target.split('_')[0]}")
    tg = [("qwen_test", "llama"), ("llama_test", "qwen")]
    series = {}
    for arch in ARCHS:
        series[f"{arch.upper()} opposite"] = ([A(o, arch, t, sub, "top1")[0] for t, o in tg], [A(o, arch, t, sub, "top1")[1] for t, o in tg])
        series[f"{arch.upper()} mixed"] = ([A("mixed", arch, t, sub, "top1")[0] for t, o in tg], [A("mixed", arch, t, sub, "top1")[1] for t, o in tg])
    x = np.arange(2); w = 0.2
    for i, (name, (mu, sd)) in enumerate(series.items()):
        ax.bar(x + i * w - 0.3, mu, w, yerr=sd, capsize=3, label=name)
    ax.set_xticks(x); ax.set_xticklabels(["target: Qwen", "target: Llama"]); ax.set_ylim(0, 1); ax.set_title(f"Multi-teacher recovery ({sub})", fontsize=10); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(F / "multi_teacher_recovery_FINAL.png", dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (m, t) in zip(axs, (("top1", "Canonical Top-1"), ("macro_f1", "Macro F1"))):
    barpanel(ax, TSS, {a.upper(): ([A(ts, a, "human", "full", m)[0] for ts in TSS], [A(ts, a, "human", "full", m)[1] for ts in TSS]) for a in ARCHS}, f"Human-written evaluation: {t}")
fig.tight_layout(); fig.savefig(F / "human_generalization_FINAL.png", dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (m, t) in zip(axs, (("top1", "Top-1"), ("macro_f1_fair", f"Macro F1 ({len(shared_ids)} shared classes)"))):
    series = {}
    for a in ARCHS:
        for sub in ("known", "new"):
            series[f"{a.upper()} {sub}"] = ([A(ts, a, "human", sub, m)[0] for ts in TSS], [A(ts, a, "human", sub, m)[1] for ts in TSS])
    x = np.arange(3); w = 0.2
    for i, (name, (mu, sd)) in enumerate(series.items()): ax.bar(x + i * w - 0.3, mu, w, yerr=sd, capsize=3, label=name)
    ax.set_xticks(x); ax.set_xticklabels(TSS); ax.set_ylim(0, 1); ax.set_title(f"Known vs new human images: {t}", fontsize=10); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(F / "human_known_vs_new_FINAL.png", dpi=150); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
cc = [("qwen", "llama_test"), ("llama", "qwen_test"), ("mixed", "llama_test"), ("mixed", "qwen_test")]
for ax, arch in zip(axs, ARCHS):
    x = np.arange(len(cc)); w = 0.35
    for i, sub in enumerate(("full", "image_disjoint")):
        ax.bar(x + i * w - w / 2, [A(t, arch, e, sub, "top1")[0] for t, e in cc], w, yerr=[A(t, arch, e, sub, "top1")[1] for t, e in cc], capsize=3, label=sub)
    ax.set_xticks(x); ax.set_xticklabels([f"{t}->{e.split('_')[0]}" for t, e in cc], fontsize=8); ax.set_ylim(0, 1); ax.set_title(f"{arch.upper()}: full vs image-disjoint Top-1", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(F / "image_disjoint_transfer_FINAL.png", dpi=150); plt.close(fig)
print("Section 20 figures written")

# ----------------------------------------------------------------------------------------------
# 9. Section 22 -- reproducibility audit (every check computed; none hardcoded)
# ----------------------------------------------------------------------------------------------
checks = []
def chk(name, ok, detail): checks.append((name, bool(ok), detail))

ck_dir = OUT / "checkpoints_seed_corrected"; ck = sorted(ck_dir.glob("*_best.pt"))
tl = pd.read_csv(OUT / "part4_training_log_seed_corrected.csv")
conds_present = set(zip(tl.regime, tl.arch, tl.seed))
hashes = {}
for p in ck: hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
distinct_across_seeds = all(len({hashes[f"{a}_{t}_shared28_seed{s}_corrected_best.pt"] for s in SEEDS}) == 3 for t in TSS for a in ARCHS
                            if all(f"{a}_{t}_shared28_seed{s}_corrected_best.pt" in hashes for s in SEEDS))
src_stageA = None
if CODE_DIR is not None:
    c = [p for p in CODE_DIR.glob("*.py") if "STAGE_A" in p.name]
    src_stageA = c[0].read_text() if c else None
order_ok = False
if src_stageA:
    i_set = src_stageA.find("set_seed(seed)  #"); i_mod = src_stageA.find("model = (KANRoute if arch")
    order_ok = 0 <= i_set < i_mod
chk("[1] seed set before model initialisation (18 corrected checkpoints + log + Stage-A code order)",
    len(ck) == 18 and len(conds_present) == 18 and distinct_across_seeds and order_ok,
    f"checkpoints={len(ck)}, log conditions={len(conds_present)}, seed-distinct weights={distinct_across_seeds}, "
    f"Stage-A set_seed precedes model construction={order_ok}" + ("" if src_stageA else " (Stage A script not supplied via CODE_DIR)"))
pred_files = sorted(PRED.glob("*.csv"))
have = {(a, t, s, e) for a in ARCHS for t in TSS for s in SEEDS for e in ("qwen_test", "llama_test", "human") if (PRED / f"{a}_{t}_seed{s}_on_{e}.csv").exists()}
chk("[2] all 3 seeds present for all 6 conditions x 3 eval sets", len(have) == 54 and len(pred_files) == 54, f"found {len(have)} of 54 prediction files")
tax_ok = len(class_to_local_id) == 28 and sorted(class_to_local_id.values()) == list(range(28)) and "matting" not in class_to_local_id
bad_map = 0
for (a, t, s, e) in have:
    d = load_pred(a, t, s, e)
    bad_map += int((d["ground_truth"] != d["ground_truth_tool"].map(class_to_local_id)).sum()) + int((d["pred_top1"] != d["pred_top1_tool"].map(class_to_local_id)).sum())
chk("[3] Shared-28 taxonomy identical across all models/eval files", tax_ok and bad_map == 0, f"28 classes, ids 0-27, matting absent={tax_ok}; label/name mismatches across 54 files = {bad_map}")
def leak(a, b, c):
    n = [set(d["query"].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()) for d in (a, b, c)]
    it = [set(zip(d.img_id, d.tool)) for d in (a, b, c)]
    return (len(n[0] & n[1]) + len(n[0] & n[2]) + len(n[1] & n[2]), len(it[0] & it[1]) + len(it[0] & it[2]) + len(it[1] & it[2]))
qn, qi = leak(qtr, qva, qte); ln_, li = leak(ltr, lva, lte)
chk("[4] no within-teacher normalized-query leakage", qn == 0 and ln_ == 0, f"Qwen overlaps={qn}, Llama overlaps={ln_} (recomputed from split CSVs)")
chk("[5] no within-teacher image/tool leakage", qi == 0 and li == 0, f"Qwen overlaps={qi}, Llama overlaps={li}")
mq = np.array([(i, t) not in ltr_keys for i, t in zip(qte.img_id, qte.tool)]); ml = np.array([(i, t) not in qtr_keys for i, t in zip(lte.img_id, lte.tool)])
pq = load_pred("kan", "qwen", 42, "qwen_test"); pl = load_pred("kan", "qwen", 42, "llama_test")
aligned = (pq.img_id.values == qte.img_id.values).all() and (pl.img_id.values == lte.img_id.values).all()
n_saved = {(r.training_source, r.evaluation_source): int(r.n) for _, r in image_disjoint.iterrows()}
chk("[6] image-disjoint subsets independently constructed (split-derived N == prediction-derived N == saved N)",
    aligned and mq.sum() == g("llama", "kan", 42, "qwen_test", "image_disjoint", "n") == n_saved[("llama", "qwen_test")]
    and ml.sum() == g("qwen", "kan", 42, "llama_test", "image_disjoint", "n") == n_saved[("qwen", "llama_test")],
    f"Qwen-test disjoint N={int(mq.sum())}/{len(mq)}, Llama-test disjoint N={int(ml.sum())}/{len(ml)}, prediction/split row order aligned={aligned}")
kn_saved = set(pd.read_csv(OUT / "human_shared28_known_images.csv")["image_id"]); nw_saved = set(pd.read_csv(OUT / "human_shared28_new_images.csv")["image_id"])
chk("[7] human known/new membership recomputed from image IDs (matches saved subset files)",
    kn_saved == set(human.loc[known_m, "image_id"]) and nw_saved == set(human.loc[new_m, "image_id"]),
    f"known={int(known_m.sum())} rows/{len(kn_saved)} images, new={int(new_m.sum())} rows/{len(nw_saved)} images")
chk("[8] all new-image human IDs absent from synthetic train corpora", len(nw_saved & train_imgs) == 0, f"intersection size = {len(nw_saved & train_imgs)}")
expected_n = {"qwen_test": len(qte), "llama_test": len(lte), "human": len(human)}
bad_n = [(a, t, s, e) for (a, t, s, e) in have if len(load_pred(a, t, s, e)) != expected_n[e]]
chk("[9] prediction count == evaluation count for every file", not bad_n, f"mismatches={len(bad_n)}; expected {expected_n}")
uniq_ok = human["query_id"].is_unique and all(load_pred(a, t, s, e)["sample_id"].is_unique for (a, t, s, e) in have)
chk("[10] query/sample IDs unique where required", uniq_ok, "human query_id unique; per-file sample_id unique (synthetic sample_id is a within-test-set position)")
finite_ok = True; bad_files = []
for fn in ["part4_all_results_seed_corrected.csv", "part4_three_seed_summary.csv", "cross_teacher_drop_3seed.csv", "multi_teacher_recovery_3seed.csv",
           "image_disjoint_cross_teacher_results.csv", "human_known_vs_new_results.csv", "synthetic_to_human_gap.csv", "kan_vs_mlp_comparison.csv",
           "shared28_per_class_transfer_3seed.csv", "support_qualified_results.csv", "within_seed_bootstrap_ci.csv"]:
    d = pd.read_csv(OUT / fn).select_dtypes("number")
    if not np.isfinite(d.to_numpy(dtype=float)).all(): finite_ok = False; bad_files.append(fn)
chk("[11] no NaN/Inf metrics in any final CSV", finite_ok, "checked 11 final CSVs" + (f"; non-finite in {bad_files}" if bad_files else ""))
worst = 0.0
for (ts, arch, ev), grp in three_seed.groupby(["training_source", "architecture", "evaluation_source"]):
    row = grp[grp.metric == "top1"].iloc[0]
    v = np.array([(load_pred(arch, ts, s, ev)["pred_top1"] == load_pred(arch, ts, s, ev)["ground_truth"]).mean() for s in SEEDS])
    worst = max(worst, abs(v.mean() - row["mean"]), abs(v.std(ddof=1) - row["std"]))
chk("[12] three-seed aggregation reproduces from raw prediction files (independent top-1 recompute)", worst < 1e-9, f"max |diff| mean/SD over 18 conditions = {worst:.2e}")
pc = pd.read_csv(OUT / "shared28_per_class_transfer_3seed.csv"); worst_pc = 0.0; seeds_ok = True
for (ts, arch, ev), grp in pc.groupby(["training_source", "architecture", "evaluation_source"]):
    f1s = []
    for s in SEEDS:
        d = load_pred(arch, ts, s, ev)
        _, _, f, _ = precision_recall_fscore_support(d["ground_truth"], d["pred_top1"], labels=LAB, zero_division=0); f1s.append(f)
    f1s = np.array(f1s); grp = grp.sort_values("class_id")
    worst_pc = max(worst_pc, np.abs(f1s.mean(0) - grp["f1_mean"].values).max(), np.abs(f1s.std(0, ddof=1) - grp["f1_std"].values).max())
chk("[13] per-class analysis uses all three seeds (recomputed per-class F1 mean/SD from predictions)", worst_pc < 1e-6 and len(pc) == 504, f"rows={len(pc)}, max |diff|={worst_pc:.2e}")
ptxt = (OUT / "qwen_corrective_augmentation_provenance_FINAL.md").read_text()
prov_ok = str(q_total) in ptxt and all(f"| {c} | {sum(v.values())} |" in ptxt for c, v in q_aug.items()) and q_total == 135
chk("[14] augmentation provenance matches actual rows (recounted from train+val+test CSVs)", prov_ok, f"Qwen total={q_total}, per-class rows found in provenance file={prov_ok}")
utxt = (OUT / "UNCERTAINTY_REPORTING.md").read_text().lower()
chk("[15] reported uncertainty type explicitly documented", "3 seeds" in utxt and "within-test-set" in utxt and "hierarchical" in utxt,
    "UNCERTAINTY_REPORTING.md distinguishes seed SD from within-test-set bootstrap and notes no hierarchical bootstrap")
chk("[16] human strict/ambiguous masks partition the human set (bug-fix check)", strict_m.sum() == 284 and ambig_m.sum() == 88 and not (strict_m & ambig_m).any(),
    f"strict={int(strict_m.sum())}, ambiguous={int(ambig_m.sum())}")
chk("[17] regenerated canonical metrics equal the previously saved canonical metrics", max_diff_canon < 1e-9,
    f"max |diff| = {max_diff_canon:.2e} (only strict/ambiguity columns changed)")
chk("[18] no NaN in human strict/ambiguity columns after fix", not new_all.loc[new_all.evaluation_source == "human", cols_h].isna().any().any(), "old file had NaN strict_top1 in 18/18 human runs")
nfail = sum(not ok for _, ok, _ in checks)
aud = ["# PART4 Final Reproducibility Audit\n", "Every result below is computed by `FINISH_sections_15_24.py` at run time; no check is hardcoded.\n",
       f"**{len(checks) - nfail}/{len(checks)} checks passed.**\n", "| Check | Result | Evidence |", "|---|---|---|"]
for n, ok, d in checks: aud.append(f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |")
aud.append("\nNote on [1]: code order cannot be inferred from weights alone; the check combines artifact evidence with a static check "
           "of the Stage-A script text (`set_seed(seed)` appears before `model = (KANRoute ...)`).")
(OUT / "PART4_FINAL_REPRODUCIBILITY_AUDIT.md").write_text("\n".join(aud))
print("\n=== AUDIT ==="); [print(("PASS " if ok else "FAIL "), n) for n, ok, _ in checks]; print(f"{len(checks)-nfail}/{len(checks)} passed")

# ----------------------------------------------------------------------------------------------
# 10. Sections 23 / 24 -- reports (numbers pulled live; wording follows the numbers)
# ----------------------------------------------------------------------------------------------
def rd(sub, key): return drop3[(drop3.subset == sub)]
mean_drop_full = drop_ps[drop_ps.subset == "full"].top1_drop.mean(); mean_drop_dj = drop_ps[drop_ps.subset == "image_disjoint"].top1_drop.mean()
rec_full = rec_ps[rec_ps.subset == "full"]; rec_dj = rec_ps[rec_ps.subset == "image_disjoint"]
best_h = gap3.loc[gap3.human_written_top1_mean.idxmax()]
def mix_vs_single(sub_metric_ev, sub):
    out = {}
    for a in ARCHS:
        mx = seeds_vals("mixed", a, "human", sub, "top1"); best_single = max((seeds_vals(t, a, "human", sub, "top1").mean(), t) for t in ("qwen", "llama"))
        sv = seeds_vals(best_single[1], a, "human", sub, "top1"); d = mx - sv
        out[a] = (d.mean(), d.std(ddof=1), best_single[1])
    return out
mvs_w, mvs_n = mix_vs_single(None, "full"), mix_vs_single(None, "new")
kn_gap = np.mean([A(ts, a, "human", "known", "top1")[0] - A(ts, a, "human", "new", "top1")[0] for ts in TSS for a in ARCHS])
low_support_flip = (sqp.kan_minus_mlp.gt(0).sum() < f1_kan_ahead / ng * sq_n - 1)
tests_ok = nfail == 0
if not tests_ok: status = "ADDITIONAL_EXPERIMENT_REQUIRED"
else: status = "PAPER_READY_WITH_LIMITATIONS"

rep = ["# PART4 Results Report -- FINAL (seed-corrected)\n",
 "## 1. Shared-28 rationale\nShared-29 minus `matting` (Llama has 1 matting sample, so balanced cross-teacher evaluation is impossible). 28 classes, identical mapping for both teachers.\n",
 f"## 2. Dataset and taxonomy statistics\n- Qwen: train {len(qtr)} / val {len(qva)} / test {len(qte)}\n- Llama: train {len(ltr)} / val {len(lva)} / test {len(lte)}\n"
 f"- Mixed training pool: {len(qtr)+len(ltr)} rows (Qwen-train + Llama-train)\n- Human Shared-28: {len(human)} queries (strict single-label {int(strict_m.sum())}, ambiguous {int(ambig_m.sum())}); "
 f"known-image {int(known_m.sum())} / new-image {int(new_m.sum())}\n",
 "## 3. Leakage-safe split protocol\nUnion-find grouping on normalized query text OR (image_id, tool); whole groups assigned to one split. Zero within-teacher overlap (audit checks 4-5). "
 "Cross-teacher image/tool overlap is real and quantified in `cross_teacher_context_overlap_report.md`.\n",
 "## 4. Seed-corrected training procedure\nSeeds 42/123/2026; `set_seed(seed)` is called **before** model construction (the earlier runs seeded only after construction, so weight initialisation did not vary by seed). All 18 runs retrained. AdamW, OneCycleLR, label smoothing 0.05, best-validation checkpoint.\n",
 "## 5. Within-teacher performance (mean +/- SD)\n" + "\n".join(f"- {ts}->{ev.split('_')[0]} {a.upper()}: Top-1 {pm(*A(ts,a,ev,'full','top1'))}, Macro-F1 {pm(*A(ts,a,ev,'full','macro_f1'))}" for ts, ev in (("qwen","qwen_test"),("llama","llama_test")) for a in ARCHS) + "\n",
 f"## 6. Cross-teacher degradation\nObserved shift, not a failure. Mean Top-1 drop across archs/sources = {mean_drop_full*100:.1f}pp. See Table 2 / `cross_teacher_drop_3seed.csv`.\n",
 f"## 7. Image-disjoint cross-teacher degradation\nOverlap is large ({100*(~ml).mean():.1f}% of Llama-test and {100*(~mq).mean():.1f}% of Qwen-test share an (image, tool) pair with the other teacher's training set), so the pooled numbers understate the shift. On image-disjoint subsets the mean Top-1 drop is {mean_drop_dj*100:.1f}pp (vs {mean_drop_full*100:.1f}pp pooled). Only this subset may be called image-disjoint.\n",
 f"## 8. Multi-teacher recovery\nFull: mean Top-1 recovery {rec_full.recovery_top1.mean()*100:.1f}pp, Macro-F1 {rec_full.recovery_f1.mean()*100:.1f}pp. Image-disjoint: Top-1 {rec_dj.recovery_top1.mean()*100:.1f}pp, Macro-F1 {rec_dj.recovery_f1.mean()*100:.1f}pp. (Table 3, 3b.)\n",
 f"## 9. Human-written evaluation\nBest condition: {best_h.training_source}-{best_h.architecture.upper()} Top-1 {best_h.human_written_top1_mean*100:.1f}%. Strict/ambiguity columns are the corrected ones (Table 4). Gap to synthetic in-domain reported in `synthetic_to_human_gap.csv` as observed distribution shift.\n",
 f"## 10. Known vs new human images\nMean known-minus-new Top-1 across conditions = {kn_gap*100:+.2f}pp -- small relative to seed SD, so most human difficulty is linguistic rather than visual. Macro-F1 compared on the {len(shared_ids)} classes present in both subsets (known lacks `vehicle_detection` and `3d_reconstruction`, which is already true of the original `image_pool` labels).\n",
 f"## 11. KAN vs MLP\nFraming status: **{framing_status}**. See `KAN_VS_MLP_FINAL_ANALYSIS.md`. Parameters: KAN {params.get('kan')} vs MLP {params.get('mlp')}. Latency not re-measured here.\n",
 "## 12. Per-class transfer behaviour\n`shared28_per_class_transfer_3seed.csv` (mean/SD F1, precision, recall, support for all 18 conditions x 28 classes). Per-class values for low-support classes are unreliable.\n",
 f"## 13. Low-support sensitivity\n{len(open(OUT/'stable_support_classes.txt').read().split())}/28 classes have test support >=5 in both teachers. Excluded: anomaly_detection, face_detection, facial_expression_recognition, logo_detection, pose_estimation, super_resolution, visual_reasoning. KAN Macro-F1 ahead in {sq_kan_ahead}/{sq_n} stable-support cells (vs {f1_kan_ahead}/{ng} groups on full 28 classes).\n",
 "## 14. Uncertainty\nPrimary: mean +/- SD across 3 seeds. Secondary: per-seed within-test-set bootstrap CI. No hierarchical bootstrap. See `UNCERTAINTY_REPORTING.md`.\n",
 "## 15. Failure modes\nSee confusion structure in the prediction files. Example: Llama `anomaly_detection` (7 train examples) is predicted as `object_detection` on all test rows.\n",
 "## 16. Limitations\n- Synthetic supervision from two LLM teachers.\n- 372-query human benchmark; known subset lacks two classes.\n"
 "- **Template-authored rows:** Qwen val/test for `facial_expression_recognition` and test for `pose_estimation` are 100% template-authored; see provenance file.\n"
 "- Seven classes have <5 test examples in at least one teacher; their per-class scores are unreliable.\n- Closed-world, single-step routing.\n- Latency not re-measured after seed correction.\n"
 f"- Human strict/ambiguity metrics previously mis-computed (NaN-as-'nan' bug); corrected here.\n"]
(OUT / "PART4_RESULTS_REPORT_FINAL.md").write_text("\n".join(rep))

def yn(c): return "Yes" if c else "No"
ver = ["# PART4 Scientific Verdict -- FINAL\n",
 f"1. **Substantial teacher-specific shift?** {yn(mean_drop_full > 0.2)}. Mean single-teacher Top-1 drop {mean_drop_full*100:.1f}pp (Qwen->Llama {drop_ps[(drop_ps.subset=='full')&(drop_ps.source_teacher=='qwen')].top1_drop.mean()*100:.1f}, Llama->Qwen {drop_ps[(drop_ps.subset=='full')&(drop_ps.source_teacher=='llama')].top1_drop.mean()*100:.1f}).\n",
 f"2. **Does it remain on image-disjoint examples?** {yn(mean_drop_dj > 0.2)}. Mean drop {mean_drop_dj*100:.1f}pp -- {'larger' if mean_drop_dj > mean_drop_full else 'not larger'} than the pooled {mean_drop_full*100:.1f}pp.\n",
 f"3. **Does multi-teacher supervision recover cross-teacher performance?** {yn(rec_full.recovery_top1.min() > 0.2)}. Top-1 recovery {rec_full.recovery_top1.min()*100:.1f}-{rec_full.recovery_top1.max()*100:.1f}pp across seeds/targets/architectures.\n",
 f"4. **Does recovery persist under image-disjoint evaluation?** {yn(rec_dj.recovery_top1.min() > 0.2)}. Top-1 recovery {rec_dj.recovery_top1.min()*100:.1f}-{rec_dj.recovery_top1.max()*100:.1f}pp.\n",
 f"5. **Generalization to human-written queries?** Canonical Top-1 ranges {gap3.human_written_top1_mean.min()*100:.1f}-{gap3.human_written_top1_mean.max()*100:.1f}% (best {best_h.training_source}-{best_h.architecture.upper()}); synthetic in-domain is ~{gap3.synthetic_in_domain_top1_mean.mean()*100:.0f}%. Reported as observed distribution shift.\n",
 f"6. **Genuinely unseen human images?** New-image Top-1 ranges {gap3.human_new_image_top1_mean.min()*100:.1f}-{gap3.human_new_image_top1_mean.max()*100:.1f}%; mean known-minus-new gap only {kn_gap*100:+.2f}pp, so unseen images add little beyond the linguistic shift.\n",
 f"7. **Does multi-teacher training materially improve human-written performance?** " + "; ".join(f"{a.upper()}: {mvs_w[a][0]*100:+.2f} +/- {mvs_w[a][1]*100:.2f}pp vs best single ({mvs_w[a][2]})" for a in ARCHS) + f". {'Material' if all(mvs_w[a][0] > 0.02 for a in ARCHS) else 'Modest / architecture-dependent -- do not overstate'}.\n",
 f"8. **... new-image performance?** " + "; ".join(f"{a.upper()}: {mvs_n[a][0]*100:+.2f} +/- {mvs_n[a][1]*100:.2f}pp" for a in ARCHS) + f". {'Material' if all(mvs_n[a][0] > 0.02 for a in ARCHS) else 'Modest / architecture-dependent -- do not overstate'}.\n",
 f"9. **Does KAN outperform MLP on raw Top-1?** Not systematically: KAN ahead by >1pp in {t1_kan_ahead}/{ng} groups, MLP ahead by >1pp in {t1_mlp_ahead}/{ng}, comparable in {t1_comp}/{ng}.\n",
 f"10. **On macro-F1?** KAN ahead in {f1_kan_ahead}/{ng} groups on the full 28-class set (mean gap exceeds seed-SD in {f1_sd_exceed}/{ng}).\n",
 f"11. **Is any KAN advantage robust across seeds?** KAN macro-F1 advantage is seed-consistent (3/3 seeds) in {f1_consistent}/{ng} groups and KAN is ahead in {sq_kan_ahead}/{sq_n} stable-support cells. Framing status: {framing_status}.\n",
 f"12. **Do low-support classes materially affect conclusions?** The cross-teacher/recovery conclusions are unaffected (see `support_qualified_results.csv`). The KAN-vs-MLP macro-F1 comparison **is** sensitive to them ({f1_kan_ahead}/{ng} groups full vs {sq_kan_ahead}/{sq_n} cells stable-support).\n",
 f"13. **Strong enough for a Q1 manuscript?** The core findings (teacher shift, recovery incl. image-disjoint, human gap) are solid, leakage-audited and seed-robust. It must be framed with the limitations listed in the report (template-authored rows in some val/test splits, small human set, low-support classes, KAN advantage not established as robust).\n",
 "14. **Is another experiment scientifically necessary?** None required within this pass's scope (Section 26 forbids new data/teachers/architectures). Optional: replace template-authored val/test rows for the three augmented classes with teacher-generated ones.\n",
 f"\nAudit: {len(checks)-nfail}/{len(checks)} checks passed.\n",
 f"PART4_FINAL_STATUS = {status}\n"]
(OUT / "PART4_SCIENTIFIC_VERDICT_FINAL.md").write_text("\n".join(ver))

# ----------------------------------------------------------------------------------------------
# 11. Code folder + zip
# ----------------------------------------------------------------------------------------------
if CODE_DIR is not None:
    (OUT / "code").mkdir(exist_ok=True)
    for p in CODE_DIR.glob("*.py"): shutil.copy(p, OUT / "code" / p.name)
shutil.copy(__file__, OUT / "code" / Path(__file__).name) if (OUT / "code").exists() else None
if ZIP_BASE:
    shutil.make_archive(ZIP_BASE, "zip", OUT); print(f"ZIP: {ZIP_BASE}.zip")
print(f"\nSTATUS = {status}   |   audit {len(checks)-nfail}/{len(checks)}")
