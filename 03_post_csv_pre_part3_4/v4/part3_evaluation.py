"""Part 3 human evaluation — PREPARED, NOT EXECUTED per Item 18."""
import pandas as pd

def strict_single_label_subset(df):
    no_multi = df["acceptable_tool_labels"].astype(str).str.strip().isin(["", "nan"])
    return df[(df["strict_eval"]==True) & no_multi].copy()

def ambiguity_aware_subset(df):
    has_multi = ~df["acceptable_tool_labels"].astype(str).str.strip().isin(["", "nan"])
    return df[(df["strict_eval"]==True) & has_multi].copy()

def score_with_alignment(model_preds_df, eval_df):
    merged = eval_df.merge(model_preds_df, on="query_id", validate="one_to_one")
    assert len(merged) == len(eval_df), "Prediction alignment mismatch!"
    return merged

def ambiguity_aware_score(merged_df):
    correct1 = merged_df.apply(lambda r: r["predicted_tool"] in [x.strip() for x in str(r["acceptable_tool_labels"]).split(",")], axis=1)
    correct3 = merged_df.apply(lambda r: len(set(r["top3_predictions"]) & set(x.strip() for x in str(r["acceptable_tool_labels"]).split(","))) > 0, axis=1)
    return dict(acceptable_top1=correct1.mean(), acceptable_top3=correct3.mean())

def stratified_visual_context_eval(preds_df, eval_df):
    merged = score_with_alignment(preds_df, eval_df)
    results = {}
    for vc in ["yes", "no"]:
        sub = merged[merged["routing_requires_visual_context"]==vc]
        results[vc] = dict(n=len(sub), top1=(sub["predicted_tool"]==sub["tool_name"]).mean())
    return results

# Execution entry point (NOT called automatically)
if __name__ == "__main__":
    print("This script is prepared but not executed. Load models and human_df, then call:")
    print("  strict_single_label_subset(), ambiguity_aware_subset(), score_with_alignment(),")
    print("  ambiguity_aware_score(), stratified_visual_context_eval()")
