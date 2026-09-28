import argparse
import ast
import json
import logging
import math
import os
import warnings

import pandas as pd
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

from src.edival.consistency_detector import evaluate_consistency, load_consistency_model
from src.edival.instruction_detector import evaluate_instruction_following, load_instruction_model
from src.edival.utils import _make_json_serializable, resolve_model_path

warnings.filterwarnings("ignore")
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

logger = logging.getLogger(__name__)

TASK_TYPES = [
    "subject_add",
    "subject_remove",
    "subject_replace",
    "color_alter",
    "material_alter",
    "text_change",
    "position_change",
    "count_change",
    "background_change",
]
CONSISTENCY_METRICS = [
    "object_dinov3_consistency",
    "object_l1_consistency",
    "background_dinov3_consistency",
    "background_l1_consistency",
]


def parse_args():
    parser = argparse.ArgumentParser(description="EdiVal instruction following (IF), content consistency (CC) and overall (O).")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATASET_PATH", "data/EdiVal"))
    parser.add_argument("--gen_dir", type=str, required=True, help="Folder with <image_index>_turn1.png edits.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <gen_dir>_eval.")
    parser.add_argument("--edival_task", type=str, default=None, choices=TASK_TYPES)
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="info")
    return parser.parse_args()


def load_metadata(data_root, edival_task=None):
    df = pd.read_csv(os.path.join(data_root, "metadata.csv"))
    df = df[df["turns"] == 1]
    if edival_task:
        df = df[df["task_type"].apply(lambda x: ast.literal_eval(x)[0]) == edival_task]
    return df


def load_models(vlm, max_memory=None):
    grounding_dino_weights = resolve_model_path("ShilongLiu/GroundingDINO", "groundingdino_swint_ogc.pth")
    grounding_model, vlm_model = load_instruction_model(grounding_dino_weights, resolve_model_path(vlm), max_memory)
    dinov3_model = load_consistency_model(resolve_model_path("facebook/dinov3-vitb16-pretrain-lvd1689m"))
    return grounding_model, vlm_model, dinov3_model


def parse_instruction(row):
    return {
        "format_instruction": ast.literal_eval(row["format_instructions"])[0],
        "instruction": ast.literal_eval(row["instructions"])[0],
        "task_type": ast.literal_eval(row["task_type"])[0],
        "unchanged_objects": ast.literal_eval(row["unchanged_objects"]),
        "all_objects": ast.literal_eval(row["all_objects"]),
        "eval_bg_consistency": bool(row.get("bg_consistency", False)),
    }


def evaluate_sample(models, src_path, gen_path, inst):
    grounding_model, vlm_model, dinov3_model = models
    score, reason = evaluate_instruction_following(
        src_path,
        gen_path,
        formatted_instruction=inst["format_instruction"],
        instruction=inst["instruction"],
        task_type=inst["task_type"],
        grounding_model=grounding_model,
        vlm_model=vlm_model,
        output_reason=True,
    )
    result = {
        "instruction": inst["instruction"],
        "format_instruction": inst["format_instruction"],
        "task_type": inst["task_type"],
        "instruction_following": score,
        "instruction_following_reason": reason,
        **{k: None for k in CONSISTENCY_METRICS},
        "object_details": None,
        "bg_details": None,
    }
    if inst["eval_bg_consistency"]:
        object_result, background_result = evaluate_consistency(
            src_path,
            gen_path,
            inst["unchanged_objects"],
            inst["all_objects"],
            grounding_model=grounding_model,
            dinov3_model=dinov3_model,
        )
        result.update(
            {
                "object_dinov3_consistency": object_result["object_dinov3_consistency_mean"],
                "object_l1_consistency": object_result["object_l1_consistency_mean"],
                "background_dinov3_consistency": background_result.get("bg_dinov3_masked_similarity"),
                "background_l1_consistency": background_result["bg_l1_consistency"],
                "object_details": object_result,
                "bg_details": background_result,
            }
        )
    return result


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def summarize(df):
    """EdiVal-CC averages object and background consistency (each the mean of DINOv3 and L1);
    EdiVal-O is the geometric mean of IF and CC. IF/CC/O are reported in percent, PQ on a 0-10 scale."""
    rows = []
    groups = [("all", df)] + [(t, df[df["task_type"] == t]) for t in TASK_TYPES if (df["task_type"] == t).any()]
    for name, group in groups:
        row = {"task_type": name, "num_samples": len(group)}
        row["instruction_following"] = _mean(group["instruction_following"].tolist())
        for k in CONSISTENCY_METRICS:
            row[k] = _mean(group[k].tolist())
        row["object_consistency"] = _mean([row["object_dinov3_consistency"], row["object_l1_consistency"]])
        row["background_consistency"] = _mean([row["background_dinov3_consistency"], row["background_l1_consistency"]])
        row["content_consistency"] = _mean([row["object_consistency"], row["background_consistency"]])
        if row["instruction_following"] is None or row["content_consistency"] is None:
            row["overall"] = None
        else:
            row["overall"] = math.sqrt(row["instruction_following"] * row["content_consistency"])
        for k in ["instruction_following", *CONSISTENCY_METRICS, "object_consistency", "background_consistency", "content_consistency", "overall"]:
            row[k] = None if row[k] is None else row[k] * 100
        if "score_naturalness" in group:
            row["perceptual_quality"] = _mean(group["score_naturalness"].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")
    transformers_logging.set_verbosity_error()
    output_dir = args.output_dir or args.gen_dir.rstrip("/") + "_eval"
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    df = load_metadata(args.data_root, args.edival_task)
    generated = {int(f[: -len("_turn1.png")]) for f in os.listdir(args.gen_dir) if f.endswith("_turn1.png")}
    missing = sorted(set(df["image_index"]) - generated)
    if missing:
        logger.warning(f"Skipping {len(missing)} samples without a generated image, e.g. {missing[:5]}")
    df = df[df["image_index"].isin(generated)]
    logger.info(f"Evaluating {len(df)} samples from {args.gen_dir}")

    models = None
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_index = int(row["image_index"])
        sample_path = os.path.join(output_dir, "samples", f"{image_index}.json")
        if os.path.exists(sample_path):
            with open(sample_path) as f:
                result = json.load(f)
        else:
            if models is None:
                models = load_models(args.vlm, args.max_memory)
            src_path = os.path.join(args.data_root, "images", f"{image_index}.png")
            gen_path = os.path.join(args.gen_dir, f"{image_index}_turn1.png")
            result = evaluate_sample(models, src_path, gen_path, parse_instruction(row))
            with open(sample_path, "w") as f:
                json.dump(_make_json_serializable(result), f, indent=2)
        records.append({"image_index": image_index, **{k: result[k] for k in ["task_type", "instruction_following", *CONSISTENCY_METRICS]}})

    records = pd.DataFrame(records)
    pq_path = os.path.join(output_dir, "pq.csv")
    if os.path.exists(pq_path):
        records = records.merge(pd.read_csv(pq_path)[["image_index", "score_naturalness"]], on="image_index", how="left")
    else:
        logger.info(f"No {pq_path}; run evaluate_pq.py to add perceptual quality to the summary.")
    records.to_csv(os.path.join(output_dir, "samples.csv"), index=False)

    summary = summarize(records)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    logger.info(f"Saved {os.path.join(output_dir, 'summary.csv')}\n{summary.round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
