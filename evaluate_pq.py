import argparse
import ast
import math
import os

import pandas as pd
import transformers
from PIL import Image
from tqdm import tqdm

from src.edival.utils import resolve_model_path
from src.viescore import VIEScore

transformers.logging.set_verbosity_error()
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Perceptual quality (PQ): VIEScore naturalness rated by Qwen2.5-VL.")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATASET_PATH", "data/EdiVal"))
    parser.add_argument("--gen_dir", type=str, required=True, help="Folder with <image_index>_turn1.png edits.")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <gen_dir>_eval.")
    parser.add_argument("--edival_task", type=str, default=None, choices=TASK_TYPES)
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    return parser.parse_args()


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return int(width), int(height)


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
    return image.resize(calculate_dimensions(512 * 512, image.width / image.height))


def main():
    args = parse_args()
    output_dir = args.output_dir or args.gen_dir.rstrip("/") + "_eval"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data_root, "metadata.csv"))
    df = df[df["turns"] == 1]
    if args.edival_task:
        df = df[df["task_type"].apply(lambda x: ast.literal_eval(x)[0]) == args.edival_task]
    df = df[df["image_index"].apply(lambda i: os.path.exists(os.path.join(args.gen_dir, f"{i}_turn1.png")))]

    vie_score = VIEScore(pretrained_model_name_or_path=resolve_model_path(args.vlm))

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_index = int(row["image_index"])
        edited = load_image(os.path.join(args.gen_dir, f"{image_index}_turn1.png"))
        result = vie_score.evaluate_PQ(edited)
        if result is None:
            print(f"Failed to get score for image index {image_index}")
            continue
        results.append({"image_index": image_index, "score_naturalness": result["score"][0]})

    results = pd.DataFrame(results, columns=["image_index", "score_naturalness"])
    save_path = os.path.join(output_dir, "pq.csv")
    results.to_csv(save_path, index=False)
    print(f"Average naturalness over {len(results)} samples: {results['score_naturalness'].mean():.4f}")
    print(f"Saved {save_path}; rerun evaluate.py with the same --gen_dir/--output_dir to add PQ to summary.csv")


if __name__ == "__main__":
    main()
