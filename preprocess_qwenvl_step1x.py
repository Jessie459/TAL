import argparse
import ast
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from src.step1x_edit.utils import resize_image, set_seed, str2bool, tensors_to_device

os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

TASK_TYPES = [
    "subject_add",
    "subject_remove",
    "subject_replace",
    "background_change",
    "color_alter",
    "material_alter",
    "text_change",
    "position_change",
    "count_change",
]


def create_pipeline():
    model_configs = [
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="vae.safetensors"),
        ModelConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct", origin_file_pattern="model-*.safetensors"),
    ]
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
    )
    return pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATASET_PATH", "data/EdiVal"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="info")
    return parser.parse_args()


def process_single_image(pipe, row, args):
    image_index = row["image_index"]
    image_path = os.path.join(args.data_root, "images", f"{image_index}.png")
    with Image.open(image_path) as image:
        edit_image = resize_image(image).convert("RGB")
    instruction = ast.literal_eval(row["instructions"])[0]
    if instruction[0] == "\"" and instruction[-1] == "\"":
        instruction = instruction[1:-1]

    inputs_posi, inputs_nega = pipe(
        prompt=instruction,
        step1x_reference_image=edit_image,
        width=edit_image.width,
        height=edit_image.height,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device="cuda",
        preprocess=True,
    )
    inputs_posi.pop("prompt")
    inputs_posi.pop("tag")
    inputs_nega.pop("negative_prompt")
    inputs_nega.pop("tag")

    tensors_to_device(inputs_posi, "cpu")
    tensors_to_device(inputs_nega, "cpu")
    torch.save(inputs_posi, os.path.join(args.output_dir, f"{image_index}_posi.pt"))
    torch.save(inputs_nega, os.path.join(args.output_dir, f"{image_index}_nega.pt"))


def main():
    args = parse_args()
    set_seed(args.seed)

    pipe = create_pipeline()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "qwenvl", "step1x_edit")

    df = pd.read_csv(os.path.join(args.data_root, "metadata.csv"))
    df = df[df["turns"] == 1]

    os.makedirs(args.output_dir, exist_ok=True)

    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        process_single_image(pipe=pipe, row=row, args=args)


if __name__ == "__main__":
    main()
