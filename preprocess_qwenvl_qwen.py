import argparse
import ast
import json
import logging
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from src.qwen_image_edit.utils import resize_image, set_seed, tensors_to_device

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


logger = logging.getLogger(__name__)


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def process_single_image(pipe, row, args):
    image_index = row["image_index"]
    instructions = ast.literal_eval(row["instructions"])
    assert isinstance(instructions, list)
    assert len(instructions) == 1
    instruction = instructions[0]

    image_path = os.path.join(args.data_root, "images", f"{image_index}.png")
    with Image.open(image_path) as image:
        edit_image = resize_image(image).convert("RGB").copy()

    inputs_posi, inputs_nega = pipe(
        prompt=instruction,
        negative_prompt="",
        edit_image=edit_image,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        height=edit_image.height,
        width=edit_image.width,
        edit_image_auto_resize=False,
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


def create_pipeline(args):
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    return pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATASET_PATH", "data/EdiVal"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="info")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=4)}")

    set_seed(args.seed)

    pipe = create_pipeline(args)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "qwenvl", "qwen_image_edit")

    df = pd.read_csv(os.path.join(args.data_root, "metadata.csv"))
    df = df[df["turns"] == 1]
    logger.info(f"Number of rows: {len(df)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        process_single_image(pipe=pipe, row=row, args=args)

    logger.info("Generation complete.")


if __name__ == "__main__":
    main()
