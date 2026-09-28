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
from src.qwen_image_edit.manager import MANAGER
from src.qwen_image_edit.utils import resize_image, set_seed, str2bool

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


INSTRUCTION_TEMPLATES = {
    "subject_add": "Add [object] on the [position] of [reference object]",
    "subject_remove": "Remove [object]",
    "subject_replace": "Replace [object] with [new object]",
    "color_alter": "Change the color of [object] to [color]",
    "material_alter": "Change the material of [object] to [material]",
    "position_change": "Change the position of [object] to [position] of [reference object]",
    "count_change": "Change the count of [object] to [number]",
    "text_change": "Add text '[text]' on the image  |  Replace the text '[old text]' on [object] with '[new text]'",
    "background_change": "Change the background to [scene]",
}

logger = logging.getLogger(__name__)


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_semant_types(task_type, instruction, qwenvl_tokens):
    if task_type is None:
        return ["img", "gen"]
    if task_type == "subject_remove":
        return ["img"]
    if task_type == "subject_add":
        return ["gen"]
    if task_type == "subject_replace":
        return ["img", "gen"]
    if task_type == "text_change":
        if qwenvl_tokens is not None:
            text_tokens = qwenvl_tokens[0]
            if "Replace" in text_tokens:
                return ["img", "gen"]
            if "Add" in text_tokens:
                return ["gen"]
            raise ValueError(f"Invalid `qwenvl_tokens`: {text_tokens}")
        if instruction.startswith("Replace"):
            return ["img", "gen"]
        if instruction.startswith("Add"):
            return ["gen"]
        raise ValueError(f"Invalid instruction for `text_change`: {instruction}")
    if task_type in {"color_alter", "material_alter"}:
        return ["img"]
    if task_type in {"background_change", "position_change", "count_change"}:
        return ["img", "gen"]
    raise ValueError(f"Invalid `task_type`: {task_type}")

def check_instruction(task_type, instruction):
    words = instruction.split()
    if task_type == "subject_add":
        valid = "on" in words
    elif task_type == "subject_replace":
        valid = "with" in words
    elif task_type == "text_change":
        valid = (words[:1] == ["Add"] and "on" in words) or (words[:1] == ["Replace"] and "on" in words and "with" in words)
    elif task_type in {"color_alter", "material_alter", "position_change", "count_change"}:
        valid = "of" in words and "to" in words and words.index("of") < words.index("to")
    else:
        valid = True
    if not valid:
        raise ValueError(f"The `{task_type}` instruction must follow the template: {INSTRUCTION_TEMPLATES[task_type]}")



def create_image_sample(args):
    return {
        "sample_id": os.path.splitext(os.path.basename(args.image))[0],
        "image_path": args.image,
        "instructions": [args.instruction],
        "task_types": [args.task_type],
        "qwenvl_dir": None,
    }


def create_edival_sample(row, args):
    image_index = row["image_index"]
    instructions = ast.literal_eval(row["instructions"])
    task_types = ast.literal_eval(row["task_type"])
    assert isinstance(instructions, list) and isinstance(task_types, list)
    assert len(instructions) == len(task_types) == 1, "Only support 1 turn"

    return {
        "sample_id": str(image_index),
        "image_path": os.path.join(args.data_root, "images", f"{image_index}.png"),
        "instructions": instructions,
        "task_types": task_types,
        "qwenvl_dir": os.path.join(args.data_root, "qwenvl", "qwen_image_edit"),
    }


def process_single_image(pipe, sample, args):
    sample_id = sample["sample_id"]
    instructions = sample["instructions"]
    task_types = sample["task_types"]

    with Image.open(sample["image_path"]) as image:
        original_size = image.size
        edit_image = resize_image(image).convert("RGB").copy()
    # edit_image.save(os.path.join(args.output_dir, f"{sample_id}.png"))

    prompt_emb = None
    prompt_emb_mask = None
    negative_prompt_emb = None
    negative_prompt_emb_mask = None
    qwenvl_tokens = None
    if args.load_qwenvl:
        qwenvl_posi = torch.load(os.path.join(sample["qwenvl_dir"], f"{sample_id}_posi.pt"))
        qwenvl_nega = torch.load(os.path.join(sample["qwenvl_dir"], f"{sample_id}_nega.pt"))
        prompt_emb = qwenvl_posi["prompt_emb"].to("cuda")
        prompt_emb_mask = qwenvl_posi["prompt_emb_mask"].to("cuda")
        negative_prompt_emb = qwenvl_nega["prompt_emb"].to("cuda")
        negative_prompt_emb_mask = qwenvl_nega["prompt_emb_mask"].to("cuda")
        if MANAGER.semant:
            qwenvl_tokens = qwenvl_posi["qwenvl_tokens"]


    for turn, (instruction, task_type) in enumerate(zip(instructions, task_types), start=1):
        MANAGER.task_type = task_type
        if args.semant:
            if args.semant_types:
                MANAGER.semant_types = args.semant_types
            else:
                MANAGER.semant_types = resolve_semant_types(task_type, instruction, qwenvl_tokens)
        if args.vis:
            MANAGER.vis_dir = os.path.join(args.output_dir, f"{sample_id}_turn{turn}")
            os.makedirs(MANAGER.vis_dir, exist_ok=True)
        else:
            MANAGER.vis_dir = None  # turn off visualization


        output_image = pipe(
            prompt=instruction,
            negative_prompt="",
            edit_image=edit_image,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            height=edit_image.height,
            width=edit_image.width,
            edit_image_auto_resize=False,
            prompt_emb=prompt_emb,
            prompt_emb_mask=prompt_emb_mask,
            negative_prompt_emb=negative_prompt_emb,
            negative_prompt_emb_mask=negative_prompt_emb_mask,
            qwenvl_tokens=qwenvl_tokens,
        )


        resized_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
        resized_image.save(os.path.join(args.output_dir, f"{sample_id}_turn{turn}.png"))
        edit_image = output_image


def create_pipeline(args):
    if args.low_vram:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        model_configs = [
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config,
            ),
        ]
        if not args.load_qwenvl:
            model_configs.append(
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="text_encoder/model*.safetensors",
                    **vram_config,
                )
            )
    else:
        model_configs = [
            ModelConfig(
                model_id="Qwen/Qwen-Image-Edit",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ]
        if not args.load_qwenvl:
            model_configs.append(
                ModelConfig(
                    model_id="Qwen/Qwen-Image",
                    origin_file_pattern="text_encoder/model*.safetensors",
                )
            )

    if args.low_vram:
        if args.vram_limit is not None:
            vram_limit = args.vram_limit
        else:
            vram_limit = torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 0.5
        logger.info(f"Setting VRAM limit to {vram_limit:.2f} GB")
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
            vram_limit=vram_limit,
        )
    else:
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        )

    return pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Edit this image instead of running on EdiVal.")
    parser.add_argument("--instruction", type=str, default=None, help="Editing instruction for --image.")
    parser.add_argument("--task_type", type=str, default=None, choices=TASK_TYPES, help="Task type of --instruction; if omitted, both image streams are used.")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DATASET_PATH", "data/EdiVal"))
    parser.add_argument("--edival_task", type=str, default=None, choices=TASK_TYPES)
    parser.add_argument("--low_vram", type=str2bool, default=False)
    parser.add_argument("--vram_limit", type=float, default=None)
    parser.add_argument("--load_qwenvl", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="outputs/edival_bench/qwen_image_edit")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semant", type=str2bool, default=False)
    parser.add_argument("--semant_types", type=str, default=None, nargs="+", choices=["img", "gen"])
    parser.add_argument("--semant_steps", type=int, default=None, nargs="+")
    parser.add_argument("--semant_blocks", type=int, default=None, nargs="+")
    parser.add_argument("--semant_threshold", type=float, default=None)
    parser.add_argument("--semant_mask_type", type=str, default="feat", choices=["feat", "attn"])
    parser.add_argument("--semant_mask_refine", type=str, default="expand_foreground", choices=["expand_foreground", "shrink_background", "none"])
    parser.add_argument("--semant_mask_kernel", type=int, default=5)
    parser.add_argument("--vis", type=str2bool, default=False)
    parser.add_argument("--log_level", type=str, default="info")
    return parser.parse_args()


def normalize_args(args):
    if args.semant:
        if not args.semant_types:
            args.semant_types = []
        else:
            args.semant_types = list(dict.fromkeys(args.semant_types))
        if not args.semant_steps:
            args.semant_steps = []
        if not args.semant_blocks:
            args.semant_blocks = []
    return args


def main():
    args = parse_args()
    setup_logging(args.log_level)
    if args.image:
        if args.instruction is None:
            raise ValueError("`--image` requires `--instruction`.")
        if args.edival_task or args.load_qwenvl:
            raise ValueError("`--edival_task` and `--load_qwenvl` only apply to EdiVal runs, not to `--image`.")
        args.instruction = args.instruction.strip().strip('"')
        if args.task_type is None:
            logger.warning(
                "No `--task_type` given: localizing with both the source (`img`) and target (`gen`) image streams "
                "and attention over the whole instruction. Pass `--task_type` for task-aware localization."
            )
        else:
            check_instruction(args.task_type, args.instruction)
    args = normalize_args(args)
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=4)}")

    set_seed(args.seed)

    MANAGER.set_parameters(
        semant=args.semant,
        semant_types=args.semant_types,
        semant_steps=args.semant_steps,
        semant_blocks=args.semant_blocks,
        semant_threshold=args.semant_threshold,
        semant_mask_type=args.semant_mask_type,
        semant_mask_refine=args.semant_mask_refine,
        semant_mask_kernel=args.semant_mask_kernel,
        num_inference_steps=args.num_inference_steps,
    )

    pipe = create_pipeline(args)

    if args.image:
        samples = [create_image_sample(args)]
    else:
        metadata_path = os.path.join(args.data_root, "metadata.csv")
        logger.info(f"Loading metadata from: {metadata_path}")
        df = pd.read_csv(metadata_path)
        df = df[df["turns"] == 1]
        if args.edival_task:
            df = df[df["task_type"].apply(lambda x: ast.literal_eval(x)[0]) == args.edival_task]
        samples = [create_edival_sample(row, args) for _, row in df.iterrows()]
    logger.info(f"Number of rows: {len(samples)}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    for sample in tqdm(samples, total=len(samples)):
        process_single_image(pipe=pipe, sample=sample, args=args)

    logger.info("Generation complete.")


if __name__ == "__main__":
    main()
