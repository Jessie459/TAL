import math
import os
from typing import Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from src.qwen_image_edit.manager import MANAGER
from src.qwen_image_edit.utils import (
    aggregate_attn,
    compute_feat_partition,
    compute_task_attn_partition,
    visualize_attn,
    visualize_binary_mask,
    visualize_partition,
)

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match_qwen_image import FlowMatchScheduler
from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE


class QwenImagePipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        from transformers import Qwen2VLProcessor

        self.scheduler = FlowMatchScheduler()
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: QwenImageVAE = None
        self.processor: Qwen2VLProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_qwen_image

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor

            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Pre-computed Qwen-VL embeddings (see `preprocess_qwenvl_qwen.py`)
        prompt_emb=None,
        prompt_emb_mask=None,
        negative_prompt_emb=None,
        negative_prompt_emb_mask=None,
        qwenvl_tokens=None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Qwen-Image-Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        # Progress bar
        progress_bar_cmd=tqdm,
        preprocess=False,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, dynamic_shift_len=(height // 16) * (width // 16))

        # Parameters
        inputs_posi = {"prompt": prompt, "tag": "posi"}
        inputs_nega = {"negative_prompt": negative_prompt, "tag": "nega"}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "edit_image": edit_image,
            "edit_image_auto_resize": edit_image_auto_resize,
        }
        if prompt_emb is not None:
            inputs_posi["prompt_emb"] = prompt_emb
            inputs_posi["prompt_emb_mask"] = prompt_emb_mask
            inputs_nega["prompt_emb"] = negative_prompt_emb
            inputs_nega["prompt_emb_mask"] = negative_prompt_emb_mask
            for unit in self.units:
                if isinstance(unit, QwenImageUnit_PromptEmbedder):
                    continue
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                    unit, self, inputs_shared, inputs_posi, inputs_nega
                )
        else:
            if preprocess:
                for unit in self.units:
                    if isinstance(unit, QwenImageUnit_PromptEmbedder):
                        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                            unit, self, inputs_shared, inputs_posi, inputs_nega
                        )
                return inputs_posi, inputs_nega
            else:
                for unit in self.units:
                    inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                        unit, self, inputs_shared, inputs_posi, inputs_nega
                    )
                qwenvl_tokens = inputs_posi["qwenvl_tokens"]
                inputs_nega.pop("qwenvl_tokens")

        if MANAGER.semant:
            MANAGER.set_tokens(qwenvl_tokens)

        latents = inputs_shared["latents"]
        edit_latents = inputs_shared["edit_latents"]
        B, _, H, W = latents.shape
        latents = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=H // 2, W=W // 2, P=2, Q=2)
        edit_latents = rearrange(edit_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=H // 2, W=W // 2, P=2, Q=2)
        img_shapes = [(B, H // 2, W // 2), (B, H // 2, W // 2)]
        inputs_shared["latents"] = latents
        inputs_shared["edit_latents"] = edit_latents
        inputs_shared["img_shapes"] = img_shapes
        noise_latents = latents.detach().clone()

        MANAGER.reset(height, width)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            MANAGER.step = progress_id
            MANAGER.img_attn.clear()
            MANAGER.gen_attn.clear()
            MANAGER.img_attn_num = 0
            MANAGER.gen_attn_num = 0
            MANAGER.img_feat.clear()
            MANAGER.gen_feat.clear()


            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep,
                progress_id=progress_id,
            )

            if MANAGER.semant and MANAGER.step in MANAGER.semant_steps:
                for semant_type in MANAGER.semant_types:
                    names = MANAGER.tokens
                    attns = MANAGER.get_attn(semant_type)
                    if MANAGER.semant_mask_type == "feat":
                        if semant_type == "gen":
                            feats = MANAGER.gen_feat
                        else:
                            feats = MANAGER.img_feat
                        assert len(feats) == 1, "Only one block is supported"
                        feats = next(iter(feats.values()))
                        edit_indices, uned_indices = compute_feat_partition(
                            names,
                            attns,
                            feats,
                            threshold=MANAGER.semant_threshold,
                            task_type=MANAGER.task_type,
                            semant_type=semant_type,
                        )
                    elif MANAGER.semant_mask_type == "attn":
                        edit_indices, uned_indices = compute_task_attn_partition(
                            names,
                            attns,
                            threshold=0.5,
                            task_type=MANAGER.task_type,
                            semant_type=semant_type,
                        )
                    else:
                        raise ValueError(f"Unsupported semant_mask_type: {MANAGER.semant_mask_type}")
                    if semant_type == "gen":
                        MANAGER.gen_edit_ids, MANAGER.gen_uned_ids = edit_indices, uned_indices
                    else:
                        MANAGER.img_edit_ids, MANAGER.img_uned_ids = edit_indices, uned_indices

                    if getattr(MANAGER, "vis_dir", None) is not None:
                        names = names[0]
                        attns = attns[0].float()
                        _edit_image = inputs_shared["edit_image"].resize((MANAGER.img_w, MANAGER.img_h))
                        _edit_image = np.array(_edit_image) / 255.0
                        _attn = aggregate_attn(names, attns, MANAGER.task_type, semant_type).cpu().numpy()
                        _attn = visualize_attn(_attn, _edit_image)
                        _path = f"step{MANAGER.step}_attn_{semant_type}.png"
                        _attn.save(os.path.join(MANAGER.vis_dir, _path))
                        _feat = visualize_partition(
                            edit_indices[0],
                            uned_indices[0],
                            MANAGER.img_h,
                            MANAGER.img_w,
                            inputs_shared["edit_image"],
                        )
                        partition_name = "feat" if MANAGER.semant_mask_type == "feat" else "attn_partition"
                        _path = f"step{MANAGER.step}_{partition_name}_{semant_type}.png"
                        _feat.save(os.path.join(MANAGER.vis_dir, _path))

                MANAGER.compute_semant_ids()

                if getattr(MANAGER, "vis_dir", None) is not None:
                    _feat = visualize_partition(
                        MANAGER.edit_ids[0],
                        MANAGER.uned_ids[0],
                        MANAGER.img_h,
                        MANAGER.img_w,
                        inputs_shared["edit_image"],
                    )
                    partition_name = "feat" if MANAGER.semant_mask_type == "feat" else "attn_partition"
                    _path = f"step{MANAGER.step}_{partition_name}.png"
                    _feat.save(os.path.join(MANAGER.vis_dir, _path))
                    _mask = visualize_binary_mask(
                        MANAGER.edit_ids[0],
                        MANAGER.uned_ids[0],
                        MANAGER.img_h,
                        MANAGER.img_w,
                        inputs_shared["edit_image"],
                    )
                    _path = f"step{MANAGER.step}_{partition_name}_mask.png"
                    _mask.save(os.path.join(MANAGER.vis_dir, _path))

            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

            if MANAGER.semant and MANAGER.step in MANAGER.semant_steps:
                sigma = self.scheduler.sigmas[MANAGER.step + 1].item() if MANAGER.step + 1 < len(self.scheduler.sigmas) else 0.0
                inv_latents = sigma * noise_latents + (1.0 - sigma) * inputs_shared["edit_latents"]
                inputs_shared["latents"] = MANAGER.post_update(inputs_shared["latents"], inv_latents)

        # Decode
        self.load_models_to_device(["vae"])
        inputs_shared["latents"] = rearrange(
            inputs_shared["latents"], "B (H W) (C P Q) -> B C (H P) (W Q)", H=H // 2, W=W // 2, P=2, Q=2
        )
        image = self.vae.decode(inputs_shared["latents"], device=self.device)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class QwenImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: QwenImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"), output_params=("latents",))

    def process(self, pipe: QwenImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise(
            (1, 16, height // 8, width // 8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"latents": noise}


class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "tag": "tag"},
            input_params_nega={"prompt": "negative_prompt", "tag": "tag"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask", "qwenvl_tokens"),
            onload_model_names=("text_encoder",),
        )

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    @staticmethod
    def extract_tokens(tokenizer, model_inputs, drop_idx):
        tokens = []
        for ids, mask in zip(model_inputs.input_ids, model_inputs.attention_mask.bool()):
            valid_ids = ids[mask]
            tokens.append(tokenizer.convert_ids_to_tokens(valid_ids))
        tokens = [t[drop_idx:] for t in tokens]
        return tokens

    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image, tag):
        template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        if tag == "posi":
            qwenvl_tokens = self.extract_tokens(pipe.processor.tokenizer, model_inputs, drop_idx)
        else:
            qwenvl_tokens = None
        return split_hidden_states, qwenvl_tokens

    def process(self, pipe: QwenImagePipeline, prompt, edit_image=None, tag=None) -> dict:
        assert isinstance(edit_image, Image.Image), "Edit image is not provided."
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            prompt = [prompt]
            split_hidden_states, qwenvl_tokens = self.encode_prompt_edit(pipe, prompt, edit_image, tag)
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
            )
            encoder_attention_mask = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
            )
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {
                "prompt_emb": prompt_embeds,
                "prompt_emb_mask": encoder_attention_mask,
                "qwenvl_tokens": qwenvl_tokens,
            }
        else:
            return {}


class QwenImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_image"),
            onload_model_names=("vae",),
        )

    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height

    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(
            1024 * 1024, edit_image.size[0] / edit_image.size[1]
        )
        return edit_image.resize((calculated_width, calculated_height))

    def process(self, pipe: QwenImagePipeline, edit_image, edit_image_auto_resize=False):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        resized_edit_image = self.edit_image_auto_resize(edit_image) if edit_image_auto_resize else edit_image
        edit_image = pipe.preprocess_image(resized_edit_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        edit_latents = pipe.vae.encode(edit_image)
        return {"edit_latents": edit_latents, "edit_image": resized_edit_image}


def model_fn_qwen_image(
    dit: QwenImageDiT = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    edit_latents=None,
    img_shapes=None,
    tag=None,
    **kwargs,
):
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000

    image = latents
    image_seq_len = image.shape[1]

    # Append the edit (source) image tokens after the generated image tokens
    image = torch.cat([image, edit_latents], dim=1)

    image = dit.img_in(image)
    conditioning = dit.time_text_embed(timestep, image.dtype)

    text = dit.txt_in(dit.txt_norm(prompt_emb))
    image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

    for block_id, block in enumerate(dit.transformer_blocks):
        MANAGER.block = block_id
        text, image = block(
            image=image,
            text=text,
            temb=conditioning,
            image_rotary_emb=image_rotary_emb,
            tag=tag,
        )

    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)
    image = image[:, :image_seq_len]

    latents = image
    return latents
