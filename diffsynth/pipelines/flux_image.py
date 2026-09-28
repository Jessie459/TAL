import os
from typing import Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.step1x_edit.manager import MANAGER
from src.step1x_edit.utils import (
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
from ..diffusion.flow_match_step1x_edit import FlowMatchScheduler
from ..models.flux_dit import FluxDiT
from ..models.flux_vae import FluxVAEDecoder, FluxVAEEncoder
from ..models.step1x_text_encoder import Step1xEditEmbedder


class FluxImagePipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler()
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.qwenvl = None
        self.step1x_connector = None
        self.in_iteration_models = ("dit", "step1x_connector")
        self.units = [
            FluxImageUnit_ShapeChecker(),
            FluxImageUnit_NoiseInitializer(),
            FluxImageUnit_ImageIDs(),
            FluxImageUnit_Step1x(),
        ]
        self.model_fn = model_fn_flux_image

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        step1x_processor_config: ModelConfig = ModelConfig(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct", origin_file_pattern=""
        ),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = FluxImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.dit = model_pool.fetch_model("flux_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        qwenvl = model_pool.fetch_model("qwen_image_text_encoder")
        if qwenvl is not None:
            from transformers import AutoProcessor

            step1x_processor_config.download_if_necessary()
            processor = AutoProcessor.from_pretrained(
                step1x_processor_config.path, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28
            )
            pipe.qwenvl = Step1xEditEmbedder(qwenvl, processor)
        pipe.step1x_connector = model_pool.fetch_model("step1x_connector")

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Scheduler
        sigma_shift: float = None,
        # Steps
        num_inference_steps: int = 30,
        # Step1x
        step1x_reference_image: Image.Image = None,
        # Progress bar
        progress_bar_cmd=tqdm,
        # Pre-computed Qwen-VL embeddings (see `preprocess_qwenvl_step1x.py`)
        inputs_posi: dict = None,
        inputs_nega: dict = None,
        preprocess: bool = False,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        if inputs_posi is not None:
            load_qwenvl = True
            inputs_posi.update({"prompt": prompt, "tag": "posi"})
            inputs_nega.update({"negative_prompt": negative_prompt, "tag": "nega"})
        else:
            load_qwenvl = False
            inputs_posi = {"prompt": prompt, "tag": "posi"}
            inputs_nega = {"negative_prompt": negative_prompt, "tag": "nega"}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "step1x_reference_image": step1x_reference_image,
        }

        if load_qwenvl:
            for unit in self.units:
                if isinstance(unit, FluxImageUnit_Step1x):
                    continue
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        else:
            if preprocess:
                for unit in self.units:
                    if isinstance(unit, FluxImageUnit_Step1x):
                        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
                return inputs_posi, inputs_nega
            else:
                for unit in self.units:
                    inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        posi_qwenvl_tokens = inputs_posi.pop("qwenvl_tokens", None)
        nega_qwenvl_tokens = inputs_nega.pop("qwenvl_tokens", None)
        if MANAGER.semant:
            MANAGER.set_tokens(posi_qwenvl_tokens)

        latents = inputs_shared["latents"]
        step1x_reference_latents_posi = inputs_posi.pop("step1x_reference_latents")
        step1x_reference_latents_nega = inputs_nega.pop("step1x_reference_latents")
        step1x_reference_latents = step1x_reference_latents_posi

        _, _, H, W = latents.shape
        latents = self.dit.patchify(latents)
        step1x_reference_image_ids = self.dit.prepare_image_ids(step1x_reference_latents)
        step1x_reference_latents = self.dit.patchify(step1x_reference_latents)

        inputs_shared["latents"] = latents
        inputs_shared["step1x_reference_latents"] = step1x_reference_latents
        inputs_shared["step1x_reference_image_ids"] = step1x_reference_image_ids
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
                progress_id=progress_id
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
                        _edit_image = inputs_shared["step1x_reference_image"].resize((MANAGER.img_w, MANAGER.img_h))
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
                            inputs_shared["step1x_reference_image"],
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
                        inputs_shared["step1x_reference_image"],
                    )
                    partition_name = "feat" if MANAGER.semant_mask_type == "feat" else "attn_partition"
                    _path = f"step{MANAGER.step}_{partition_name}.png"
                    _feat.save(os.path.join(MANAGER.vis_dir, _path))
                    _mask = visualize_binary_mask(
                        MANAGER.edit_ids[0],
                        MANAGER.uned_ids[0],
                        MANAGER.img_h,
                        MANAGER.img_w,
                        inputs_shared["step1x_reference_image"],
                    )
                    _path = f"step{MANAGER.step}_{partition_name}_mask.png"
                    _mask.save(os.path.join(MANAGER.vis_dir, _path))

            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

            if MANAGER.semant and MANAGER.step in MANAGER.semant_steps:
                sigma = self.scheduler.sigmas[MANAGER.step + 1].item() if MANAGER.step + 1 < len(self.scheduler.sigmas) else 0.0
                inv_latents = sigma * noise_latents + (1.0 - sigma) * inputs_shared["step1x_reference_latents"]
                inputs_shared["latents"] = MANAGER.post_update(inputs_shared["latents"], inv_latents)

        # Decode
        self.load_models_to_device(["vae_decoder"])
        inputs_shared["latents"] = self.dit.unpatchify(inputs_shared["latents"], H, W)
        image = self.vae_decoder(inputs_shared["latents"], device=self.device)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class FluxImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: FluxImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class FluxImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"), output_params=("latents",))

    def process(self, pipe: FluxImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height // 8, width // 8), seed=seed, rand_device=rand_device)
        return {"latents": noise}


class FluxImageUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents",), output_params=("image_ids",))

    def process(self, pipe: FluxImagePipeline, latents):
        latent_image_ids = pipe.dit.prepare_image_ids(latents)
        return {"image_ids": latent_image_ids}


class FluxImageUnit_Step1x(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("step1x_reference_image", "prompt", "negative_prompt"),
            output_params=("step1x_llm_embedding", "step1x_mask", "step1x_reference_latents", "qwenvl_tokens"),
            onload_model_names=("qwenvl", "vae_encoder"),
        )

    def process(self, pipe: FluxImagePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict):
        image = inputs_shared.get("step1x_reference_image", None)
        if image is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            pipe.load_models_to_device(self.onload_model_names)
            prompt = inputs_posi["prompt"]
            nega_prompt = inputs_nega["negative_prompt"]
            captions = [prompt, nega_prompt]
            ref_images = [image, image]
            embs, masks, tokens = pipe.qwenvl(captions, ref_images, return_tokens=True)
            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae_encoder(image)  # [1, 16, 148, 112]
            inputs_posi.update(
                {
                    "step1x_llm_embedding": embs[0:1],
                    "step1x_mask": masks[0:1],
                    "step1x_reference_latents": image,
                    "qwenvl_tokens": tokens[0:1],
                }
            )
            if inputs_shared.get("cfg_scale", 1) != 1:
                inputs_nega.update(
                    {
                        "step1x_llm_embedding": embs[1:2],
                        "step1x_mask": masks[1:2],
                        "step1x_reference_latents": image,
                        "qwenvl_tokens": None,
                    }
                )
            return inputs_shared, inputs_posi, inputs_nega


def model_fn_flux_image(
    dit: FluxDiT,
    step1x_connector=None,
    latents=None,
    timestep=None,
    image_ids=None,
    step1x_llm_embedding=None,
    step1x_mask=None,
    step1x_reference_latents=None,
    step1x_reference_image_ids=None,
    tag=None,
    **kwargs
):
    hidden_states = latents

    # Step1x
    prompt_emb, pooled_prompt_emb = step1x_connector(step1x_llm_embedding, timestep / 1000, step1x_mask)
    text_ids = torch.zeros((1, prompt_emb.shape[1], 3), dtype=prompt_emb.dtype, device=prompt_emb.device)

    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)

    # Step1x: append the reference image tokens after the generated image tokens
    image_ids = torch.concat([image_ids, step1x_reference_image_ids], dim=-2)
    hidden_states = torch.concat([hidden_states, step1x_reference_latents], dim=1)

    hidden_states = dit.x_embedder(hidden_states)

    prompt_emb = dit.context_embedder(prompt_emb)
    image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))  # [1, 1, 8928, 64, 2, 2]

    # Joint Blocks
    for block_id, block in enumerate(dit.blocks):
        MANAGER.block = block_id
        hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, tag=tag)

    # Single Blocks
    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
    num_joint_blocks = len(dit.blocks)
    for block_id, block in enumerate(dit.single_blocks):
        MANAGER.block = block_id + num_joint_blocks
        hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, tag=tag)
    hidden_states = hidden_states[:, prompt_emb.shape[1] :]

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)

    # Step1x: drop the reference image tokens
    hidden_states = hidden_states[:, : hidden_states.shape[1] // 2]

    return hidden_states
