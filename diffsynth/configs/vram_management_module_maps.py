flux_general_vram_config = {
    "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
    "torch.nn.Embedding": "diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.LayerNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.GroupNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    "diffsynth.models.general_modules.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
}

VRAM_MANAGEMENT_MODULE_MAPS = {
    "diffsynth.models.qwen_image_dit.QwenImageDiT": {
        "diffsynth.models.qwen_image_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.qwen_image_vae.QwenImageVAE": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.qwen_image_vae.QwenImageRMS_norm": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.flux_dit.FluxDiT": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "diffsynth.models.flux_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.flux_vae.FluxVAEEncoder": flux_general_vram_config,
    "diffsynth.models.flux_vae.FluxVAEDecoder": flux_general_vram_config,
    "diffsynth.models.step1x_connector.Qwen2Connector": flux_general_vram_config,
}
