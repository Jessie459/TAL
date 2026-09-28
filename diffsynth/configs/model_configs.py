qwen_image_series = [
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "0319a1cb19835fb510907dd3367c95ff",
        "model_name": "qwen_image_dit",
        "model_class": "diffsynth.models.qwen_image_dit.QwenImageDiT",
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
        # Example: ModelConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct", origin_file_pattern="model-*.safetensors")
        "model_hash": "8004730443f55db63092006dd9f7110e",
        "model_name": "qwen_image_text_encoder",
        "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
        "model_hash": "ed4ea5824d55ec3107b09815e318123a",
        "model_name": "qwen_image_vae",
        "model_class": "diffsynth.models.qwen_image_vae.QwenImageVAE",
    },
]

step1x_edit_series = [
    {
        # Example: ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="vae.safetensors")
        "model_hash": "21ea55f476dfc4fd135587abb59dfe5d",
        "model_name": "flux_vae_encoder",
        "model_class": "diffsynth.models.flux_vae.FluxVAEEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_vae.FluxVAEEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="vae.safetensors")
        "model_hash": "21ea55f476dfc4fd135587abb59dfe5d",
        "model_name": "flux_vae_decoder",
        "model_class": "diffsynth.models.flux_vae.FluxVAEDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_vae.FluxVAEDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors")
        "model_hash": "d30fb9e02b1dbf4e509142f05cf7dd50",
        "model_name": "step1x_connector",
        "model_class": "diffsynth.models.step1x_connector.Qwen2Connector",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.step1x_connector.Qwen2ConnectorStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors")
        "model_hash": "d30fb9e02b1dbf4e509142f05cf7dd50",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
        "extra_kwargs": {"disable_guidance_embedder": True},
    },
]

MODEL_CONFIGS = qwen_image_series + step1x_edit_series
