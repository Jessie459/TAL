from .prompts import _context_no_delimit, _prompts_0shot_rule_PQ
from .qwen25vl import Qwen25VL
from .utils import mllm_output_to_dict


class VIEScore:
    """VIEScore perceptual quality (PQ) for image editing with a Qwen2.5-VL judge."""

    def __init__(self, **kwargs):
        self.model = Qwen25VL(**kwargs)
        self.PQ_prompt = "\n".join([_context_no_delimit, _prompts_0shot_rule_PQ])

    def evaluate_PQ(self, image_prompts):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]

        PQ_prompt_final = self.model.prepare_prompt(image_prompts[-1], self.PQ_prompt)
        PQ_output = self.model.get_parsed_output(PQ_prompt_final)
        PQ_dict = mllm_output_to_dict(PQ_output, give_up_parsing=False)

        return PQ_dict if isinstance(PQ_dict, dict) else None
