import logging
import torch
from .utils import intersect, refine_token_ids, replace

logger = logging.getLogger(__name__)


class Manager:
    def __init__(self):
        self.num_inference_steps: int = None
        self.task_type: str = None


        self.step: int = None
        self.block: int = None

        self.pix_h: int = None
        self.pix_w: int = None
        self.img_h: int = None
        self.img_w: int = None
        self.img_seq_len: int = None

        self.semant: bool = False
        self.semant_types: list[str] = None
        self.semant_steps: list[int] = None
        self.semant_blocks: list[int] = None
        self.semant_threshold: float = None
        self.semant_mask_type: str = None
        self.semant_mask_refine: str = None
        self.semant_mask_kernel: int = 5

        self.tokens: list[list[str]] = None
        self.ranges: list[list[int]] = None

        self.gen_attn: list[torch.Tensor] = None
        self.img_attn: list[torch.Tensor] = None
        self.gen_attn_num: int = None
        self.img_attn_num: int = None
        self.gen_feat: dict[int, torch.Tensor] = None
        self.img_feat: dict[int, torch.Tensor] = None

        self.gen_edit_ids: torch.Tensor = None
        self.gen_uned_ids: torch.Tensor = None
        self.img_edit_ids: torch.Tensor = None
        self.img_uned_ids: torch.Tensor = None
        self.edit_ids: torch.Tensor = None
        self.uned_ids: torch.Tensor = None

    def set_parameters(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.semant:
            if self.semant_steps is None:
                self.semant_steps = []
            self.semant_steps = sorted(list(set(self.semant_steps)))
            logger.info(f"semant_steps: {self.semant_steps}")

    def reset(self, pix_h, pix_w):
        scale_factor = 16
        assert pix_h % scale_factor == 0
        assert pix_w % scale_factor == 0
        self.pix_h = pix_h
        self.pix_w = pix_w
        self.img_h = pix_h // scale_factor
        self.img_w = pix_w // scale_factor
        self.img_seq_len = self.img_h * self.img_w

        self.gen_attn = []
        self.img_attn = []
        self.gen_attn_num = 0
        self.img_attn_num = 0
        self.gen_feat = {}
        self.img_feat = {}

        self.gen_edit_ids = None
        self.gen_uned_ids = None
        self.img_edit_ids = None
        self.img_uned_ids = None
        self.edit_ids = None
        self.uned_ids = None

    def compute_semant_ids(self):
        if len(self.semant_types) == 2:
            uned_ids = intersect(self.gen_uned_ids, self.img_uned_ids)
        elif len(self.semant_types) == 1 and self.semant_types[0] == "gen":
            uned_ids = self.gen_uned_ids
        elif len(self.semant_types) == 1 and self.semant_types[0] == "img":
            uned_ids = self.img_uned_ids
        else:
            raise RuntimeError("No unedited ids found")

        uned_ids = uned_ids[0]
        mask = torch.ones(self.img_seq_len).to(uned_ids)
        mask[uned_ids] = False
        edit_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)

        if self.semant_mask_refine and self.semant_mask_refine != "none":
            edit_ids, uned_ids = refine_token_ids(
                edit_ids,
                self.img_h,
                self.img_w,
                kernel=self.semant_mask_kernel,
                refine=self.semant_mask_refine,
            )

        self.uned_ids = uned_ids.unsqueeze(0)
        self.edit_ids = edit_ids.unsqueeze(0)

    def post_update(self, latents, inv_latents):
        replace(self.uned_ids, inv_latents, latents)
        return latents

    def set_tokens(self, tokens: list[list[str]]):
        self.tokens = []
        self.ranges = []
        for tok in tokens:
            if "<|vision_end|>" in tok:
                i = tok.index("<|vision_end|>") + 1
            else:
                logger.warning(f"<|vision_end|> not found: {tok}")
                i = 0
            if "<|im_end|>" in tok:
                j = tok.index("<|im_end|>")
            else:
                logger.warning(f"<|im_end|> not found: {tok}")
                j = len(tok)
            self.ranges.append([i, j])
            self.tokens.append(tok[i:j])

        if self.task_type == "background_change":
            for i in range(len(self.tokens)):
                if "," in self.tokens[i]:
                    j = self.tokens[i].index(",")
                    self.tokens[i] = self.tokens[i][:j]
                    self.ranges[i][1] = self.ranges[i][0] + j

    def get_attn(self, semant_type, device=None, dtype=None):
        attn = eval(f"self.{semant_type}_attn")
        attn_num = eval(f"self.{semant_type}_attn_num")
        device = device or attn[0].device
        dtype = dtype or attn[0].dtype
        attn = [(a / attn_num).to(device=device, dtype=dtype) for a in attn]
        return attn

    @staticmethod
    def _refine_attn(self_attn: torch.Tensor, cross_attn: torch.Tensor, num_steps=1):
        self_attn = self_attn / (self_attn.sum(dim=-1, keepdim=True) + 1e-8)
        result = cross_attn
        for _ in range(num_steps):
            result = self_attn @ result
        return result

    def set_attn(self, attn: torch.Tensor, semant_type: str, txt_len: int, img_len: int):
        batch_size = attn.shape[0]
        assert attn.shape[-1] == txt_len + img_len * 2

        cross_attn = attn[..., :txt_len]
        if semant_type == "gen":
            self_attn = attn[..., txt_len : txt_len + img_len]
        else:
            self_attn = attn[..., txt_len + img_len :]
        attn = self._refine_attn(self_attn, cross_attn)

        for b in range(batch_size):
            a = attn[b].sum(0).reshape(self.img_h, self.img_w, txt_len)
            i, j = self.ranges[b]
            if semant_type == "gen":
                if len(self.gen_attn) == batch_size:
                    self.gen_attn[b] += a[..., i:j]
                else:
                    self.gen_attn.append(a[..., i:j])
            else:
                if len(self.img_attn) == batch_size:
                    self.img_attn[b] += a[..., i:j]
                else:
                    self.img_attn.append(a[..., i:j])

        if semant_type == "gen":
            self.gen_attn_num += 1
        else:
            self.img_attn_num += 1


MANAGER = Manager()
