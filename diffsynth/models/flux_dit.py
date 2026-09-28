import math

import torch
from einops import rearrange

from .general_modules import AdaLayerNorm, RMSNorm, TimestepEmbeddings
from src.step1x_edit.manager import MANAGER


def update_attn(q: torch.Tensor, k: torch.Tensor, txt_len: int, img_len: int):
    """
    Args:
        q (torch.Tensor): shape [batch_size, num_heads, seq_length, head_dim]
        k (torch.Tensor): shape [batch_size, num_heads, seq_length, head_dim]
    """
    assert q.shape[-2] == txt_len + img_len * 2

    scale = 1.0 / math.sqrt(q.shape[-1])
    if "gen" in MANAGER.semant_types:
        q_chunk = q[:, :, txt_len : txt_len + img_len, :]
        attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        MANAGER.set_attn(attn, "gen", txt_len=txt_len, img_len=img_len)
    if "img" in MANAGER.semant_types:
        q_chunk = q[:, :, txt_len + img_len :, :]
        attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        MANAGER.set_attn(attn, "img", txt_len=txt_len, img_len=img_len)


class RoPEEmbedding(torch.nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0, "The dimension must be even."

        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)

        batch_size, seq_length = pos.shape
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)

        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
        return out.float()

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([self.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class FluxJointAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_q_b = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_b = RMSNorm(head_dim, eps=1e-6)

        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        self.b_to_out = torch.nn.Linear(dim_b, dim_b)

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def forward(self, hidden_states_a, hidden_states_b, image_rotary_emb, tag=None):
        batch_size = hidden_states_a.shape[0]

        # Part A
        qkv_a = self.a_to_qkv(hidden_states_a)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        # Part B
        qkv_b = self.b_to_qkv(hidden_states_b)
        qkv_b = qkv_b.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=1)
        q_b, k_b = self.norm_q_b(q_b), self.norm_k_b(k_b)

        q = torch.concat([q_b, q_a], dim=2)
        k = torch.concat([k_b, k_a], dim=2)
        v = torch.concat([v_b, v_a], dim=2)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        if tag == "posi" and MANAGER.semant and MANAGER.step in MANAGER.semant_steps:
            update_attn(q, k, txt_len=MANAGER.txt_seq_len, img_len=MANAGER.img_seq_len)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_b, hidden_states_a = (
            hidden_states[:, : hidden_states_b.shape[1]],
            hidden_states[:, hidden_states_b.shape[1] :],
        )
        hidden_states_a = self.a_to_out(hidden_states_a)
        hidden_states_b = self.b_to_out(hidden_states_b)
        return hidden_states_a, hidden_states_b


class FluxJointTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim)

        self.attn = FluxJointAttention(dim, dim, num_attention_heads, dim // num_attention_heads)

        self.norm2_a = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(dim * 4, dim)
        )

        self.norm2_b = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_b = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(dim * 4, dim)
        )

    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb, tag=None):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(
            norm_hidden_states_a,
            norm_hidden_states_b,
            image_rotary_emb,
            tag=tag,
        )
        if tag == "posi" and MANAGER.semant and MANAGER.step in MANAGER.semant_steps and MANAGER.block in MANAGER.semant_blocks:
            img_len = MANAGER.img_seq_len
            if "gen" in MANAGER.semant_types:
                MANAGER.gen_feat[MANAGER.block] = attn_output_a[:, :img_len].detach().clone()
            if "img" in MANAGER.semant_types:
                MANAGER.img_feat[MANAGER.block] = attn_output_a[:, img_len:].detach().clone()

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b


class AdaLayerNormSingle(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(dim, 3 * dim, bias=True)
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class FluxSingleTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.dim = dim

        self.norm = AdaLayerNormSingle(dim)
        self.to_qkv_mlp = torch.nn.Linear(dim, dim * (3 + 4))
        self.norm_q_a = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(self.head_dim, eps=1e-6)

        self.proj_out = torch.nn.Linear(dim * 5, dim)

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def process_attention(self, hidden_states, image_rotary_emb, tag=None):
        batch_size = hidden_states.shape[0]

        qkv = hidden_states.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        q, k = self.norm_q_a(q), self.norm_k_a(k)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        if tag == "posi" and MANAGER.semant and MANAGER.step in MANAGER.semant_steps:
            update_attn(q, k, txt_len=MANAGER.txt_seq_len, img_len=MANAGER.img_seq_len)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states

    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb, tag=None):
        residual = hidden_states_a
        norm_hidden_states, gate = self.norm(hidden_states_a, emb=temb)
        hidden_states_a = self.to_qkv_mlp(norm_hidden_states)
        attn_output, mlp_hidden_states = hidden_states_a[:, :, : self.dim * 3], hidden_states_a[:, :, self.dim * 3 :]

        attn_output = self.process_attention(attn_output, image_rotary_emb, tag=tag)
        if tag == "posi" and MANAGER.semant and MANAGER.step in MANAGER.semant_steps and MANAGER.block in MANAGER.semant_blocks:
            txt_len = MANAGER.txt_seq_len
            img_len = MANAGER.img_seq_len
            if "gen" in MANAGER.semant_types:
                MANAGER.gen_feat[MANAGER.block] = attn_output[:, txt_len:][:, :img_len].detach().clone()
            if "img" in MANAGER.semant_types:
                MANAGER.img_feat[MANAGER.block] = attn_output[:, txt_len:][:, img_len:].detach().clone()
        mlp_hidden_states = torch.nn.functional.gelu(mlp_hidden_states, approximate="tanh")

        hidden_states_a = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states_a = gate.unsqueeze(1) * self.proj_out(hidden_states_a)
        hidden_states_a = residual + hidden_states_a

        return hidden_states_a, hidden_states_b


class AdaLayerNormContinuous(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(dim, dim * 2, bias=True)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning))
        shift, scale = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None] + shift[:, None]
        return x


class FluxDiT(torch.nn.Module):
    def __init__(self, disable_guidance_embedder=False, input_dim=64, num_blocks=19):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.time_embedder = TimestepEmbeddings(256, 3072)
        self.guidance_embedder = None if disable_guidance_embedder else TimestepEmbeddings(256, 3072)
        self.pooled_text_embedder = torch.nn.Sequential(
            torch.nn.Linear(768, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072)
        )
        self.context_embedder = torch.nn.Linear(4096, 3072)
        self.x_embedder = torch.nn.Linear(input_dim, 3072)

        self.blocks = torch.nn.ModuleList([FluxJointTransformerBlock(3072, 24) for _ in range(num_blocks)])
        self.single_blocks = torch.nn.ModuleList([FluxSingleTransformerBlock(3072, 24) for _ in range(38)])

        self.final_norm_out = AdaLayerNormContinuous(3072)
        self.final_proj_out = torch.nn.Linear(3072, 64)

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states

    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(
            hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=height // 2, W=width // 2
        )
        return hidden_states

    def prepare_image_ids(self, latents):
        batch_size, _, height, width = latents.shape
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(device=latents.device, dtype=latents.dtype)

        return latent_image_ids
