import argparse
import gc
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def tensors_to_device(data: dict, device: str | torch.device):
    if isinstance(data, dict):
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device)
    else:
        raise ValueError(f"Expected input to be a dict, but got {type(data)}")


def minmax_normalize(tensor: torch.Tensor, eps=1e-8) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def replace(ids: torch.Tensor, source: torch.Tensor, target: torch.Tensor):
    batch_indices = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1)
    target[batch_indices, ids] = source[batch_indices, ids]


def intersect(ids1: torch.Tensor, ids2: torch.Tensor):
    results = []
    for b in range(len(ids1)):
        mask = torch.isin(ids1[b], ids2[b])
        result = torch.unique(ids1[b][mask])
        results.append(result)
    if len(results) == 1:
        results = results[0].unsqueeze(0)
    return results


def masked_average_pooling(embs: torch.Tensor, mask: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Args:
        embs (torch.Tensor): shape: [B D H W] or [D H W]
        mask (torch.Tensor): shape: [B K H W] or [K H W]
    """
    batched = embs.ndim == 4
    if not batched:
        embs = embs.unsqueeze(0)
        mask = mask.unsqueeze(0)
    denom = mask.sum(dim=(-1, -2), keepdim=True).clamp(min=eps)
    result = torch.einsum("bdhw,bkhw->bkd", embs, mask / denom)
    if not batched:
        result = result.squeeze(0)
    return result


def aggregate_attn(names, attns, task_type, semant_type):
    if task_type is None:
        i = 0
        j = len(names)
    elif task_type == "subject_remove":
        i = 1
        j = len(names)
    elif task_type == "subject_add":
        i = 1
        j = names.index("\u0120on")
    elif task_type == "subject_replace":
        if semant_type == "img":
            i = 1
            j = names.index("\u0120with")
        else:
            i = names.index("\u0120with") + 1
            j = len(names)
    elif task_type == "text_change":
        if names[0] == "Replace":
            if semant_type == "img":
                i = 0
                j = names.index("\u0120on")
            else:
                i = names.index("\u0120with") + 1
                j = len(names)
        elif names[0] == "Add":
            # Remove "on the image"
            i = 0
            j = names.index("\u0120on")
        else:
            raise ValueError
    elif task_type == "color_alter":
        # between "of" and "to"
        i = names.index("\u0120of") + 1
        j = names.index("\u0120to")
    elif task_type == "material_alter":
        # between "of" and "to"
        i = names.index("\u0120of") + 1
        j = names.index("\u0120to")
    elif task_type == "background_change":
        i = 0
        j = len(names)
    elif task_type == "position_change":
        # between "of" and "to"
        i = names.index("\u0120of") + 1
        j = names.index("\u0120to")
    elif task_type == "count_change":
        # between "of" and "to"
        i = names.index("\u0120of") + 1
        j = names.index("\u0120to")
    else:
        raise ValueError
    attn = attns[:, :, i:j].mean(-1)
    attn = minmax_normalize(attn)
    return attn


def compute_feat_partition(names, attns, feats, threshold, task_type, semant_type):
    B, N, D = feats.shape
    h, w = attns[0].shape[:2]

    attn_masks = []
    for name, attn in zip(names, attns):
        attn = aggregate_attn(name, attn, task_type=task_type, semant_type=semant_type)
        attn_masks.append((attn > threshold).int())

    feats = rearrange(feats, "b (h w) d -> b d h w", h=h, w=w).float()
    feats = F.normalize(feats, dim=1)

    edit_indices, uned_indices = [], []
    for b in range(B):
        pooled_feats = torch.stack(
            [masked_average_pooling(feats[b], (attn_masks[b] == c).unsqueeze(0)).squeeze(0) for c in range(2)],
            dim=0,
        )
        scores = pooled_feats @ feats[b].reshape(D, -1)  # [c d] @ [d hw] -> [c hw]
        mask = torch.argmax(scores, dim=0).bool()
        full_indices = torch.arange(N, device=feats.device)
        edit_indices.append(full_indices[mask])
        uned_indices.append(full_indices[~mask])
    if B == 1:
        edit_indices = edit_indices[0].unsqueeze(0)
        uned_indices = uned_indices[0].unsqueeze(0)
    return edit_indices, uned_indices


def compute_attn_partition(names, attns, threshold):
    edit_indices = []
    uned_indices = []
    for name, attn in zip(names, attns):
        attn = minmax_normalize(attn.mean(-1))
        mask = (attn > threshold).bool().view(-1)
        full_indices = torch.arange(mask.shape[0], device=mask.device)
        edit_indices.append(full_indices[mask])
        uned_indices.append(full_indices[~mask])
    if len(attns) == 1:
        edit_indices = edit_indices[0].unsqueeze(0)
        uned_indices = uned_indices[0].unsqueeze(0)
    return edit_indices, uned_indices


def compute_task_attn_partition(names, attns, threshold, task_type, semant_type):
    edit_indices = []
    uned_indices = []
    for name, attn in zip(names, attns, strict=False):
        attn = aggregate_attn(name, attn, task_type=task_type, semant_type=semant_type)
        mask = (attn > threshold).bool().view(-1)
        full_indices = torch.arange(mask.shape[0], device=mask.device)
        edit_indices.append(full_indices[mask])
        uned_indices.append(full_indices[~mask])
    if len(attns) == 1:
        edit_indices = edit_indices[0].unsqueeze(0)
        uned_indices = uned_indices[0].unsqueeze(0)
    return edit_indices, uned_indices


def visualize_attn(attn: np.ndarray, img: np.ndarray | Image.Image):
    if isinstance(img, Image.Image):
        img = np.array(img) / 255.0
    cmap = plt.get_cmap("jet")(attn)[..., :3]
    cmap = cmap * 0.5 + img * 0.5
    cmap = (cmap * 255).astype(np.uint8)
    return Image.fromarray(cmap)


def visualize_partition(edit_ids: torch.Tensor, uned_ids: torch.Tensor, h: int, w: int, image: Image.Image, color=(255, 0, 0)):
    mask = torch.zeros(edit_ids.shape[0] + uned_ids.shape[0])
    mask[edit_ids] = 1
    mask = rearrange(mask, "(h w) -> h w", h=h, w=w)
    mask = F.interpolate(mask[None, None], size=[image.height, image.width], mode="nearest")[0, 0]
    mask = mask.bool().cpu().numpy()

    image = np.array(image, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    image[mask] = (image[mask] + color) * 0.5
    return Image.fromarray(image.astype(np.uint8))


def visualize_binary_mask(edit_ids: torch.Tensor, uned_ids: torch.Tensor, h: int, w: int, image: Image.Image):
    mask = torch.zeros(edit_ids.shape[0] + uned_ids.shape[0], device=edit_ids.device)
    mask[edit_ids] = 255
    mask = rearrange(mask, "(h w) -> h w", h=h, w=w)
    mask = F.interpolate(mask[None, None], size=[image.height, image.width], mode="nearest")[0, 0]
    return Image.fromarray(mask.byte().cpu().numpy(), mode="L")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def resize_image(image: Image.Image, target_area: int = 1024 * 1024, multiple: int = 32) -> Image.Image:
    width, height = image.size
    half = multiple // 2
    if (
        width % multiple == 0
        and height % multiple == 0
        and (width - half) * (height - half) <= target_area <= (width + half) * (height + half)
    ):
        return image
    ratio = width / height
    new_width = math.sqrt(target_area * ratio)
    new_height = new_width / ratio
    new_width = round(new_width / multiple) * multiple
    new_height = round(new_height / multiple) * multiple
    return image.resize((new_width, new_height), resample=Image.LANCZOS)


def largest_connected_component(mask):
    H, W = mask.shape
    visited = torch.zeros(H, W, dtype=torch.bool, device=mask.device)
    best_mask = torch.zeros(H, W, dtype=torch.bool, device=mask.device)
    best_size = 0
    coords = torch.nonzero(mask)
    for y, x in coords:
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        component = []
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= H or cx < 0 or cx >= W:
                continue
            if visited[cy, cx] or not mask[cy, cx]:
                continue
            visited[cy, cx] = True
            component.append((cy, cx))
            stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
        if len(component) > best_size:
            best_size = len(component)
            best_mask.zero_()
            ys = torch.tensor([c[0] for c in component], device=mask.device)
            xs = torch.tensor([c[1] for c in component], device=mask.device)
            best_mask[ys, xs] = True
    return best_mask


def fill_holes(mask):
    H, W = mask.shape
    inv = ~mask
    visited = torch.zeros(H, W, dtype=torch.bool, device=mask.device)
    stack = []
    for i in range(H):
        for j_idx, j in [(0, 0), (1, W - 1)]:
            if inv[i, j] and not visited[i, j]:
                visited[i, j] = True
                stack.append((i, j))
    for j in range(W):
        for i_idx, i in [(0, 0), (1, H - 1)]:
            if inv[i, j] and not visited[i, j]:
                visited[i, j] = True
                stack.append((i, j))
    while stack:
        y, x = stack.pop()
        for ny, nx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if visited[ny, nx] or not inv[ny, nx]:
                continue
            visited[ny, nx] = True
            stack.append((ny, nx))
    holes = inv & (~visited)
    return mask | holes


def dilate(mask, kernel=5, iterations=1):
    x = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        x = F.max_pool2d(x, kernel, stride=1, padding=kernel // 2)
    return x.squeeze(0).squeeze(0) > 0


def erode(mask, kernel=5, iterations=1):
    return ~dilate(~mask, kernel=kernel, iterations=iterations)


def refine_token_ids(edit_ids, H, W, kernel=5, dilation_iter=1, refine="expand_foreground"):
    device = edit_ids.device
    N = H * W

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[edit_ids] = True
    mask = mask.view(H, W)

    if refine == "expand_foreground":
        original_mask = mask.clone()
        lcc = largest_connected_component(mask)
        lcc = fill_holes(lcc)
        lcc = dilate(lcc, kernel, dilation_iter)
        final_mask = original_mask | lcc
    elif refine == "shrink_background":
        background = largest_connected_component(~mask)
        background = erode(background, kernel=kernel, iterations=dilation_iter)
        final_mask = ~background
    else:
        raise ValueError(f"Invalid refine: {refine}")

    final_mask = final_mask.view(-1)

    new_edit_ids = torch.nonzero(final_mask).squeeze(1).long()
    new_uned_ids = torch.nonzero(~final_mask).squeeze(1).long()

    return new_edit_ids, new_uned_ids
