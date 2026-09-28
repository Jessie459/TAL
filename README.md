# Rethinking Where to Edit: Task-Aware Localization for Instruction-Based Image Editing

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue.svg)](https://2026.acmmm.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.20258-b31b1b.svg)](https://arxiv.org/abs/2604.20258)

Official implementation of **Task-Aware Localization (TAL)** (ACM MM 2026), a training-free framework that reduces over-editing in instruction-based image editing (IIE).

Different editing operations change different parts of the scene: an added object appears only in the **target** image stream, a removed object lives in the **source** image stream, and a replacement involves both. TAL exploits this inside dual-stream diffusion transformers:

1. **Attention-based semantic estimation.** Instruction-to-image attention in each image stream, propagated once through the stream's self-attention, gives a coarse edit cue.
2. **Feature-based semantic assignment.** The coarse cue defines two feature centroids (edit / non-edit) by masked average pooling; every token is assigned to the nearest centroid.
3. **Task-aware mask construction.** The source-stream and target-stream masks are selected or combined according to the editing task.
4. **Mask-guided latent preservation.** At a few denoising steps, latents outside the edit mask are re-anchored to the source image, preserving non-edit regions.

TAL is applied to [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) and [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) and evaluated on [EdiVal-Bench](https://arxiv.org/abs/2509.13399).

## Installation

Tested with Python 3.10, PyTorch 2.6.0 (CUDA 12.4) and Transformers 4.57.6.

```bash
conda create -n tal python=3.10 -y
conda activate tal
pip install -r requirements.txt
```

[FlashAttention-3](https://github.com/Dao-AILab/flash-attention) is optional and is used by Qwen-Image-Edit when installed; otherwise PyTorch SDPA is used.

Evaluation additionally needs FlashAttention-2 and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), which are built against the installed PyTorch:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation

git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO
git checkout 856dde20aee659246248e20734ef9ba5214f5e44
# PyTorch >= 2.6 removed Tensor.type() in dispatch macros
sed -i "s/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/" \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
pip install --no-build-isolation --no-deps -e .   # needs nvcc and GCC >= 9
```

## Model Weights

Weights are looked up under `$DIFFSYNTH_MODEL_BASE_PATH/<model_id>/` (default: `./models`) and downloaded from Hugging Face automatically when missing. To download them yourself:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=./models

# Step1X-Edit
hf download stepfun-ai/Step1X-Edit step1x-edit-i1258.safetensors vae.safetensors --local-dir $DIFFSYNTH_MODEL_BASE_PATH/stepfun-ai/Step1X-Edit
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir $DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen2.5-VL-7B-Instruct

# Qwen-Image-Edit
hf download Qwen/Qwen-Image-Edit --include "transformer/*" --include "processor/*" --local-dir $DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen-Image-Edit
hf download Qwen/Qwen-Image --include "vae/*" --include "text_encoder/*" --local-dir $DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen-Image
```

Set `DIFFSYNTH_SKIP_DOWNLOAD=true` to disable automatic downloads.

## Data

We evaluate on EdiVal-Bench. Download our preprocessed copy from [Google Drive](https://drive.google.com/drive/folders/1JXa2-Z2UAIHe6IylupPOfg3er4W26rSF?usp=drive_link) and place it at `data/EdiVal`, or point `DATASET_PATH` to its location:

```
data/EdiVal/
├── images/          # {image_index}.png, 606 source images
└── metadata.csv     # EdiVal-Bench instructions and task types
```

The source images come from [GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) (English split) and are resized to about 1024×1024 pixels, preserving the aspect ratio, with both sides rounded to multiples of 32 (LANCZOS). The scripts apply the same rule to any other input image; images already in this format are used unchanged. Only the first editing turn of each EdiVal-Bench sample is used.

## Usage

### 1. Generate edits

Settings used in the paper (28 denoising steps, attention threshold 0.5, latent preservation at steps 4, 9 and 14, feature layer 44 for Step1X-Edit and 49 for Qwen-Image-Edit; steps and layers are 0-indexed):

```bash
# Step1X-Edit + TAL
python infer_step1x.py --output_dir outputs/step1x_edit_tal \
    --semant true --semant_threshold 0.5 --semant_blocks 44 --semant_steps 4 9 14

# Qwen-Image-Edit + TAL
python infer_qwen.py --output_dir outputs/qwen_image_edit_tal \
    --semant true --semant_threshold 0.5 --semant_blocks 49 --semant_steps 4 9 14
```

Omit the `--semant*` options to run the unmodified base model. Edited images are written to `<output_dir>/<image_index>_turn1.png` at the original input resolution.

To edit your own image, pass `--image`, `--instruction` and optionally `--task_type` instead of running on EdiVal:

```bash
python infer_step1x.py --output_dir outputs/demo \
    --image path/to/image.png --instruction "Replace wool white sheep with goats" --task_type subject_replace \
    --semant true --semant_threshold 0.5 --semant_blocks 44 --semant_steps 4 9 14
```

The result is saved as `<output_dir>/<image name>_turn1.png`. TAL decides which image stream to localize from the task type, and reads the edited object from the instruction, so with `--task_type` the instruction must follow the EdiVal template of its task:

| `--task_type` | Instruction template |
|---|---|
| `subject_add` | Add [object] on the [position] of [reference object] |
| `subject_remove` | Remove [object] |
| `subject_replace` | Replace [object] with [new object] |
| `color_alter` | Change the color of [object] to [color] |
| `material_alter` | Change the material of [object] to [material] |
| `position_change` | Change the position of [object] to [position] of [reference object] |
| `count_change` | Change the count of [object] to [number] |
| `text_change` | Add text '[text]' on the image / Replace the text '[old text]' on [object] with '[new text]' |
| `background_change` | Change the background to [scene] |

Without `--task_type`, the script prints a warning and falls back to task-agnostic localization: it combines the masks of both the source and the target image streams and uses attention over the whole instruction, so any instruction wording is accepted but localization is less precise.

By default, the Qwen2.5-VL text encoder runs alongside the diffusion model. Step1X-Edit (about 40 GB of weights in total) fits on a 48 GB GPU this way. Qwen-Image-Edit (about 55 GB) does not: either cache the text embeddings first (see below), which leaves about 40 GB, or add `--low_vram true`.

### 2. Cache the text embeddings (optional)

Pre-computing the Qwen2.5-VL embeddings lets generation load only the diffusion model and the VAE:

```bash
python preprocess_qwenvl_step1x.py   # -> $DATASET_PATH/qwenvl/step1x_edit/
python preprocess_qwenvl_qwen.py     # -> $DATASET_PATH/qwenvl/qwen_image_edit/
```

Then add `--load_qwenvl true` to the generation commands. Results are identical to computing the embeddings on the fly.

Useful options:

| Option | Description |
|---|---|
| `--edival_task` | Run a single EdiVal task type, e.g. `subject_add`, `subject_remove`, `subject_replace` |
| `--image`, `--instruction`, `--task_type` | Edit one image instead of EdiVal; `--task_type` is optional (see above) |
| `--semant_blocks` | 0-based DiT layer used for feature-based assignment (44 for Step1X-Edit, 49 for Qwen-Image-Edit) |
| `--semant_steps` | 0-based denoising steps at which masks are computed and non-edit latents are preserved |
| `--semant_threshold` | Attention threshold τ |
| `--semant_types` | Override the task-aware stream selection: `img` (source), `gen` (target), or both |
| `--vis true` | Save attention maps and edit masks for each sample |
| `--load_qwenvl true` | Read cached text embeddings from `$DATASET_PATH/qwenvl/` instead of running the text encoder |
| `--low_vram true` | Offload weights and store them in FP8 to reduce GPU memory (results differ slightly from full precision) |

### 3. Evaluate

We report the EdiVal-Bench metrics EdiVal-IF (instruction following), EdiVal-CC (content consistency: L1 and DINOv3 similarity over non-target objects and background) and EdiVal-O (geometric mean of IF and CC), plus perceptual quality (PQ): the naturalness score (0-10) of [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore) rated by Qwen2.5-VL.

```bash
python evaluate_pq.py --gen_dir outputs/step1x_edit_tal   # PQ  -> outputs/step1x_edit_tal_eval/pq.csv
python evaluate.py    --gen_dir outputs/step1x_edit_tal   # IF, CC, O (+ PQ) -> outputs/step1x_edit_tal_eval/summary.csv
```

`summary.csv` has one row for all samples and one per task type; IF, CC and O are in percent. `evaluate.py` stores a JSON per sample and resumes from them, so rerunning it after `evaluate_pq.py` only refreshes the summary. The judges are Qwen2-VL-7B-Instruct (IF, via `--vlm`) and Qwen2.5-VL-7B-Instruct (PQ); GroundingDINO and DINOv3 weights are fetched like the editing models. `facebook/dinov3-vitb16-pretrain-lvd1689m` is gated on Hugging Face: accept its license and log in with `hf auth login`, or place it under `$DIFFSYNTH_MODEL_BASE_PATH` yourself.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DIFFSYNTH_MODEL_BASE_PATH` | Root directory of the model weights (editing models and evaluation judges) | `./models` |
| `DATASET_PATH` | EdiVal data directory (`images/`, `metadata.csv`) for inference and evaluation | `data/EdiVal` |
| `DIFFSYNTH_DOWNLOAD_SOURCE` | `huggingface` or `modelscope` | `huggingface` |
| `DIFFSYNTH_SKIP_DOWNLOAD` | Set to `true` to disable automatic downloads | unset |

## Acknowledgements

This code builds on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [EdiVal-Bench](https://arxiv.org/abs/2509.13399) (evaluation code in `src/edival/`), [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore) (`src/viescore/`) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

## Citation

```bibtex
@inproceedings{he2026rethinking,
  title     = {Rethinking Where to Edit: Task-Aware Localization for Instruction-Based Image Editing},
  author    = {He, Jingxuan and Wang, Xiyu and Zheng, Mengyu and Zeng, Xiangyu and Wang, Yunke and Xu, Chang},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
```
