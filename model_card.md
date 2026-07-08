---
license: mit
tags:
  - medical-imaging
  - segmentation
  - mri
  - swin-transformer
  - pytorch
  - abdominal-organs
datasets:
  - CHAOS
metrics:
  - dice
---

# SwinDAF — Abdominal MRI Segmentation (CHAOS T2-SPIR)

Swin Transformer encoder (timm, ImageNet-pretrained) + Dual Attention Fusion decoder
(PAM, CAM, semantic guidance, multi-scale deep supervision) for 5-class segmentation
of abdominal organs on T2-SPIR MRI slices.

> ⚠️ **Research and education demo. Not a medical device. Not for diagnostic use.**

## Model details

- **Architecture:** Swin-Tiny (`swin_tiny_patch4_window7_224`) encoder + Dual Attention
  Fusion (DAF) decoder — position + channel attention modules, semantic guidance, and
  multi-scale deep supervision.
- **Classes (5):** background, liver, right kidney, left kidney, spleen.
- **Input:** single-channel 224×224 T2-SPIR MRI slice.
- **Training data:** [CHAOS](https://chaos.grand-challenge.org/) T2-SPIR MRI, 20 subjects,
  subject-level split — 16 train / 2 val / 2 test (497 / 64 / 62 slices). Test subjects are
  never seen in training or validation.
- **Training settings:** 30 epochs, batch size 16, lr 0.001, seed 42, bf16 AMP, best
  validation-Dice checkpoint selection. Trained locally on an RTX 5080.
- **License:** MIT (code and weights).

## Test-set results (held-out, 2 subjects, 62 slices)

2D per-slice Dice (mean ± std):

| Organ | Dice |
|---|---|
| Liver | 0.689 ± 0.428 |
| Right Kidney | 0.812 ± 0.348 |
| Left Kidney | 0.816 ± 0.342 |
| Spleen | 0.731 ± 0.403 |
| **Mean** | **0.762** |

3D per-subject volumetric Dice (mean ± std, n=2 subjects):

| Organ | Dice |
|---|---|
| Liver | 0.479 ± 0.465 |
| Right Kidney | 0.910 ± 0.017 |
| Left Kidney | 0.898 ± 0.014 |
| Spleen | 0.393 ± 0.393 |
| **Mean** | **0.670** |

**Known failure mode — read before use.** With only 2 held-out test subjects, one atypical
subject dominates the aggregate. One test subject has a genuine, disclosed liver/spleen
segmentation failure at the volumetric level (Liver 3D Dice ~0.01, Spleen 3D Dice 0.00 — the
model predicts almost no liver/spleen pixels at all for that subject) despite scoring well on
both kidneys. This was investigated directly (ruled out label errors and data leakage;
an intensity-jitter augmentation fix was tried and did not resolve it) — see the project
README's Failure analysis section for the full writeup. Full numbers, methodology, and
reproduction commands: <https://github.com/MounishAllam/Abdominal-Vision-Transformers>.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# assumes this repo's src/models/swin_danet.py is importable
from models.swin_danet import SwinDAF

weights_path = hf_hub_download(repo_id="MounishAllam/swin-daf-chaos-mri",
                                filename="Best_SwinDAF-CHAOS.pth")

net = SwinDAF(num_classes=5, encoder_name="swin_tiny_patch4_window7_224", pretrained=False)
net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
net.eval()

# x: (1, 3, 224, 224) preprocessed T2-SPIR slice
logits = net(x)
pred_mask = logits.softmax(dim=1).argmax(dim=1)  # 0=bg, 1=liver, 2=R kidney, 3=L kidney, 4=spleen
```

## Intended use and limitations

This model is a portfolio/research demo, not a validated clinical tool. It has not been
evaluated for regulatory clearance, was trained on a small public research dataset (CHAOS,
20 subjects), and its failure mode above is not fully understood. Do not use it for
diagnosis, treatment decisions, or any clinical purpose. See the linked GitHub repository
for the full evaluation methodology, failure analysis, and an interactive demo with
confidence/entropy visualization and an LLM-generated (Groq, Llama 3.3 70B) clinical report
that is explicitly not a medical finding.
