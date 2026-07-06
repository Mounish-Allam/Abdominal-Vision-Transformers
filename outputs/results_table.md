### Test-set Dice score (2D per-slice, mean ± std)

| Organ | DAF baseline (ResNeXt-101) | SwinDAF (Swin-Tiny + DAF) |
|---|---|---|
| Liver | 0.565 ± 0.462 | 0.654 ± 0.445 |
| Right Kidney | 0.799 ± 0.341 | 0.871 ± 0.292 |
| Left Kidney | 0.738 ± 0.384 | 0.808 ± 0.350 |
| Spleen | 0.706 ± 0.421 | 0.757 ± 0.397 |
| **Mean** | **0.702** | **0.772** |

### Test-set Dice score (3D per-subject, mean ± std)

| Organ | DAF baseline (ResNeXt-101) | SwinDAF (Swin-Tiny + DAF) |
|---|---|---|
| Liver | 0.484 ± 0.440 | 0.472 ± 0.463 |
| Right Kidney | 0.777 ± 0.127 | 0.906 ± 0.039 |
| Left Kidney | 0.743 ± 0.070 | 0.864 ± 0.002 |
| Spleen | 0.353 ± 0.353 | 0.404 ± 0.404 |
| **Mean** | **0.589** | **0.661** |

*3D scores computed per-subject over 2 held-out test subjects.*

### Provenance

- **DAF baseline (ResNeXt-101)**: checkpoint `model\Best_DAF-baseline.pth`, seed 42, 62 test slices, evaluated 20260704T052916Z
- **SwinDAF (Swin-Tiny + DAF)**: checkpoint `model\Best_SwinDAF-CHAOS.pth`, seed 42, 62 test slices, evaluated 20260704T052900Z
