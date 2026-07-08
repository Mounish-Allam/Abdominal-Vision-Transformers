### Test-set Dice score (2D per-slice, mean ± std)

| Organ | DAF baseline (ResNeXt-101) | SwinDAF (Swin-Tiny + DAF) |
|---|---|---|
| Liver | 0.565 ± 0.462 | 0.689 ± 0.428 |
| Right Kidney | 0.799 ± 0.341 | 0.812 ± 0.348 |
| Left Kidney | 0.738 ± 0.384 | 0.816 ± 0.342 |
| Spleen | 0.706 ± 0.421 | 0.731 ± 0.403 |
| **Mean** | **0.702** | **0.762** |

### Test-set Dice score (3D per-subject, mean ± std)

| Organ | DAF baseline (ResNeXt-101) | SwinDAF (Swin-Tiny + DAF) |
|---|---|---|
| Liver | 0.484 ± 0.440 | 0.479 ± 0.465 |
| Right Kidney | 0.777 ± 0.127 | 0.910 ± 0.017 |
| Left Kidney | 0.743 ± 0.070 | 0.898 ± 0.014 |
| Spleen | 0.353 ± 0.353 | 0.393 ± 0.393 |
| **Mean** | **0.589** | **0.670** |

*3D scores computed per-subject over 2 held-out test subjects.*

### Provenance

- **DAF baseline (ResNeXt-101)**: checkpoint `model\Best_DAF-baseline.pth`, seed 42, 62 test slices, evaluated 20260704T052916Z
- **SwinDAF (Swin-Tiny + DAF)**: checkpoint `model\Best_SwinDAF-CHAOS.pth`, seed 42, 62 test slices, evaluated 20260708T022402Z
