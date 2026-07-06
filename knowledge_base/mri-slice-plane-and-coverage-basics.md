---
source_url: "N/A"
license: "Original reference notes, no external source"
topic: "Why single-slice coverage percentage is not the same as organ volume"
---

# Single-Slice Coverage vs. Whole-Organ Volume

This project analyzes individual 2D axial MRI slices, and reports each
organ's coverage as a percentage of that single slice's pixels. This is a
deliberately local measurement and should not be interpreted as a statement
about the organ's total volume, its size relative to "normal" whole-organ
reference ranges, or its size in 3D.

A few concrete consequences of this slice-based approach:

- The same organ will show very different coverage percentages at different
  slice levels — a mid-liver slice shows far more liver than a slice near the
  liver's superior or inferior edge.
- An organ can be completely or partially absent from a given slice simply
  because that slice does not intersect the organ at all, not because the
  organ is abnormal or the segmentation failed.
- Comparing one organ's coverage percentage to another organ's coverage
  percentage on the *same* slice is only meaningful as a rough relative
  indicator (e.g., "the liver takes up much more of this slice than the
  spleen does"), not as a precise volumetric comparison.

Any clinical-sounding statement about whether an organ is "enlarged" or
"reduced" in size should be understood as relating only to this single slice,
and any full assessment of organ size would require a 3D volumetric analysis
across the entire scan, which is outside the scope of a per-slice report.
