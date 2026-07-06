---
source_url: "N/A"
license: "Original reference notes, no external source"
topic: "What T2-SPIR MRI is and why fat suppression is used"
---

# T2-SPIR MRI — Basics

T2-weighted MRI is a sequence type that produces strong contrast based on
tissue T2 relaxation times — fluid-containing structures (such as bile,
urine, or cysts) appear bright, while most solid organ parenchyma appears at
an intermediate signal level. This makes T2-weighted imaging useful for
distinguishing fluid from solid tissue and for delineating organ boundaries
against surrounding structures.

SPIR (Spectral Presaturation with Inversion Recovery) is a fat-suppression
technique applied on top of a T2-weighted sequence. Because the abdomen
contains substantial amounts of fat — both subcutaneous and around and
between internal organs — that fat can otherwise appear bright and compete
visually with anatomy of interest. SPIR selectively suppresses the fat
signal, which increases the contrast between organ parenchyma and the
surrounding fat, making organ borders sharper and easier to trace, whether by
a human reader or an automated segmentation model.

For a segmentation project like this one, T2-SPIR is a favorable input
modality precisely because of this improved organ-versus-fat contrast — it is
one of the reasons boundary delineation between adjacent structures (e.g.
liver against surrounding fat, or spleen against perisplenic fat) tends to be
more consistent than on non-fat-suppressed sequences.
