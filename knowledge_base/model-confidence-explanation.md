---
source_url: "N/A"
license: "Original reference notes, no external source"
topic: "Lay explanation of softmax confidence for AI segmentation reports"
---

# Model Confidence — What It Means

For each pixel, a segmentation model like the one in this project produces a
probability for each possible organ class (background, liver, right kidney,
left kidney, spleen), and these probabilities sum to 1 across the classes.
The "confidence" for a pixel is the probability assigned to whichever class
the model ultimately predicts for that pixel — a confidence near 1.0 means the
model considered that prediction highly likely relative to the alternatives;
a confidence closer to an even split across classes (e.g. near 0.2 for 5
classes) means the model was much less certain.

Averaging this per-pixel confidence across all the pixels predicted as a
given organ gives that organ's mean confidence for the slice. A high mean
confidence suggests the model's prediction for that organ's boundary and
extent was reasonably decisive; a lower mean confidence suggests the model
found the pixels genuinely ambiguous — often at organ boundaries, in regions
with unusual signal, or when an organ is barely present in the slice.

Confidence is a statement about the model's own certainty, not a guarantee of
correctness — a model can be confidently wrong, particularly on cases unlike
its training data. It is best used as one signal among several (alongside
entropy and visual review) for deciding whether a region deserves closer human
attention, not as a stand-alone accuracy guarantee.
