---
source_url: "N/A"
license: "Original reference notes, no external source"
topic: "Lay explanation of entropy/uncertainty and why high entropy flags review"
---

# Model Entropy — What It Means

Entropy is a measure of how spread out a model's predicted probabilities are
across the possible classes for a given pixel. If a model assigns nearly all
of the probability to a single class, entropy is low (the model is decisive).
If the model spreads probability more evenly across several classes, entropy
is higher (the model is uncertain which class is correct). Entropy and
confidence are related but not identical: confidence looks only at the top
predicted class's probability, while entropy accounts for the full
distribution across all classes, so it can catch cases where a pixel is
genuinely torn between two or more plausible classes even if the top class's
probability isn't especially low.

Averaging per-pixel entropy across all pixels predicted as a given organ gives
that organ's mean entropy for the slice. A high mean entropy for an organ
suggests its predicted region contains many ambiguous pixels — commonly at
boundaries between adjacent organs, or where the organ is thin, partially cut
by the slice plane, or has an atypical appearance.

In this project, an organ with unusually high mean entropy (or low mean
confidence) is explicitly flagged in the generated report as warranting human
review, rather than having the report state a finding about that organ with
unwarranted certainty. This is a deliberate design choice to keep automated
uncertainty signals visible to the reader instead of hidden inside a
confident-sounding sentence.
