---
source_url: "N/A"
license: "Original reference notes, no external source"
topic: "General limitations of automated/AI image segmentation"
---

# AI-Assisted Segmentation — General Caveats

Automated segmentation models, including the one used in this project,
predict a per-pixel organ label from a trained neural network. Several
general limitations apply to this kind of output, independent of any specific
model's measured accuracy:

- **No clinical judgment.** The model outputs a pixel classification based on
  patterns learned from training data; it does not incorporate patient
  history, symptoms, prior imaging, or laboratory results the way a
  radiologist's interpretation would.
- **Training-distribution dependence.** Performance is strongest on cases that
  resemble the data the model was trained and evaluated on. Unusual anatomy,
  pathology not represented in training data, or different scanner/protocol
  characteristics can degrade accuracy in ways that aren't always visible from
  the output alone.
- **Errors can look confident.** A segmentation mask does not automatically
  communicate uncertainty — a wrong prediction can still look visually
  plausible unless uncertainty measures (like per-pixel confidence or entropy)
  are explicitly computed and surfaced, as this project attempts to do.
- **Single-slice scope.** As with any per-slice analysis, a single 2D result
  does not capture whether an error is isolated to one slice or consistent
  across the full 3D volume.

Because of these limitations, automated segmentation output — and any report
generated from it — is best treated as a decision-support aid that a qualified
human should review, rather than a stand-alone diagnostic result.
