"""
Abdominal MRI Organ Segmentation — Gradio Dashboard
  Encoder : Swin Transformer (timm, ImageNet pretrained)
  Decoder : Dual Attention Fusion  (PAM + CAM + Semantic Module)
  Report  : Groq LLM  (llama-3.3-70b-versatile, free tier)

Run:
    python src/demo.py
    python src/demo.py --weights model/Best_SwinDAF-CHAOS.pth --share
"""

import sys, os, io, argparse, glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import sobel
import torch
import torch.nn.functional as F
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from models.swin_danet import SwinDAF
from report_generator import (
    generate_report,
    ORGAN_NAMES,
    CONFIDENCE_THRESHOLD,
    ENTROPY_THRESHOLD,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ORGAN_COLORS_NORM = {
    1: (1.00, 0.24, 0.24),    # Liver        — red
    2: (0.24, 0.86, 0.24),    # Right Kidney — green
    3: (0.24, 0.47, 1.00),    # Left Kidney  — blue
    4: (1.00, 0.82, 0.00),    # Spleen       — yellow
}
OVERLAY_RGBA = {
    0: (0,   0,   0,   0),
    1: (255, 60,  60,  180),
    2: (60,  220, 60,  180),
    3: (60,  120, 255, 180),
    4: (255, 210, 0,   180),
}
BOUNDARY_COLORS = {
    1: (255, 60,  60),
    2: (60,  220, 60),
    3: (60,  120, 255),
    4: (255, 210, 0),
}

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

DESCRIPTION = """
# 🏥 Abdominal MRI Organ Segmentation

**Architecture** — Swin Transformer encoder + Dual Attention Fusion decoder (PAM · CAM · Semantic Module)
**Dataset** — CHAOS MRI · 5 classes: Background · Liver · Right Kidney · Left Kidney · Spleen
**Report** — Groq LLM clinical impression (Llama 3.3 70B · free tier)
"""

# ── Model ─────────────────────────────────────────────────────────────────────
_cache: dict = {}

def load_model(weights_path: str, encoder: str, device: torch.device) -> SwinDAF:
    model = SwinDAF(num_classes=5, encoder_name=encoder, pretrained=False)
    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            # Encoder variant mismatch — auto-detect from checkpoint and reload
            ckpt_stage2_blocks = sum(
                1 for k in state if "encoder.backbone.layers_2.blocks." in k
                and k.endswith("norm1.weight")
            )
            variant_map = {6: "swin_tiny_patch4_window7_224",
                           18: "swin_small_patch4_window7_224"}
            detected = variant_map.get(ckpt_stage2_blocks, encoder)
            if detected != encoder:
                print(f"[load_model] Encoder mismatch — checkpoint uses {detected}, reloading.")
                model = SwinDAF(num_classes=5, encoder_name=detected, pretrained=False)
                model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, size: int = 224) -> torch.Tensor:
    arr = np.array(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

def run_model(model, tensor: torch.Tensor, device: torch.device):
    with torch.no_grad():
        logits = model(tensor.to(device))          # (1,5,H,W)
    probs = F.softmax(logits, dim=1).squeeze(0)    # (5,H,W)
    mask  = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    probs = probs.cpu().numpy()                    # (5,H,W)
    return mask, probs

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── Visualisations ────────────────────────────────────────────────────────────
def make_overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    base  = img.convert("L").resize((w, h)).convert("RGBA")
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, rgba in OVERLAY_RGBA.items():
        layer[mask == cls_id] = rgba
    return Image.alpha_composite(base, Image.fromarray(layer, "RGBA")).convert("RGB")


def make_boundary_overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Organ edges (Sobel) drawn in organ colour on the grayscale MRI."""
    h, w = mask.shape
    gray  = np.array(img.convert("L").resize((w, h)), dtype=np.float32)
    # Normalise to 0-255 for the RGB canvas
    gray  = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)
    canvas = np.stack([gray, gray, gray], axis=2).copy()

    for cls_id, color in BOUNDARY_COLORS.items():
        region = (mask == cls_id).astype(np.float32)
        if region.sum() < 10:
            continue
        ex = sobel(region, axis=1)
        ey = sobel(region, axis=0)
        edge = np.hypot(ex, ey) > 0.15
        canvas[edge] = color

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(canvas)
    ax.set_title("Organ Boundaries", fontsize=12, fontweight="bold")
    ax.axis("off")
    patches = [mpatches.Patch(color=[c/255 for c in col], label=ORGAN_NAMES[i])
               for i, col in BOUNDARY_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=7,
              framealpha=0.7, ncol=2)
    plt.tight_layout()
    return fig_to_pil(fig)


def make_entropy_map(probs: np.ndarray) -> Image.Image:
    """Shannon entropy per pixel — high entropy = model is uncertain."""
    eps     = 1e-8
    entropy = -(probs * np.log(probs + eps)).sum(axis=0)   # (H,W)
    max_ent = np.log(probs.shape[0])                        # log(num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left: entropy heatmap
    im = axes[0].imshow(entropy, cmap="inferno", vmin=0, vmax=max_ent)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Entropy (uncertainty)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Right: histogram of entropy values
    axes[1].hist(entropy.ravel(), bins=40, color="#e05c5c", edgecolor="white",
                 linewidth=0.5)
    axes[1].axvline(entropy.mean(), color="white", linestyle="--",
                    linewidth=1.2, label=f"mean={entropy.mean():.2f}")
    axes[1].set_xlabel("Entropy", fontsize=10)
    axes[1].set_ylabel("Pixel count", fontsize=10)
    axes[1].set_title("Entropy Distribution", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_facecolor("#1a1a2e")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model Uncertainty", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig_to_pil(fig)


def make_confidence_map(probs: np.ndarray) -> Image.Image:
    confidence = probs.max(axis=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Model Confidence", fontsize=12, fontweight="bold")
    ax.axis("off")
    return fig_to_pil(fig)


def make_probability_maps(probs: np.ndarray) -> Image.Image:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    cmaps = ["Reds", "Greens", "Blues", "YlOrBr"]
    for i, (cls_id, name) in enumerate(ORGAN_NAMES.items()):
        ax = axes[i]
        ax.imshow(probs[cls_id], cmap=cmaps[i], vmin=0, vmax=1)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Per-Organ Probability Maps", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig_to_pil(fig)


def make_prob_histograms(probs: np.ndarray) -> Image.Image:
    """Distribution of softmax scores for each organ class."""
    colors = ["#ff3c3c", "#3cdc3c", "#3c78ff", "#ffd200"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    axes = axes.ravel()

    for i, (cls_id, name) in enumerate(ORGAN_NAMES.items()):
        p = probs[cls_id].ravel()
        ax = axes[i]
        ax.hist(p, bins=50, color=colors[i], edgecolor="white", linewidth=0.4)
        ax.axvline(p.mean(), color="white", linestyle="--",
                   linewidth=1.2, label=f"mean={p.mean():.3f}")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Probability", fontsize=8)
        ax.set_ylabel("Pixels", fontsize=8)
        ax.legend(fontsize=8)
        ax.set_facecolor("#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Probability Score Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig_to_pil(fig)


def make_bar_chart(stats: dict) -> Image.Image:
    names  = [ORGAN_NAMES[i] for i in range(1, 5)]
    values = [stats[i]["pct"] for i in range(1, 5)]
    colors = [ORGAN_COLORS_NORM[i] for i in range(1, 5)]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlabel("Slice Coverage (%)", fontsize=10)
    ax.set_title("Organ Coverage", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(max(values) * 1.2, 5))
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig_to_pil(fig)


def make_pixel_donut(stats: dict) -> Image.Image:
    labels = ["Background"] + [ORGAN_NAMES[i] for i in range(1, 5)]
    sizes  = [stats[0]["pct"]] + [stats[i]["pct"] for i in range(1, 5)]
    colors = ["#cccccc", "#ff3c3c", "#3cdc3c", "#3c78ff", "#ffd200"]
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not non_zero:
        non_zero = [("Background", 100, "#cccccc")]
    labels_, sizes_, colors_ = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(
        sizes_, colors=colors_, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.5),
    )
    ax.legend(wedges, [f"{l} ({s:.1f}%)" for l, s in zip(labels_, sizes_)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8, frameon=False, ncol=2)
    ax.set_title("Pixel Distribution", fontsize=12, fontweight="bold")
    return fig_to_pil(fig)


def compute_stats(mask: np.ndarray, probs: np.ndarray | None = None) -> dict:
    total = mask.size
    stats = {"total": total}

    confidence_map = probs.max(axis=0) if probs is not None else None
    entropy_map = (
        -(probs * np.log(probs + 1e-8)).sum(axis=0) if probs is not None else None
    )

    for i in range(5):
        organ_mask = mask == i
        count = int(organ_mask.sum())
        entry = {"pixels": count, "pct": count / total * 100}

        if probs is not None:
            if count > 0:
                mean_confidence = float(confidence_map[organ_mask].mean())
                mean_entropy = float(entropy_map[organ_mask].mean())
                low_confidence = (
                    mean_confidence < CONFIDENCE_THRESHOLD
                    or mean_entropy > ENTROPY_THRESHOLD
                )
            else:
                mean_confidence = None
                mean_entropy = None
                low_confidence = True
            entry["mean_confidence"] = mean_confidence
            entry["mean_entropy"] = mean_entropy
            entry["low_confidence"] = low_confidence

        stats[i] = entry
    return stats


def stats_to_markdown(stats: dict) -> str:
    rows = "\n".join(
        f"| {ORGAN_NAMES[i]} | {stats[i]['pixels']:,} | {stats[i]['pct']:.1f}% |"
        for i in range(1, 5)
    )
    return "| Organ | Pixels | Coverage |\n|-------|--------|----------|\n" + rows

# ── Main analysis function ────────────────────────────────────────────────────
def run_analysis(image, weights_path, encoder_name, api_key, use_rag):
    if image is None:
        return (None,) * 8 + ("Upload an MRI image first.", "", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (weights_path.strip(), encoder_name)
    if key not in _cache:
        try:
            _cache.clear()
            _cache[key] = load_model(weights_path.strip(), encoder_name, device)
        except Exception as e:
            return (None,) * 8 + (f"**Model error:** {e}", "", "")

    model  = _cache[key]
    pil    = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    tensor = preprocess(pil)
    mask, probs = run_model(model, tensor, device)

    overlay    = make_overlay(pil, mask)
    boundary   = make_boundary_overlay(pil, mask)
    conf_map   = make_confidence_map(probs)
    entropy    = make_entropy_map(probs)
    prob_maps  = make_probability_maps(probs)
    prob_hist  = make_prob_histograms(probs)
    stats      = compute_stats(mask, probs)
    bar_chart  = make_bar_chart(stats)
    donut      = make_pixel_donut(stats)
    stats_md   = stats_to_markdown(stats)
    report, passages_md = generate_report(stats, api_key=api_key, use_rag=use_rag)

    return overlay, boundary, conf_map, entropy, prob_maps, prob_hist, bar_chart, donut, stats_md, report, passages_md

# ── UI ────────────────────────────────────────────────────────────────────────
def build_ui(default_weights="", default_encoder="swin_tiny_patch4_window7_224", default_use_rag=True):
    with gr.Blocks(title="MRI Segmentation — SwinDAF") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # ── Inputs ───────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                image_in = gr.Image(label="Upload MRI Slice", type="pil", height=260)

                example_files = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.png")))
                if example_files:
                    gr.Examples(
                        examples=[[f] for f in example_files],
                        inputs=[image_in],
                        label="Try a bundled test-set slice (no upload needed)",
                    )

                with gr.Accordion("Model", open=True):
                    encoder_dd = gr.Dropdown(
                        choices=["swin_tiny_patch4_window7_224",
                                 "swin_small_patch4_window7_224",
                                 "swin_base_patch4_window7_224"],
                        value=default_encoder,
                        label="Swin Variant",
                        info="Tiny → fastest   Base → most accurate",
                    )
                    weights_in = gr.Textbox(
                        value=default_weights,
                        label="Checkpoint Path",
                        placeholder="model/Best_SwinDAF-CHAOS.pth",
                        info="Leave blank for random-weight demo mode",
                    )

                with gr.Accordion("LLM Report — Groq (free)", open=False):
                    api_key_in = gr.Textbox(
                        label="Groq API Key",
                        placeholder="gsk_...",
                        type="password",
                        info="Free key at console.groq.com or set GROQ_API_KEY env var",
                    )
                    use_rag_in = gr.Checkbox(
                        value=default_use_rag,
                        label="Use RAG grounding",
                        info="Ground the report in retrieved reference passages (recommended)",
                    )

                run_btn = gr.Button("▶  Analyse", variant="primary", size="lg")

            # ── Outputs ───────────────────────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs():

                    with gr.Tab("🎨 Segmentation"):
                        with gr.Row():
                            overlay_out  = gr.Image(label="Colour Overlay", height=300)
                            boundary_out = gr.Image(label="Organ Boundaries", height=300)

                    with gr.Tab("🌡️ Confidence & Uncertainty"):
                        with gr.Row():
                            conf_out    = gr.Image(label="Confidence Map", height=300)
                            entropy_out = gr.Image(label="Entropy / Uncertainty", height=300)

                    with gr.Tab("📊 Probability Maps"):
                        prob_out = gr.Image(label="Per-Organ Softmax Maps", height=250)
                        prob_hist_out = gr.Image(label="Score Distributions", height=280)

                    with gr.Tab("📈 Coverage Charts"):
                        with gr.Row():
                            bar_out   = gr.Image(label="Bar Chart", height=280)
                            donut_out = gr.Image(label="Pixel Distribution", height=280)

                stats_out  = gr.Markdown(label="Organ Statistics")
                report_out = gr.Textbox(label="Clinical Report (Groq LLM)", lines=7, interactive=False)
                with gr.Accordion("Retrieved passages (grounding)", open=False):
                    passages_out = gr.Markdown(label="Retrieved Passages")

        gr.Markdown(
            "---\n"
            "**Legend** — 🔴 Liver · 🟢 Right Kidney · 🔵 Left Kidney · 🟡 Spleen  \n"
            "*SwinDAF · PyTorch · timm · Gradio · Groq Llama 3.3 70B*\n\n"
            "⚠️ **Research and education demo. Not a medical device. Not for diagnostic use.**"
        )

        run_btn.click(
            fn=run_analysis,
            inputs=[image_in, weights_in, encoder_dd, api_key_in, use_rag_in],
            outputs=[
                overlay_out, boundary_out,
                conf_out, entropy_out,
                prob_out, prob_hist_out,
                bar_out, donut_out,
                stats_out, report_out, passages_out,
            ],
        )

    return demo

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",   action="store_true")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--weights", default="")
    parser.add_argument("--encoder", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--no_rag", action="store_true",
                         help="Disable RAG grounding by default (still toggleable in the UI)")
    args = parser.parse_args()

    build_ui(
        default_weights=args.weights,
        default_encoder=args.encoder,
        default_use_rag=not args.no_rag,
    ).launch(share=args.share, server_port=args.port, theme=gr.themes.Soft())
