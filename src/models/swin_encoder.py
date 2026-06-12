import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


SWIN_CHANNELS = {
    'swin_tiny_patch4_window7_224':  [96,  192,  384,  768],
    'swin_small_patch4_window7_224': [96,  192,  384,  768],
    'swin_base_patch4_window7_224':  [128, 256,  512, 1024],
}


class SwinEncoder(nn.Module):
    """
    Wraps a timm Swin Transformer to produce 4 hierarchical feature maps
    at strides [4, 8, 16, 32] relative to the input, each in (B, C, H, W) format.
    """

    def __init__(self, model_name: str = 'swin_tiny_patch4_window7_224', pretrained: bool = True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm>=0.9.0")

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.channels = SWIN_CHANNELS.get(model_name, [96, 192, 384, 768])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C_in, H, W) — input image tensor
        Returns:
            list of 4 tensors in (B, C, H, W) format:
            [(B, 96, H/4, W/4), ..., (B, 768, H/32, W/32)] for swin_tiny.
        """
        # timm Swin returns (B, H, W, C) — permute to (B, C, H, W) for conv layers
        return [f.permute(0, 3, 1, 2).contiguous() for f in self.backbone(x)]
