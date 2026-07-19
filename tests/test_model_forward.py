"""Forward-pass shape tests for the segmentation network - CPU only, no pretrained download."""

from __future__ import annotations

import torch

from models.swin_danet import SwinDAF


def test_swindaf_forward_shape_rgb_input():
    net = SwinDAF(num_classes=5, encoder_name="swin_tiny_patch4_window7_224", pretrained=False)
    net.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = net(x)

    assert out.shape == (1, 5, 224, 224)


def test_swindaf_forward_shape_grayscale_input_is_repeated_to_rgb():
    net = SwinDAF(num_classes=5, encoder_name="swin_tiny_patch4_window7_224", pretrained=False)
    net.eval()

    x = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        out = net(x)

    assert out.shape == (1, 5, 224, 224)


def test_swindaf_forward_batch_dimension_preserved():
    net = SwinDAF(num_classes=5, encoder_name="swin_tiny_patch4_window7_224", pretrained=False)
    net.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(x)

    assert out.shape == (2, 5, 224, 224)


def test_swindaf_eval_output_is_softmax_ready_logits_not_probabilities():
    # In eval mode the forward pass averages 4 raw prediction heads - it should
    # not already be a normalized probability distribution over classes.
    net = SwinDAF(num_classes=5, encoder_name="swin_tiny_patch4_window7_224", pretrained=False)
    net.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = net(x)

    per_pixel_sum = out.sum(dim=1)
    assert not torch.allclose(per_pixel_sum, torch.ones_like(per_pixel_sum), atol=1e-3)
