"""Shape tests for the PAM / CAM / semantic attention modules - small tensors, CPU only."""

from __future__ import annotations

import torch

from models.attention import CAM_Module, PAM_Module, semanticModule


def test_pam_module_preserves_shape():
    module = PAM_Module(in_dim=8)
    x = torch.randn(2, 8, 8, 8)

    out = module(x)

    assert out.shape == x.shape


def test_cam_module_preserves_shape():
    module = CAM_Module(in_dim=8)
    x = torch.randn(2, 8, 8, 8)

    out = module(x)

    assert out.shape == x.shape


def test_pam_module_is_identity_at_init_since_gamma_starts_at_zero():
    # gamma is nn.Parameter(torch.zeros(1)), so out = gamma * attn(x) + x == x
    # until gamma is trained away from zero - a real, checkable invariant.
    module = PAM_Module(in_dim=8)
    x = torch.randn(2, 8, 8, 8)

    out = module(x)

    assert torch.allclose(out, x)


def test_cam_module_is_identity_at_init_since_gamma_starts_at_zero():
    module = CAM_Module(in_dim=8)
    x = torch.randn(2, 8, 8, 8)

    out = module(x)

    assert torch.allclose(out, x)


def test_semantic_module_output_shapes():
    module = semanticModule(in_dim=8)
    x = torch.randn(2, 8, 16, 16)

    sem_vector, decoded = module(x)

    # enc2 is (B, in_dim*4, H/4, W/4) flattened to 1D
    assert sem_vector.shape == (2 * (8 * 4) * 4 * 4,)
    # dec1 is projected back to in_dim channels at the original spatial size
    assert decoded.shape == (2, 8, 16, 16)
