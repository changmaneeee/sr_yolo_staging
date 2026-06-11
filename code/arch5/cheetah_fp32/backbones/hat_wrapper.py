"""
HAT Wrapper — Unified interface for HAT super-resolution backbone.

Reference:
    Chen et al. "Activating More Pixels in Image Super-Resolution Transformer"
    CVPR 2023. https://github.com/XPixelGroup/HAT
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

# basicsr stub (same pattern as drct_wrapper.py)
_fake_basicsr = types.ModuleType("basicsr")
_fake_utils = types.ModuleType("basicsr.utils")
_fake_registry = types.ModuleType("basicsr.utils.registry")
_fake_archs = types.ModuleType("basicsr.archs")
_fake_arch_util = types.ModuleType("basicsr.archs.arch_util")


class _FakeRegistry:
    def register(self, cls=None):
        if cls is not None:
            return cls

        def wrapper(c):
            return c
        return wrapper


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


_fake_arch_util.to_2tuple = _to_2tuple
_fake_arch_util.trunc_normal_ = _trunc_normal_
_fake_registry.ARCH_REGISTRY = _FakeRegistry()
_fake_utils.scandir = lambda *a, **kw: []
_fake_utils.registry = _fake_registry

sys.modules["basicsr"] = _fake_basicsr
sys.modules["basicsr.utils"] = _fake_utils
sys.modules["basicsr.utils.registry"] = _fake_registry
sys.modules["basicsr.archs"] = _fake_archs
sys.modules["basicsr.archs.arch_util"] = _fake_arch_util

HAT_PATH = Path(__file__).parent.parent.parent / "third_party" / "HAT"
sys.path.insert(0, str(HAT_PATH))

import importlib.util
_arch_path = HAT_PATH / "hat" / "archs" / "hat_arch.py"
_spec = importlib.util.spec_from_file_location("hat_arch", _arch_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_HAT = _mod.HAT


# Official SRx4 ImageNet-pretrain config (HAT/options/test/HAT_SRx4_ImageNet-pretrain.yml)
HAT_CONFIGS = {
    "base": dict(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    ),
}


class HATWrapper(nn.Module):
    """Drop-in SR backbone wrapper for HAT. Same interface as DRCTWrapper."""

    def __init__(self, scale=4, pretrained_path=None, variant="base",
                 input_range="0-1", device="cuda", use_checkpoint=False):
        super().__init__()
        self.scale = scale
        self.input_range = input_range

        cfg = HAT_CONFIGS[variant].copy()
        cfg["upscale"] = scale
        self.model = _HAT(**cfg)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

        # gradient checkpointing은 unfreeze_for_phase3()에서 enable_gradient_checkpointing() 호출로 활성화
        # (use_checkpoint 인자는 호환성 위해 받지만, arch의 dead 옵션이라 실제 적용은 아래 메서드가 함)
        if use_checkpoint:
            self.enable_gradient_checkpointing()

    def load_pretrained(self, path):
        print(f"[HAT] Loading weights: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        weights = state.get("params_ema", state.get("params", state))
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if missing:
            print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)} keys")
        print(f"[HAT] ✓ Weights loaded")

    def enable_gradient_checkpointing(self):
        """Gradient checkpointing 활성화 (Phase 3 full-unfreeze backward OOM 방지).

        self.model.forward_features의 `for layer in self.layers` 루프에서
        각 RHAG block을 torch.utils.checkpoint로 감싸 activation을 저장 대신
        backward 때 재계산 → 메모리 40-50%↓ (속도 20-30%↓, 결과는 수학적으로 동일).

        third_party 원본 파일은 건드리지 않고 런타임에 forward_features를 교체.
        추론(no_grad) 시엔 checkpoint가 자동으로 일반 forward로 동작 → 영향 없음.
        """
        import torch.utils.checkpoint as cp
        m = self.model

        def forward_features_ckpt(x):
            x_size = (x.shape[2], x.shape[3])
            attn_mask = m.calculate_mask(x_size).to(x.device)
            params = {'attn_mask': attn_mask,
                      'rpi_sa': m.relative_position_index_SA,
                      'rpi_oca': m.relative_position_index_OCA}
            x = m.patch_embed(x)
            if m.ape:
                x = x + m.absolute_pos_embed
            x = m.pos_drop(x)
            for layer in m.layers:
                if self.training and torch.is_grad_enabled():
                    # use_reentrant=False: params(dict) 같은 non-tensor 인자 안전
                    x = cp.checkpoint(layer, x, x_size, params, use_reentrant=False)
                else:
                    x = layer(x, x_size, params)
            x = m.norm(x)
            x = m.patch_unembed(x, x_size)
            return x

        m.forward_features = forward_features_ckpt
        self._gradient_checkpointing = True
        print("[HAT] ✓ Gradient checkpointing 활성화 (RHAG layers, backward OOM 방지)")

    @property
    def feature_channels(self):
        """Feature channel count for Arch5 fusion compatibility (HAT: embed_dim=180)."""
        return self.model.embed_dim

    def forward_features(self, lr):
        """Extract intermediate features (before upsampling).
        Input [B,3,H,W] in [0,1] → Features [B, embed_dim, H, W].

        HAT의 main forward()와 동일한 전반부 흐름 (DRCT와 같은 패턴):
            x = (lr - mean) * img_range
            x = conv_first(x)
            x = conv_after_body(forward_features(x)) + x
        """
        if self.input_range == "0-255":
            lr = lr / 255.0
        self.model.mean = self.model.mean.type_as(lr)
        x = (lr - self.model.mean) * self.model.img_range
        x = self.model.conv_first(x)
        x = self.model.conv_after_body(self.model.forward_features(x)) + x
        return x  # [B, 180, H, W]

    def forward_reconstruct(self, features):
        """Reconstruct HR image from features.
        Features [B, embed_dim, H, W] → HR [B,3,H*scale,W*scale] in [0,1]."""
        x = self.model.conv_before_upsample(features)
        x = self.model.conv_last(self.model.upsample(x))
        x = x / self.model.img_range + self.model.mean.type_as(x)
        if self.input_range == "0-255":
            x = x * 255.0
        return x

    def forward(self, lr):
        if self.input_range == "0-255":
            lr = lr / 255.0
        hr = self.model(lr)
        if self.input_range == "0-255":
            hr = hr * 255.0
        return hr

    @torch.no_grad()
    def inference(self, lr):
        self.eval()
        return self.forward(lr)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
