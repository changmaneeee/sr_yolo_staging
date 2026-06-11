"""
DRCT Wrapper — Unified interface for DRCT super-resolution backbone.

Reference:
    Hsu et al. "DRCT: Saving Image Super-resolution away from Information
    Bottleneck" CVPR-W 2024 NTIRE.
    https://github.com/ming053l/DRCT

Note: basicsr ARCH_REGISTRY has compatibility issues with modern torchvision.
      We import the DRCT class directly and bypass the registry.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Stub out basicsr registry to avoid import errors
import types
_fake_basicsr = types.ModuleType("basicsr")
_fake_utils = types.ModuleType("basicsr.utils")
_fake_registry = types.ModuleType("basicsr.utils.registry")

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

# Build complete basicsr stub
_fake_archs = types.ModuleType("basicsr.archs")
_fake_arch_util = types.ModuleType("basicsr.archs.arch_util")
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

# Now import DRCT arch directly
DRCT_PATH = Path(__file__).parent.parent.parent / "third_party" / "DRCT"
sys.path.insert(0, str(DRCT_PATH))

import importlib.util
_arch_path = DRCT_PATH / "drct" / "archs" / "DRCT_arch.py"
_spec = importlib.util.spec_from_file_location("drct_arch", _arch_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_DRCT = _mod.DRCT


# Model variant configs (from official test YAMLs)
DRCT_CONFIGS = {
    "base": dict(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv", gc=32,
    ),
    "large": dict(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.,
        depths=[6]*12, embed_dim=180,
        num_heads=[6]*12, mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv", gc=32,
    ),
}


class DRCTWrapper(nn.Module):
    """Drop-in SR backbone wrapper for DRCT. Same interface as RFDN."""

    def __init__(self, scale=4, pretrained_path=None, variant="base",
                 input_range="0-1", device="cuda", use_checkpoint=False):
        super().__init__()
        self.scale = scale
        self.input_range = input_range

        cfg = DRCT_CONFIGS[variant].copy()
        cfg["upscale"] = scale
        self.model = _DRCT(**cfg)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

        # gradient checkpointing은 unfreeze_for_phase3()에서 enable_gradient_checkpointing() 호출로 활성화
        if use_checkpoint:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self):
        """Gradient checkpointing 활성화 (Phase 3 full-unfreeze backward OOM 방지).

        self.model.forward_features의 layer 루프를 torch.utils.checkpoint로 감싸
        activation을 backward 때 재계산 → 메모리↓ (속도↓, 결과는 수학적으로 동일).
        third_party 원본 불변, 런타임에 forward_features 교체. 추론(no_grad)엔 영향 없음.
        """
        import torch.utils.checkpoint as cp
        m = self.model

        def forward_features_ckpt(x):
            x_size = (x.shape[2], x.shape[3])
            x = m.patch_embed(x)
            if m.ape:
                x = x + m.absolute_pos_embed
            x = m.pos_drop(x)
            for layer in m.layers:
                if self.training and torch.is_grad_enabled():
                    x = cp.checkpoint(layer, x, x_size, use_reentrant=False)
                else:
                    x = layer(x, x_size)
            x = m.norm(x)
            x = m.patch_unembed(x, x_size)
            return x

        m.forward_features = forward_features_ckpt
        self._gradient_checkpointing = True
        print("[DRCT] ✓ Gradient checkpointing 활성화 (RDG layers, backward OOM 방지)")

    def load_pretrained(self, path):
        print(f"[DRCT] Loading weights: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        weights = state.get("params_ema", state.get("params", state))
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if missing:
            print(f"  Missing: {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)} keys")
        print(f"[DRCT] ✓ Weights loaded")

    @property
    def feature_channels(self):
        """Feature channel count for Arch5 fusion compatibility."""
        return self.model.embed_dim  # 180 for base/large

    def forward_features(self, lr):
        """Extract intermediate features (before upsampling).
        Input [B,3,H,W] in [0,1] → Features [B, embed_dim, H, W]."""
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
        return x

    def forward(self, lr):
        """Full forward: Input [B,3,H,W] in [0,1] → HR [B,3,H*scale,W*scale] in [0,1]."""
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


if __name__ == "__main__":
    import time

    weight_path = Path("/home/changmin/dark_vessel_sr_yolo/weights/drct/DRCT_SRx4.pth")

    wrapper = DRCTWrapper(scale=4, variant="base",
                          pretrained_path=str(weight_path) if weight_path.exists() else None)
    wrapper = wrapper.cuda().eval()

    print(f"\nParams: {wrapper.num_parameters / 1e6:.2f}M")

    lr = torch.rand(1, 3, 64, 64).cuda()
    with torch.no_grad():
        hr = wrapper(lr)
    print(f"Input: {lr.shape} → Output: {hr.shape}")
    assert hr.shape == (1, 3, 256, 256), f"Shape mismatch: {hr.shape}"

    # Latency
    for _ in range(10):
        wrapper(lr)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        wrapper(lr)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"Latency (64×64): {ms:.2f} ms")

    print("\n✅ DRCT wrapper smoke test passed")
