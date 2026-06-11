"""
MAN Wrapper — Unified interface for MAN super-resolution backbone.

Reference:
    Wang et al. "Multi-scale Attention Network for Single Image Super-Resolution"
    CVPRW 2024. https://github.com/icandle/MAN
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


class _FakeRegistry:
    def register(self, cls=None):
        if cls is not None:
            return cls

        def wrapper(c):
            return c
        return wrapper


_fake_registry.ARCH_REGISTRY = _FakeRegistry()
_fake_utils.scandir = lambda *a, **kw: []
_fake_utils.registry = _fake_registry

sys.modules["basicsr"] = _fake_basicsr
sys.modules["basicsr.utils"] = _fake_utils
sys.modules["basicsr.utils.registry"] = _fake_registry

MAN_PATH = Path(__file__).parent.parent.parent / "third_party" / "MAN"
sys.path.insert(0, str(MAN_PATH))

import importlib.util
_arch_path = MAN_PATH / "archs" / "MAN_arch.py"
_spec = importlib.util.spec_from_file_location("man_arch", _arch_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_MAN = _mod.MAN


# MAN base config (matches MANx4_DF2K.pth, ~8.7M params)
MAN_CONFIGS = {
    "base": dict(
        n_resblocks=36, n_resgroups=1, n_colors=3,
        n_feats=180, scale=4, res_scale=1.0,
    ),
    "light": dict(
        n_resblocks=24, n_resgroups=1, n_colors=3,
        n_feats=60, scale=4, res_scale=1.0,
    ),
    "tiny": dict(
        n_resblocks=5, n_resgroups=1, n_colors=3,
        n_feats=48, scale=4, res_scale=1.0,
    ),
}


class MANWrapper(nn.Module):
    """Drop-in SR backbone wrapper for MAN. Same interface as DRCTWrapper."""

    def __init__(self, scale=4, pretrained_path=None, variant="base",
                 input_range="0-1", device="cuda"):
        super().__init__()
        self.scale = scale
        self.input_range = input_range

        cfg = MAN_CONFIGS[variant].copy()
        cfg["scale"] = scale
        self.model = _MAN(**cfg)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        print(f"[MAN] Loading weights: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        weights = state.get("params_ema", state.get("params", state))
        # MAN class overrides load_state_dict and returns None
        self.model.load_state_dict(weights, strict=False)
        print(f"[MAN] ✓ Weights loaded")

    def enable_gradient_checkpointing(self):
        """Gradient checkpointing 활성화 (Phase 3 full-unfreeze 메모리 절감).

        forward_features의 body resblock 루프(n_resblocks개)를 torch.utils.checkpoint로
        감싸 activation을 backward 때 재계산 → 메모리↓ (속도 약간↓, 결과 수학적 동일).
        HAT/DRCT와 동일 취지. third_party 원본 불변, 추론(no_grad)엔 영향 없음.
        """
        self._use_checkpoint = True
        print("[MAN] ✓ Gradient checkpointing 활성화 (body resblocks, backward 메모리 절감)")

    @property
    def feature_channels(self):
        """Feature channel count for Arch5 fusion compatibility (MAN base: n_feats=180)."""
        return self.model.head.out_channels

    def forward_features(self, lr):
        """Extract intermediate features (before upsampling).
        Input [B,3,H,W] in [0,1] → Features [B, n_feats, H, W].

        MAN의 main forward()와 동일한 전반부 흐름:
            x = sub_mean(lr)
            x = head(x)         # n_colors → n_feats
            res = x
            for block in body: res = block(res)
            if n_resgroups > 1: res = body_t(res) + x
        """
        if self.input_range == "0-255":
            lr = lr / 255.0
        x = self.model.sub_mean(lr)
        x = self.model.head(x)
        res = x
        if getattr(self, "_use_checkpoint", False) and self.training and torch.is_grad_enabled():
            # ⚠️ checkpoint는 ResGroup 통째가 아니라 그 안의 개별 MAB 블록 단위로 해야
            #    backward 재계산 peak가 1개 블록으로 줄어듦 (통째로 하면 36블록이 한꺼번에
            #    살아나 효과 0). ResGroup.forward(clone→36×MAB→body_t+residual)를 동일 복제.
            import torch.utils.checkpoint as cp
            for rg in self.model.body:          # ResGroup(s) — base는 1개
                rg_in = res
                r = rg_in.clone()
                for mab in rg.body:             # 36개 MAB 블록 개별 checkpoint
                    r = cp.checkpoint(mab, r, use_reentrant=False)
                res = rg.body_t(r) + rg_in
        else:
            for block in self.model.body:
                res = block(res)
        if self.model.n_resgroups > 1:
            res = self.model.body_t(res) + x
        return res  # [B, n_feats, H, W]

    def forward_reconstruct(self, features):
        """Reconstruct HR image from features.
        Features [B, n_feats, H, W] → HR [B,3,H*scale,W*scale] in [0,1]."""
        x = self.model.tail(features)
        x = self.model.add_mean(x)
        if self.input_range == "0-255":
            x = x * 255.0
        return x

    def forward(self, lr):
        if self.input_range == "0-255":
            lr = lr / 255.0
        out = self.model(lr)
        # MAN forward returns (sr, feature) tuple — take SR output only
        hr = out[0] if isinstance(out, tuple) else out
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
