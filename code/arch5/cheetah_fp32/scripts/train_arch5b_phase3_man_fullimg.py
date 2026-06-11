#!/usr/bin/env python3
"""
Arch5B Phase 3 with MAN backbone -- Joint Training (FULL IMAGE)
================================================================
Problem: MAN was trained on 64x64 LR crops (256x256 HR) due to OOM on 4060.
         This doesn't generalize to 192x192 full images at inference.
Solution: RTX 4090 (24GB) can handle full 192x192 LR -> 768x768 HR images.

Key differences from train_arch5b_phase3_man.py:
  - SRDetPairDataset (full images, NO cropping)
  - Resume from Phase 2 fullimg best checkpoint
  - batch_size=2, grad_accum=1 (effective batch=2) + warmup 5ep -- RFDN 레시피 정렬
  - OUT_DIR = iac_lab/runs/arch5b_phase3_man_fullimg

Phase 3: Full unfreeze, joint end-to-end training.
"""

import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from PIL import Image
import cv2

PROJECT_ROOT = Path("/home/jovyan/changmin/dark_vessel_research/handoff_cheetah_20260525/code")
sys.path.insert(0, str(PROJECT_ROOT))
import yaml

SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
HR_ROOT = Path("/home/jovyan/changmin/dark_vessel_research/smart_airbus_data")
LR_ROOT = Path("/home/jovyan/changmin/dark_vessel_research/smart_airbus_data_lr")
OUT_DIR = PROJECT_ROOT / "iac_lab/runs/arch5b_phase3_man_fullimg"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class SRDetPairDataset(Dataset):
    """Full-image LR/HR pair dataset with YOLO labels (no cropping)."""

    def __init__(self, split="train", max_images=0):
        hr_img_dir = HR_ROOT / "images" / split
        lr_img_dir = LR_ROOT / "images" / split
        label_dir = HR_ROOT / "labels" / split

        # Only images with non-empty labels
        label_stems = set()
        for lf in label_dir.glob("*.txt"):
            if lf.stat().st_size > 0:
                with open(lf) as f:
                    if any(l.strip() for l in f):
                        label_stems.add(lf.stem)

        lr_stems = {f.stem for f in lr_img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
        hr_stems = {f.stem for f in hr_img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
        valid = sorted(label_stems & lr_stems & hr_stems)

        if max_images > 0:
            valid = valid[:max_images]

        self.samples = valid
        self.hr_img_dir = hr_img_dir
        self.lr_img_dir = lr_img_dir
        self.label_dir = label_dir
        print(f"[Dataset] {split}: {len(self.samples)} paired images (full image, positive only)")

    def __len__(self):
        return len(self.samples)

    def _find(self, directory, stem):
        for ext in [".jpg", ".png", ".jpeg"]:
            p = directory / (stem + ext)
            if p.exists():
                return p
        return None

    def __getitem__(self, idx):
        stem = self.samples[idx]
        lr_path = self._find(self.lr_img_dir, stem)
        hr_path = self._find(self.hr_img_dir, stem)

        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0

        lr_t = torch.from_numpy(lr).permute(2, 0, 1).contiguous()
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).contiguous()

        # Labels: YOLO format (normalized coords, no remapping needed for full images)
        labels = []
        label_path = self.label_dir / f"{stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([int(float(parts[0]))] + [float(p) for p in parts[1:5]])

        labels_t = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        return lr_t, hr_t, labels_t, stem


def collate_fn(batch):
    lrs = torch.stack([b[0] for b in batch])
    hrs = torch.stack([b[1] for b in batch])
    targets = []
    for i, b in enumerate(batch):
        if b[2].shape[0] > 0:
            bi = torch.full((b[2].shape[0], 1), i, dtype=torch.float32)
            targets.append(torch.cat([bi, b[2]], dim=1))
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    stems = [b[3] for b in batch]
    return lrs, hrs, targets, stems


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    epochs = 50
    batch_size = 2   # RFDN 정렬: real 2 (accum 없음) -> effective 2
    grad_accum = 1   # RFDN 정렬: accum 없음 (effective = real batch = 2)
    lr_rate = 5e-5
    patience = 10  # RFDN 베이스라인과 동일
    warmup_epochs = 5  # RFDN 정렬: 5 epoch linear warmup

    print("=" * 60)
    print("Arch5B Phase 3 -- MAN Backbone (Joint Training, FULL IMAGE)")
    print(f"Full 192x192 LR -> 768x768 HR (NO cropping)")
    print(f"Batch: {batch_size} x grad_accum {grad_accum} = effective {batch_size*grad_accum}")
    print(f"Epochs: {epochs}, LR: {lr_rate}, Patience: {patience}, Warmup: {warmup_epochs}ep")
    print(f"Target GPU: RTX 4090 (24GB)")
    print("=" * 60)

    train_ds = SRDetPairDataset("train")
    val_ds = SRDetPairDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True, collate_fn=collate_fn,
                              drop_last=True, prefetch_factor=6, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn,
                            prefetch_factor=2, persistent_workers=True)

    cfg_path = PROJECT_ROOT / "configs/experiment/arch5b_phase3_man.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from src.models.pipelines.arch5b_fusion import Arch5BFusion
    model = Arch5BFusion(cfg)
    model._sr_weight = 0.3
    model._det_weight = 1.0

    # Resume: prefer Phase 3 fullimg best (crash recovery), else Phase 2 fullimg best
    phase3_ckpt = OUT_DIR / "phase3_man_fullimg_best.pt"
    phase2_ckpt = PROJECT_ROOT / "iac_lab/runs/arch5b_phase2_man_fullimg/phase2_man_fullimg_best.pt"
    start_epoch = 0

    if phase3_ckpt.exists():
        print(f"[Resume] Loading Phase 3 fullimg best (crash recovery): {phase3_ckpt}")
        state = torch.load(str(phase3_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        hist_path = OUT_DIR / "history.json"
        if hist_path.exists():
            with open(hist_path) as _f:
                prev_hist = json.load(_f)
            start_epoch = len(prev_hist)
            print(f"[Resume] Continuing from epoch {start_epoch}")
    elif phase2_ckpt.exists():
        print(f"[Resume] Loading Phase 2 fullimg best: {phase2_ckpt}")
        state = torch.load(str(phase2_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        print(f"[Resume] Phase 2 fullimg fusion weights loaded successfully")
    else:
        print(f"[WARNING] No fullimg checkpoint found, training from scratch")

    model.to(device)

    # Phase 3: Full unfreeze
    param_groups = model.unfreeze_for_phase3()
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable: {total_trainable/1e6:.2f}M")
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    optimizer = AdamW([
        {"params": [p for p in model.sr_model.parameters() if p.requires_grad], "lr": lr_rate * 0.2},
        {"params": [p for p in model.detector.parameters() if p.requires_grad], "lr": lr_rate * 0.6},
        {"params": [p for p in model.fusion.parameters() if p.requires_grad], "lr": lr_rate},
    ], weight_decay=0.01)
    # RFDN 정렬: 5 epoch linear warmup -> cosine. per-group lr에 동일 배율 적용.
    _eta_min_ratio = 0.01
    def _lr_factor(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / warmup_epochs
        prog = float(ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return _eta_min_ratio + (1.0 - _eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_factor)

    # AMP (Mixed Precision)
    scaler = torch.cuda.amp.GradScaler()
    print("[AMP] Full fp32 training (AMP disabled)")

    best_val = float("inf"); best_epoch = -1; no_improve = 0
    history = []

    # Load previous history if resuming
    if start_epoch > 0:
        hist_path = OUT_DIR / "history.json"
        if hist_path.exists():
            with open(hist_path) as _f:
                history = json.load(_f)
            best_rec = min(history, key=lambda x: x["val_loss"])
            best_val = best_rec["val_loss"]
            best_epoch = best_rec["epoch"]
            print(f"[Resume] Previous best: epoch {best_epoch}, val={best_val:.4f}")
        # Step scheduler forward
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        eloss = edet = esr = 0; nb = 0; t0 = time.perf_counter()

        for bi, (lr_imgs, hr_imgs, targets, stems) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device); hr_imgs = hr_imgs.to(device); targets = targets.to(device)
            try:
                with torch.cuda.amp.autocast():
                    dets, feats = model.forward(lr_imgs, return_features=True)
                    ld = model.compute_loss((dets, feats), targets, lr_image=lr_imgs, hr_gt=hr_imgs)
                    loss = ld["total"] / grad_accum  # Scale for accumulation
                if loss.requires_grad and torch.isfinite(loss):
                    scaler.scale(loss).backward()
                if (bi + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                eloss += ld["total"].item(); edet += ld.get("det_loss", torch.tensor(0)).item()
                esr += ld.get("sr_loss", torch.tensor(0)).item(); nb += 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA" in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise
            if (bi+1) % 1000 == 0:
                print(f"  [{bi+1}/{len(train_loader)}] loss={ld['total'].item():.4f}", flush=True)

        scheduler.step()
        avg = eloss/max(1,nb); adet = edet/max(1,nb); asr = esr/max(1,nb)
        elapsed = time.perf_counter() - t0

        alphas = {}
        for n, p in model.named_parameters():
            if "alpha" in n: alphas[n.split(".")[-2]] = round(torch.sigmoid(p).item(), 4)

        # Val (also AMP)
        model.eval(); vloss = vdet = vsr = vn = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs, targets, stems in val_loader:
                lr_imgs = lr_imgs.to(device); hr_imgs = hr_imgs.to(device); targets = targets.to(device)
                try:
                    with torch.cuda.amp.autocast():
                        d, f = model.forward(lr_imgs, return_features=True)
                        ld = model.compute_loss((d,f), targets, lr_image=lr_imgs, hr_gt=hr_imgs)
                    vloss += ld["total"].item(); vdet += ld.get("det_loss", torch.tensor(0)).item()
                    vsr += ld.get("sr_loss", torch.tensor(0)).item(); vn += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); continue

        va = vloss/max(1,vn)
        rec = {"epoch": epoch, "train_loss": round(avg,4), "val_loss": round(va,4),
               "train_det": round(adet,4), "train_sr": round(asr,4),
               "val_sr": round(vsr/max(1,vn),4), "alphas": alphas,
               "time_s": round(elapsed,1), "batches": nb}
        history.append(rec)

        astr = " ".join(f"{v:.3f}" for v in alphas.values())
        print(f"Epoch {epoch:>3d}/{epochs}: train={avg:.4f} val={va:.4f} "
              f"det={adet:.4f} sr={asr:.4f}/{vsr/max(1,vn):.4f} "
              f"alpha=[{astr}] ({elapsed:.0f}s)", flush=True)

        if va < best_val:
            best_val = va; best_epoch = epoch; no_improve = 0
            torch.save(model.state_dict(), OUT_DIR / "phase3_man_fullimg_best.pt")
            print(f"  -> New best! val={va:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stop at epoch {epoch}"); break

        # Save history every epoch (crash-safe)
        with open(OUT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        # Save last.pt every epoch so stop-and-resume loses at most one epoch
        torch.save(model.state_dict(), OUT_DIR / "phase3_man_fullimg_last.pt")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), OUT_DIR / f"phase3_man_fullimg_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), OUT_DIR / "phase3_man_fullimg_last.pt")
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PHASE 3 COMPLETE (FULL IMAGE). Best epoch {best_epoch}, val={best_val:.4f}")
    print(f"Checkpoint: {OUT_DIR / 'phase3_man_fullimg_best.pt'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
