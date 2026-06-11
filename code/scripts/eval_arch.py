#!/usr/bin/env python3
"""
eval_arch.py — 표준 SR 백본 교체 실험 공통 평가 스크립트
=======================================================

모든 SR 백본(RFDN/DRCT/HAT/MAN) × 모든 Architecture(0/2/4/5) 평가를
단일 스크립트로 실행. 고정 요소는 fixed_protocol.yaml에서 읽어 강제.

Usage:
    # RFDN Arch4
    python eval_arch.py --sr-backbone rfdn --arch 4 \
        --sr-weight weights/rfdn/model_best.pt \
        --sniper-weight weights/yolohr/8s/best.pt

    # DRCT Arch0
    python eval_arch.py --sr-backbone drct --arch 0 \
        --sr-weight weights/sr_finetuned/drct/best.pt

    # RFDN Arch5 (checkpoint 방식)
    python eval_arch.py --sr-backbone rfdn --arch 5 \
        --arch5-checkpoint iac_lab/runs/.../phase3_best.pt

근거: Notion '[표준 프로토콜] SR 백본 교체 실험' (2026-05-18)
"""

import sys
import time
import json
import hashlib
import argparse
import logging
from pathlib import Path

import yaml
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from iac_scripts.arch4_eval_ultralytics import (
    load_yaml_dict, patch_arch4_config, list_images, cv2_to_tensor,
    load_yolo_label_file, xywhn_to_xyxy_pixels, box_iou_torch, process_batch,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ============================================================================
# Pre-flight Check
# ============================================================================

def preflight_check(protocol: dict, args) -> dict:
    """실험 시작 전 고정 요소를 검증하고 로그에 기록."""
    log.info("=" * 60)
    log.info(" PRE-FLIGHT CHECK")
    log.info("=" * 60)

    checks = {}
    all_pass = True

    # 1. Scout weight (Arch4 only)
    scout_cfg = protocol.get("scout", {})
    scout_path = Path(scout_cfg.get("weight", ""))
    expected_md5 = scout_cfg.get("md5_prefix", "")
    if args.arch == 4:
        if scout_path.exists():
            with open(scout_path, "rb") as f:
                actual_md5 = hashlib.md5(f.read()).hexdigest()[:12]
            match = actual_md5 == expected_md5
            checks["scout_weight"] = {"path": str(scout_path), "md5_prefix": actual_md5, "match": match}
            log.info(f"  Scout weight: {scout_path}")
            log.info(f"  Scout MD5:    {actual_md5} (expected {expected_md5}) {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            checks["scout_weight"] = {"path": str(scout_path), "error": "NOT FOUND"}
            log.error(f"  Scout weight NOT FOUND: {scout_path}")
            all_pass = False
    else:
        checks["scout_weight"] = "N/A (arch != 4)"
        log.info(f"  Scout weight: N/A (arch {args.arch})")

    # 2. Dataset
    ds_cfg = protocol.get("dataset", {})
    lr_images_dir = Path(ds_cfg.get("lr_images", ""))
    hr_labels_dir = Path(ds_cfg.get("hr_labels", ""))
    expected_count = ds_cfg.get("expected_count", 6418)

    labeled_stems = set()
    if hr_labels_dir.exists():
        labeled_stems = {f.stem for f in hr_labels_dir.glob("*.txt") if f.stat().st_size > 0}
    img_paths = [p for p in list_images(lr_images_dir) if p.stem in labeled_stems]
    actual_count = len(img_paths)
    count_match = actual_count == expected_count

    checks["dataset"] = {"count": actual_count, "expected": expected_count, "match": count_match}
    log.info(f"  Dataset:      {actual_count} images (expected {expected_count}) {'PASS' if count_match else 'FAIL'}")
    if not count_match:
        all_pass = False

    # 3. Thresholds
    thr_cfg = protocol.get("threshold", {})
    checks["threshold"] = dict(thr_cfg)
    log.info(f"  Thresholds:   pass1={thr_cfg.get('pass1_conf')}, high={thr_cfg.get('high_conf')}, "
             f"final={thr_cfg.get('final_conf')}, sniper={thr_cfg.get('sniper_conf')}")

    # 4. Merge
    merge_cfg = protocol.get("merge", {})
    checks["merge"] = dict(merge_cfg)
    log.info(f"  Merge:        iou={merge_cfg.get('merge_iou')}, expansion={merge_cfg.get('roi_expansion')}, "
             f"crop={merge_cfg.get('crop_size_lr')}")

    # 5. Sniper weight
    if args.sniper_weight:
        sniper_exists = Path(args.sniper_weight).exists()
        checks["sniper_weight"] = {"path": args.sniper_weight, "exists": sniper_exists}
        log.info(f"  Sniper:       {args.sniper_weight} ({'FOUND' if sniper_exists else 'NOT FOUND'})")
    else:
        checks["sniper_weight"] = "N/A"
        log.info(f"  Sniper:       N/A")

    # 6. SR weight
    sr_exists = Path(args.sr_weight).exists() if args.sr_weight else False
    checks["sr_weight"] = {"path": args.sr_weight or "N/A", "exists": sr_exists}
    log.info(f"  SR weight:    {args.sr_weight} ({'FOUND' if sr_exists else 'NOT FOUND'})")

    # 7. Experiment info
    log.info(f"  SR backbone:  {args.sr_backbone}")
    log.info(f"  Architecture: {args.arch}")

    status = "ALL PASS" if all_pass else "CHECKS FAILED"
    checks["status"] = status
    log.info(f"  Status:       {status}")
    log.info("=" * 60)

    if not all_pass:
        log.warning("Pre-flight check failed. Proceeding with warnings.")

    return checks


# ============================================================================
# Model loading
# ============================================================================

def build_arch4_config(protocol: dict, args) -> dict:
    """fixed_protocol.yaml + SR/Sniper weight로 Arch4 config 생성.
    IAC 0.7986 기준: Arch4RoiAwareNMS + 6개 추가 파라미터 포함."""
    thr = protocol["threshold"]
    merge = protocol["merge"]
    scout = protocol["scout"]
    nms = protocol.get("arch4_nms", {})

    cfg = {
        "data": {"upscale_factor": 4},
        "model": {
            "sr": {
                "type": args.sr_backbone,
                "weights": args.sr_weight,
                "rfdn": {"nf": 50, "num_modules": 4},
            },
            "yolo": {
                "weights_hr": args.sniper_weight or "",
                "weights_lr": scout["weight"],
                "classes": 1,
                "num_classes": 1,
            },
            "arch4": {
                "pass1_conf": thr["pass1_conf"],
                "high_conf": thr["high_conf"],
                "pass2_conf": thr["high_conf"],
                "final_conf": thr["final_conf"],
                "sniper_conf": thr["sniper_conf"],
                "merge_iou": merge["merge_iou"],
                "roi_expansion": merge["roi_expansion"],
                "crop_size_lr": merge["crop_size_lr"],
                "batch_size_sr": merge["batch_size_sr"],
                # 6개 ROI-aware NMS 파라미터
                "scout_nms_iou": nms.get("scout_nms_iou", 0.5),
                "roi_merge_iou": nms.get("roi_merge_iou", 0.3),
                "roi_center_ratio": nms.get("roi_center_ratio", 0.35),
                "sniper_nms_iou": nms.get("sniper_nms_iou", 0.45),
                "final_nms_iou": nms.get("final_nms_iou", 0.5),
                "drop_uncertain_if_sniper_hits": nms.get("drop_uncertain_if_sniper_hits", True),
                "sniper_score_bonus": nms.get("sniper_score_bonus", 0.0),
                "merge_policy": nms.get("merge_policy", "size_cond"),
                "final_fusion_method": nms.get("final_fusion_method", "soft_nms"),
                "soft_nms_sigma": nms.get("soft_nms_sigma", 0.3),
            },
        },
    }
    return patch_arch4_config(cfg)


def _load_sr_model(sr_backbone: str, sr_weight: str, device: str):
    """SR 모델만 단독 로드 (Arch0/2 SR cache 생성용)."""
    if sr_backbone == "rfdn":
        from src.models.sr_models.rfdn import RFDN
        sr = RFDN(in_channels=3, out_channels=3, nf=50, num_modules=4, upscale=4, input_range="0-255")
        sr.load_pretrained(sr_weight)
        return sr.to(device).eval()
    elif sr_backbone == "drct":
        from sci_lab.backbones.drct_wrapper import DRCTWrapper
        sr = DRCTWrapper(scale=4, pretrained_path=None, variant="base", input_range="0-255")
        ckpt = torch.load(sr_weight, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            sr.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "params_ema" in ckpt:
            sr.model.load_state_dict(ckpt["params_ema"], strict=False)
        else:
            sr.load_state_dict(ckpt, strict=False)
        return sr.to(device).eval()
    elif sr_backbone == "hat":
        from sci_lab.backbones.hat_wrapper import HATWrapper
        sr = HATWrapper(scale=4, pretrained_path=None, variant="base", input_range="0-255")
        ckpt = torch.load(sr_weight, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            sr.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "params_ema" in ckpt:
            sr.model.load_state_dict(ckpt["params_ema"], strict=False)
        else:
            sr.load_state_dict(ckpt, strict=False)
        return sr.to(device).eval()
    elif sr_backbone == "man":
        from sci_lab.backbones.man_wrapper import MANWrapper
        sr = MANWrapper(scale=4, pretrained_path=None, variant="base", input_range="0-255")
        ckpt = torch.load(sr_weight, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            sr.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "params_ema" in ckpt:
            sr.model.load_state_dict(ckpt["params_ema"], strict=False)
        else:
            sr.load_state_dict(ckpt, strict=False)
        return sr.to(device).eval()
    else:
        raise ValueError(f"Unsupported SR backbone for standalone loading: {sr_backbone}")


def load_model(arch: int, protocol: dict, args, device: str):
    """Architecture에 맞는 모델 로드."""
    if arch == 0 and args.sr_backbone != "rfdn":
        # Arch0 only: DRCT/HAT/MAN은 SR cache 방식.
        # Arch2는 Gate pipeline이 필수이므로 SR cache 불가.
        return None

    if arch == 0:
        from src.models.pipelines.arch0_sequential import Arch0Sequential
        cfg = _build_arch0_config(protocol, args)
        model = Arch0Sequential(cfg).to(device)
        model.eval()
        return model

    elif arch == 2:
        from src.models.pipelines.arch2_softgate import Arch2SoftGate
        cfg = _build_arch2_config(protocol, args)
        model = Arch2SoftGate(cfg).to(device)
        model.eval()
        return model

    elif arch == 4:
        from src.models.pipelines.arch4_roi_awareNMS import Arch4RoiAwareNMS
        cfg = build_arch4_config(protocol, args)
        model = Arch4RoiAwareNMS(cfg)
        model.eval()
        return model

    elif arch == 5:
        from src.models.pipelines.arch5b_fusion import Arch5BFusion
        cfg = _build_arch5_config(protocol, args)
        model = Arch5BFusion(cfg).to(device)
        if args.arch5_checkpoint:
            ckpt = torch.load(args.arch5_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt, strict=False)
            log.info(f"  Loaded Arch5 checkpoint: {args.arch5_checkpoint}")
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown arch: {arch}")


def _build_arch0_config(protocol, args):
    from types import SimpleNamespace
    return SimpleNamespace(
        data=SimpleNamespace(upscale_factor=4, scale_factor=4),
        model=SimpleNamespace(
            sr_model=args.sr_backbone,
            sr_type=args.sr_backbone,
            weights=SimpleNamespace(
                sr_model=args.sr_weight,
                detector=args.detector_weight or "",
            ),
            sr_config=SimpleNamespace(input_range="0-255", nf=50, scale=4),
            yolo=SimpleNamespace(
                weights_path=args.detector_weight or "",
                num_classes=1,
                imgsz=640,
            ),
            arch0=SimpleNamespace(detach_sr=True),
            rfdn=SimpleNamespace(nf=50, num_modules=4),
        ),
        training=SimpleNamespace(),
    )


def _build_arch2_config(protocol, args):
    from types import SimpleNamespace
    return SimpleNamespace(
        data=SimpleNamespace(upscale_factor=4),
        model=SimpleNamespace(
            sr_type=args.sr_backbone,
            sr=SimpleNamespace(weights=args.sr_weight, pretrain_path=args.sr_weight),
            rfdn=SimpleNamespace(nf=50, num_modules=4, pretrain_path=args.sr_weight),
            yolo=SimpleNamespace(
                weights_path=args.detector_weight or "",
                num_classes=1,
                imgsz=640,
            ),
            gate=SimpleNamespace(
                base_channels=32, num_layers=4, in_channels=3,
                weights_path=args.gate_weight or "",
                use_selective_inference=True,
                inference_threshold=0.0,          # IAC 기준: 모든 이미지 soft blend
                blend_selected_inference=True,     # IAC 기준: gate*SR + (1-gate)*bilinear
            ),
        ),
    )


def _build_arch5_config(protocol, args):
    return {
        "data": {"upscale_factor": 4},
        "device": "cuda",
        "model": {
            "sr_type": args.sr_backbone,
            "sr": {
                "type": args.sr_backbone,
                "weights": args.sr_weight,
                "rfdn": {"nf": 50, "num_modules": 4},
            },
            "yolo": {
                "weights": args.detector_weight or "",
                "weights_path": args.detector_weight or "",
                "num_classes": 1,
            },
            "detector_input": "sr",
            "fusion": {
                "sr_channels": 50,
                "det_channels": [128, 256, 512],
                "use_cross_attn": [False, True, True],
            },
        },
        "evaluation": {
            "conf_threshold": protocol["evaluation"]["conf_threshold"],
            "iou_threshold": protocol["evaluation"]["iou_threshold"],
        },
    }


# ============================================================================
# Evaluation loop
# ============================================================================

def evaluate_sr_cache(arch: int, protocol: dict, args, device: str):
    """Arch0/2 + non-RFDN SR: SR cache 방식 평가.
    SR 모델로 이미지 생성 → YOLO val()로 mAP 측정."""
    import gc
    from ultralytics import YOLO
    import torchvision.transforms as T

    ds_cfg = protocol["dataset"]
    lr_images_dir = Path(ds_cfg["lr_images"])
    hr_images_dir = Path(ds_cfg["hr_images"])
    hr_labels_dir = Path(ds_cfg["hr_labels"])

    labeled_stems = {f.stem for f in Path(hr_labels_dir).glob("*.txt") if f.stat().st_size > 0}
    img_paths = sorted([p for p in list_images(lr_images_dir) if p.stem in labeled_stems])

    log.info(f"\n[SR Cache Eval] {args.sr_backbone.upper()} Arch{arch}, {len(img_paths)} images")

    # 1. SR cache 생성
    import tempfile, shutil, yaml as _yaml
    cache_dir = Path(tempfile.mkdtemp(prefix=f"eval_{args.sr_backbone}_arch{arch}_"))
    cache_img = cache_dir / "images" / "val"
    cache_lbl = cache_dir / "labels" / "val"
    cache_img.mkdir(parents=True)
    cache_lbl.mkdir(parents=True)

    sr_model = _load_sr_model(args.sr_backbone, args.sr_weight, device)
    to_t = T.ToTensor()
    from PIL import Image

    log.info(f"  Generating SR cache → {cache_dir}")
    t0 = time.perf_counter()
    for i, lp in enumerate(img_paths):
        lr = to_t(Image.open(lp).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            if args.sr_backbone == "rfdn":
                sr = torch.clamp(sr_model(lr * 255.0) / 255.0, 0, 1)
            else:
                sr = torch.clamp(sr_model(lr * 255.0) / 255.0, 0, 1)
        arr = (sr[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(cache_img / f"{lp.stem}.png")
        src = hr_labels_dir / f"{lp.stem}.txt"
        if src.exists():
            shutil.copy(src, cache_lbl / f"{lp.stem}.txt")
        del lr, sr
        torch.cuda.empty_cache()
        if (i + 1) % 1000 == 0:
            log.info(f"    {i+1}/{len(img_paths)}")
            gc.collect()
            torch.cuda.empty_cache()

    del sr_model
    gc.collect()
    torch.cuda.empty_cache()
    cache_time = time.perf_counter() - t0
    log.info(f"  SR cache done ({cache_time:.0f}s)")

    # data.yaml
    with open(cache_dir / "sr_data.yaml", "w") as f:
        _yaml.safe_dump({"path": str(cache_dir), "train": "images/val", "val": "images/val",
                         "names": {0: "ship"}, "nc": 1}, f)

    # 2. YOLO val
    detector_weight = args.detector_weight or ""
    log.info(f"  YOLO eval: {detector_weight}")
    yolo = YOLO(detector_weight)
    results = yolo.val(data=str(cache_dir / "sr_data.yaml"), imgsz=640,
                       conf=0.001, iou=0.6, max_det=300, device=device,
                       batch=8, half=False, verbose=False)

    mAP50 = float(results.box.map50)
    mAP5095 = float(results.box.map)
    prec = float(results.box.mp)
    rec = float(results.box.mr)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)

    return {
        "mAP50": round(mAP50, 4),
        "mAP50-95": round(mAP5095, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "n_images": len(img_paths),
        "method": "sr_cache",
        "sr_cache_time_s": round(cache_time, 1),
    }


def evaluate(model, arch: int, protocol: dict, args, device: str):
    """공통 평가 루프. 모든 Arch에서 동일한 metric 계산."""

    # Arch0 + non-RFDN only: SR cache 방식 (Arch2는 Gate 필수 → pipeline 실행)
    if model is None and arch == 0:
        return evaluate_sr_cache(arch, protocol, args, device)

    from ultralytics.utils.metrics import ap_per_class

    ds_cfg = protocol["dataset"]
    lr_images_dir = Path(ds_cfg["lr_images"])
    hr_images_dir = Path(ds_cfg["hr_images"])
    hr_labels_dir = Path(ds_cfg["hr_labels"])

    labeled_stems = {f.stem for f in Path(hr_labels_dir).glob("*.txt") if f.stat().st_size > 0}
    img_paths = sorted([p for p in list_images(lr_images_dir) if p.stem in labeled_stems])

    upscale = 4
    eval_space = "hr"
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    eval_cfg = protocol.get("evaluation", {})
    conf_threshold = eval_cfg.get("conf_threshold", 0.25)
    iou_threshold = eval_cfg.get("iou_threshold", 0.5)

    log.info(f"\n[Evaluation] {len(img_paths)} images, conf={conf_threshold}, iou={iou_threshold}")

    stats = []
    tp50, fp50, fn50 = 0, 0, 0
    t0 = time.perf_counter()
    total_infer_ms = 0.0

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lr_tensor = cv2_to_tensor(img).to(device)

        # GT labels
        lb = hr_labels_dir / f"{img_path.stem}.txt"
        labels_xywhn = load_yolo_label_file(lb)
        hr_img_path = hr_images_dir / img_path.name
        hr_img = cv2.imread(str(hr_img_path))
        if hr_img is not None:
            H_hr, W_hr = hr_img.shape[0], hr_img.shape[1]
        else:
            H_hr, W_hr = img.shape[0] * upscale, img.shape[1] * upscale
        gt_xyxy = xywhn_to_xyxy_pixels(labels_xywhn, W_hr, H_hr)

        # Inference
        infer_t0 = time.perf_counter()
        with torch.no_grad():
            if arch == 5:
                result = model.inference(lr_tensor, conf_threshold=conf_threshold,
                                         iou_threshold=iou_threshold, return_features=False)
                dets = result["detections"]
            elif arch == 4:
                out = model.forward(lr_tensor, debug=False)
                dets = out["detections"]
            elif arch == 0:
                out = model.forward(lr_tensor)
                # Arch0: returns (sr_image, list_of_det_dicts)
                dets = out[1] if isinstance(out, tuple) else out
                if isinstance(dets, dict) and "detections" in dets:
                    dets = dets["detections"]
                if not isinstance(dets, list):
                    dets = [dets]
            elif arch == 2:
                out = model.forward(lr_tensor)
                # Arch2SoftGate: returns dict with 'detections' key
                dets = out["detections"]
                if not isinstance(dets, list):
                    dets = [dets]
        total_infer_ms += (time.perf_counter() - infer_t0) * 1000.0

        det = dets[0]
        boxes = det["boxes"].detach().to(device).float()
        scores = det["scores"].detach().to(device).float()
        classes = det["classes"].detach().to(device).float()

        # Scale to HR space if needed
        # Arch4: boxes are in LR space → scale to HR
        # Arch0/2: _predict_detector already returns boxes in SR (768) space → no scaling
        # Arch5: inference() handles internally
        if arch == 4 and eval_space == "hr":
            H_lr, W_lr = img.shape[0], img.shape[1]
            sx = W_hr / W_lr
            sy = H_hr / H_lr
            if boxes.numel() > 0:
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

        # Build prediction tensor
        if boxes.numel() == 0:
            pred = torch.zeros((0, 6), device=device)
        else:
            pred = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)

        # GT tensor
        gt_t = torch.from_numpy(gt_xyxy).to(device).float() if gt_xyxy.shape[0] > 0 else \
            torch.zeros((0, 5), device=device)

        # Direct IoU@0.5 metrics
        n_gt = gt_t.shape[0]
        if pred.shape[0] > 0 and n_gt > 0:
            iou_mat = box_iou_torch(gt_t[:, 1:5], pred[:, :4])
            matched = set()
            for g in range(n_gt):
                best_iou, best_p = iou_mat[g].max(0)
                if best_iou.item() >= 0.5 and best_p.item() not in matched:
                    tp50 += 1
                    matched.add(best_p.item())
                else:
                    fn50 += 1
            fp50 += pred.shape[0] - len(matched)
        elif pred.shape[0] > 0:
            fp50 += pred.shape[0]
        elif n_gt > 0:
            fn50 += n_gt

        # AP stats
        correct = process_batch(pred, gt_t, iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], gt_t[:, 0]))

        if (idx + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t0
            log.info(f"  {idx+1}/{len(img_paths)} ({elapsed:.0f}s)")

    # Compute AP
    total_time = time.perf_counter() - t0
    stats_np = [torch.cat(x, 0).cpu().numpy() if isinstance(x[0], torch.Tensor) else np.concatenate(x, 0)
                for x in zip(*stats)]

    if len(stats_np) == 4 and len(stats_np[0]) > 0:
        tp_cat, conf_cat, pred_cls, target_cls = stats_np
        results = ap_per_class(tp_cat, conf_cat, pred_cls, target_cls, plot=False)
        if isinstance(results, tuple) and len(results) >= 6:
            tp_r, fp_r, p_r, r_r, f1_r, ap_r = results[:6]
            ap50 = float(ap_r[:, 0].mean()) if ap_r.ndim == 2 else float(ap_r.mean())
            ap5095 = float(ap_r.mean())
            precision = float(p_r.mean())
            recall = float(r_r.mean())
            f1 = float(f1_r.mean())
        else:
            ap50 = ap5095 = precision = recall = f1 = 0.0
    else:
        ap50 = ap5095 = precision = recall = f1 = 0.0

    avg_ms = total_infer_ms / max(len(img_paths), 1)

    return {
        "mAP50": round(ap50, 4),
        "mAP50-95": round(ap5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp50": tp50,
        "fp50": fp50,
        "fn50": fn50,
        "n_images": len(img_paths),
        "avg_ms_per_img": round(avg_ms, 2),
        "total_time_s": round(total_time, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Standard SR backbone swap evaluation")

    # Variable (SR backbone)
    p.add_argument("--sr-backbone", required=True, choices=["rfdn", "drct", "hat", "man"],
                   help="SR backbone type")
    p.add_argument("--arch", required=True, type=int, choices=[0, 2, 4, 5],
                   help="Architecture number")
    p.add_argument("--sr-weight", required=True, help="SR model weight path")
    p.add_argument("--sniper-weight", default=None, help="Sniper YOLO weight (Arch4, from-scratch)")
    p.add_argument("--gate-weight", default=None, help="Gate weight (Arch2)")
    p.add_argument("--detector-weight", default=None, help="YOLO detector weight (Arch0/2/5)")
    p.add_argument("--arch5-checkpoint", default=None, help="Arch5 full checkpoint path")

    # Fixed (protocol)
    p.add_argument("--protocol", default="configs/fixed_protocol.yaml",
                   help="Fixed protocol config (DO NOT change for main results)")

    # Output
    p.add_argument("--out-json", default=None, help="Output JSON path (auto-generated if omitted)")

    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load protocol
    protocol_path = PROJECT_ROOT / args.protocol
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    protocol = load_yaml_dict(str(protocol_path))

    # Pre-flight check
    pf = preflight_check(protocol, args)

    # Load model
    log.info(f"\nLoading {args.sr_backbone.upper()} Arch{args.arch}...")
    model = load_model(args.arch, protocol, args, device)

    # Evaluate
    metrics = evaluate(model, args.arch, protocol, args, device)

    # Print results
    log.info(f"\n{'='*60}")
    log.info(f" RESULT: {args.sr_backbone.upper()} Arch{args.arch}")
    log.info(f"{'='*60}")
    log.info(f"  mAP@50:    {metrics['mAP50']}")
    log.info(f"  mAP@50-95: {metrics['mAP50-95']}")
    log.info(f"  F1:        {metrics['f1']}")
    log.info(f"  Precision: {metrics['precision']}")
    log.info(f"  Recall:    {metrics['recall']}")
    if 'tp50' in metrics:
        log.info(f"  TP/FP/FN:  {metrics['tp50']}/{metrics['fp50']}/{metrics['fn50']}")
    if 'avg_ms_per_img' in metrics:
        log.info(f"  Avg ms:    {metrics['avg_ms_per_img']}")
    log.info(f"  Images:    {metrics['n_images']}")
    log.info(f"{'='*60}")

    # Save
    out_path = args.out_json or f"sci_lab/results/eval_{args.sr_backbone}_arch{args.arch}.json"
    out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "standard_protocol_eval",
        "sr_backbone": args.sr_backbone,
        "architecture": args.arch,
        "preflight": pf,
        "metrics": metrics,
        "args": {
            "sr_weight": args.sr_weight,
            "sniper_weight": args.sniper_weight,
            "gate_weight": args.gate_weight,
            "detector_weight": args.detector_weight,
            "arch5_checkpoint": args.arch5_checkpoint,
            "protocol": args.protocol,
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
