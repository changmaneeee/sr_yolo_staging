# 06-01. fixed_protocol.yaml

## 위치

`configs/fixed_protocol.yaml` (메인 repo)

## 역할

**모든 SR 백본 실험에서 동일해야 하는 hyperparameter를 강제**. CLI로 override 불가.

## 전체 내용

```yaml
# --- Scout YOLO (Arch4 전용) ---
scout:
  weight: "/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt"
  md5_prefix: "f9f175f7f758"
  description: "YOLOv8s stage2, LR improved, IAC 0.7986 기준"

# --- 평가 데이터셋 ---
dataset:
  lr_images: "/home/changmin/smart_airbus_data_lr/images/val"
  hr_images: "/home/changmin/smart_airbus_data/images/val"
  hr_labels: "/home/changmin/smart_airbus_data/labels/val"
  lr_labels: "/home/changmin/smart_airbus_data_lr/labels/val"
  filter: "labeled_only"
  expected_count: 6418

# --- Threshold (Arch4 Confidence-Adaptive) ---
threshold:
  pass1_conf: 0.0075
  high_conf: 0.45
  final_conf: 0.3       # 또는 0.25 (실험별)
  sniper_conf: 0.001

# --- Merge 정책 (Arch4) ---
merge:
  merge_iou: 0.5
  roi_expansion: 1.75
  crop_size_lr: 64
  batch_size_sr: 32

# --- Arch4 ROI-aware NMS 추가 파라미터 ---
arch4_nms:
  scout_nms_iou: 0.5
  roi_merge_iou: 0.3
  roi_center_ratio: 0.35
  sniper_nms_iou: 0.45
  final_nms_iou: 0.5
  drop_uncertain_if_sniper_hits: true
  sniper_score_bonus: 0.0
  merge_policy: "size_cond"
  final_fusion_method: "soft_nms"
  soft_nms_sigma: 0.3

# --- 평가 공통 설정 ---
evaluation:
  conf_threshold: 0.25
  iou_threshold: 0.5
  eval_space: "hr"
  device: "cuda"

# --- 측정 설정 (latency) ---
measurement:
  warmup: 30
  n_iter: 200
  sync: true
```

## 강제 항목

| 항목 | 값 | 이유 |
|:--|:--|:--|
| Scout weight | yolo_lr_improved/stage2/best.pt | 모든 SR 공통 |
| Scout MD5 | f9f175f7f758 | 변조 방지 |
| Val set | subset6418 (labeled_only) | 평가 set 통일 |
| Expected count | 6418 | 자동 검증 |
| Conf threshold | 0.25 | 모든 Arch 공통 NMS |
| IoU threshold | 0.5 | 모든 Arch 공통 NMS |
| Eval space | hr | 좌표 공간 통일 |

## 변경 차단

- CLI로 override 불가 → 변경 시 `fixed_protocol.yaml` 자체를 수정 + commit
- Pre-flight check가 매 실행 시 검증 (scout MD5, expected_count 등)

## 사고 기록 (final_conf 차이)

- Properbase Phase E mAP eval: `final_conf=0.25`
- NMS sweep v1: `final_conf=0.3` (잘못)
- NMS sweep v2: `final_conf=0.25` (수정)

v2에서 properbase와 일치 → 비교 가능.

## 검증 포인트

- [ ] fixed_protocol.yaml이 모든 실험에서 로드되는가
- [ ] CLI override가 차단되는가
- [ ] MD5 검증이 실행되는가
- [ ] expected_count: 6418이 검증되는가
