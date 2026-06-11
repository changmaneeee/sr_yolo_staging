# 05-01. Scout YOLO 학습

## 개요

Arch4의 Scout YOLO는 LR 이미지 검출용. **모든 SR backbone 실험에서 공통 사용**되는 단일 weight.

## 학습 절차

### Stage 1: deadline_try
- 데이터: `/home/changmin/smart_airbus_data_lr/` (LR)
- 모델: YOLOv8s
- 입력 해상도: 192 (LR 그대로)
- Epoch: deadline 모드 (early stop on plateau)
- Augmentation: Mosaic, HSV, hflip, scale (Ultralytics 기본)
- Optimizer: SGD (Ultralytics default)
- 저장: `weights/yolo_lr_improved/8s_aug_deadline_try/weights/best.pt`

### Stage 2: stage2 refinement
- Stage 1 weights를 starting point
- 추가 epoch (fine-tune)
- 저장: `weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt`

## 사용된 데이터

- Train: `smart_airbus_data_lr/images/train/` (108,414장)
- Val: `smart_airbus_data_lr/images/val/` (28,884장)
- Label: `smart_airbus_data_lr/labels/train/`, `val/`

→ **subset6418과 무관한 표준 train/val split 사용**.

## 공정성 보장

- 모든 SR backbone 실험에서 같은 Scout weight (`stage2/best.pt`) 사용
- Pre-flight check 함수가 MD5 prefix `f9f175f7f758`로 검증
- fixed_protocol.yaml에 명시되어 변경 차단

## 검증 포인트

- [ ] LR train set만 사용 (val 누설 없음)
- [ ] Stage 1 → Stage 2 단계 명확
- [ ] 단일 weight를 모든 실험에서 공유
- [ ] MD5 검증으로 weight 변조 방지
