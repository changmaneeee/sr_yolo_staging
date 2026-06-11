# 05-02. Sniper YOLO from-scratch 학습 (Fair eval용)

## 개요

Fair eval에서 사용한 Sniper YOLO. 각 SR backbone마다 **별도 학습**.

## 학습 절차

### 데이터 준비
1. Train HR 이미지(`smart_airbus_data/images/train/`)에서 LR 생성
2. LR → 해당 SR backbone → HR 출력 → 저장
   - 예: `weights/sr_dump_drct/train_hr/*.jpg`
3. Label은 원본 HR label 그대로 사용 (좌표는 0~1 정규화이므로 호환)

### YOLO 학습
- 모델: YOLOv8s
- 입력 해상도: 640 (또는 768)
- Optimizer: SGD (Ultralytics default)
- Augmentation: Ultralytics 기본
- Random seed: (시드 명시되면 추가, 없으면 Ultralytics default)
- Epoch: deadline 모드

### 각 SR별 결과
- `weights/yolo_8s_rfdn/weights/best.pt`
- `weights/yolo_8s_drct/weights/best.pt`
- `weights/yolo_8s_hat/best.pt`
- `weights/yolo_8s_man/best.pt`

## 공정성 보장

- 같은 train HR set으로 모든 SR이 SR 생성
- 같은 label 사용 (HR 좌표)
- 같은 YOLOv8s + 같은 augmentation
- 차이는 오직 **각 SR이 만든 HR 이미지의 특성** → 본 비교의 의도

## Fair eval에서의 사용

Fair eval은 이 from-scratch Sniper를 사용:
```bash
python eval_arch.py --sr-backbone rfdn --arch 4 \
  --sr-weight weights/rfdn_arch4/model_best.pt \
  --sniper-weight weights/yolo_8s_rfdn/weights/best.pt
```

자세히: [docs/06_evaluation/02_eval_arch_py.md](../06_evaluation/02_eval_arch_py.md)

## 검증 포인트

- [ ] 각 SR 전용 Sniper가 from-scratch로 학습
- [ ] 같은 train HR set 사용
- [ ] 같은 YOLOv8s 모델 구조
- [ ] 같은 augmentation
- [ ] val 누설 없음 (train만 사용)
