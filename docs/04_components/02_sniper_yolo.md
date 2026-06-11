# 04-02. Sniper YOLO (Arch4 HR Crop Detector)

## 역할

Arch4의 두 번째 검출기. **SR로 만든 HR crop(256×256)에서 정밀 검출**.

## SR별 별도 학습 (중요)

Sniper는 SR 출력에서 동작하므로 **각 SR backbone마다 별도 학습**.
이를 어기면 공정성 위배 (rfdn base 사고 참조).

## Weight 종류

### From-scratch Sniper (Fair eval 용)
- `weights/yolo_8s_rfdn/weights/best.pt`
- `weights/yolo_8s_drct/weights/best.pt`
- `weights/yolo_8s_hat/best.pt`
- `weights/yolo_8s_man/best.pt`

각각 자기 SR의 출력으로 from-scratch 학습.

### Hardneg-finetuned Sniper (Properbase, NMS sweep 용)
- RFDN: `weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/weights/best.pt`
- DRCT: `weights/yolo_sniper_hardneg/20260609_070148_hardneg_drct_pb/weights/best.pt`
- HAT: `weights/yolo_sniper_hardneg/20260609_*_hardneg_hat_pb/weights/best.pt`
- MAN: `weights/yolo_sniper_hardneg/20260609_*_hardneg_man_pb/weights/best.pt`

각각 Old Pipeline Phase B+C로 강화 학습.

## 학습 절차 (Old Pipeline)

각 SR별로 동일한 5단계:

### Phase A — Crop Dump
SR로 만든 HR crop들을 디스크에 저장.

### Phase B — Crop Fine-tune (50ep)
- Base: 각 SR 전용 `yolo_8s_{sr}/best.pt` (rfdn base 사고 후 수정)
- 50 epochs fine-tune on the dumped crops

### Phase C — Hardneg Mining (25ep)
- Phase B 결과에서 high-confidence FP를 hard negatives로 식별
- Negative crops를 추가하여 25 epochs fine-tune

### Phase D — Interpolation (6개 alpha)
- Phase B와 Phase C weights의 linear interpolation
- α ∈ {0.20, 0.30, 0.40, 0.50, 0.60, 0.70}

### Phase E — Bonus (3개 추가 sweep)
- 추가 hyperparameter sweep
- Best (bonus_000) 선정

자세히: [docs/05_training/04_sniper_old_pipeline.md](../05_training/04_sniper_old_pipeline.md)

## Inference 절차

Arch4의 Pass 2:
```python
sniper_results = self.sniper_detector.predict(
    sr_crop_batch,           # SR로 만든 HR crops (256×256)
    conf=self.config["sniper_conf"],   # 0.001
    iou=self.config["sniper_nms_iou"]  # 0.45
)
```

각 crop당 detection을 원본 좌표 공간으로 변환 후 Scout detection과 ROI-aware NMS로 합침.

## sniper_max_det_per_crop 제한

각 crop당 최대 3개 detection만 유지.
- v2 sweep에서 누락됐다가 추가됨 (사고 기록 참조)

## 검증 포인트

- [ ] 각 SR마다 적절한 Sniper weight 사용 (rfdn base 사고 재발 방지)
- [ ] Fair eval은 from-scratch, Properbase/sweep은 hardneg 사용 구분
- [ ] Phase B의 base detector가 각 SR 전용인가 (yolo_8s_{sr})
- [ ] sniper_max_det_per_crop=3이 적용되는가
