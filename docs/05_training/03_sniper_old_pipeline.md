# 05-03. Sniper Old Pipeline (Phase A~E, Properbase)

## 개요

Properbase 실험에서 사용된 강화 학습 파이프라인. From-scratch Sniper보다 +2~3pp 향상.

## 핵심: BASE_SNIPER 선택 (사고 재발 방지)

각 SR마다 **자기 전용 detector**를 base로 사용 (rfdn base 사고 후 수정):

| SR | BASE_SNIPER |
|:--|:--|
| RFDN | `weights/yolo_8s_rfdn/weights/best.pt` ✅ |
| DRCT | `weights/yolo_8s_drct/weights/best.pt` ✅ |
| HAT | `weights/yolo_8s_hat/best.pt` ✅ |
| MAN | `weights/yolo_8s_man/best.pt` ✅ |

→ 1차 실행에서 모두 `yolo_8s_rfdn`을 잘못 사용했던 사고 후 수정.

## Phase A — Crop Dump

각 train 이미지에 대해:
1. SR로 HR 생성
2. label이 있는 ship 위치 + 주변 ROI 잘라서 256×256 crop 저장
3. Negative (background) crop도 일부 저장
4. `data/arch4_sniper_crops_{sr}_old/data.yaml`로 dataset 구성

소요: ~1~2h/SR (SR inference 속도에 의존)

## Phase B — Crop Fine-tune (50 epoch)

```bash
yolo train \
  model=weights/yolo_8s_{sr}/best.pt \   # BASE_SNIPER
  data=data/arch4_sniper_crops_{sr}_old/data.yaml \
  epochs=50 imgsz=256
```

저장: `weights/yolo_sniper_cropft_{sr}/weights/best.pt`

소요: ~3h/SR

## Phase C — Hardneg Mining + Fine-tune (25 epoch)

1. Phase B weight로 train set inference
2. 높은 confidence인 FP를 hard negative로 식별
3. 원래 crop dataset에 negative들 추가
4. 25 epoch fine-tune

저장: `weights/yolo_sniper_hardneg_{sr}_pb/weights/best.pt`

소요: ~2.5h/SR

## Phase D — Interpolation (6 alpha)

Phase B weight와 Phase C weight의 linear interpolation:
```
W(α) = α * W_phaseC + (1-α) * W_phaseB
```

α ∈ {0.20, 0.30, 0.40, 0.50, 0.60, 0.70}

각 α로 subset6418에서 eval → mAP@50 측정.

소요: ~1h/SR

## Phase E — Bonus (3 sweep)

추가 hyperparameter sweep:
- bonus_000: 기본 설정
- bonus_003: 다른 conf threshold
- bonus_005: 또 다른 변형

각각 6418장에서 eval. **Best 선택 (대부분 bonus_000)**.

소요: ~0.6h/SR

## 전체 소요시간

| Phase | Time/SR |
|:--|:--|
| A | 0.8~1.7h |
| B | ~3.3h |
| C | ~2.5h |
| D | ~0.9h |
| E | ~0.6h |
| **Total** | **~8~9h/SR** |

## Properbase 결과 (subset6418 mAP@50)

| SR | Best Phase | mAP@50 |
|:--|:--|--:|
| RFDN | bonus_000 | **0.8007** |
| DRCT | bonus_000 | **0.7973** |
| HAT | bonus_000 | **0.7905** |
| MAN | bonus_000 | **0.7918** |

자세한 결과: [docs/07_experiments/02_properbase_old_pipeline.md](../07_experiments/02_properbase_old_pipeline.md)

## 스크립트 위치

- 1차 (rfdn base, 잘못): `iac_runs/run_drct_old_sniper_pipeline.sh` 등
- 2차 (proper base): `iac_runs/run_old_pipeline_properbase.sh`, `_hatman.sh`

## 검증 포인트

- [ ] 각 SR이 자기 전용 BASE_SNIPER 사용 (rfdn base 사고 재발 방지)
- [ ] Phase A~E 절차가 모든 SR에서 동일
- [ ] Train set만 사용, val 누설 없음
- [ ] mAP eval은 subset6418에서만
- [ ] Best Phase 선정 기준이 일관
