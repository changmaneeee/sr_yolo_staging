# 03-04. Arch5b — Feature-level Fusion

## 구조 한 줄

> LR feature + SR feature를 attention 기반으로 fusion → 한 YOLO head로 통합 검출

이미지 레벨이 아닌 **feature 레벨에서 LR/SR 정보를 결합**. Arch5는 SOTA 후보로 학습되었으며, Arch4와는 다른 접근.

## 흐름

```
LR (1, 3, 192, 192)
   ├─ Path 1: LR → YOLO backbone → LR feature
   └─ Path 2: LR → SR → HR → YOLO backbone → HR feature
   ↓
[Fusion] Attention Fusion Module이 두 feature를 결합
   ↓
fused feature
   ↓
YOLO head → detection
```

## 코드

`src/models/pipelines/arch5b_fusion.py` (`Arch5BFusion` 클래스)

Fusion module: `src/models/fusion/attention_fusion.py`

## 학습 단계 (Phase 2 + Phase 3)

Arch5는 single-stage 학습이 어려워서 2단계로 학습:

### Phase 2 — Fusion warm-up (4090 서버에서 학습)
- LR/SR feature를 결합하는 attention module만 학습
- 다른 부분(SR backbone, YOLO backbone, head)은 frozen
- 목표: fusion이 두 feature를 효과적으로 결합하는 방식 학습

### Phase 3 — Joint full-training (4090 서버에서 학습)
- 모든 컴포넌트가 함께 fine-tune
- learnable α 파라미터로 SR vs LR 가중치 학습
- 목표: end-to-end optimization

자세한 학습 절차: [docs/05_training/06_arch5_phase2_warmup.md](../05_training/06_arch5_phase2_warmup.md), [07_arch5_phase3_joint.md](../05_training/07_arch5_phase3_joint.md)

## Phase 3 checkpoint 저장 방식

전체 모델의 weight를 단일 .pt 파일로 저장:
- `checkpoint["model_state_dict"]`에 SR + backbone + fusion + head + α 포함
- 평가 시 `--arch5-checkpoint` 인자로 로드

```python
# eval_arch.py line 274~277
if args.arch5_checkpoint:
    ckpt = torch.load(args.arch5_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)
```

## SR backbone별 Arch5

각 SR backbone에 대해 Arch5 학습 가능:
- Arch5b RFDN (선행 SOTA)
- Arch5b DRCT (Phase 2, 3 진행)
- Arch5b HAT (선택)
- Arch5b MAN (선택)

## 학습 위치

**Arch5의 학습은 4090 서버에서 수행됨** (현재 working dir의 WSL2 GPU(4060)는 학습용으로 너무 작음).

서버 코드 동기화: [docs/PROMPTS/server_arch5_request.md](../PROMPTS/server_arch5_request.md)

## 결과 (예시)

| SR | Arch5b mAP@50 |
|:--|--:|
| RFDN | 0.9102 (SOTA 기록) |
| DRCT | (Phase 3 학습 중/완료) |

자세한 결과: [docs/08_results/](../08_results/)

## 검증 포인트

- [ ] Phase 2와 Phase 3의 학습 구분이 명확한가
- [ ] Phase 3 checkpoint에 learnable α가 포함되는가
- [ ] 서버에서 학습된 weights가 추적 가능한가
- [ ] Arch5 평가 시 동일한 subset6418 사용
- [ ] Fusion module 구조가 모든 SR에 동일하게 적용되는가
