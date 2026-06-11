# 05-06. Arch5b Phase 3 — Joint Training

## 개요

Arch5b의 모든 컴포넌트(SR, fusion, backbone, head)를 end-to-end 학습.

## 환경

- **위치**: 4090 서버
- 시작점: Phase 2 checkpoint

## 학습 절차

```
1. Phase 2 checkpoint load (fusion module 포함)
2. 모든 컴포넌트 unfreeze (선택적으로 SR은 lower lr)
3. Learnable α 파라미터도 함께 학습 (gradient descent)
4. Loss: detection loss + (선택) SR reconstruction loss
5. Optimizer: Adam (lr=1e-5, weight_decay=1e-5) — Phase 2보다 작음
6. Epoch: 약 100
```

## Phase 3 checkpoint

- 전체 모델 weight + α를 단일 .pt에 저장
- `iac_lab/runs/{timestamp}_arch5b_{sr}_phase3/checkpoints/phase3_best.pt`

## 평가 시 로드

```bash
python eval_arch.py --sr-backbone drct --arch 5 \
  --arch5-checkpoint iac_lab/runs/.../phase3_best.pt
```

```python
# eval_arch.py line 274~277
ckpt = torch.load(args.arch5_checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt, strict=False)
```

## Arch5b RFDN 결과 (SOTA)

- mAP@50 = **0.9102** (SR-YOLO 프로젝트 SOTA)
- TRT FP16 latency: 12ms
- Small vessel recall +84% (Phase 2 → Phase 3)

## Arch5b DRCT 결과

(서버 학습 중/완료) — server prompt로 가져옴

## 검증 포인트

- [ ] Phase 2 → Phase 3 시작점 일관
- [ ] α 파라미터가 학습 가능 (gradient flow)
- [ ] 평가 시 checkpoint 로드가 정상
- [ ] 학습 데이터셋 train만 사용
- [ ] Best checkpoint 선정 기준 명시
