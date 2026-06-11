# 05-05. Arch5b Phase 2 — Fusion Warmup

## 개요

Arch5b의 fusion module만 학습. 다른 컴포넌트(SR, YOLO backbone/head)는 frozen.

## 환경

- **위치**: 4090 서버 (WSL2 4060는 부족)
- 코드 위치 (서버): `/home/changmin/dark_vessel_sr_yolo/...` (메인 repo 동기화)

## 학습 절차

```
1. SR backbone load (pretrained, frozen)
2. YOLOv8 backbone + head load (pretrained, frozen)
3. Fusion module 초기화 (학습 대상)
4. Loss: detection loss (YOLOv8 표준)
5. Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
6. Epoch: 약 50~100
```

## 저장 형식

- `iac_lab/runs/{timestamp}_arch5b_{sr}_phase2/checkpoints/best.pt`
- `model_state_dict`에 fusion module의 weight만 변동

## DRCT Phase 2 결과

(서버에서 학습 완료) — checkpoint 경로는 server prompt 참조.

## 검증 포인트

- [ ] 다른 컴포넌트가 정말 frozen인가
- [ ] Fusion module만 학습됨
- [ ] 학습 데이터셋이 train만 사용
- [ ] α 파라미터가 Phase 2에서는 초기화 단계
