# 05. Training Procedures

각 컴포넌트의 학습 절차. **공정성 보장**의 핵심 영역.

| 문서 | 내용 |
|:--|:--|
| [01_scout_training.md](01_scout_training.md) | Scout YOLO 학습 (LR detector, 모든 SR 공통) |
| [02_sniper_from_scratch.md](02_sniper_from_scratch.md) | Sniper from-scratch (Fair eval용) |
| [03_sniper_old_pipeline.md](03_sniper_old_pipeline.md) | Sniper Old Pipeline (Phase A~E, Properbase) |
| [04_gate_training.md](04_gate_training.md) | Gate network 학습 (BCE 250K iter) |
| [05_arch5_phase2_warmup.md](05_arch5_phase2_warmup.md) | Arch5b Phase 2 (fusion warmup) |
| [06_arch5_phase3_joint.md](06_arch5_phase3_joint.md) | Arch5b Phase 3 (joint training) |
| [07_sr_finetune.md](07_sr_finetune.md) | SR backbone fine-tune (DRCT crop64 등) |

## 핵심 사고 기록 (학습 공정성)

### 1. rfdn base 사고 (2026-06-07~09)
- DRCT/HAT/MAN의 Sniper fine-tune에서 base detector를 모두 yolo_8s_rfdn으로 잘못 사용
- 발견 후 각 SR 전용 detector로 재학습 (properbase pipeline)
- 자세히: [docs/10_integrity_audit/04_changes_history.md](../10_integrity_audit/04_changes_history.md)

### 2. Mamba loader 사고
- Mamba SR weight를 RFDN loader로 로드하면 F1~0.0003 실패
- 각 SR은 반드시 자기 wrapper로 로드

### 3. NMS sweep v1 RFDN weight 사고
- RFDN의 SR weight 경로 잘못 (`weights/rfdn/` vs `weights/rfdn_arch4/`)
- Sniper도 baseline 사용 → 0.7450 으로 비정상 낮음
- 발견 후 정확한 weight로 재실행 → 0.7981

## 검증 체크리스트

- [ ] 각 SR에 적절한 base detector 사용 (yolo_8s_{sr})
- [ ] 학습 데이터셋이 train split만 사용 (val 누설 없음)
- [ ] Augmentation이 모든 SR backbone에서 동일
- [ ] Optimizer / LR / batch size 일관성
- [ ] Random seed 명시 (reproducibility)
