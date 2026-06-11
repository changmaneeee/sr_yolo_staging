# 04-04. Attention Fusion Module (Arch5b)

## 역할

Arch5b에서 LR feature와 SR feature를 attention 기반으로 결합.

## 위치

`src/models/fusion/attention_fusion.py`

## 핵심 메커니즘

- LR/SR feature 각각의 채널 attention 계산
- Cross-attention으로 두 feature를 정렬
- Learnable α 파라미터로 가중치 결정 (Phase 3에서 학습)

## 단계별 변화

### Phase 2 (warmup)
- Fusion module만 학습
- α 파라미터 초기화 후 학습 시작
- 다른 컴포넌트는 모두 frozen

### Phase 3 (joint)
- 모든 컴포넌트가 함께 fine-tune
- α도 함께 학습 → SR vs LR의 동적 가중치

## SR backbone별 별도 학습

Arch5는 SR backbone과 강하게 결합되므로 각 SR마다 별도 학습 필요.

| SR | Arch5b checkpoint |
|:--|:--|
| RFDN | (SOTA, Phase 3 완료) |
| DRCT | (서버에서 진행) |
| HAT | (선택) |
| MAN | (선택) |

자세한 학습: [docs/05_training/06_arch5_phase2_warmup.md](../05_training/06_arch5_phase2_warmup.md), [07_arch5_phase3_joint.md](../05_training/07_arch5_phase3_joint.md)

## 검증 포인트

- [ ] Phase 2와 Phase 3 학습 분리가 명확
- [ ] Phase 3 checkpoint에 α 포함
- [ ] Fusion 구조가 모든 SR에 동일
- [ ] 서버에서 학습된 weight 추적 가능
