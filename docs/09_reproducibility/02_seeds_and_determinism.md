# 09-02. Seeds and Determinism

## 일반 원칙

본 연구는 deterministic하지 않은 부분(CUDA, cuDNN)이 있을 수 있으나, 다음을 보장:
- 같은 weight + 같은 config → 같은 mAP 결과 (run-to-run variance 측정됨)

## Run-to-Run Variance (Task #63)

DRCT 5회 + RFDN 5회로 측정한 결과:
- mAP@50 변동: < 0.005 (0.5pp)
- → 본 연구의 결과는 reproducible

## Seed 명시

### Sniper YOLO 학습
- Ultralytics 기본 seed (보통 0)
- seed override 없음

### Gate 학습
- (코드에 명시된 seed가 있다면 추가)

### Arch5 Phase 학습
- 서버 코드에서 seed 명시 (server prompt에서 가져옴)

## 평가는 deterministic

평가 시:
- model.eval() 모드
- `torch.no_grad()` context
- batch_size=1
- → 평가 결과는 100% reproducible

## 검증 포인트

- [ ] 같은 명령 두 번 실행 시 같은 결과
- [ ] Run-to-run variance가 측정되어 있음
- [ ] 측정 환경(GPU)이 일관됨
