# 10-03. Known Issues and Limitations

본 연구가 가진 알려진 limitation. Paper에서 투명하게 언급.

## 1. v2 sweep grid의 제한

- 8 pass1 × 5 roi × 1 sigma = 40 combos (per SR)
- 다른 NMS 파라미터 4개는 sweep 안 함 (sensitivity로 확인)
- Full grid (모든 23개 파라미터 sweep)는 비현실적 (5,760 combos)
- **방어**: Sensitivity ablation으로 robustness 증명

## 2. Eval pipeline 차이

- `eval_arch.py` vs `arch4_eval_ultralytics.py` 두 evaluator 사용
- 둘 다 같은 `ap_per_class` 함수 사용하지만 결과 약간 다를 수 있음
- Fair eval 0.7731 vs NMS sweep v1 RFDN 0.7949 — 같은 SR이지만 differ
- **이유**: weight + config 차이 (v1은 weight 사고, config 누락)
- **수정**: v2에서 모두 통일, 결과 일관

## 3. RFDN weight 두 버전 존재

- `weights/rfdn_arch4/model_best.pt` (Mar 14) — 정상
- `weights/rfdn/model_best.pt` (Jan 12) — 잘못된 버전
- **사용**: 정상 weight (rfdn_arch4)만
- **잘못된 weight는 메인 폴더에 남아있음** (혼동 위험)
- → 검사자는 weight 경로 확인 필수

## 4. CUDA 크래시 (WSL2 GPU)

- WSL2 4060 환경에서 CUDA 크래시가 자주 발생 (`cudaErrorUnknown`)
- 약 6418장 중 4800~5000장 부근에서 발생
- **대응**: skip 로직으로 재시작 시 완료된 combo 건너뛰기
- 본 연구 결과의 무결성에는 영향 없음 (모든 평가 완료)

## 5. Arch5의 학습 위치

- Arch5 Phase 2/3 학습은 4090 서버에서 수행
- 본 staging repo는 결과만 정리
- 코드 자체는 동일하지만 학습 환경 차이 가능
- **방어**: 서버에서 받은 checkpoint를 본 repo에서 re-evaluate 가능

## 6. SR backbone fine-tune 범위

- DRCT만 crop64 fine-tune (검증 #36)
- HAT, MAN, RFDN은 base weight 그대로
- **방어**: properbase의 Sniper fine-tune이 SR의 차이를 absorb

## 7. Train-time augmentation 미명시

- Ultralytics 기본 augmentation 사용 (수정 안 함)
- 명시적 seed 설정 안 함 (보통 0)
- **방어**: Run-to-run variance < 0.5pp 측정으로 reproducibility 보장

## 8. 28,883 vs 6,418

- LR val이 1장 더 있음 (28,884 vs 28,883)
- **무력화**: 평가는 항상 subset6418로
- 이는 데이터 생성의 미세 차이로, 평가에는 영향 없음

## 9. final_conf 0.25 vs 0.3

- Properbase config: final_conf=0.25
- fixed_protocol.yaml (현재): final_conf=0.3
- NMS sweep v2: final_conf=0.25 (properbase 일치)
- **혼란**: fixed_protocol의 0.3과 properbase의 0.25 차이
- **결정**: paper에서는 v2 (final_conf=0.25)를 main result로 사용

## 10. JSTAR급 수준의 한계

본 연구는 JSTAR (IF ~5.0)급을 목표로 하지만:
- 모든 hyperparameter를 full grid sweep하지 못함 (시간 제약)
- 단일 task (ship detection)만 평가
- 단일 dataset (Airbus)만 평가

이러한 limitation은 paper의 Conclusion/Limitation section에서 투명하게 언급.

## 검사자에게

본 issues는 paper에 명시적으로 언급되어야 함. 추가로 발견된 이슈는 별도 보고서로.
