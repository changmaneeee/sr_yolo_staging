# 10-01. Audit Checklist (검사 체크리스트)

본 체크리스트를 따라 본 연구의 무결성을 종합 검증.

---

## A. Dataset 검증

- [ ] **A1**: HR/LR dataset 경로 존재 (docs/01_dataset/01_source.md)
- [ ] **A2**: HR 768×768, LR 192×192 해상도 확인
- [ ] **A3**: HR train 108,414장, LR train 108,414장 일치
- [ ] **A4**: subset6418 정확히 6,418장
- [ ] **A5**: subset6418 HR/LR 1:1 매칭 (같은 stem)
- [ ] **A6**: Label format이 YOLO 표준 (5 columns, 0~1 normalized)
- [ ] **A7**: 클래스가 1개 (ship)
- [ ] **A8**: subset 선정 로직이 reproducible (`label.size > 0`)

## B. Preprocessing 검증

- [ ] **B1**: LR 생성이 bicubic + 4× downscale
- [ ] **B2**: 모든 SR backbone이 같은 LR 입력
- [ ] **B3**: input_range="0-255" 통일
- [ ] **B4**: 평가 시 BGR→RGB→0-1 변환 동일
- [ ] **B5**: SR 출력이 모두 0-1 RGB 텐서

## C. Architecture 검증

- [ ] **C1**: 4 Arch가 같은 입력 인터페이스 (LR tensor)
- [ ] **C2**: 4 Arch가 같은 출력 인터페이스 (detection dict)
- [ ] **C3**: 4 SR backbone이 같은 Wrapper 패턴
- [ ] **C4**: Arch4의 23개 NMS 파라미터가 명시
- [ ] **C5**: Arch5b의 Phase 2/3 구분 명확

## D. Component 검증

- [ ] **D1**: Scout가 모든 SR에서 공통 weight (LR만 처리)
- [ ] **D2**: Sniper가 각 SR마다 별도 (SR 출력 처리)
- [ ] **D3**: Gate가 SR별로 별도 (BCE 250K iter)
- [ ] **D4**: YOLO wrapper가 일관된 인터페이스
- [ ] **D5**: ROI-aware NMS의 11개 파라미터가 fixed_protocol에 통제

## E. Training 검증

- [ ] **E1**: 모든 학습이 train set만 사용 (val 누설 없음)
- [ ] **E2**: 같은 train set이 4 SR에 적용
- [ ] **E3**: 각 SR 전용 Sniper base detector 사용 (yolo_8s_{sr}) — rfdn base 사고 재발 방지
- [ ] **E4**: Augmentation이 4 SR에 동일
- [ ] **E5**: Optimizer, lr, batch_size 일관
- [ ] **E6**: Phase B (50ep), Phase C (25ep) 절차 통일
- [ ] **E7**: Gate 4종 모두 BCE 250K iter (detection-loss 폐기)

## F. Evaluation 검증

- [ ] **F1**: 모든 평가가 subset6418 (6,418장)
- [ ] **F2**: 모든 평가가 ultralytics ap_per_class 사용
- [ ] **F3**: mAP@50이 메인 metric
- [ ] **F4**: Eval space "hr" 통일
- [ ] **F5**: Pre-flight check가 MD5 + count 검증
- [ ] **F6**: NMS 파라미터가 fixed_protocol에 통제

## G. Experiment 공정성 검증

- [ ] **G1**: Fair eval에 from-scratch Sniper 사용
- [ ] **G2**: Properbase에 hardneg Sniper 사용
- [ ] **G3**: NMS sweep v2에 properbase Sniper 사용
- [ ] **G4**: 4 SR 모두 같은 v2 grid (8 pass1 × 5 roi)
- [ ] **G5**: Sensitivity가 각 SR best config에서 동일 ±값
- [ ] **G6**: RFDN sweep과 다른 SR sweep의 grid가 동일

## H. Result 검증

- [ ] **H1**: 모든 수치가 JSON 파일에 존재
- [ ] **H2**: Fair / Properbase / v2 mAP 수치 일관
- [ ] **H3**: Sensitivity Min/Max가 12 evals에서 계산
- [ ] **H4**: Ranking 변동이 robustness analysis와 일치
- [ ] **H5**: Paper에 들어갈 수치가 4자리 (Changmin profile)

## I. Code Integrity 검증

- [ ] **I1**: eval_arch.py 코드가 명세와 일치
- [ ] **I2**: arch4_eval_ultralytics.py 코드가 명세와 일치
- [ ] **I3**: fixed_protocol.yaml 내용이 명세와 일치
- [ ] **I4**: 각 Arch pipeline 코드가 docs와 일치
- [ ] **I5**: SR wrapper 코드가 일관

## J. 사고 기록 검증

- [ ] **J1**: rfdn base 사고가 문서화되었고 수정됨
- [ ] **J2**: NMS sweep v1 weight 사고가 문서화되었고 수정됨
- [ ] **J3**: NMS sweep v1 누락 6개 파라미터가 v2에서 추가됨
- [ ] **J4**: Mamba loader 사고가 문서화됨
- [ ] **J5**: 모든 사고의 detection → fix 흐름이 추적 가능

## K. Reproducibility 검증

- [ ] **K1**: 환경 (python 3.12, torch 2.9.1+cu128) 명시
- [ ] **K2**: Weight 경로 정확
- [ ] **K3**: 재현 명령 (run_commands.md) 동작 확인 가능
- [ ] **K4**: Run-to-run variance 측정됨

---

## 검사자에게

각 항목에 대해:
1. **OK**: 명세대로 구현됨
2. **WARN**: 일부 의문 (자세한 검토 필요)
3. **FAIL**: 문제 발견

발견된 모든 FAIL은 [docs/10_integrity_audit/03_known_issues.md](03_known_issues.md)에 기록.

전체 검사 결과 보고서 작성을 권장.
