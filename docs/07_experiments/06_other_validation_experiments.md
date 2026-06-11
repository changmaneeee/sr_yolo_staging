# 07-06. 기타 검증 실험

본 연구 진행 중 수행한 검증/ablation 실험 목록. (Task #31~#46)

## 검증 #31 — Arch2 DRCT +6.61pp 재확인
- DRCT가 Arch2에서 RFDN 대비 +6.61pp 향상의 재현 확인
- 결과: 일관됨

## 검증 #32 — Crop size sweep (Arch4 DRCT)
- crop_size_lr를 {32, 64, 96, 128}로 변동
- 결과: 64가 최적 (fixed_protocol에 명시)

## 검증 #33 — Swin Window Boundary PSNR
- DRCT의 Swin window 경계가 PSNR에 미치는 영향
- 결과: 작은 영향만 있음

## 검증 #34 — FP 메커니즘 + Per-image breakdown
- False positive의 발생 메커니즘 분석
- 이미지별 P/R 변동 확인

## 검증 #35 — Arch2 vs Arch4 DRCT 비교
- 두 Arch의 DRCT 결과 직접 비교

## 검증 #36 — DRCT 64×64 crop 전용 fine-tune
- DRCT를 64×64 crop으로 fine-tune
- 결과: `weights/sr_finetuned/drct/best.pt`로 사용

## 실험 #37 — Cross-Experiment 2×2 (SR × Sniper)
- {RFDN SR, DRCT SR} × {RFDN Sniper, DRCT Sniper}
- 결과: 각자 자기 매칭이 최적 (cross는 불공정)

## 실험 #38 — Gate Cross-Swap (Arch2 SR × Gate)
- SR과 Gate를 cross-swap 시 결과
- 결과: 각자 자기 매칭이 최적

## 실험 #39~42 — Gate Decision, Oracle Gate, FP-Area Scaling, PSNR-mAP Curve
- 다양한 ablation. 자세한 내용은 Notion에 있음.

## 실험 #43~46 — DRCT 추가 검증
- crop64 fine-tune 후 재평가
- Scout YOLO 영향 점검 (RFDN vs DRCT)
- DRCT Sniper from-scratch 학습 + Arch4 재평가
- 등

## 실험 #47~51 — Exp1 재측정, σ sweep, 통계 검정 등
- Multi-seed run-to-run variance (DRCT 5회 + RFDN 5회) — Task #63

## 실험 #52~56 — fixed_protocol, eval_arch, Pre-flight, Arch4 통일, 중복 제거
- 평가 인프라 정비

## 실험 #57~64 — H6 (bbox precision), H8 (crop boundary), H9 (hallucination FFT)
- 더 깊은 분석. Notion에 자세히 기록.

## 실험 #65~68 — HAT/MAN Arch0/2/4 평가
- HAT/MAN을 Arch0/Arch2/Arch4에 모두 적용
- Gate 학습, Sniper 학습, 평가까지 완료

## 모든 검증 실험에 공통

- 같은 subset6418 평가
- 같은 fixed_protocol 적용
- 같은 metric (ap_per_class)
- Notion에 결과 기록 (자세한 수치)

## 검증 포인트

- [ ] 모든 검증 실험이 subset6418에서 평가
- [ ] fixed_protocol을 따름
- [ ] 결과가 Notion에 기록되어 추적 가능
