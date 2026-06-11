# 10-02. Fairness Log (공정성 보장 기록)

## 1. Dataset 공정성

| 항목 | 보장 |
|:--|:--|
| 4 SR이 같은 LR 입력 | ✅ bicubic 4× downscale, 모든 실험 공통 |
| 4 SR이 같은 HR target | ✅ 같은 768×768 원본 |
| 평가 set | ✅ subset6418 (모든 실험 공통) |

## 2. Training 공정성

| 항목 | 보장 |
|:--|:--|
| YOLO 학습 augmentation | ✅ Ultralytics 기본, 모든 SR 동일 |
| Sniper base detector | ✅ 각 SR 전용 yolo_8s_{sr} (rfdn base 사고 후 수정) |
| Old Pipeline Phase | ✅ A→B→C→D→E 동일 절차, SR별 |
| Gate 학습 | ✅ BCE 250K iter, 4 SR 동일 |
| val 누설 | ✅ 없음 (train set만 사용) |

## 3. Evaluation 공정성

| 항목 | 보장 |
|:--|:--|
| Val set | ✅ subset6418 (Pre-flight check로 강제) |
| Metric | ✅ ultralytics ap_per_class (일관) |
| NMS protocol | ✅ Fair/Properbase는 fixed_protocol, v2는 sweep grid 동일 |
| Sniper weight | ✅ Fair는 from-scratch, Properbase/v2는 hardneg (실험별 명시) |

## 4. NMS Sweep 공정성

| 항목 | 보장 |
|:--|:--|
| Grid 통일 | ✅ 8 pass1 × 5 roi × 1 sigma = 40 combos, 4 SR 동일 |
| Sniper 통일 | ✅ properbase Sniper (각 SR 전용) |
| 누락 파라미터 | ✅ v1 → v2에서 6개 추가 |
| RFDN weight 정정 | ✅ rfdn_arch4 (Mar 14) 사용 |
| 평가 set | ✅ subset6418 |

## 5. Sensitivity 공정성

| 항목 | 보장 |
|:--|:--|
| 변동 범위 | ✅ 4 SR 모두 같은 ±값 |
| Starting point | ✅ 각 SR의 v2 best config |
| Sweep 파라미터 | ✅ 같은 4개 (max_det, final_conf, roi_small, replace_margin) |

## 6. 코드 공정성

| 항목 | 보장 |
|:--|:--|
| Wrapper 패턴 | ✅ 4 SR 동일 인터페이스 |
| eval_arch.py | ✅ 4 Arch가 같은 코드 경로 |
| arch4_eval_ultralytics.py | ✅ 모든 NMS sweep 공통 |
| Pre-flight check | ✅ 모든 평가에서 자동 실행 |

## 7. 사고 발생 → 수정 흐름

| 사고 | 발견 | 수정 |
|:--|:--|:--|
| rfdn base | 2026-06-09 결과 검토 | properbase pipeline 재실행 |
| NMS sweep v1 RFDN weight | 2026-06-10 결과 분석 | rfdn_arch4 weight로 재실행 |
| NMS sweep v1 누락 6개 | 2026-06-10 properbase 비교 | v2에 추가 + 전체 재실행 |
| Mamba loader | 검증 #45 | wrapper 분리 강제 |

자세히: [04_changes_history.md](04_changes_history.md)

## 8. Run-to-Run Variance 측정

DRCT 5회 + RFDN 5회 측정 (Task #63):
- 변동: < 0.005 (0.5pp)
- → 본 연구의 측정 차이가 ~1pp인데 noise 범위 < 0.5pp

→ Ranking 변동이 통계적으로 의미 있는 수치 (∆ > noise).
