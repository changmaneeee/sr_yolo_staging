# 07-04. Sensitivity Ablation (Section IV-D)

## 목적

NMS sweep에서 sweep한 변수(pass1, roi_expansion) 외의 4개 NMS 파라미터에 대한 robustness 확인.

JSTAR 리뷰어가 "왜 그것들도 sweep 안 했냐?" 묻는 것을 방어.

## 절차

각 SR의 v2 best config에서 4개 파라미터를 ±변동:

| 파라미터 | values |
|:--|:--|
| sniper_max_det_per_crop | {2, 3, 5} |
| final_conf | {0.20, 0.25, 0.30} |
| roi_small_thresh | {16, 32, 48} |
| sniper_replace_margin | {0.05, 0.10, 0.20} |

= 4 × 3 = 12 evals/SR
= 4 SR × 12 = **48 evals 총**

각각 subset6418 (6418장) 평가, ~12분/eval ≈ ~10h 소요.

## 결과

| SR | v2 best | Sens Min | Sens Max | Range |
|:--|--:|--:|--:|:--|
| RFDN | 0.7981 | 0.7949 | 0.8006 | ±0.57pp |
| DRCT | 0.7990 | 0.7963 | 0.8004 | ±0.41pp |
| **HAT** | **0.8003** | 0.7980 | **0.8029** | ±0.49pp |
| MAN | 0.7940 | 0.7902 | 0.7961 | ±0.60pp |

→ **모든 SR이 ±0.6pp 이내**. 매우 robust.

## 결과 파일

```
iac_runs/nms_sensitivity/{drct,hat,man,rfdn}/
  ├── md2.json, md3.json, md5.json
  ├── fc0.20.json, fc0.25.json, fc0.30.json
  ├── rst16.json, rst32.json, rst48.json
  └── rm0.05.json, rm0.10.json, rm0.20.json
= 12 files per SR
```

## Paper용 framing

> "Sensitivity analysis of 4 ancillary NMS parameters confirms robustness (Δ ≤ 0.6pp under ±50% perturbation), demonstrating that the v2 sweep results are not sensitive to specific choices of these fixed parameters."

## 자동 실행

- Robust pipeline 스크립트 (`run_robust_pipeline.sh`)가 v2 완료 후 자동 실행
- tmux 죽음 자동 복구 (max 5회 retry)
- 결과 무결성 자동 검증

## 공정성 보장

- 각 SR이 자기 v2 best config에서 시작 (공평한 출발점)
- 같은 ±값 범위에서 sweep
- 같은 subset6418
- 같은 metric

## 검증 포인트

- [ ] 4 SR 모두 동일한 ±값으로 sweep
- [ ] 48/48 evals 완료
- [ ] 결과 JSON 무결성 (corrupt 없음)
- [ ] ranking robustness가 paper 주장과 일치
