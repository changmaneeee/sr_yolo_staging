# 07-03. NMS Sweep v2 (Section IV-C)

## 목적

각 SR이 자기 최적 NMS를 가지면 어떻게 되는가? Ranking이 바뀌는가?

## 절차 (2-Stage)

RFDN 선행 sweep과 동일 프로토콜:

### Stage 1 (Quick scan, 200장)
- 40 combos / SR
- pass1_conf × roi_expansion × sigma 그리드

### Stage 2 (Full eval, 6418장)
- Top-10 combos만
- Stage 1 결과 기준 정렬

### Grid (v2)
- pass1_conf: {0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025} (8개)
- roi_expansion: {1.5, 1.75, 2.0, 2.25, 2.5} (5개)
- soft_nms_sigma: {0.3} (1개, sigma exclude 확정 후)
- 총 = 40 combos

### v1 → v2 변경 (사고 후 수정)

v1에서 누락됐던 6개 파라미터를 v2에 모두 포함 (properbase config과 일치):

| 파라미터 | v1 | v2 |
|:--|--:|--:|
| final_conf | 0.3 (잘못) | **0.25** (properbase 일치) |
| roi_small_thresh | 없음 | **32.0** |
| roi_large_thresh | 없음 | **96.0** |
| large_roi_score_thresh | 없음 | **0.5** |
| sniper_replace_margin | 없음 | **0.1** |
| sniper_max_det_per_crop | 없음 | **3** |

## v1 결과 (참고용)

| SR | v1 best | Config |
|:--|--:|:--|
| HAT | 0.7980 | p1=0.0175, roi=1.5 |
| DRCT | 0.7963 | p1=0.0175, roi=2.0 |
| RFDN | 0.7949 (수정 후) | p1=0.0125, roi=1.75 |
| MAN | 0.7902 | p1=0.01, roi=1.75 |

→ v1은 누락 6개로 인해 신뢰성 부족.

## v2 결과 (최종)

| SR | v2 best | Best Config |
|:--|--:|:--|
| **HAT** | **0.8003** 🥇 | p1=0.0175, roi=1.5 |
| **DRCT** | **0.7990** | p1=0.0175, roi=2.0 |
| **RFDN** | **0.7981** | p1=0.0125, roi=1.75 |
| **MAN** | **0.7940** | p1=0.01, roi=1.75 |

- Range: 0.63pp
- Ranking: **HAT > DRCT > RFDN > MAN**

## 결과 파일

```
iac_runs/nms_sweep_2stage/{drct,hat,man,rfdn}/
  ├── stage1/
  │   ├── p10.005_roi1.5_s0.3.json
  │   ├── ... (40개 200장 평가)
  │   └── summary.json
  └── stage2/
      ├── p10.0175_roi2.0_s0.3.json
      ├── ... (10개 6418장 평가)
      └── summary.json
```

## 사고 기록 (v1 → v2)

- **RFDN weight 사고**: v1에서 RFDN SR을 `weights/rfdn/model_best.pt`로 잘못 지정 (정상은 `weights/rfdn_arch4/model_best.pt`). Sniper도 baseline 사용.
  - 발견: 결과 0.7450이 너무 낮아서 의심 → weight 비교 → 발견
  - 수정: 정확한 weight로 재실행 → 0.7949

- **누락 6개 파라미터**: v1 config이 properbase보다 6개 부족 → mAP 0.5pp 이상 차이
  - 발견: properbase 0.8007 vs v1 RFDN 0.7949 비교
  - 수정: v2에서 모두 포함 + 재실행

자세히: [docs/10_integrity_audit/04_changes_history.md](../10_integrity_audit/04_changes_history.md)

## 공정성 보장

- 같은 grid (8×5)에 모든 SR 적용
- 같은 properbase Sniper 사용 (각 SR 전용)
- 같은 Scout
- subset6418 평가
- v2 config이 23개 NMS 파라미터 모두 명시 (누락 없음)

## 검증 포인트

- [ ] v2 grid가 4 SR에 동일하게 적용
- [ ] v2 config에 누락 파라미터가 없음 (v1 사고 재발 방지)
- [ ] 정확한 weight 사용 (RFDN: rfdn_arch4 + hardneg_newscout sniper)
- [ ] Stage 2가 6418장에서 정밀 측정
