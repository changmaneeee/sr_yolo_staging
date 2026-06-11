# 07-02. Properbase Old Pipeline (Section IV-B)

## 목적

Sniper를 hardneg fine-tune으로 강화하여 각 SR의 잠재력을 끌어낸 후 비교.

## 1차 시도 (사고 — rfdn base, 폐기)

- DRCT/HAT/MAN 모두 BASE_SNIPER=`yolo_8s_rfdn`으로 잘못 사용
- 결과는 나왔지만 paper에 못 씀

| SR | 1차 mAP@50 |
|:--|--:|
| RFDN | 0.8007 |
| DRCT | 0.7927 (rfdn base, 잘못) |
| HAT | 0.7914 (rfdn base, 잘못) |
| MAN | 0.7955 (rfdn base, 잘못) |

## 2차 시도 (properbase)

각 SR 자기 전용 detector로 base 변경 후 재실행.

### 절차 (Phase A 재사용, B~E만 재실행)
```bash
bash iac_runs/run_old_pipeline_properbase.sh   # DRCT/HAT/MAN
```

### 각 SR config

| SR | BASE_SNIPER (2차 수정) | SR_WEIGHT |
|:--|:--|:--|
| DRCT | weights/yolo_8s_drct/weights/best.pt | weights/sr_finetuned/drct/best.pt |
| HAT | weights/yolo_8s_hat/best.pt | weights/sr_finetuned/hat/best.pt |
| MAN | weights/yolo_8s_man/best.pt | weights/sr_finetuned/man/best.pt |

### 소요 시간

- DRCT: ~7.3h (Phase A~E)
- HAT: ~8.2h
- MAN: ~7.3h

## 결과 (Properbase 최종)

| SR | mAP@50 | Best Config |
|:--|--:|:--|
| **RFDN** | **0.8007** | bonus_000 |
| **DRCT** | **0.7973** | bonus_000 |
| **HAT** | **0.7905** | bonus_000 |
| **MAN** | **0.7918** | bonus_000 |

- Range: 1.02pp
- Ranking: **RFDN > DRCT > MAN > HAT**

## 결과 파일

| SR | Path |
|:--|:--|
| RFDN | `iac_runs/20260325_023318_newscout_sniper_pipeline/phase_e_bonus_000_map.json` |
| DRCT | `iac_runs/20260609_070148_drct_properbase_pipeline/final_map_result.json` |
| HAT | `iac_runs/20260609_143533_hat_properbase_pipeline/final_map_result.json` |
| MAN | `iac_runs/20260609_143533_man_properbase_pipeline/final_map_result.json` |

## 공정성 보장

- 같은 protocol (Phase A~E)
- 각 SR 자기 base detector 사용 ← **핵심 fix**
- 같은 train set으로 fine-tune
- subset6418에서만 mAP 평가
- 같은 metric

## 사고 기록 (rfdn base)

- 발생: 2026-06-07~09
- 원인: 스크립트의 `BASE_SNIPER` 하드코딩
- 발견: 결과 분석 중 NMS sweep과 비교에서 명확화
- 수정: 2차 properbase 스크립트로 재실행

자세히: [docs/10_integrity_audit/04_changes_history.md](../10_integrity_audit/04_changes_history.md)

## 검증 포인트

- [ ] 각 SR이 자기 전용 BASE_SNIPER 사용 (1차 실수 재발 방지)
- [ ] 같은 Phase A~E 절차
- [ ] subset6418에서 mAP 측정
- [ ] 1차 결과는 폐기되고 2차만 paper에 사용
