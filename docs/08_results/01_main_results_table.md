# 08-01. Main Results Table

모든 결과는 **subset6418 mAP@50** 기준 (다른 명시가 없는 한).

## Arch4 × SR (3가지 NMS protocol)

| SR | Fair NMS (IV-A) | Properbase (IV-B) | v2 NMS-opt (IV-C) |
|:--|--:|--:|--:|
| RFDN | 0.7731 | **0.8007** | 0.7981 |
| DRCT | **0.7806** | 0.7973 | 0.7990 |
| HAT | 0.7733 | 0.7905 | **0.8003** 🥇 |
| MAN | 0.7720 | 0.7918 | 0.7940 |
| **Range** | **0.86pp** | **1.02pp** | **0.63pp** |

## Ranking 비교

```
Fair NMS:       DRCT > HAT > RFDN > MAN
Properbase:     RFDN > DRCT > MAN > HAT
v2 NMS-opt:     HAT > DRCT > RFDN > MAN
```

→ Protocol에 따라 ranking 변동, 그러나 모든 protocol에서 range ≤ 1pp.

## Sensitivity Ablation (IV-D)

| SR | v2 best | Min | Max | Range |
|:--|--:|--:|--:|:--|
| RFDN | 0.7981 | 0.7949 | 0.8006 | ±0.57pp |
| DRCT | 0.7990 | 0.7963 | 0.8004 | ±0.41pp |
| HAT | 0.8003 | 0.7980 | 0.8029 | ±0.49pp |
| MAN | 0.7940 | 0.7902 | 0.7961 | ±0.60pp |

→ 4개 NMS 파라미터 변동에도 ±0.6pp 이내. Robust.

## Arch0 / Arch2 / Arch4 / Arch5 비교 (선택 SR)

| Arch | RFDN | DRCT | 비고 |
|:--|--:|--:|:--|
| Arch0 (Sequential) | 0.x | +2.48pp | DRCT가 약간 향상 |
| Arch2 (SoftGate) | 0.x | +6.61pp | DRCT가 크게 향상 |
| Arch4 (Dual-YOLO, properbase) | 0.8007 | 0.7973 | RFDN이 약간 높음 |
| Arch4 (NMS-opt v2) | 0.7981 | 0.7990 | DRCT가 약간 높음 |
| Arch5b (Fusion) | **0.9102** 🥇 | (서버 학습 결과 추가 예정) | SOTA |

## 결과 출처 파일

| 결과 | JSON 위치 |
|:--|:--|
| Fair eval | `results/json/arch4_evals/fair_{sr}.json` |
| Properbase | `results/json/properbase/{sr}_final_map.json` |
| NMS sweep v2 | `results/json/nms_sweep_v2/{sr}/stage2/summary.json` |
| Sensitivity | `results/json/sensitivity/{sr}/*.json` |
| Arch5b RFDN | `results/json/arch5/rfdn_phase3_eval.json` |

## 검증 포인트

- [ ] 모든 결과가 subset6418에서 측정
- [ ] mAP@50이 메인 metric
- [ ] 모든 결과가 JSON으로 추적 가능
- [ ] Ranking 변동이 paper finding과 일치
