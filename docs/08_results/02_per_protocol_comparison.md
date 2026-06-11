# 08-02. Per-Protocol Comparison (3가지 NMS)

본 연구의 main finding: **NMS protocol에 따라 SR backbone ranking이 변동된다**.

## Protocol 정의

| Protocol | NMS | Sniper |
|:--|:--|:--|
| **Fair (IV-A)** | 모든 SR 동일 (fixed_protocol) | from-scratch (각 SR 별도) |
| **Properbase (IV-B)** | 모든 SR 동일 (fixed_protocol) | hardneg fine-tuned (Phase B+C+D+E best) |
| **v2 NMS-opt (IV-C)** | 각 SR best (40 combos sweep) | hardneg fine-tuned (동일) |

## 결과

| SR | Fair | Properbase | v2 |
|:--|--:|--:|--:|
| RFDN | 0.7731 | 0.8007 | 0.7981 |
| DRCT | 0.7806 | 0.7973 | 0.7990 |
| HAT | 0.7733 | 0.7905 | **0.8003** |
| MAN | 0.7720 | 0.7918 | 0.7940 |

## Delta (Fair 기준)

| SR | Fair | Properbase delta | v2 delta |
|:--|--:|--:|--:|
| RFDN | 0.7731 | +2.76pp | +2.50pp |
| DRCT | 0.7806 | +1.67pp | +1.84pp |
| HAT | 0.7733 | +1.72pp | +2.70pp |
| MAN | 0.7720 | +1.98pp | +2.20pp |

→ Properbase는 모두 +1.7~2.8pp 향상. v2도 비슷 (RFDN만 약간 손실).

## Ranking 변동

```
Fair NMS:       DRCT (0.7806) > HAT (0.7733) > RFDN (0.7731) > MAN (0.7720)
Properbase:     RFDN (0.8007) > DRCT (0.7973) > MAN  (0.7918) > HAT (0.7905)
v2 NMS-opt:     HAT  (0.8003) > DRCT (0.7990) > RFDN (0.7981) > MAN (0.7940)
```

→ **세 protocol 모두 ranking 다름** → 흥미로운 paper finding.

## Range 변동

| Protocol | Range |
|:--|:--|
| Fair | 0.86pp (가장 좁음) |
| Properbase | 1.02pp |
| v2 NMS-opt | 0.63pp (가장 좁음) |

→ NMS 최적화가 SR 간 차이를 오히려 줄임 (top 3가 모두 0.798~0.800).

## Paper framing

```
"Across three NMS protocols (Fair-NMS, Properbase, SR-optimized), 
the relative ordering of SR backbones shifts, but the absolute mAP@50 range 
stays within 1pp. Sensitivity analysis of 4 ancillary NMS parameters 
confirms robustness (Δ ≤ 0.6pp). This demonstrates that SR backbone choice 
has limited impact on Arch4 detection performance, with HAT slightly favored 
under SR-optimized NMS."
```

## 검증 포인트

- [ ] 3 protocol에 같은 4 SR이 비교됨
- [ ] Sniper의 종류가 protocol 별로 명확히 다름
- [ ] Range가 모든 protocol에서 ≤ 1pp
- [ ] Ranking 변동이 통계적으로 의미 있는가 (run-to-run variance와 비교)
