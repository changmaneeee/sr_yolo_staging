# 07-01. Fair Comparison (Section IV-A)

## 목적

모든 SR backbone에 동일한 NMS 설정(`fixed_protocol.yaml`)을 적용하여 first-pass 비교.

## 방법

```bash
# 4개 SR 모두 같은 명령어 패턴
python eval_arch.py --sr-backbone {rfdn,drct,hat,man} --arch 4 \
  --sr-weight weights/{sr_backbone_specific} \
  --sniper-weight weights/yolo_8s_{sr}/weights/best.pt   # from-scratch Sniper
```

## 조건

- **NMS**: fixed_protocol의 모든 파라미터 (4 SR에 동일)
- **Sniper**: from-scratch (각 SR 전용 yolo_8s_{sr})
- **Scout**: 공통 yolo_lr_improved/stage2
- **Val set**: subset6418 (6418장)
- **Metric**: mAP@50

## 결과

| SR | mAP@50 |
|:--|--:|
| **DRCT** | **0.7806** |
| HAT | 0.7733 |
| RFDN | 0.7731 |
| MAN | 0.7720 |

- Range: 0.86pp
- Ranking: DRCT > HAT > RFDN > MAN

## 결과 파일

| SR | 위치 |
|:--|:--|
| RFDN | `iac_runs/*/eval_arch_rfdn_arch4.json` |
| DRCT | `iac_runs/*/eval_arch_drct_arch4.json` |
| HAT | `iac_runs/*/eval_arch_hat_arch4.json` |
| MAN | `iac_runs/*/eval_arch_man_arch4.json` |

## 공정성 보장

- 같은 fixed_protocol → 같은 NMS, 같은 conf threshold
- 같은 Scout
- 각 SR이 자기 from-scratch Sniper 사용 (공평한 출발점)
- 같은 subset6418 평가
- 같은 ap_per_class metric

## 검증 포인트

- [ ] 4 SR 모두 같은 fixed_protocol 적용
- [ ] from-scratch Sniper 사용 (hardneg 아님)
- [ ] subset6418에서만 평가
- [ ] 같은 metric 계산
