# 09-03. Weight Locations (모든 weight의 정확한 경로)

본 staging repo에는 weights가 **포함되지 않는다**. 검사자/재현자는 메인 폴더에서 직접 접근.

**메인 폴더 위치**: `/home/changmin/dark_vessel_sr_yolo/`

---

## 1. SR Backbone Weights

| SR | 정확한 경로 | 비고 |
|:--|:--|:--|
| **RFDN** | `/home/changmin/dark_vessel_sr_yolo/weights/rfdn_arch4/model_best.pt` | Mar 14, MD5 `539f72b2` |
| ⚠️ RFDN 잘못된 weight | ~~`weights/rfdn/model_best.pt`~~ | Jan 12, MD5 `0087ca54`. **사용 금지** |
| **DRCT** | `/home/changmin/dark_vessel_sr_yolo/weights/sr_finetuned/drct/best.pt` | crop64 fine-tune |
| **HAT** | `/home/changmin/dark_vessel_sr_yolo/weights/sr_finetuned/hat/best.pt` | |
| **MAN** | `/home/changmin/dark_vessel_sr_yolo/weights/sr_finetuned/man/best.pt` | |

---

## 2. Scout YOLO (Arch4 LR detector) — 모든 SR 공통

```
/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt
```

- MD5 prefix: `f9f175f7f758`
- 4 SR backbone 실험 모두에서 동일 weight 사용
- fixed_protocol.yaml의 `scout.weight`에 명시

---

## 3. Sniper YOLO (Arch4 HR crop detector)

### 3-A. From-scratch (Fair eval용)

| SR | 정확한 경로 |
|:--|:--|
| RFDN | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/weights/best.pt` |
| DRCT | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_drct/weights/best.pt` |
| HAT | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_hat/best.pt` |
| MAN | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_man/best.pt` |

### 3-B. Hardneg fine-tuned (Properbase + NMS sweep용)

| SR | 정확한 경로 |
|:--|:--|
| RFDN | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/weights/best.pt` |
| DRCT | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/20260609_070148_hardneg_drct_pb/weights/best.pt` |
| HAT | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/20260609_*_hardneg_hat_pb/weights/best.pt` |
| MAN | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/20260609_*_hardneg_man_pb/weights/best.pt` |

⚠️ HAT/MAN의 정확한 timestamp는 properbase_pipeline 디렉토리에서 확인:
```bash
ls /home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/ | grep -E "hat_pb|man_pb"
```

---

## 4. Gate Network (Arch2)

| SR | 정확한 경로 |
|:--|:--|
| RFDN | `/home/changmin/dark_vessel_sr_yolo/weights/gate_arch2/best.pt` |
| DRCT | `/home/changmin/dark_vessel_sr_yolo/weights/gate_drct/best.pt` |
| HAT | `/home/changmin/dark_vessel_sr_yolo/weights/gate_hat/best.pt` |
| MAN | `/home/changmin/dark_vessel_sr_yolo/weights/gate_man/best.pt` |

---

## 5. HR Detector (Arch0/2/5)

| SR | 정확한 경로 |
|:--|:--|
| RFDN | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_rfdn/weights/best.pt` (Arch0/2/5 공통) |
| DRCT | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_drct/weights/best.pt` |
| HAT | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_hat/best.pt` |
| MAN | `/home/changmin/dark_vessel_sr_yolo/weights/yolo_8s_man/best.pt` |

---

## 6. Arch5b Checkpoints (4090 서버)

| SR | Phase 3 checkpoint |
|:--|:--|
| RFDN | (서버 경로: `/path/to/arch5b_rfdn_phase3_best.pt`) |
| DRCT | (서버 경로) |

서버에서 가져오는 명령: [docs/PROMPTS/server_arch5_request.md](../PROMPTS/server_arch5_request.md)

---

## 검증 명령 (모든 weight 존재 확인)

```bash
cd /home/changmin/dark_vessel_sr_yolo
for w in \
  weights/rfdn_arch4/model_best.pt \
  weights/sr_finetuned/drct/best.pt \
  weights/sr_finetuned/hat/best.pt \
  weights/sr_finetuned/man/best.pt \
  weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt \
  weights/yolo_8s_rfdn/weights/best.pt \
  weights/yolo_8s_drct/weights/best.pt \
  weights/yolo_8s_hat/best.pt \
  weights/yolo_8s_man/best.pt \
  weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/weights/best.pt \
  weights/yolo_sniper_hardneg/20260609_070148_hardneg_drct_pb/weights/best.pt; do
  if [ -f "$w" ]; then
    echo "OK: $w"
  else
    echo "MISSING: $w"
  fi
done
```

## MD5 검증

Scout weight는 `fixed_protocol.yaml`의 MD5 prefix와 일치해야 함:
```bash
md5sum weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt | cut -c1-12
# → f9f175f7f758
```

## 검증 포인트

- [ ] 모든 weight 경로 존재
- [ ] RFDN은 `rfdn_arch4/model_best.pt` (Mar 14)
- [ ] Scout MD5가 fixed_protocol과 일치
- [ ] 잘못된 weight (rfdn/, yolohr/8s) 사용하지 않음
- [ ] Properbase Sniper가 각 SR 전용 (rfdn base 사고 재발 방지)
