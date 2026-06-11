# 03-03. Arch4 — Dual-YOLO with ROI-aware NMS

## 구조 한 줄

> LR → Scout YOLO (LR 직접 검출) → 신뢰도 낮은 ROI만 SR → Sniper YOLO (HR crop) → ROI-aware NMS

본 연구 **paper Section IV의 핵심 아키텍처**. SR을 이미지 전체에 적용하지 않고 필요한 ROI에만 적용하는 효율적인 구조.

## 흐름

```
LR (1, 3, 192, 192)
   ↓
[Pass 1] Scout YOLO (LR에서 직접 검출)
   ↓
Scout boxes with scores
   ├─ score > high_conf (0.45) → confident (그대로 사용, SR 안 거침)
   └─ pass1_conf < score < high_conf → uncertain (ROI 후보)
   ↓
[ROI processing] uncertain boxes를 ROI로 그룹핑, expansion (1.5~2.5×)
   ↓
[Pass 2] 각 ROI에 대해
   ├─ LR crop → SR → HR crop (64×64 LR → 256×256 HR)
   └─ Sniper YOLO (HR crop에서 정밀 검출)
   ↓
[Final NMS] ROI-aware NMS로 confident + Sniper detection 합침
   ↓
final detections
```

## 코드

`src/models/pipelines/arch4_roi_awareNMS.py` (`Arch4RoiAwareNMS` 클래스)

또는 ablation 버전: `arch4_roi_awareNMS_ablation.py`

## 핵심 파라미터 (fixed_protocol 기준)

| 파라미터 | 값 | 의미 |
|:--|--:|:--|
| pass1_conf | 0.0075 | Scout 최소 confidence |
| high_conf | 0.45 | 즉시 확정 threshold |
| final_conf | 0.25 | 최종 출력 threshold |
| sniper_conf | 0.001 | Sniper detection threshold |
| merge_iou | 0.5 | Scout+Sniper merge IoU |
| roi_expansion | 1.75 | ROI 박스 확장 비율 |
| crop_size_lr | 64 | LR crop 크기 (HR로 256) |

### NMS 6개 추가 파라미터 (ROI-aware NMS)

| 파라미터 | 값 | 의미 |
|:--|--:|:--|
| scout_nms_iou | 0.5 | Scout NMS IoU |
| roi_merge_iou | 0.3 | ROI 그룹핑 IoU |
| roi_center_ratio | 0.35 | ROI 중심 매칭 비율 |
| sniper_nms_iou | 0.45 | Sniper NMS IoU |
| final_nms_iou | 0.5 | 최종 NMS IoU |
| drop_uncertain_if_sniper_hits | true | Sniper에서 검출 시 Scout uncertain 제거 |

### 추가 5개 파라미터 (size_cond merge 정책, sniper 후처리)

| 파라미터 | 값 | 의미 |
|:--|--:|:--|
| merge_policy | size_cond | 크기 조건 merge |
| roi_small_thresh | 32.0 | 작은 ROI 임계값 |
| roi_large_thresh | 96.0 | 큰 ROI 임계값 |
| large_roi_score_thresh | 0.5 | 큰 ROI 최소 score |
| sniper_replace_margin | 0.1 | Sniper가 Scout 대체 시 margin |
| sniper_max_det_per_crop | 3 | 각 crop당 최대 detection 수 |

이 6+5개 파라미터가 v2 sweep에서 모두 포함됨 (v1에서는 일부 누락).

## 컴포넌트별 weight

각 SR backbone마다 별도 weight를 가짐:

| SR | Scout (LR detector) | Sniper (HR detector) |
|:--|:--|:--|
| RFDN | yolo_lr_improved (공통) | yolo_8s_rfdn, yolo_sniper_hardneg (Old Pipeline) |
| DRCT | yolo_lr_improved (공통) | yolo_8s_drct, properbase hardneg sniper |
| HAT | yolo_lr_improved (공통) | yolo_8s_hat, properbase hardneg sniper |
| MAN | yolo_lr_improved (공통) | yolo_8s_man, properbase hardneg sniper |

**중요**: Scout는 LR 입력만 사용하므로 SR backbone과 무관 → **모든 SR이 같은 Scout 사용**.
Sniper는 SR 출력으로 만든 crop을 사용하므로 SR backbone별 별도 학습 필요.

## SR backbone별 차이는 어디서 오는가

Arch4의 SR backbone 비교에서 차이는 다음 경로에서만 발생:
1. **SR weight 자체** (RFDN vs DRCT vs HAT vs MAN)
2. **Sniper weight** (각 SR의 crop으로 학습됨)
3. **(선택) NMS 파라미터** (Section IV-C에서 sweep)

Scout, 이미지 로딩, dataset, 평가 metric은 모두 동일.

## 결과

| SR | Fair | Properbase | v2 NMS-opt |
|:--|--:|--:|--:|
| RFDN | 0.7731 | 0.8007 | 0.7981 |
| DRCT | 0.7806 | 0.7973 | 0.7990 |
| HAT | 0.7733 | 0.7905 | **0.8003** 🥇 |
| MAN | 0.7720 | 0.7918 | 0.7940 |

## 검증 포인트

- [ ] Scout가 모든 SR에서 공통(`yolo_lr_improved`) 사용
- [ ] 각 SR마다 적절한 Sniper weight 사용 (rfdn base 사고 재발 방지)
- [ ] 11개 NMS 파라미터가 fixed_protocol에 명시되는가
- [ ] v2 sweep에서 누락 6개 파라미터가 모두 포함되는가
- [ ] ROI expansion이 같은 grid에서 비교되는가
