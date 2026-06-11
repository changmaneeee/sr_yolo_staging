# 04-05. ROI-aware NMS (Arch4)

## 역할

Arch4의 최종 단계. Scout detection과 Sniper detection을 합쳐 최종 결과 생성.

## 위치

`src/models/pipelines/arch4_roi_awareNMS.py` (`Arch4RoiAwareNMS` 클래스 내부)

## 핵심 절차

```
1. Scout detection
   ├─ confident (score > 0.45): 그대로 keep
   └─ uncertain (0.0075 < score < 0.45): ROI 후보
   
2. ROI grouping
   - 가까운 uncertain box들을 그룹핑 (roi_merge_iou=0.3)
   - 각 그룹의 중심점 + roi_expansion으로 ROI 박스 생성

3. Per-ROI processing
   - LR crop → SR → HR crop (crop_size_lr=64)
   - Sniper inference (각 crop에서 최대 3 detection)

4. Merge
   - merge_policy="size_cond": ROI 크기에 따라 다른 처리
     · small (< 32): 작은 ROI는 Sniper만 신뢰
     · medium (32~96): Sniper + Scout 가중치 평균
     · large (> 96): Scout만 신뢰 (score > large_roi_score_thresh=0.5)
   - sniper_replace_margin=0.1: Sniper score가 Scout보다 0.1+ 높아야 대체
   - drop_uncertain_if_sniper_hits=True: Sniper에서 hit 시 Scout uncertain 제거

5. Final NMS
   - final_nms_iou=0.5
   - final_fusion_method ∈ {nms, soft_nms, wbf}
     · v2: soft_nms (sigma=0.3)
```

## 11개 파라미터 (모두 fixed_protocol 통제)

| 파라미터 | 값 | 의미 |
|:--|--:|:--|
| pass1_conf | 0.0075 | Scout 최소 conf |
| high_conf | 0.45 | 즉시 확정 threshold |
| final_conf | 0.25 (or 0.3) | 최종 출력 conf |
| sniper_conf | 0.001 | Sniper min conf |
| merge_iou | 0.5 | merge IoU |
| roi_expansion | 1.75 | ROI 확장 |
| crop_size_lr | 64 | LR crop 크기 |
| scout_nms_iou | 0.5 | Scout NMS |
| roi_merge_iou | 0.3 | ROI grouping |
| roi_center_ratio | 0.35 | ROI 중심 매칭 |
| sniper_nms_iou | 0.45 | Sniper NMS |
| final_nms_iou | 0.5 | 최종 NMS |
| drop_uncertain_if_sniper_hits | true | Sniper hit 시 uncertain drop |
| sniper_score_bonus | 0.0 | Sniper score 보너스 |
| merge_policy | "size_cond" | merge 정책 |
| final_fusion_method | "soft_nms" | NMS 방식 |
| soft_nms_sigma | 0.3 | soft NMS sigma |
| roi_small_thresh | 32.0 | 작은 ROI 경계 |
| roi_large_thresh | 96.0 | 큰 ROI 경계 |
| large_roi_score_thresh | 0.5 | 큰 ROI min score |
| sniper_replace_margin | 0.1 | replace margin |
| sniper_max_det_per_crop | 3 | crop당 max det |

## v2 sweep에서 6개 누락 → 추가

NMS sweep v1에서 다음 6개가 누락 → properbase config 대비 불일치 → v2에서 모두 포함:
- `final_conf`
- `roi_small_thresh`
- `roi_large_thresh`
- `large_roi_score_thresh`
- `sniper_replace_margin`
- `sniper_max_det_per_crop`

자세히: [docs/07_experiments/03_nms_sweep_v2.md](../07_experiments/03_nms_sweep_v2.md)

## 검증 포인트

- [ ] 23개 파라미터 모두 fixed_protocol에 명시
- [ ] v2 sweep config과 properbase config의 파라미터 일치
- [ ] merge_policy="size_cond"의 분기 로직이 결정론적
- [ ] sniper_max_det_per_crop=3 적용 확인
