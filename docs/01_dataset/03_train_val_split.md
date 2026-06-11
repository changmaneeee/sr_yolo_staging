# 01-03. Train / Val Split

## 분할 기준

### Train (108,414 images)
- 위치: `/home/changmin/smart_airbus_data/images/train/`
- 라벨: `/home/changmin/smart_airbus_data/labels/train/`
- 용도: 모든 모델(SR backbone, YOLO Scout/Sniper, Gate, Arch5 phase)의 학습
- Label coverage: 거의 100% (background-only image도 일부 포함)

### Val (28,883 images, evaluation에는 subset6418 사용)
- 위치: `/home/changmin/smart_airbus_data/images/val/`
- 라벨: `/home/changmin/smart_airbus_data/labels/val/` (6,418개만 존재)
- 용도: 학습 중 검증, 최종 mAP 측정 (subset6418로)

## Subset6418 (평가 메인)

본 연구에서 모든 메인 평가 결과는 **subset6418** 기준으로 산출됨.

### 선정 기준
HR `labels/val/` 디렉토리에서 label 파일이 존재하고 **non-empty** (적어도 1개 ship instance가 있는) 이미지만.

### 정의 (logic)
```python
hr_labels_dir = Path("/home/changmin/smart_airbus_data/labels/val")
labeled_stems = {f.stem for f in hr_labels_dir.glob("*.txt") if f.stat().st_size > 0}
img_paths = [p for p in list_images(lr_images_dir) if p.stem in labeled_stems]
assert len(img_paths) == 6418
```

이 logic은 `eval_arch.py`(line 471~472)에서 직접 확인 가능하며, `arch4_eval_ultralytics.py`의 `--labeled_only` 옵션도 동일한 로직을 사용함.

### subset 위치
- HR: `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr/`
- LR: `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr/`
- 생성 시점: 2026-03-16

### 왜 28883 전체를 안 쓰는가?
- 28,883장 중 22,465장이 label 없음 (negative image)
- 만약 28,883 전체를 평가에 쓰면 false positive 다수 발생 → precision/mAP 왜곡
- labeled_only로 평가하면 precision/recall 측정이 더 신뢰성 있음
- 이는 `fixed_protocol.yaml`의 `filter: "labeled_only"`로 강제됨

## 모든 실험에서 동일 적용

| 실험 | Val set |
|:--|:--|
| Fair eval | subset6418 (eval_arch.py + labeled_only filter) |
| Properbase Phase D/E mAP | subset6418 (arch4_eval_ultralytics.py + `--labeled_only` 효과) |
| NMS sweep v1/v2 Stage 2 | subset6418 (`--max_images 0` + labeled_only) |
| Sensitivity ablation | subset6418 (동일) |

**모든 평가에서 정확히 같은 6,418장이 사용됨.**

## 검증 포인트

- [ ] train과 val이 disjoint한가
- [ ] subset6418의 6418장 ≡ {x : label(x) exists and non-empty}
- [ ] subset 선정 로직이 코드에서 추적 가능한가
- [ ] 모든 실험에서 동일 subset 사용됨이 명시되는가
