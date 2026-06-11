# 06-05. Validation Set 강제

## 모든 평가 = subset6418

본 연구의 **모든** 메인 평가는 subset6418 (6,418장)에서 수행된다.

### 구체적 사용 위치

| 실험 | 평가 set 지정 |
|:--|:--|
| Fair eval (eval_arch.py) | `fixed_protocol.yaml`의 `dataset.filter: "labeled_only"` |
| Properbase Phase E mAP (arch4_eval_ultralytics.py) | `--hr_data_yaml subset6418_hr_data.yaml` |
| NMS sweep v1/v2 Stage 2 | 동일 |
| Sensitivity ablation | 동일 |

### 6418장 확인 (Pre-flight check)

```python
# eval_arch.py
labeled_stems = {f.stem for f in Path(hr_labels_dir).glob("*.txt") if f.stat().st_size > 0}
img_paths = sorted([p for p in list_images(lr_images_dir) if p.stem in labeled_stems])

expected_count = protocol["dataset"]["expected_count"]   # 6418
assert len(img_paths) == expected_count, f"Expected {expected_count} images, got {len(img_paths)}"
```

### Stage 1 (200장) 예외

NMS sweep Stage 1만 200장 quick scan 사용 (Stage 2는 6418장 정밀 평가).
Stage 1 결과는 Top-K 선정에만 사용되고, **최종 mAP@50은 항상 Stage 2 (6418장)**.

## 28,883장 전체를 안 쓰는 이유

- 28,883 - 6418 = 22,465장은 label 없음 (negative image)
- 전체로 평가 시 FP가 다수 발생 → precision 왜곡
- labeled_only로 평가 시 precision/recall이 의미있는 지표가 됨

이는 `docs/01_dataset/05_subset_6418.md`에서도 설명.

## 검증 포인트

- [ ] 모든 메인 결과가 6,418장에서 측정
- [ ] Pre-flight check가 expected_count를 강제
- [ ] subset6418의 HR/LR이 동일 stem
- [ ] Stage 1 200장이 최종 결과로 사용되지 않음
