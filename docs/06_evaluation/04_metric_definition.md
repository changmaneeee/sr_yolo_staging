# 06-04. Metric Definition

## Metric 정의

본 연구는 ultralytics 표준 metric을 사용한다.

### Primary: mAP@50

- Mean Average Precision at IoU=0.5
- COCO standard / Ultralytics 표준
- 본 연구의 main result에서 사용

### Secondary: mAP@50-95

- mAP averaged over IoU 0.5 ~ 0.95 (step 0.05)
- 10개 IoU threshold에 대한 mean
- 더 strict한 평가

### Precision / Recall

- IoU=0.5 기준
- best F1 confidence에서 계산
- `metrics/precision(B)`, `metrics/recall(B)`

### Direct (P/R/F1 @ IoU=0.5)

- TP/FP/FN을 직접 계산 (사용된 confidence threshold 기준)
- `direct/precision50`, `direct/recall50`
- best confidence threshold가 아닌, 실제 사용된 threshold 기준의 metric

## 계산 코드

### eval_arch.py / arch4_eval_ultralytics.py 공통

```python
from ultralytics.utils.metrics import ap_per_class

correct = np.concatenate([s[0] for s in stats], 0)  # (N, 10) — 10 IoU thresholds
conf = np.concatenate([s[1] for s in stats], 0)
pred_cls = np.concatenate([s[2] for s in stats], 0)
target_cls = np.concatenate([s[3] for s in stats], 0)

ap_results = ap_per_class(correct, conf, pred_cls, target_cls, plot=False)
# Returns: (_, _, p, r, f1, ap, ap_class, ...)

map50 = ap[:, 0].mean()      # mAP@50
map5095 = ap.mean()           # mAP@50-95
mp = p.mean()                 # Precision
mr = r.mean()                 # Recall
```

## TP/FP/FN 계산

```python
# process_batch: predicted box vs GT box matching
correct = process_batch(pred, gt, iouv)   # (N, 10) for 10 IoU thresholds

# At IoU=0.5
tp50 = int(correct[:, 0].sum())
fp50 = num_predictions - tp50
fn50 = num_gts - tp50
```

## Eval space (좌표 공간)

본 연구는 모든 평가를 **HR space**에서 진행:
- Arch4의 LR coordinate detection은 HR로 변환 후 비교
- `eval_space: "hr"` 명시
- 768×768 픽셀 좌표계에서 IoU 계산

## Class

`num_classes=1` (ship). multi-class metric은 사용 안 함.

## 검증 포인트

- [ ] 모든 실험에서 ap_per_class가 일관되게 호출
- [ ] mAP@50이 메인 metric으로 통일
- [ ] HR space에서 좌표 비교
- [ ] num_classes=1로 동일
