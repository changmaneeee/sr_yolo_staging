# 04-06. YOLO Wrapper (공통 검출기)

## 역할

Ultralytics YOLOv8을 wrapper로 감싸서 모든 Arch에서 일관된 인터페이스 제공.

## 위치

`src/models/detectors/yolo_wrapper.py` (`YOLOWrapper` 클래스)

## 주요 기능

```python
class YOLOWrapper:
    def __init__(self, weights_path, num_classes=1, imgsz=640):
        self.yolo = YOLO(weights_path)
        ...
    
    def predict(self, x, conf=0.25, iou=0.5):
        # x: (B, 3, H, W) tensor
        results = self.yolo.predict(x, conf=conf, iou=iou)
        return [{
            "boxes": r.boxes.xyxy,
            "scores": r.boxes.conf,
            "classes": r.boxes.cls,
        } for r in results]
```

## 모든 Arch에서 동일 인터페이스

| Arch | Wrapper 사용 |
|:--|:--|
| Arch0 | YOLOWrapper(hr_detector_weight) → predict(sr_image) |
| Arch2 | YOLOWrapper(hr_detector_weight) → predict(fused_image) |
| Arch4 Scout | YOLOWrapper(lr_detector_weight) → predict(lr_image) |
| Arch4 Sniper | YOLOWrapper(sniper_weight) → predict(sr_crop_batch) |
| Arch5b | YOLO head 직접 사용 (wrapper 안 거침, fused feature 입력) |

## 공정성 보장

- 모든 YOLO 호출이 같은 wrapper로 동작 → 후처리, NMS, classes 등 일관
- conf/iou는 Arch별로 다를 수 있으나 fixed_protocol로 통제
- num_classes=1 (ship) 고정

## 검증 포인트

- [ ] 모든 Arch가 같은 wrapper를 사용
- [ ] predict 출력 형식이 일관됨 (boxes/scores/classes dict)
- [ ] num_classes=1 통일
- [ ] conf/iou가 fixed_protocol로 통제
