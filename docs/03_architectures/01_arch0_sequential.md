# 03-01. Arch0 — Sequential SR → YOLO

## 구조 한 줄

> LR 이미지 → SR backbone → HR 이미지 → YOLOv8 → detection

가장 단순한 결합 방식. SR이 HR 이미지를 만들고, 그 HR을 YOLO가 처리.

## 흐름

```
LR (1, 3, 192, 192) [0-1 RGB]
   ↓
SR backbone (RFDN/DRCT/HAT/MAN)
   ↓
HR (1, 3, 768, 768) [0-1 RGB]
   ↓
YOLOv8s detector (HR로 학습된 yolo_8s_{sr}/best.pt)
   ↓
detection {boxes, scores, classes}
```

## 코드

`src/models/pipelines/arch0_sequential.py` (`Arch0Sequential` 클래스)

핵심 forward:
```python
def forward(self, lr_tensor):
    # 1. SR
    sr_image = self.sr_model(lr_tensor)   # (1, 3, 768, 768)
    
    # 2. Detection
    detections = self._predict_detector(sr_image)
    
    return sr_image, detections
```

## 학습 방식

본 Arch는 별도 학습이 없다. 두 컴포넌트를 사용:
1. **SR**: pretrained (RFDN model_best.pt, DRCT base, HAT base, MAN base)
2. **YOLO**: 각 SR backbone별로 학습된 detector
   - `weights/yolo_8s_rfdn/weights/best.pt` (RFDN SR 출력으로 학습)
   - `weights/yolo_8s_drct/weights/best.pt` (DRCT SR 출력으로 학습)
   - `weights/yolo_8s_hat/best.pt`
   - `weights/yolo_8s_man/best.pt`

각 SR 전용 YOLO를 사용함으로써 **각 SR의 특성에 맞게 detector가 학습됨**.

## 평가 시 사용 NMS

`fixed_protocol.yaml`의 NMS 설정 (모든 Arch 공통):
- `conf_threshold: 0.25`
- `iou_threshold: 0.5`

`evaluate()` 함수에서 inference 후 그대로 적용.

## 모든 SR backbone에 동일 적용

Arch0의 평가 방식은 SR backbone에 따라 다르지 않다:
- 입력: 항상 LR (1, 3, 192, 192)
- SR: 해당 backbone (RFDN/DRCT/HAT/MAN)
- YOLO: 해당 backbone 전용 detector
- 출력: detection dict (boxes, scores, classes)

→ **4 SR backbone이 같은 코드 경로를 거쳐 평가됨**.

## 결과 (예시)

| SR | Arch0 mAP@50 (subset6418) |
|:--|--:|
| RFDN | (이전 측정값) |
| DRCT | (이전 측정값, +2.48pp 개선) |

상세 결과: [docs/08_results/](../08_results/)

## 검증 포인트

- [ ] Arch0의 평가 흐름이 모든 SR backbone에서 동일한가
- [ ] 각 SR에 적절한 YOLO weight가 사용되는가 (rfdn 출력에 yolo_8s_rfdn 등)
- [ ] NMS 파라미터가 공통(fixed_protocol)인가
- [ ] 별도 학습 없이 pretrained만 사용함이 명확한가
