# 04-01. Scout YOLO (Arch4 LR Detector)

## 역할

Arch4의 첫 번째 검출기. **LR 이미지(192×192)에서 직접 ROI 후보를 검출**.

## Weight

**모든 SR backbone에서 공통 사용** (Scout는 LR만 처리하므로 SR 출력과 무관):

```
/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt
```

- 모델: YOLOv8s
- 입력 해상도: 192
- MD5 prefix: `f9f175f7f758` (fixed_protocol에 명시)
- 학습 단계: 2단계 (deadline + stage2 refinement)

## fixed_protocol.yaml 명시

```yaml
scout:
  weight: "/home/changmin/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt"
  md5_prefix: "f9f175f7f758"
  description: "YOLOv8s stage2, LR improved, IAC 0.7986 기준"
```

→ Pre-flight check 함수가 매 실행 시 MD5 확인하여 Scout 일관성 강제.

## 모든 SR backbone에 공통 적용

| SR | Scout |
|:--|:--|
| RFDN | yolo_lr_improved/stage2/best.pt ✅ |
| DRCT | yolo_lr_improved/stage2/best.pt ✅ |
| HAT | yolo_lr_improved/stage2/best.pt ✅ |
| MAN | yolo_lr_improved/stage2/best.pt ✅ |

이는 Scout가 SR 출력과 무관하므로 공정성에 기여.

## Inference 절차

Arch4의 Pass 1:
```python
scout_results = self.scout_detector.predict(
    lr_tensor,
    conf=self.config["pass1_conf"],   # 0.0075
    iou=self.config["scout_nms_iou"]  # 0.5
)
```

출력: `boxes (N, 4), scores (N,), classes (N,)`

각 box를 `high_conf=0.45`로 분류:
- `score > 0.45` → confident (그대로 사용)
- `0.0075 < score < 0.45` → uncertain (ROI 후보)
- `score < 0.0075` → drop

## 검증 포인트

- [ ] Scout가 모든 SR backbone 실험에서 같은 weight 사용
- [ ] MD5 prefix가 fixed_protocol과 일치
- [ ] Pre-flight check 함수가 검증 강제
- [ ] LR 192×192 입력으로 동작
