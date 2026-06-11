# 03-02. Arch2 — Soft Gate (SR + Bilinear Blend)

## 구조 한 줄

> LR 이미지 → Gate network가 픽셀별 SR/bilinear 가중치 결정 → blend → YOLO → detection

SR이 모든 픽셀에 동등하게 좋은 게 아니라는 가정 하에, Gate가 어떤 영역은 SR을 더 쓰고 어떤 영역은 bilinear를 더 쓰도록 결정.

## 흐름

```
LR (1, 3, 192, 192)
   ↓
   ├─ SR backbone → sr_hr (1, 3, 768, 768)
   ├─ bilinear upsample → bilinear_hr (1, 3, 768, 768)
   └─ Gate network → gate_mask (1, 1, 768, 768) ∈ [0, 1]
   ↓
fused_hr = gate * sr_hr + (1 - gate) * bilinear_hr
   ↓
YOLOv8 → detection
```

## 코드

`src/models/pipelines/arch2_softgate.py` (`Arch2SoftGate` 클래스)

## Gate Network 구조

- 입력: LR tensor (또는 LR upsampled)
- 출력: 0~1 mask, HR 해상도
- 학습 방식: **Label-based BCE training** (250K iterations)
- 학습 데이터: SR vs bilinear의 detection 결과를 비교한 pseudo-label
- 자세히: [docs/04_components/03_gate_network.md](../04_components/03_gate_network.md)

## Gate 학습이 SR backbone별로 별도

각 SR backbone마다 Gate를 별도로 학습해야 한다 (SR의 특성이 다르므로):
- `gate_rfdn/weights/best.pt`
- `gate_drct/weights/best.pt`
- `gate_hat/weights/best.pt`
- `gate_man/weights/best.pt`

→ 모든 Arch2 평가에서 **각 SR과 일치하는 gate를 사용해야 공정**.

## 학습 방식 (요약)

```
1. 각 train 이미지에 대해 SR과 bilinear 두 버전 생성
2. 각 버전에서 YOLO inference
3. 어느 버전이 GT와 더 잘 맞는지 픽셀 단위로 결정 → gate label (0/1)
4. Gate network를 BCE loss로 250K iter 학습
```

자세한 학습 절차: [docs/05_training/05_gate_training.md](../05_training/05_gate_training.md)

## 평가 시 inference

```python
# arch2_softgate.py
def forward(self, lr_tensor):
    sr_hr = self.sr_model(lr_tensor)
    bilinear_hr = F.interpolate(lr_tensor, scale_factor=4, mode='bilinear')
    
    gate = self.gate_model(lr_tensor)   # ∈ [0, 1]
    fused = gate * sr_hr + (1 - gate) * bilinear_hr
    
    detections = self._predict_detector(fused)
    return {"detections": detections, "gate": gate}
```

### Soft vs Hard 모드
- `inference_threshold: 0.0` (fixed_protocol 기본) → 모든 픽셀 soft blend
- `blend_selected_inference: True` → 항상 gate * sr + (1-gate) * bilinear 적용

→ 모든 SR backbone에 동일한 inference 모드 적용.

## 결과 (예시)

| SR | Arch2 mAP@50 (subset6418) |
|:--|--:|
| RFDN | (이전 측정값) |
| DRCT | (이전 측정값, +6.61pp 개선) |

이 +6.61pp는 paper Section IV의 main finding 중 하나로, DRCT가 Arch2에서 RFDN 대비 큰 향상을 보였다는 것을 의미.

## 검증 포인트

- [ ] 각 SR에 맞는 gate weight를 사용하는가
- [ ] Gate 학습이 동일한 protocol(BCE, 250K iter)인가
- [ ] SR과 bilinear의 blend 공식이 일관되는가
- [ ] Gate threshold/blend 설정이 fixed_protocol에 의해 강제되는가
