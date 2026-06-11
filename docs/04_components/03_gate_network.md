# 04-03. Gate Network (Arch2)

## 역할

Arch2에서 SR과 bilinear upsampling 사이의 픽셀별 blend 비율 결정.

## 구조

- 입력: LR tensor (1, 3, 192, 192)
- 중간: convolution backbone (base_channels=32, num_layers=4)
- 출력: HR 해상도의 1-channel mask, 0~1

## SR별 별도 학습

Gate는 SR 출력의 특성을 학습하므로 **각 SR마다 별도 학습**.

| SR | Gate weight |
|:--|:--|
| RFDN | `weights/gate_arch2/best.pt` |
| DRCT | `weights/gate_drct/best.pt` |
| HAT | `weights/gate_hat/best.pt` |
| MAN | `weights/gate_man/best.pt` |

## 학습 방식

### Label-based BCE (250K iterations)
- **이 방식만 사용 가능** (detection-loss 방식은 검증 후 폐기)
- 자세히: [feedback_gate_training](../../memory_reference.md)

### Pseudo-label 생성
```
for each train image:
    1. SR로 sr_hr 생성
    2. bilinear로 bilinear_hr 생성
    3. 각각 YOLO inference → sr_dets, bilinear_dets
    4. GT와 IoU 비교 → 어느 버전이 더 좋은지 픽셀 단위 결정
    5. → 0/1 gate label
```

### 학습 loop
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Batch: 32
- Iterations: 250,000
- Loss: BCE between predicted gate and pseudo-label

자세히: [docs/05_training/05_gate_training.md](../05_training/05_gate_training.md)

## Inference 시 설정

`fixed_protocol.yaml`의 Gate 설정:
```python
gate=SimpleNamespace(
    base_channels=32, num_layers=4, in_channels=3,
    weights_path=args.gate_weight or "",
    use_selective_inference=True,
    inference_threshold=0.0,          # 모든 픽셀 soft blend
    blend_selected_inference=True,     # gate*SR + (1-gate)*bilinear
)
```

→ 모든 SR backbone에 동일한 inference 모드.

## 검증 포인트

- [ ] 각 SR에 맞는 Gate weight 사용
- [ ] 모든 Gate가 같은 protocol(BCE 250K iter)로 학습됨
- [ ] inference_threshold=0.0, blend_selected_inference=True 통일
- [ ] Pseudo-label 생성 방식이 일관됨
