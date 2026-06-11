# 05-04. Gate Network 학습 (Arch2)

## 개요

Arch2의 Gate network는 **label-based BCE training**으로만 학습된다 (detection-loss 방식은 검증 후 폐기).

## Pseudo-label 생성 (1회)

각 train 이미지에 대해:
1. SR로 sr_hr 생성
2. LR을 bilinear upsample → bilinear_hr
3. Pretrained YOLO로 두 버전 각각 inference
4. GT bbox와 IoU 계산
5. 픽셀 단위로 어느 버전이 더 정확한지 판정
6. → binary mask (1 = SR이 더 좋음, 0 = bilinear가 더 좋음)
7. `data/gate_labels_{sr}/train/{stem}.npy`로 저장

## 학습 loop

```
Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
Batch size: 32
Iterations: 250,000
Loss: BCE(predicted_gate, pseudo_label)
Data: LR → gate inference → BCE with pseudo_label
```

각 SR backbone마다 별도 학습 (SR 특성이 다르므로).

## 결과

| SR | Gate weight |
|:--|:--|
| RFDN | `weights/gate_arch2/best.pt` |
| DRCT | `weights/gate_drct/best.pt` |
| HAT | `weights/gate_hat/best.pt` |
| MAN | `weights/gate_man/best.pt` |

## Detection-loss 방식 폐기

이전 검증에서 detection-loss로 gate를 학습하려고 시도했으나:
- mAP@50이 BCE 방식 대비 -4pp~5pp 낮음
- 수렴 불안정
- → 최종 모든 SR에서 **BCE 250K iter 방식만 사용**

자세히: [feedback_gate_training](../../memory_reference)

## 검증 포인트

- [ ] Gate 4종 모두 BCE 250K로 학습 (방식 통일)
- [ ] 같은 hyperparameter (lr, batch, optimizer)
- [ ] Pseudo-label이 train set에서만 생성 (val 누설 없음)
- [ ] Detection-loss 방식 사용 안 됨이 명시
