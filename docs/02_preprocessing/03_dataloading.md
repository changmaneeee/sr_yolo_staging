# 02-03. Data Loading & Augmentation

## 평가 시 데이터 로딩 (모든 실험)

`eval_arch.py`, `arch4_eval_ultralytics.py` 둘 다 동일한 로딩 방식:

```python
img = cv2.imread(str(img_path))         # BGR uint8 (H,W,3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0     # 0-1 float32
img = img.transpose(2, 0, 1)             # (3, H, W)
lr_tensor = torch.from_numpy(img).unsqueeze(0).to(device)   # (1, 3, H, W)
```

- BGR → RGB 변환 ✅
- 0-255 → 0-1 정규화 ✅
- (H,W,C) → (C,H,W) 채널 변환 ✅
- batch 차원 추가 ✅

→ **모든 SR backbone, 모든 Arch가 같은 입력 텐서 형식을 받는다.**

## 학습 시 augmentation

### YOLO Scout/Sniper 학습
Ultralytics 기본 augmentation 사용:
- Mosaic
- HSV jitter
- Horizontal flip
- Random scale
- 모든 학습에서 동일 설정 (각 SR backbone별 별도 fine-tune 시에도)

### SR backbone fine-tune
SR backbone들은 본 연구에서 직접 학습하지 않고 **pretrained weight를 그대로 사용**한다.
단, fine-tune이 필요한 경우(예: DRCT의 crop64 fine-tune)는 다음 augmentation:
- Random crop
- HFlip
- 동일 batch size, optimizer 사용

### Gate (Arch2) 학습
- Label-based BCE training (250K steps)
- LR/HR pair를 입력으로 받아 0-1 mask 출력

## 평가 시 inference 흐름

각 Arch별 평가 흐름:

### Arch0
1. LR tensor 입력 → SR → HR tensor
2. HR tensor → YOLO → detection

### Arch2
1. LR tensor 입력 → SR + bilinear blend (Gate 비율) → HR tensor
2. HR tensor → YOLO → detection

### Arch4
1. LR tensor 입력 → Scout YOLO → ROI candidates
2. 각 ROI에 대해 LR crop → SR → HR crop
3. HR crop → Sniper YOLO → fine detection
4. Scout + Sniper detection을 ROI-aware NMS로 합침

### Arch5b
1. LR tensor 입력 → LR feature 추출 (YOLO backbone)
2. LR tensor → SR → HR feature 추출 (YOLO backbone)
3. Attention fusion으로 두 feature 결합
4. Fused feature → YOLO head → detection

상세 구현: [docs/03_architectures/](../03_architectures/)

## 검증 포인트

- [ ] 학습/평가 데이터 로딩이 동일한 BGR→RGB→0-1 변환
- [ ] 모든 SR backbone이 같은 입력 텐서를 받는가
- [ ] YOLO 학습 시 augmentation이 SR backbone별로 다르지 않은가
- [ ] 평가 시 inference 흐름이 각 Arch에서 일관되는가
