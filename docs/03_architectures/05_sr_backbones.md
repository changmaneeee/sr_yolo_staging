# 03-05. SR Backbones (RFDN / DRCT / HAT / MAN)

본 연구가 비교한 4개의 SR backbone에 대한 상세 명세.

## 공통 사양

| 항목 | 값 |
|:--|:--|
| Scale factor | 4× (192→768) |
| Input | RGB (3 channel) |
| input_range | "0-255" (모든 backbone) |
| Output | (1, 3, 768, 768), [0-1] |

## 1. RFDN

- **이름**: Residual Feature Distillation Network
- **참조**: AIM 2020 SR Challenge
- **파라미터**: ~0.5M (가장 가벼움)
- **구조**: 여러 RFDB block, residual distillation
- **본 연구의 wrapper**: `src/models/sr_models/rfdn.py`

### 본 연구 사용 설정
```python
RFDN(
    in_channels=3, out_channels=3, 
    nf=50, num_modules=4,
    upscale=4, input_range="0-255"
)
```

### Weights
- 주: `weights/rfdn_arch4/model_best.pt` (Mar 14, MD5 `539f72b2`) — **모든 실험에 사용**
- 다른 weight (혼동 주의): `weights/rfdn/model_best.pt` (Jan 12, MD5 `0087ca54`) — **사용 금지**, NMS sweep v1 실수 원인

자세한 weight 명세: [docs/09_reproducibility/03_weight_locations.md](../09_reproducibility/03_weight_locations.md)

## 2. DRCT

- **이름**: Dilated Residual Convolutional Transformer
- **참조**: CVPR 2024
- **파라미터**: ~14M
- **구조**: Swin Transformer 기반 + dilated convolution
- **본 연구의 wrapper**: `sci_lab/backbones/drct_wrapper.py`

### 본 연구 사용 설정
```python
DRCTWrapper(
    scale=4, pretrained_path=None,
    variant="base", input_range="0-255"
)
```

### Weights
- `weights/sr_finetuned/drct/best.pt`

### 특이사항
- Swin window 단위 처리로 인해 작은 ROI(<32px)에 weakness 가능
- crop64 fine-tune 별도 진행 (검증 #36)

## 3. HAT

- **이름**: Hybrid Attention Transformer
- **참조**: CVPR 2023
- **파라미터**: ~21M (가장 무거움)
- **구조**: Channel attention + window self-attention
- **본 연구의 wrapper**: `sci_lab/backbones/hat_wrapper.py`

### 본 연구 사용 설정
```python
HATWrapper(
    scale=4, pretrained_path=None,
    variant="base", input_range="0-255"
)
```

### Weights
- `weights/sr_finetuned/hat/best.pt`

### 특이사항
- v2 NMS sweep에서 **mAP@50 0.8003으로 1위**
- Window 크기가 DRCT와 다르며, boundary effect 검증됨 (검증 #33)

## 4. MAN

- **이름**: Multi-scale Attention Network
- **참조**: 2023
- **파라미터**: ~9M (중간)
- **구조**: Multi-scale attention block
- **본 연구의 wrapper**: `sci_lab/backbones/man_wrapper.py`

### 본 연구 사용 설정
```python
MANWrapper(
    scale=4, pretrained_path=None,
    variant="base", input_range="0-255"
)
```

### Weights
- `weights/sr_finetuned/man/best.pt`

### 특이사항
- DRCT, HAT 대비 약간 낮은 성능
- v2 sweep에서 4위

## 공정 비교를 위한 통일

| 항목 | 모든 SR 통일 |
|:--|:--|
| Scale factor | 4× |
| Input shape | (B, 3, 192, 192) |
| Input range | 0-255 (wrapper에서 내부 변환) |
| Output shape | (B, 3, 768, 768) |
| Output range | 0-1 |
| Pretrained weight | 각자 base weight 그대로 (재학습 없음) |

→ **4 SR이 같은 인터페이스로 호환됨. 코드 경로가 동일.**

## Mamba SR 위험

이전 사고(검증 #45): Mamba 기반 SR weight를 RFDN loader로 로드하면 F1~0.0003으로 완전 실패.
→ **각 SR backbone은 반드시 자기 wrapper로 로드해야 함.**

이는 `eval_arch.py`의 `_load_sr_model()` 함수에서 backbone 이름으로 분기하여 강제됨.

## 검증 포인트

- [ ] 4 SR이 모두 같은 입/출력 shape와 range 보장
- [ ] 같은 input_range="0-255" 사용
- [ ] 각 SR에 적절한 wrapper와 weight 사용 (Mamba loader 사고 재발 방지)
- [ ] base weight를 재학습 없이 그대로 사용
