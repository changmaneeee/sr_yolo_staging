# 02-02. Input Normalization (input_range)

## 개요

SR backbone마다 학습된 input range가 다를 수 있다 (`0-1` vs `0-255`). 본 연구는 **모든 SR backbone에 `input_range="0-255"`를 사용**한다.

## 코드 증거

`eval_arch.py` (lines 200~237):

```python
# RFDN
sr = RFDN(in_channels=3, out_channels=3, nf=50, num_modules=4, upscale=4, input_range="0-255")

# DRCT
sr = DRCTWrapper(scale=4, pretrained_path=None, variant="base", input_range="0-255")

# HAT
sr = HATWrapper(scale=4, pretrained_path=None, variant="base", input_range="0-255")

# MAN
sr = MANWrapper(scale=4, pretrained_path=None, variant="base", input_range="0-255")
```

`input_range="0-255"` → 4 SR backbone 모두 동일.

## 동작 방식

각 wrapper(예: `sci_lab/backbones/drct_wrapper.py`) 내부:
```python
if self.input_range == "0-255":
    x = x * 255.0      # 0-1 텐서를 0-255로 변환
sr_out = self.model(x)
if self.input_range == "0-255":
    sr_out = sr_out / 255.0   # 다시 0-1로
```

즉:
- **input**: PIL/cv2에서 읽은 0-1 텐서를 입력으로 받음
- **SR 모델 내부**: 0-255 범위에서 동작
- **output**: 0-1 텐서로 변환되어 반환

→ 일관된 텐서 범위를 SR 출력에서 보장.

## 학습 시점에서도 동일

각 SR backbone의 학습/fine-tune 시에도 같은 `input_range="0-255"`가 사용됨. 이는 `iac_jetson/train_*.py` 학습 스크립트에서 확인 가능 (해당 디렉토리는 검사용으로 메인 repo에 보존).

## 평가 시 SR 출력 후처리

SR 출력(0-1 텐서)은:
- 표준 RGB로 해석되어 YOLO에 전달
- YOLO도 0-1 텐서를 받음 (Ultralytics 기본)
- 어떤 SR backbone이든 출력 형태가 동일하므로 YOLO와의 인터페이스가 같음

## 검증 포인트

- [ ] 4 SR backbone 모두 `input_range="0-255"`인가
- [ ] SR 출력이 항상 0-1 범위의 RGB 텐서인가
- [ ] YOLO 입력이 모든 SR에서 동일 형식인가
- [ ] 학습/평가 시 동일한 normalization이 적용되는가
