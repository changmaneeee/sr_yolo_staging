# 09-01. Environment

## 시스템

- OS: Linux (WSL2 Ubuntu) — 평가용 (4060)
- OS: Linux Ubuntu — 학습용 (4090 서버)
- GPU: NVIDIA RTX 4060 (8GB) — 평가
- GPU: NVIDIA RTX 4090 (24GB) — 학습 (Arch5 등)

## Python

- Python 3.12

## Conda Environment

```bash
conda activate dark_vessel_mamba
```

## 핵심 라이브러리

```
torch==2.9.1+cu128
torchvision
ultralytics (YOLOv8)
opencv-python
PIL
numpy
yaml
hashlib
```

## 의존성 (간략)

```
# environment.yml
name: dark_vessel_mamba
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - pytorch
  - torchvision
  - cuda-toolkit=12.8
  - pip:
    - ultralytics
    - opencv-python
    - timm
```

자세한 패키지 목록: `environment/requirements.txt`, `environment/environment.yml`

## CUDA

- CUDA 12.8
- Driver 591.86

## 검증 명령

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.9.1+cu128 True

python -c "from ultralytics import YOLO; print('YOLO OK')"
```

## 검증 포인트

- [ ] Python 3.12 사용
- [ ] PyTorch 2.9.1+cu128
- [ ] Ultralytics 설치됨
- [ ] CUDA 12.8
