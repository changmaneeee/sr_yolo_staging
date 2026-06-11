# 03. 데이터셋 + 전처리 / 데이터 처리 (CHEETAH)

> 데이터 코드 검증분 (`scripts/.../SRDetPairDataset`, `data.yaml`). 결과 없음.

---

## 1. 데이터셋 개요

- **출처**: Smart Airbus ship detection (위성 영상 선박 검출).
- **클래스**: `nc=1`, `names=['ship']` (단일 클래스 = 선박/vessel).
- **SR scale**: ×4. **LR 192×192 → HR 768×768**.
- 두 폴더로 분리:
  - HR: `smart_airbus_data/` (768², 라벨 보유)
  - LR: `smart_airbus_data_lr/` (192²)
- ⚠️ `data.yaml`의 `path:`는 `/home/changmin/...`(4060/4090 경로)로 **stale**. CHEETAH 실제 경로는 학습 스크립트 상단 상수로 지정:
  - `HR_ROOT = /home/jovyan/changmin/dark_vessel_research/smart_airbus_data`
  - `LR_ROOT = /home/jovyan/changmin/dark_vessel_research/smart_airbus_data_lr`

## 2. 이미지 / 라벨 수 (실측)

| split | HR 이미지 | LR 이미지 | **라벨(비어있지 않음)** |
|---|---|---|---|
| train | 108,414 | 108,414 | 108,414 |
| val | 28,883 | 28,884 | **6,418** |

- **train**: 108,414장 (모두 라벨 보유).
- **val**: 이미지는 28,883장이지만 **선박이 있는(positive) 라벨은 6,418장** → 이것이 평가셋 **subset6418**.
- 라벨 형식: **YOLO format** (`class cx cy w h`, 모두 [0,1] 정규화 좌표).

## 3. 전처리 / 데이터로딩 (코드 검증: `SRDetPairDataset`)

### 3.1 핵심 특징
- **Full image, NO cropping**: LR 192² / HR 768² 통째 사용 (스크립트명 `_fullimg`). 4090 초기엔 64² crop이었으나 24GB GPU에서 full image로 전환.
- **Positive-only 필터링**: 라벨이 **비어있지 않은** stem만 사용. 정확히:
  ```python
  valid = label_stems(내용있음) & lr_stems & hr_stems
  ```
  → LR·HR·라벨 3자가 모두 있는 양성 이미지만 학습/검증. (배경-only 이미지 제외)
- **LR/HR 페어링**: 파일명 stem 기준 (`_find`로 .jpg/.png/.jpeg 탐색). 라벨은 HR_ROOT/labels에서.

### 3.2 정규화 / 텐서 변환 (코드)
```python
lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0   # [0,1]
hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0   # [0,1]
lr_t = torch.from_numpy(lr).permute(2,0,1).contiguous()   # HWC→CHW
hr_t = ...
```
- **RGB, [0,1] 정규화** (÷255). CHW. (SR backbone 내부에서 mean-shift 등 추가 정규화는 각 wrapper가 처리.)

### 3.3 라벨 / collate (코드)
```python
# 라벨: [class, cx, cy, w, h] (정규화, full image라 remap 불필요)
labels_t = torch.tensor([...]) or zeros((0,5))
# collate_fn: batch index 부여 → [batch_idx, class, cx, cy, w, h]
targets = cat([full(N,1, i), labels])   # YOLOv8 detection loss 형식
```
- DataLoader: `num_workers=6(train)/2(val)`, `pin_memory=True`, `persistent_workers=True`, `drop_last=True(train)`.

## 4. Train / Val 분리 (누설 방지)

- 학습(Phase 2/3)은 **`split="train"` 만** 사용 (108,414 양성).
- 검증(val_loss)은 **`split="val"`** (6,418 양성, subset6418).
- train/val 폴더가 물리적으로 분리 → **val 누설 없음**. 최종 mAP 평가도 동일 subset6418에서 수행 예정.

## 5. SR backbone 입출력 정리

| 단계 | 입력 | 출력 |
|---|---|---|
| `forward_features` | LR [B,3,192,192] [0,1] | feature [B,180,192,192] |
| `forward_reconstruct` | feature [B,180,192,192] | SR 이미지 [B,3,768,768] [0,1] |
| detector(YOLO) | SR 이미지 [B,3,768,768] | pyramid P3/P4/P5 |

## 6. 4090과 차이 (데이터 측면)

- **데이터셋 자체는 동일**(같은 이미지/라벨). 차이는 **경로**(CHEETAH `/home/jovyan/...`)와 **crop→fullimg 전환** 정도.
- 기존 4090 dataset/preprocessing 문서: `docs/01_dataset/`, `docs/02_preprocessing/` (본 문서는 CHEETAH 코드 재검증판).
