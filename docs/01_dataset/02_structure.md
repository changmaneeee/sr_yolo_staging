# 01-02. Dataset Structure

## 디렉토리 구조

### HR Dataset
```
/home/changmin/smart_airbus_data/
├── data.yaml
├── images/
│   ├── train/         (108,414 jpg, 768×768 RGB)
│   └── val/           (28,883 jpg, 768×768 RGB)
└── labels/
    ├── train/         (108,414 txt)
    └── val/           (6,418 txt — labeled only, 22,465 empty/missing)
```

### LR Dataset
```
/home/changmin/smart_airbus_data_lr/
├── data.yaml
├── images/
│   ├── train/         (108,414 jpg, 192×192 RGB)
│   └── val/           (28,884 jpg, 192×192 RGB)
└── labels/
    ├── train/         (... txt, normalized 0-1)
    └── val/           (... txt)
```

### Subset 6418 (labeled_only)
```
/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/
├── subset6418_hr/
│   ├── images/val/    (6,418 jpg, 768×768)
│   └── labels/val/    (6,418 txt)
├── subset6418_lr/
│   ├── images/val/    (6,418 jpg, 192×192)
│   └── labels/val/    (6,418 txt)
├── subset6418_hr_data.yaml
└── subset6418_lr_data.yaml
```

## 이미지 사양

| 항목 | HR | LR | 비율 |
|:--|:--|:--|:--|
| 해상도 | 768×768 | 192×192 | 4× |
| 색공간 | RGB | RGB | 동일 |
| 포맷 | JPEG (.jpg) | JPEG (.jpg) | 동일 |
| 파일명 stem | 9자리 hex (예: `00003e153`) | **동일 stem 사용** | 1:1 매칭 |

### 검증 명령
```bash
# HR/LR 이미지 개수 일치 확인
HR_TRAIN=$(ls /home/changmin/smart_airbus_data/images/train/ | wc -l)         # 108414
LR_TRAIN=$(ls /home/changmin/smart_airbus_data_lr/images/train/ | wc -l)      # 108414
HR_VAL=$(ls /home/changmin/smart_airbus_data/images/val/ | wc -l)             # 28883
LR_VAL=$(ls /home/changmin/smart_airbus_data_lr/images/val/ | wc -l)          # 28884

# subset6418
SUBSET_HR=$(ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr/images/val/ | wc -l)   # 6418
SUBSET_LR=$(ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr/images/val/ | wc -l)   # 6418

# 같은 stem 확인
diff \
  <(ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr/images/val/ | sort) \
  <(ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr/images/val/ | sort)
# 출력 없으면 100% 같은 stem
```

## 확인된 통계

| 데이터셋 | 이미지 개수 | Label 개수 |
|:--|--:|--:|
| HR train | 108,414 | 108,414 |
| HR val (전체) | 28,883 | 6,418 (있는 것만) |
| LR train | 108,414 | (학습 시 사용 안 함, HR label과 연동) |
| LR val (전체) | 28,884 | (평가 시 사용 안 함) |
| **Subset6418 HR val** | **6,418** | **6,418** ✅ |
| **Subset6418 LR val** | **6,418** | **6,418** ✅ |

### 주의: HR val 28883 vs LR val 28884
- LR val에 1장 더 있는 것은 데이터 생성 과정의 차이(미세 차이)이며, 평가는 항상 **subset6418로 수행**하므로 실제 측정에는 영향 없음.
- subset6418은 label이 있는 것만 선별되었기 때문에 HR/LR 모두 정확히 6418장이고 1:1 매칭됨.

## 검증 포인트

- [ ] HR train/val 개수 = label train/val 개수 (train만)
- [ ] HR/LR stem 매칭 100% (subset6418)
- [ ] 해상도가 정확히 768×768 (HR), 192×192 (LR) — 4× 비율
- [ ] subset 6418 모두 label 파일 존재 (non-empty)
- [ ] LR val 28884의 1장 초과는 subset6418 사용으로 무력화됨
