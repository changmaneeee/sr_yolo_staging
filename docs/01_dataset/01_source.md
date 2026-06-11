# 01-01. Dataset Source

## 출처

본 연구에서 사용한 데이터셋은 **Airbus Ship Detection Challenge** 기반의 dark vessel 데이터셋이다.

- 원본 출처: Airbus Ship Detection Challenge (Kaggle 등)
- 라벨링: ship instance에 대한 bbox annotation
- 클래스: 1개 (`ship`)
- 도메인: 항공/위성 이미지 (RGB)

## 로컬 경로

| 데이터셋 | 절대 경로 |
|:--|:--|
| HR (원본) | `/home/changmin/smart_airbus_data/` |
| LR (4× downscale) | `/home/changmin/smart_airbus_data_lr/` |
| labeled_only subset (6418) | `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_{hr,lr}/` |

## data.yaml (Ultralytics 호환)

### HR data.yaml
경로: `/home/changmin/smart_airbus_data/data.yaml`
```yaml
path: /home/changmin/smart_airbus_data
train: /home/changmin/smart_airbus_data/images/train
val: /home/changmin/smart_airbus_data/images/val

nc: 1
names: ['ship']
```

### LR data.yaml
경로: `/home/changmin/smart_airbus_data_lr/data.yaml`
```yaml
path: /home/changmin/smart_airbus_data_lr
train: /home/changmin/smart_airbus_data_lr/images/train
val: /home/changmin/smart_airbus_data_lr/images/val

nc: 1
names: ['ship']
```

### Subset6418 HR data.yaml
경로: `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr_data.yaml`
```yaml
path: /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr
train: images/val
val: images/val
names:
  0: ship
nc: 1
```

### Subset6418 LR data.yaml
경로: `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr_data.yaml`
```yaml
path: /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr
train: images/val
val: images/val
names:
  0: ship
nc: 1
```

**주의**: subset6418 yaml에서 `train: images/val`로 설정한 것은 의도된 것이다 (subset이 val 분할만 갖고 있고, train 슬롯에 어떤 값이든 필요하기 때문). 실제 학습에는 사용되지 않으며, 평가 시 `val` 슬롯만 참조된다.

## 검증 포인트

검사자는 다음을 확인:
- [ ] HR과 LR이 다른 폴더에 있는가 (혼동 위험 차단)
- [ ] 두 dataset의 nc=1, names='ship' 일치
- [ ] subset6418의 path가 별도로 분리되어 있는가
- [ ] 모든 경로가 절대 경로로 명시되어 있는가
