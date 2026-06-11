# 09-04. Data Paths

## 메인 dataset

| Dataset | 경로 | 비고 |
|:--|:--|:--|
| HR | `/home/changmin/smart_airbus_data/` | 768×768, 137,297장 |
| LR | `/home/changmin/smart_airbus_data_lr/` | 192×192, 137,298장 (LR val 1장 초과) |

## subset6418 (메인 평가)

```
/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/
├── subset6418_hr/
│   ├── images/val/    (6418 jpg)
│   └── labels/val/    (6418 txt)
├── subset6418_lr/
│   ├── images/val/    (6418 jpg)
│   └── labels/val/    (6418 txt)
├── subset6418_hr_data.yaml
└── subset6418_lr_data.yaml
```

## fixed_protocol.yaml에 명시

```yaml
dataset:
  lr_images: "/home/changmin/smart_airbus_data_lr/images/val"
  hr_images: "/home/changmin/smart_airbus_data/images/val"
  hr_labels: "/home/changmin/smart_airbus_data/labels/val"
  lr_labels: "/home/changmin/smart_airbus_data_lr/labels/val"
  filter: "labeled_only"
  expected_count: 6418
```

## Arch4 sniper crop datasets (Old Pipeline용)

```
/home/changmin/dark_vessel_sr_yolo/data/
├── arch4_sniper_crops_rfdn_old/
├── arch4_sniper_crops_drct_old/
├── arch4_sniper_crops_hat_old/
├── arch4_sniper_crops_man_old/
└── (각각 data.yaml + images + labels)
```

자세히: [docs/05_training/03_sniper_old_pipeline.md](../05_training/03_sniper_old_pipeline.md)

## 검증 명령

```bash
# subset6418 path 검증
[ -d /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr ] && echo "HR OK"
[ -d /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr ] && echo "LR OK"

# 이미지 개수
ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr/images/val/ | wc -l   # 6418
ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr/images/val/ | wc -l   # 6418
```

## 검증 포인트

- [ ] 모든 dataset 경로 존재
- [ ] subset6418 HR/LR 6418장 일치
- [ ] fixed_protocol.yaml에 경로 명시
- [ ] Arch4 crop dataset이 각 SR별로 분리
