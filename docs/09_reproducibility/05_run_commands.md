# 09-05. Run Commands (재현 명령)

## Fair eval

```bash
cd /home/changmin/dark_vessel_sr_yolo
conda activate dark_vessel_mamba

# RFDN Arch4
python eval_arch.py --sr-backbone rfdn --arch 4 \
  --sr-weight weights/rfdn_arch4/model_best.pt \
  --sniper-weight weights/yolo_8s_rfdn/weights/best.pt \
  --out-json iac_runs/eval_arch_results/fair_rfdn_arch4.json

# DRCT Arch4
python eval_arch.py --sr-backbone drct --arch 4 \
  --sr-weight weights/sr_finetuned/drct/best.pt \
  --sniper-weight weights/yolo_8s_drct/weights/best.pt

# HAT Arch4
python eval_arch.py --sr-backbone hat --arch 4 \
  --sr-weight weights/sr_finetuned/hat/best.pt \
  --sniper-weight weights/yolo_8s_hat/best.pt

# MAN Arch4
python eval_arch.py --sr-backbone man --arch 4 \
  --sr-weight weights/sr_finetuned/man/best.pt \
  --sniper-weight weights/yolo_8s_man/best.pt
```

## Properbase Old Pipeline

```bash
# 1차 (잘못된 rfdn base) — 사용 금지
# bash iac_runs/run_drct_old_sniper_pipeline.sh

# 2차 (proper base) — 실제 사용
bash iac_runs/run_old_pipeline_properbase.sh   # DRCT 시작
bash iac_runs/run_old_pipeline_properbase_hatman.sh   # HAT + MAN
```

## NMS Sweep v2

```bash
bash iac_runs/run_nms_sweep_2stage.sh
# 결과: iac_runs/nms_sweep_2stage/{drct,hat,man,rfdn}/
```

## Sensitivity Ablation

```bash
# v2 sweep 완료 후
bash iac_runs/run_nms_sensitivity.sh
# 결과: iac_runs/nms_sensitivity/{drct,hat,man,rfdn}/
```

## Robust Pipeline (자동 실행)

```bash
bash iac_runs/run_robust_pipeline.sh
# v2 완료 폴링 → sensitivity 자동 시작 → 종합 분석
```

## Arch4 단일 config 평가

```bash
python iac_scripts/arch4_eval_ultralytics.py \
  --arch4_config configs/experiment/arch4_drct_old_pipeline_config.yaml \
  --hr_data_yaml iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr_data.yaml \
  --lr_data_yaml iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr_data.yaml \
  --eval_space hr --max_images 0 --device cuda --batch 1 \
  --out_json result.json
```

## Arch5 평가 (서버 checkpoint 사용)

```bash
python eval_arch.py --sr-backbone drct --arch 5 \
  --arch5-checkpoint /path/to/server/arch5b_drct_phase3_best.pt
```

## 검증 포인트

- [ ] 모든 명령이 conda env에서 실행
- [ ] 경로가 정확하고 weight 파일 존재
- [ ] 결과 JSON이 정상 생성됨
