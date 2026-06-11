# 06-03. arch4_eval_ultralytics.py 동작

## 위치

`iac_scripts/arch4_eval_ultralytics.py` (메인 repo)

## 역할

**Arch4 전용** 평가 스크립트. NMS sweep, Properbase Phase D/E mAP eval 등에서 사용.

## eval_arch.py와의 관계

| 항목 | eval_arch.py | arch4_eval_ultralytics.py |
|:--|:--|:--|
| 지원 Arch | 0/2/4/5 | 4만 |
| Config 로드 | fixed_protocol.yaml | YAML config 직접 |
| Sniper weight | --sniper-weight | YAML model.yolo.weights_hr |
| Subset 강제 | filter: labeled_only | --labeled_only or --hr_data_yaml |

**평가 metric (ap_per_class)는 두 스크립트가 동일.**

## 사용 예

```bash
python iac_scripts/arch4_eval_ultralytics.py \
  --arch4_config configs/experiment/arch4_drct_old_pipeline_config.yaml \
  --hr_data_yaml subset6418_hr_data.yaml \
  --lr_data_yaml subset6418_lr_data.yaml \
  --eval_space hr \
  --max_images 0 \         # 전체 (subset6418의 6418장)
  --device cuda \
  --batch 1 \
  --out_json result.json
```

## YAML config 구조

```yaml
data:
  upscale_factor: 4
model:
  sr:
    type: drct
    weights: weights/sr_finetuned/drct/best.pt
    rfdn: {nf: 50, num_modules: 4}
  yolo:
    weights_lr: weights/yolo_lr_improved/.../best.pt
    weights_hr: weights/yolo_sniper_hardneg/.../best.pt
    classes: 1
  arch4:
    pass1_conf: 0.0075
    high_conf: 0.45
    ...
    # 23개 NMS 파라미터
```

## 결과 JSON 구조

```json
{
  "meta": {
    "time": "...",
    "arch": "arch4_adaptive",
    "num_images": 6418,
    "avg_ms_per_image": 89.3,
    ...
  },
  "runs": [
    {
      "tag": "ARCH4",
      "results_dict": {
        "metrics/precision(B)": 0.7807,
        "metrics/recall(B)": 0.7325,
        "metrics/mAP50(B)": 0.7918,
        "metrics/mAP50-95(B)": 0.6087,
        "direct/tp50": 4823,
        "direct/fp50": 1543,
        "direct/fn50": 1595,
        ...
      }
    }
  ]
}
```

## NMS sweep / Sensitivity에서의 활용

```bash
# Sweep 한 combo
python iac_scripts/arch4_eval_ultralytics.py \
  --arch4_config sweep_dir/p10.0175_roi2.0_s0.3.yaml \
  --hr_data_yaml subset6418_hr_data.yaml \
  --lr_data_yaml subset6418_lr_data.yaml \
  --eval_space hr --max_images 0 --device cuda --batch 1 \
  --out_json sweep_dir/p10.0175_roi2.0_s0.3.json
```

## 검증 포인트

- [ ] 같은 ap_per_class metric 사용 (eval_arch.py와 동일)
- [ ] 6418장 평가 (subset6418 yaml + max_images 0)
- [ ] 23개 NMS 파라미터 모두 YAML에서 통제
- [ ] 결과 JSON 형식이 일관됨
