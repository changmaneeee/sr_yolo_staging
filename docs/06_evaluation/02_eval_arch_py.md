# 06-02. eval_arch.py 동작

## 위치

메인 repo: `eval_arch.py`

## 역할

**모든 SR 백본 × 모든 Architecture(0/2/4/5) 평가의 공통 entry point**.

## 사용 예

```bash
# RFDN Arch4
python eval_arch.py --sr-backbone rfdn --arch 4 \
    --sr-weight weights/rfdn_arch4/model_best.pt \
    --sniper-weight weights/yolo_8s_rfdn/weights/best.pt

# DRCT Arch0
python eval_arch.py --sr-backbone drct --arch 0 \
    --sr-weight weights/sr_finetuned/drct/best.pt

# RFDN Arch5b
python eval_arch.py --sr-backbone rfdn --arch 5 \
    --arch5-checkpoint iac_lab/runs/.../phase3_best.pt
```

## 인자

```
--sr-backbone {rfdn,drct,hat,man}    SR backbone type
--arch {0,2,4,5}                     Architecture number
--sr-weight                          SR model weight path
--sniper-weight                      Sniper YOLO weight (Arch4)
--gate-weight                        Gate weight (Arch2)
--detector-weight                    YOLO detector weight (Arch0/2/5)
--arch5-checkpoint                   Arch5 full checkpoint path
--protocol                           Fixed protocol config (default: configs/fixed_protocol.yaml)
--out-json                           Output JSON path
```

## 동작 흐름

```
1. fixed_protocol.yaml 로드
2. Pre-flight check (scout MD5, dataset paths, etc.)
3. SR backbone 로드 (각자 wrapper)
4. YOLO/Gate weights 로드
5. Arch 별 모델 빌드:
   - Arch0Sequential
   - Arch2SoftGate
   - Arch4RoiAwareNMS
   - Arch5BFusion
6. 6418장 evaluation
7. ultralytics ap_per_class로 mAP 계산
8. 결과를 JSON으로 저장
```

## 6418장 평가 logic

```python
# eval_arch.py line 467~472
ds_cfg = protocol["dataset"]
lr_images_dir = Path(ds_cfg["lr_images"])
hr_labels_dir = Path(ds_cfg["hr_labels"])

# labeled_only filter
labeled_stems = {f.stem for f in Path(hr_labels_dir).glob("*.txt") if f.stat().st_size > 0}
img_paths = sorted([p for p in list_images(lr_images_dir) if p.stem in labeled_stems])
# 결과: 정확히 6418장
```

## Inference

Arch별 inference 흐름은 [docs/03_architectures/](../03_architectures/) 참조.

각 이미지에 대해:
```python
with torch.no_grad():
    if arch == 4:
        out = model.forward(lr_tensor, debug=False)
        dets = out["detections"]
    elif arch == 0:
        out = model.forward(lr_tensor)
        dets = out[1] if isinstance(out, tuple) else out
    elif arch == 2:
        out = model.forward(lr_tensor)
        dets = out["detections"]
    elif arch == 5:
        result = model.inference(lr_tensor, conf_threshold=0.25, iou_threshold=0.5)
        dets = result["detections"]
```

## Metric 계산

```python
# eval_arch.py line 463-490
from ultralytics.utils.metrics import ap_per_class

correct, conf, pred_cls, target_cls = [torch.cat(x, 0).numpy() for x in zip(*stats)]
ap_results = ap_per_class(correct, conf, pred_cls, target_cls, plot=False)

# mAP@50, mAP@50-95, precision, recall, F1
```

→ **모든 Arch가 같은 ap_per_class를 통과**.

## 검증 포인트

- [ ] fixed_protocol.yaml이 강제 로드되는가
- [ ] Pre-flight check가 실행되는가
- [ ] 정확히 6418장이 평가되는가
- [ ] 모든 Arch가 같은 ap_per_class 사용
- [ ] CLI override가 차단되는가 (CLI는 weight만 지정)
