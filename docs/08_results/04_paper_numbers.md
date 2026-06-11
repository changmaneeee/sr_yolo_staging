# 08-04. Paper에 들어갈 정확한 수치

Paper 작성 시 정확하게 인용할 수치 모음. 모두 subset6418 (6,418장), mAP@50 기준.

## Section IV-A: Fair Comparison

```
RFDN  : 0.7731
DRCT  : 0.7806  ← max
HAT   : 0.7733
MAN   : 0.7720
Range : 0.86pp
```

## Section IV-B: Properbase (Old Pipeline)

```
RFDN  : 0.8007  ← max
DRCT  : 0.7973
HAT   : 0.7905
MAN   : 0.7918
Range : 1.02pp
```

## Section IV-C: NMS Sweep v2 (Best per SR)

Best configs:
- RFDN: pass1_conf=0.0125, roi_expansion=1.75, soft_nms_sigma=0.3
- DRCT: pass1_conf=0.0175, roi_expansion=2.0, soft_nms_sigma=0.3
- HAT:  pass1_conf=0.0175, roi_expansion=1.5, soft_nms_sigma=0.3
- MAN:  pass1_conf=0.01, roi_expansion=1.75, soft_nms_sigma=0.3

Best mAP@50:
```
RFDN  : 0.7981
DRCT  : 0.7990
HAT   : 0.8003  ← max
MAN   : 0.7940
Range : 0.63pp
```

## Section IV-D: Sensitivity Range

```
RFDN  : ±0.57pp (0.7949 ~ 0.8006)
DRCT  : ±0.41pp (0.7963 ~ 0.8004)
HAT   : ±0.49pp (0.7980 ~ 0.8029)
MAN   : ±0.60pp (0.7902 ~ 0.7961)
Max   : 0.60pp
```

## Section V: Arch5b (SOTA, 참조)

```
RFDN Arch5b Phase 3:    0.9102 mAP@50, 12ms TRT FP16 latency
Small vessel recall:    +84% (Phase 2 → Phase 3)
```

## 메인 finding 한 줄

> "Across three NMS protocols and sensitivity analysis, the mAP@50 range across 4 SR backbones (RFDN, DRCT, HAT, MAN) stays within 1pp on Arch4. HAT achieves the highest mAP@50 of 0.8003 under SR-optimized NMS, but the result is robust to NMS choice."

## 검증 포인트

- [ ] 모든 수치가 4자리 (Changmin profile 따름)
- [ ] 각 수치가 JSON 파일에 존재
- [ ] Best config의 hyperparameter 정확
- [ ] Sensitivity Range가 (Max-Min) * 100 형식
