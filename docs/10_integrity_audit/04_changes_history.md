# 10-04. Changes History (모든 fix/correction 기록)

본 연구 진행 중 발견된 문제와 수정 기록.

---

## Fix #1: rfdn base 사고 (2026-06-07~09)

### 문제
DRCT/HAT/MAN의 Sniper fine-tune (Phase B)에서 base detector로 `weights/yolo_8s_rfdn/best.pt`를 모든 SR에 공통 사용.

### 영향
- 각 SR이 RFDN 이미지에 적응된 detector를 base로 시작
- 결과의 신뢰성 손상

### 1차 결과 (잘못, rfdn base)
| SR | mAP@50 |
|:--|--:|
| DRCT | 0.7927 |
| HAT | 0.7914 |
| MAN | 0.7955 |

### 발견 과정
2026-06-09 결과 검토 중 detector의 base 경로 확인 → 모두 yolo_8s_rfdn 발견.

### 수정
- 스크립트 `iac_runs/run_old_pipeline_properbase.sh` 작성
- 각 SR 전용 detector를 BASE_SNIPER로 사용:
  - DRCT: `weights/yolo_8s_drct/weights/best.pt`
  - HAT: `weights/yolo_8s_hat/best.pt`
  - MAN: `weights/yolo_8s_man/best.pt`
- Phase A (crop dump) 재사용, Phase B~E만 재실행 (~7~8h per SR)

### 2차 결과 (properbase)
| SR | mAP@50 |
|:--|--:|
| DRCT | 0.7973 |
| HAT | 0.7905 |
| MAN | 0.7918 |

### 산물
- 1차 결과는 폐기
- 2차 결과를 paper Section IV-B에 사용

---

## Fix #2: NMS sweep v1 RFDN weight 사고 (2026-06-10)

### 문제 1
NMS sweep 스크립트에서 RFDN의 SR weight를 `weights/rfdn/model_best.pt`로 잘못 지정.
- 정상: `weights/rfdn_arch4/model_best.pt` (Mar 14, MD5 539f72b2)
- 잘못: `weights/rfdn/model_best.pt` (Jan 12, MD5 0087ca54) — 다른 weight!

### 문제 2
NMS sweep에서 RFDN Sniper를 `weights/yolohr/8s/best.pt` (fine-tune 안 된 baseline)로 사용.
- Properbase에서는 `weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/best.pt` 사용 (hardneg fine-tune)

### 결과
RFDN NMS sweep v1 best = **0.7450** (다른 SR은 0.79대인데 비정상 낮음)

### 발견 과정
Changmin이 "RFDN은 똑같은 weight를 쓰는데 왜 sweep 결과가 다르지?" 질문 → weight 경로 비교 → 잘못된 weight 발견.

### 수정
- `iac_runs/run_nms_sweep_2stage.sh`의 SR_WEIGHT[rfdn], SNIPER_WEIGHT[rfdn] 정정
- 잘못된 결과는 `nms_sweep_2stage_v1_missing_params/rfdn_WRONG_baseline_sniper/`로 백업
- RFDN sweep 단독 재실행

### 수정 후 결과
RFDN sweep v1 (정확한 weight) = **0.7949**

---

## Fix #3: NMS sweep v1 → v2 누락 6개 파라미터 (2026-06-10)

### 문제
NMS sweep v1의 config가 properbase config 대비 6개 파라미터 누락:

| 파라미터 | properbase | v1 |
|:--|--:|--:|
| final_conf | 0.25 | 0.3 (잘못) |
| roi_small_thresh | 32.0 | (없음) |
| roi_large_thresh | 96.0 | (없음) |
| large_roi_score_thresh | 0.5 | (없음) |
| sniper_replace_margin | 0.1 | (없음) |
| sniper_max_det_per_crop | 3 | (없음) |

### 영향
같은 weights + 같은 핵심 NMS인데도 v1 RFDN 0.7949 vs properbase 0.8007 → -0.58pp 차이.

### 발견 과정
v1 결과와 properbase 결과 비교 중 같은 weights에서 결과 차이를 의심 → config 차이 발견.

### 수정
- `iac_runs/run_nms_sweep_2stage.sh`의 `make_config` 함수에 6개 파라미터 추가
- v1 결과 백업 (`nms_sweep_2stage_v1_missing_params/`)
- v2 sweep 전체 재실행 (4 SR × 50 combos)

### v2 결과
| SR | v2 best | v1 best | 차이 |
|:--|--:|--:|--:|
| RFDN | 0.7981 | 0.7949 | +0.32pp |
| DRCT | 0.7990 | 0.7963 | +0.27pp |
| HAT | 0.8003 | 0.7980 | +0.23pp |
| MAN | 0.7940 | 0.7902 | +0.38pp |

→ 모든 SR이 향상. v2가 paper에 사용될 main result.

---

## Fix #4: Mamba loader 사고 (검증 #45 이전)

### 문제
Mamba SR weight를 RFDN loader로 로드하면 F1~0.0003 (사실상 실패).

### 원인
SR backbone마다 architecture가 다르므로 wrapper도 달라야 함.

### 수정
`eval_arch.py`의 `_load_sr_model()` 함수가 backbone 이름으로 분기하여 강제:
```python
if sr_backbone == "rfdn":
    from src.models.sr_models.rfdn import RFDN
    sr = RFDN(...)
elif sr_backbone == "drct":
    from sci_lab.backbones.drct_wrapper import DRCTWrapper
    sr = DRCTWrapper(...)
# ...
```

### 결과
잘못된 wrapper 호출 시 에러 발생 (강제 검증).

---

## Fix #5: 평가 인프라 정비 (Task #52~56)

### 작업
- `configs/fixed_protocol.yaml` 생성 (Task #52)
- `eval_arch.py` 공통 평가 스크립트 작성 (Task #53)
- Pre-flight check 함수 추가 (Task #54)
- Arch4 클래스 통일 (Task #55)
- 중복 함수 제거 (Task #56)

### 결과
- 모든 SR/Arch가 같은 entry point (`eval_arch.py`) 사용
- fixed_protocol이 hyperparameter 강제
- Pre-flight check가 weight/dataset 검증

---

## 모든 사고의 공통 패턴

1. **사고 발생**: 코드/config의 미세한 차이
2. **발견 트리거**: 결과가 예상과 다름 (검토 중 의심)
3. **분석**: weight, config, code 비교
4. **수정**: 스크립트/config 정정 + 재실행
5. **기록**: Notion + memory + 본 문서

→ 이 패턴이 본 연구의 견고함을 보여줌.
