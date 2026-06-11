# 00. SR-YOLO 프로젝트 전체 개요

## 한 줄 요약

> 저해상도(192×192) 위성/항공 이미지에서 작은 선박(dark vessel)을 정확히 검출하기 위해, **Super-Resolution (SR) → YOLO 검출**의 다양한 결합 방식(Arch0~5)을 비교하고, 그 중 어떤 SR backbone(RFDN/DRCT/HAT/MAN)을 쓰는 게 가장 효과적인지를 공정하게 평가하는 연구.

---

## 1. 연구 배경

작은 선박은 LR 이미지에서 픽셀 몇 개 수준으로만 보이기 때문에 일반 YOLO만으로는 검출이 매우 어렵다.
SR(Super-Resolution)으로 해상도를 4배 올린 후 YOLO를 적용하면 검출 성능이 크게 향상되는데,
- 어떤 방식으로 SR과 YOLO를 결합하느냐 (Arch0, 2, 4, 5)
- 어떤 SR backbone을 쓰느냐 (RFDN, DRCT, HAT, MAN)

가 성능과 속도에 큰 영향을 미친다.

본 연구는 두 차원을 모두 공정하게 비교하여 dark vessel 검출의 best practice를 정립하려는 것이 목적.

---

## 2. 데이터셋

- **HR (high-resolution)**: 768×768 RGB, 총 137,297장
- **LR (low-resolution)**: 192×192 RGB, HR에서 4× downscale로 생성
- **클래스**: 1개 (`ship`)
- **Train**: 108,414장 (HR/LR 동일 개수)
- **Val (전체)**: 28,883장 (HR), 28,884장 (LR)
- **Val (labeled_only subset6418)**: **6,418장** — label 파일이 있는 것만 (메인 평가 기준)
- 데이터 위치: `/home/changmin/smart_airbus_data/`, `/home/changmin/smart_airbus_data_lr/`
- subset 위치: `/home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_{hr,lr}/`

자세한 내용: [docs/01_dataset/](01_dataset/)

---

## 3. 아키텍처 5종 비교

본 연구에서 비교한 SR-YOLO 결합 방식:

| 아키텍처 | 구조 한 줄 설명 |
|:--|:--|
| **Arch0** (Sequential) | LR → SR → HR이미지 → YOLO 한 번 |
| **Arch2** (Soft-Gate) | LR → (Gate가 결정한 비율로) SR + bilinear blend → YOLO |
| **Arch4** (Dual-YOLO ROI-aware) | LR → Scout YOLO → ROI만 SR → Sniper YOLO (crop별 정밀) |
| **Arch5b** (Feature Fusion) | LR feature + SR feature 융합 → YOLO head |

자세한 구현: [docs/03_architectures/](03_architectures/)

---

## 4. SR Backbone 4종

| Backbone | 한 줄 설명 | 파라미터 |
|:--|:--|--:|
| **RFDN** | Residual Feature Distillation Network, 경량 | ~0.5M |
| **DRCT** | Dilated Residual Convolutional Transformer, Swin 기반 | ~14M |
| **HAT** | Hybrid Attention Transformer, swin window | ~21M |
| **MAN** | Multi-scale Attention Network, 중간 복잡도 | ~9M |

자세한 명세: [docs/03_architectures/05_sr_backbones.md](03_architectures/05_sr_backbones.md)

---

## 5. 핵심 실험 흐름

### Phase 1. Fair Evaluation (Section IV-A)
모든 SR에 동일한 NMS 설정(`fixed_protocol.yaml`)을 적용하여 첫 비교 측정. From-scratch Sniper 사용.

### Phase 2. Properbase Old Pipeline (Section IV-B)
Sniper를 crop fine-tune (Phase B) + hardneg mining (Phase C) + interpolation/bonus (Phase D/E)으로 강화. 각 SR 전용 base detector 사용.

### Phase 3. NMS Sweep (Section IV-C)
각 SR마다 최적 NMS parameter를 찾는 grid search. 2-stage (200장 quick → 6418장 full).

### Phase 4. Sensitivity Ablation (Section IV-D)
NMS sweep의 grid에 없던 6개 파라미터를 ±변동시켜 robustness 확인.

자세한 절차: [docs/07_experiments/](07_experiments/)

---

## 6. 최종 결과 (간단)

| SR | Fair NMS | Properbase | v2 (NMS-opt) | Sensitivity range |
|:--|--:|--:|--:|:--|
| RFDN | 0.7731 | 0.8007 | 0.7981 | ±0.57pp |
| DRCT | 0.7806 | 0.7973 | 0.7990 | ±0.41pp |
| HAT | 0.7733 | 0.7905 | **0.8003** 🥇 | ±0.49pp |
| MAN | 0.7720 | 0.7918 | 0.7940 | ±0.60pp |

- 모든 measurement에서 4 SR의 range ≤ 1pp
- Ranking은 protocol에 따라 다름 (Fair: DRCT 1위 / Properbase: RFDN 1위 / v2: HAT 1위)
- Sensitivity 분석: ±0.6pp 이하 → robust

전체 표: [docs/08_results/](08_results/)

---

## 7. 사고 기록 (중요)

1. **rfdn base 사고 (2026-06-07~09)**: DRCT/HAT/MAN의 Sniper fine-tune을 시작할 때 base로 RFDN 전용 detector를 잘못 사용 → 1차 Old Pipeline 결과 무효 → 2차 properbase로 재실행.

2. **NMS sweep v1 RFDN weight 오류 (2026-06-10)**: RFDN의 SR weight를 `weights/rfdn/`로 잘못 지정 (다른 모든 실험은 `weights/rfdn_arch4/`). 또한 Sniper도 baseline을 사용. → 수정 후 재측정.

3. **NMS sweep v1 → v2 누락 파라미터 (2026-06-10)**: Sweep config에 `final_conf`, `roi_small_thresh`, `roi_large_thresh`, `large_roi_score_thresh`, `sniper_replace_margin`, `sniper_max_det_per_crop` 6개가 properbase config 대비 누락 → v2에서 모두 포함 + sweep 재실행.

자세한 사고 기록: [docs/10_integrity_audit/04_changes_history.md](10_integrity_audit/04_changes_history.md)

---

## 8. 문서 진입점 (검사자용)

| 검증 주제 | 시작 문서 |
|:--|:--|
| 데이터셋 출처/구조 | [docs/01_dataset/](01_dataset/) |
| 전처리 일관성 | [docs/02_preprocessing/](02_preprocessing/) |
| 아키텍처 구현 | [docs/03_architectures/](03_architectures/) |
| 컴포넌트 코드 | [docs/04_components/](04_components/) |
| 학습 절차 | [docs/05_training/](05_training/) |
| 평가 프로토콜 | [docs/06_evaluation/](06_evaluation/) |
| 모든 실험 절차 | [docs/07_experiments/](07_experiments/) |
| 최종 결과 표 | [docs/08_results/](08_results/) |
| 재현성 (env, weight 경로) | [docs/09_reproducibility/](09_reproducibility/) |
| 검사 체크리스트 | [docs/10_integrity_audit/](10_integrity_audit/) |
