# 03-06. Architecture Comparison Table

4개 Arch의 구조, 학습, 속도, 정확도 비교.

## 구조 비교

| 항목 | Arch0 | Arch2 | Arch4 | Arch5b |
|:--|:--|:--|:--|:--|
| SR 적용 | 전체 이미지 | 전체 + Gate blend | ROI만 (부분) | Feature 레벨 |
| YOLO 개수 | 1 (HR detector) | 1 (HR detector) | **2** (Scout + Sniper) | 1 (fused feature) |
| 추가 모듈 | 없음 | Gate network | ROI-aware NMS | Attention fusion |
| 학습 단계 | 컴포넌트별 별도 | Gate + YOLO 별도 | Scout + Sniper 별도 | Phase 2 + Phase 3 |

## 데이터 흐름 비교

| Arch | LR | SR | HR | Detection |
|:--|:--|:--|:--|:--|
| 0 | 입력 | 전체 적용 | YOLO 입력 | output |
| 2 | 입력 | 일부 영역 강조 | gate*SR + (1-gate)*bilinear | output |
| 4 | Scout 입력 | ROI만 | crop 단위 | Scout + Sniper merge |
| 5b | 양쪽 feature | feature 추출 후 fusion | (사용 안 함, feature만) | fused feature → head |

## 학습 비용

| Arch | 학습 컴포넌트 | 학습 시간 (예) |
|:--|:--|:--|
| 0 | YOLO (HR detector) | ~수시간 |
| 2 | YOLO + Gate (label-based BCE 250K) | ~12h |
| 4 | Scout YOLO + Sniper YOLO (별도 + Old Pipeline B/C/D/E) | ~8h (per SR) |
| 5b | Phase 2 (fusion warmup) + Phase 3 (joint) | ~수일 (4090) |

## Inference 속도 (4060, FP16, 추정)

| Arch | Latency |
|:--|:--|
| 0 | ~15ms |
| 2 | ~17ms (Gate inference 추가) |
| 4 | ~50ms (Scout + Sniper, ROI 수에 의존) |
| 5b | ~12ms (single forward, TRT 최적화) |

Arch5b가 가장 빠른 이유: 한 번의 inference (Arch4처럼 두 번이 아님), feature 레벨 fusion.

## 정확도 요약 (subset6418 mAP@50)

| Arch | RFDN | DRCT | HAT | MAN |
|:--|--:|--:|--:|--:|
| Arch0 | 0.x | +2.48pp | (HAT/MAN baseline 결과) | |
| Arch2 | 0.x | +6.61pp | | |
| Arch4 properbase | 0.8007 | 0.7973 | 0.7905 | 0.7918 |
| Arch4 v2 (NMS-opt) | 0.7981 | 0.7990 | **0.8003** | 0.7940 |
| Arch5b | 0.9102 (SOTA) | (Phase 3 진행) | | |

자세한 표: [docs/08_results/01_main_results_table.md](../08_results/01_main_results_table.md)

## 공정 비교 보장 항목

다음 항목들이 4 Arch와 4 SR backbone에 걸쳐 동일하다:
1. **데이터셋**: subset6418 (6,418장)
2. **이미지 로딩**: BGR→RGB→0-1 변환 (cv2)
3. **Label format**: YOLO 표준 (0~1 정규화)
4. **Metric**: ultralytics `ap_per_class` (mAP@50, mAP@50-95)
5. **NMS**: fixed_protocol.yaml의 conf=0.25, IoU=0.5 (Arch0/2는 그대로, Arch4는 추가 11개 파라미터)
6. **input_range**: 모든 SR backbone "0-255"

## 검증 포인트

- [ ] 4 Arch가 같은 평가 기준 (subset6418, 동일 metric)에서 비교되는가
- [ ] 학습 단계가 명확히 구분되는가
- [ ] 각 Arch의 weight 종속성이 명확한가
- [ ] Inference 속도가 같은 환경에서 측정되는가
