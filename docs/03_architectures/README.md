# 03. Architectures

SR과 YOLO를 결합하는 5가지 아키텍처(Arch0/2/4/5)와 4개 SR backbone의 설계/구현을 다룬다.

| 문서 | 내용 |
|:--|:--|
| [01_arch0_sequential.md](01_arch0_sequential.md) | Arch0: LR → SR → YOLO 단일 흐름 |
| [02_arch2_softgate.md](02_arch2_softgate.md) | Arch2: Gate가 SR과 bilinear 비율 결정 |
| [03_arch4_dual_yolo.md](03_arch4_dual_yolo.md) | Arch4: Scout + Sniper + ROI-aware NMS |
| [04_arch5b_fusion.md](04_arch5b_fusion.md) | Arch5b: LR/SR feature fusion → YOLO head |
| [05_sr_backbones.md](05_sr_backbones.md) | RFDN/DRCT/HAT/MAN 4종 명세 |
| [06_architecture_comparison.md](06_architecture_comparison.md) | 4 Arch 비교 표 (구조, 속도, 정확도) |

---

## 메인 repo 코드 위치

| 컴포넌트 | 메인 파일 |
|:--|:--|
| Arch0 | `src/models/pipelines/arch0_sequential.py` |
| Arch2 | `src/models/pipelines/arch2_softgate.py` |
| Arch4 | `src/models/pipelines/arch4_roi_awareNMS.py` |
| Arch4 ablation | `src/models/pipelines/arch4_roi_awareNMS_ablation.py` |
| Arch5b | `src/models/pipelines/arch5b_fusion.py` |
| Base | `src/models/pipelines/base_pipeline.py` |
| SR — RFDN | `src/models/sr_models/rfdn.py` |
| SR — DRCT | `sci_lab/backbones/drct_wrapper.py` |
| SR — HAT | `sci_lab/backbones/hat_wrapper.py` |
| SR — MAN | `sci_lab/backbones/man_wrapper.py` |
| Yolo Wrapper | `src/models/detectors/yolo_wrapper.py` |

→ 위 파일들이 `staging_public/code/` 디렉토리로 복사됨 (Step 7에서 진행).

---

## 검증 체크리스트

- [ ] 4 Arch 모두 일관된 입력 인터페이스 (LR tensor)
- [ ] 4 Arch 모두 일관된 출력 인터페이스 (detection dict)
- [ ] 같은 SR backbone이 모든 Arch에서 같은 weight로 로드되는가
- [ ] 같은 YOLO weight가 일관되게 사용되는가
- [ ] 4 SR backbone이 같은 인터페이스(Wrapper)를 통해 통합되는가
