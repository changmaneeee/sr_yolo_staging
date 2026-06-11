# 04. Components

각 아키텍처를 구성하는 핵심 컴포넌트의 명세.

| 문서 | 내용 |
|:--|:--|
| [01_scout_yolo.md](01_scout_yolo.md) | Arch4의 Scout YOLO (LR detector) |
| [02_sniper_yolo.md](02_sniper_yolo.md) | Arch4의 Sniper YOLO (HR crop detector) |
| [03_gate_network.md](03_gate_network.md) | Arch2의 Gate network |
| [04_fusion_module.md](04_fusion_module.md) | Arch5b의 Attention Fusion module |
| [05_roi_aware_nms.md](05_roi_aware_nms.md) | Arch4의 ROI-aware NMS |
| [06_yolo_wrapper.md](06_yolo_wrapper.md) | YOLO 통합 wrapper |

## 검증 체크리스트

- [ ] 각 컴포넌트가 모든 SR backbone에서 동일하게 작동
- [ ] Scout는 SR과 무관하게 공통 weight
- [ ] Sniper는 SR별 별도 학습 (rfdn base 사고 재발 방지)
- [ ] NMS 파라미터가 fixed_protocol에 의해 통제
