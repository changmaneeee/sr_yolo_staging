# 06. Evaluation Procedures

평가 절차의 무결성. **공정 비교의 마지막 보루**.

| 문서 | 내용 |
|:--|:--|
| [01_fixed_protocol.md](01_fixed_protocol.md) | fixed_protocol.yaml 명세 |
| [02_eval_arch_py.md](02_eval_arch_py.md) | eval_arch.py 동작 (Fair eval, Arch0/2/5) |
| [03_arch4_eval_ultralytics.md](03_arch4_eval_ultralytics.md) | arch4_eval_ultralytics.py 동작 (Arch4 전용) |
| [04_metric_definition.md](04_metric_definition.md) | mAP, P, R 정확한 정의 |
| [05_validation_set.md](05_validation_set.md) | 평가 set (subset6418) 강제 |
| [06_preflight_check.md](06_preflight_check.md) | Pre-flight check 함수 |

## 핵심 사실

1. **모든 평가**: subset6418 (6,418장) 사용
2. **모든 metric**: ultralytics `ap_per_class` 사용
3. **모든 NMS**: fixed_protocol에 통제
4. **MD5 검증**: Pre-flight check가 scout/sniper weight 검증

## 검증 체크리스트

- [ ] 6,418장이 정확히 평가되는가
- [ ] 동일한 metric 계산이 일관되는가
- [ ] fixed_protocol이 모든 hyperparam을 통제
- [ ] Pre-flight check가 weight 일관성 강제
