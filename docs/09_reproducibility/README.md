# 09. Reproducibility

본 연구를 재현하기 위한 모든 정보.

| 문서 | 내용 |
|:--|:--|
| [01_environment.md](01_environment.md) | Python/PyTorch 버전, conda env |
| [02_seeds_and_determinism.md](02_seeds_and_determinism.md) | Seed, determinism 보장 |
| [03_weight_locations.md](03_weight_locations.md) | 모든 weight의 정확한 경로 |
| [04_data_paths.md](04_data_paths.md) | 데이터셋 경로 명세 |
| [05_run_commands.md](05_run_commands.md) | 각 실험 실행 명령 |

## Weights 처리

본 staging repo에는 weights가 **포함되지 않는다**.
검사자/재현자는 메인 폴더(`/home/changmin/dark_vessel_sr_yolo/weights/`)에서 직접 접근.

각 weight의 정확한 경로는 [03_weight_locations.md](03_weight_locations.md)에 명시.
