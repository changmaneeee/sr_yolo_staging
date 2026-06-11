# SR-YOLO Staging Package (Pre-Public Audit)

> **목적**: SR-YOLO 연구의 모든 구성요소(데이터셋, 전처리, 아키텍처, 학습, 평가, 결과)를 한 곳에 정리하여 **무결성 검사**를 받기 위한 패키지.
>
> **상태**: 공개 전 임시 단계 (private GitHub: `changmaneeee/sr_yolo_staging`).
>
> **검사자**: 별도 모델/agent가 본 패키지를 처음부터 끝까지 읽고 검증합니다.

---

## 검사자에게 — 무엇을 검증할 것인가

본 패키지는 다음 5개 축의 무결성 검증을 목적으로 합니다.

### 검증 축 1: 데이터셋 (`docs/01_dataset/`)
- [ ] 데이터 출처가 명확하고 추적 가능한가
- [ ] HR/LR/labels의 경로, 개수, 통계가 일치하는가
- [ ] Train/Val split의 기준이 명시되어 있는가
- [ ] subset6418의 선정 기준이 합리적인가

### 검증 축 2: 전처리 (`docs/02_preprocessing/`)
- [ ] LR 이미지 생성 과정이 모든 SR backbone에서 동일한가
- [ ] 정규화/normalize 방식이 학습/평가 간 일관되는가
- [ ] labeled_only filter가 올바르게 작동하는가

### 검증 축 3: 아키텍처 코드 무결성 (`docs/03_architectures/`, `code/`)
- [ ] 각 아키텍처(Arch0/2/4/5)가 의도대로 구현되어 있는가
- [ ] SR backbone 4종(RFDN/DRCT/HAT/MAN)이 같은 방식으로 통합되는가
- [ ] 공통 컴포넌트(Scout, Sniper, Gate, Fusion)가 재사용되는가
- [ ] 코드 간 비교에서 어떤 SR이든 동일한 절차가 적용되는가

### 검증 축 4: 학습 절차의 공정성 (`docs/05_training/`)
- [ ] 동일한 데이터셋/seed/hyperparameter로 학습됐는가
- [ ] Sniper의 base detector가 각 SR마다 올바르게 선택됐는가 (rfdn base 사고 재발 방지)
- [ ] 학습 중 임의로 weights/데이터를 바꾸지 않았는가

### 검증 축 5: 평가 절차의 공정성 (`docs/06_evaluation/`, `docs/07_experiments/`)
- [ ] 모든 SR이 동일한 validation set(6418장)에서 평가되었는가
- [ ] 동일한 metric (mAP@50, Precision, Recall) 정의를 사용하는가
- [ ] 같은 NMS protocol 하에서 비교되는가 (또는 SR별 다른 경우 사유가 명시되는가)

---

## 디렉토리 구조

```
staging_public/
├── README.md                   ← (현재 파일) 검사자 entry point
├── docs/                       ← 모든 과정 설명
│   ├── 00_overview.md          ← 프로젝트 전체 개요
│   ├── 01_dataset/             ← 데이터셋 명세
│   ├── 02_preprocessing/       ← 전처리 과정
│   ├── 03_architectures/       ← Arch0/2/4/5 + SR backbone
│   ├── 04_components/          ← Scout/Sniper/Gate/Fusion/NMS
│   ├── 05_training/            ← 학습 절차 (각 컴포넌트별)
│   ├── 06_evaluation/          ← 평가 프로토콜 + metric
│   ├── 07_experiments/         ← 모든 실험의 절차 + 결과
│   ├── 08_results/             ← 최종 표 + 수치
│   ├── 09_reproducibility/     ← 환경, seed, weight 경로
│   ├── 10_integrity_audit/     ← 검사용 체크리스트 + 이슈 로그
│   └── PROMPTS/                ← 서버 동기화/검사 모델 prompt
├── code/                       ← 핵심 코드 (weights 제외)
│   ├── shared/                 ← SR backbones, yolo_wrapper
│   ├── arch0/, arch2/, arch4/, arch5/
│   ├── scripts/                ← eval_arch.py 등
│   └── configs/                ← yaml 설정
├── results/                    ← 작은 산출물
│   ├── tables/                 ← csv summary
│   ├── json/                   ← 핵심 eval JSON
│   └── figures_data/           ← plot용 csv
└── environment/                ← 재현 환경
```

---

## 검사 권장 순서

1. **시작**: `docs/00_overview.md` — 프로젝트 전체 그림 이해
2. **데이터**: `docs/01_dataset/` 모두 읽기 → `docs/02_preprocessing/`
3. **구조**: `docs/03_architectures/` → `docs/04_components/`
4. **학습**: `docs/05_training/` — 각 컴포넌트별 학습 절차
5. **평가**: `docs/06_evaluation/` — 측정 방식
6. **실험**: `docs/07_experiments/` — 모든 실험 절차
7. **결과**: `docs/08_results/` — 최종 표
8. **재현**: `docs/09_reproducibility/` — 환경, weight 경로
9. **검증 체크리스트**: `docs/10_integrity_audit/01_audit_checklist.md` — 검사 항목

---

## 메인 repo와의 관계

본 폴더는 메인 작업 폴더(`dark_vessel_sr_yolo/`) 안에 있지만 **독립된 git repo**입니다.

- 메인 repo: 실제 작업 (학습 스크립트 출력, weights, 데이터 캐시 등)
- staging_public (현재): 검사용 정리본 (코드 + 문서 + 작은 산출물)

**Weights는 본 폴더에 포함되지 않습니다.** 정확한 경로는 `docs/09_reproducibility/03_weight_locations.md`에 명시되어 있으며, 검사자가 메인 폴더에서 직접 접근할 수 있습니다.

---

## 진행 상태

| 섹션 | 상태 |
|:--|:--|
| 00_overview | ✅ 완료 |
| 01_dataset | ✅ 완료 (5 files) |
| 02_preprocessing | ✅ 완료 (3 files) |
| 03_architectures | ✅ 완료 (6 files) |
| 04_components | ✅ 완료 (6 files) |
| 05_training | ✅ 완료 (7 files) |
| 06_evaluation | ✅ 완료 (6 files) |
| 07_experiments | ✅ 완료 (6 files) |
| 08_results | ✅ 완료 (4 files) |
| 09_reproducibility | ✅ 완료 (5 files) |
| 10_integrity_audit | ✅ 완료 (5 files) |
| code/ 복사 | ✅ 완료 (메인 컴포넌트 25 files) |
| results/ 복사 | ✅ 완료 (JSON 핵심 + CSV) |
| 서버 동기화 prompt | ✅ 완료 (PROMPTS/server_arch5_request.md) |
| 검사 prompt | ✅ 완료 (PROMPTS/audit_prompt.md) |
| Arch5 서버 자료 동기화 | 대기 (서버에서 수행) |

---

## 연락
- Maintainer: changmaneeee (changmin3702@gmail.com)
- Project: SR-YOLO (Dark Vessel Detection)
