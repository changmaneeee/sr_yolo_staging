# 02. 학습 설정 / 레시피 + 각 설정의 이유 (CHEETAH) ★핵심

> 전부 **스크립트 코드 검증분** (`scripts/train_arch5b_phase{2,3}_*_fullimg.py`). 결과 수치 없음.

---

## 0. 두 줄 요약

- **모든 backbone 완전 fp32** (AMP/fp16 미사용).
- **Phase 3는 RFDN(4060 baseline) 레시피에 정렬**: 유효배치 2 · per-group lr · warmup 5ep · patience 10 · gradient checkpointing.

## 1. 2-Phase 구조

| | Phase 2 (fusion warm-up) | Phase 3 (joint fine-tune) |
|---|---|---|
| 학습 대상 | **Fusion 모듈만** (SR + YOLO frozen) | **전체 unfreeze** (SR+YOLO+Fusion) |
| 시작점 | SR/YOLO 사전학습 weight + fusion 랜덤 | **Phase 2 best.pt에서 resume** |
| 목적 | fusion을 안정적으로 켜기(detector 보존) | end-to-end 공동 최적화 |

**왜 2-phase인가**: 처음부터 전체를 풀면 랜덤 초기화된 fusion이 사전학습 detector를 망가뜨릴 위험 → Phase 2에서 fusion만 먼저 데워(α_s≈0.12로 거의 identity) 안정화한 뒤, Phase 3에서 전체를 미세조정.

## 2. Phase 2 설정 (완료) — ⚠️ RFDN 정렬 **이전** 완료

| 항목 | HAT | MAN | DRCT | 비고/이유 |
|---|---|---|---|---|
| precision | fp32 | fp32 | fp32 | `GradScaler(enabled=False)` |
| real batch | 8 | 12 | 12 | 모델별 메모리 한도 내 최대 (HAT가 가장 큰 transformer라 8) |
| grad_accum | 1 | 1 | 1 | |
| lr | 1e-4 | 1e-4 | 1e-4 | 단일 lr |
| patience | 7 | 7 | 7 | |
| warmup | 없음 | 없음 | 없음 | |
| trainable | Fusion만(~7.7M) | | | SR+YOLO frozen |

→ Phase 2는 **fusion warm-up**이라 frozen-SR 상태(메모리 여유) + RFDN 레시피 정렬 결정 전에 완료됨. 비교의 핵심인 Phase 3만 RFDN에 정렬(아래).

## 3. Phase 3 설정 (진행중) — RFDN 레시피 정렬 ★

| 항목 | 값 (HAT/MAN/DRCT 공통) | **이유** |
|---|---|---|
| precision | **완전 fp32** | transformer(HAT/DRCT)가 **fp16 AMP에서 attention softmax overflow→NaN/발산**. CNN(MAN/RFDN)은 robust. RFDN baseline도 fp32 → **공정 비교 + 안정성** 위해 전부 fp32 통일 |
| real batch | **2** | RFDN 레시피 정렬 (아래) |
| grad_accum | **1** → 유효배치 **2** | RFDN(4060)이 유효배치 2 |
| lr (base) | **5e-5**, per-group | SR×0.2=**1e-5** / detector×0.6=**3e-5** / fusion×1.0=**5e-5**. SR/detector는 사전학습됐으니 작은 lr, fusion(신규)은 큰 lr |
| scheduler | **5ep linear warmup → cosine** (LambdaLR) | RFDN 정렬 + 초반 안정(특히 transformer 발산 방지) |
| patience | **10** | RFDN(4060)과 동일 |
| grad checkpointing | **ON** (HAT/DRCT arch-level, MAN per-MAB) | full-unfreeze fp32가 backward에서 24GB 초과 OOM → activation 재계산으로 메모리↓ (결과 수학적 동일) |
| loss weight | λ_det=**1.0**, λ_sr=**0.3** | detector 주목적, SR 복원은 보조 regularizer |
| detector_input | **'sr'** | detector가 SR 복원 이미지 위에서 동작 |
| SR forward precision | **항상 fp32** (`autocast(enabled=False)`) | SR이 fp16이면 transformer attention NaN. frozen/일부라 메모리 부담 작음 |

### 3.1 RFDN 레시피 정렬의 이유 (왜 유효배치 2·warmup·patience를 맞췄나)
- 공정 비교의 핵심은 **유효 배치**(linear scaling rule: 적정 lr ∝ 유효배치). RFDN(4060) baseline이 **유효배치 2, lr 5e-5(per-group), warmup 5ep, patience 10**.
- 우리 신규 backbone(HAT/MAN/DRCT)을 RFDN에 맞추면 RFDN 재학습(0.9102 매트릭스 무효화) 없이 **동일 조건 비교** 성립. 하드웨어(4060↔A5000) 차이는 같은 레시피면 시드노이즈 수준.
- (정렬 전에는 유효배치 4·warmup 없음·patience 7로 어긋나 있었음 — DRCT 4090 fullimg 스크립트 복사 잔재였음.)

### 3.2 gradient checkpointing 이유/방식
- full-unfreeze fp32에서 SR backbone backward activation이 폭증 → 24GB OOM.
- checkpointing = forward activation을 저장 대신 backward 때 재계산 → **메모리 대폭↓, 결과 수학적 동일, 속도 20-30%↓**.
- HAT/DRCT: arch의 `forward_features` layer 루프(RHAG/RDG 6개)를 checkpoint.
- **MAN: ResGroup 통째로 하면 효과 0**(backward 때 36 MAB가 한꺼번에 부활) → **ResGroup 내부 36 MAB를 개별 checkpoint**해야 함 (24GB→5GB). 자세히는 `04_code_changes_and_rationale.md`.

## 4. 하드웨어 / 시간

- GPU: RTX A5000 ×2 (각 24GB), 컨테이너 한도 CPU 20 vCPU / RAM 126GB.
- epoch당: P2 — HAT 4.5h / MAN 2.3h / DRCT 3.0h. P3 — HAT 16.8h / MAN 9.9h (fp32+checkpointing 비용).
- Phase 3 batch=2의 메모리 실측: HAT 13~15GB, MAN 5GB(per-MAB checkpoint), DRCT ~ (실행 시 측정 예정).

## 5. 안전장치 / 알려진 함정

- train loop에 `except (CUDA OOM): continue`가 있어 **OOM을 조용히 삼켜 0-step 학습**이 될 수 있음 → **batch=2 메모리 사전 테스트로 회피** 후 실행 (과거 HAT 23시간 0-step 사고 교훈).
- NFS 홈 볼륨이라 **로그 mtime/내용이 캐시로 옛값처럼 보일 수 있음** → 가동 판단은 GPU 전력 + 프로세스 CPU시간 + history.json 내용으로.

## 6. 결과/평가 (별도)

- **mAP eval(subset6418) 미실행** — 현재 학습 `val_loss`만. `*_eval.json`(0.91 형식)은 best.pt로 eval 돌려야 생성.
- 수렴 α_s 값/최종 수치는 학습 종료 후 확정 (본 문서는 설정만).
