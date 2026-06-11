# 04. 4090 → CHEETAH 코드 변경점 + 제작 이유 (해석)

> 본 디렉토리의 CHEETAH 코드가 본 repo의 4090 버전(`code/arch5/`, `code/shared/`)과 **무엇이/왜 다른지**. diff 규모는 실측.

---

## 0. 변경 규모 (diff, 4090 vs CHEETAH)

| 파일 | 변경 라인 | 요약 |
|---|---|---|
| `arch5b_fusion.py` | **+144** | HAT/MAN 지원 + SR fp32 강제 + Phase3 checkpointing 활성 |
| `attention_fusion.py` | **0 (동일)** | Fusion 모듈은 4090과 완전 동일 (α_s 정의 불변) |
| `hat_wrapper.py` | ~61 | `forward_features/reconstruct/feature_channels` + `enable_gradient_checkpointing` |
| `man_wrapper.py` | ~45 | 동일 + **per-MAB** checkpointing |
| `drct_wrapper.py` | ~35 | `enable_gradient_checkpointing` (RDG) |
| `train_arch5b_phase{2,3}_*` | (신규/재작성) | 완전 fp32 + (Phase3) RFDN 레시피 |

→ **Fusion 알고리즘 자체는 안 바뀜**(attention_fusion 0 라인). 바뀐 것은 ① backbone 지원 확대 ② precision ③ 메모리(checkpointing) ④ 학습 레시피.

---

## 1. 변경의 전체 맥락 (왜 이렇게 됐나 — 시간순)

이 변경들은 **CHEETAH에서 HAT/MAN을 새로 돌리며 마주친 문제를 해결한 결과**입니다:

1. **HAT/MAN wrapper에 feature 메서드가 없었음** → Arch5 fusion이 `forward_features`를 요구하는데 DRCT만 있었음. → HAT/MAN wrapper에 추가.
2. **HAT batch=12 OOM** → batch=8로.
3. **HAT fp16에서 처음부터 loss=NaN** → transformer attention softmax가 fp16 AMP에서 overflow. → **SR을 fp32로 강제**, 이어 **전체 fp32 통일**.
4. **Phase 3 full-unfreeze fp32가 backward에서 24GB OOM** (forward는 통과). 게다가 train loop의 `except(CUDA):continue`가 OOM을 삼켜 **23시간 동안 0-step**(모델 안 변함)이었던 사고. → **gradient checkpointing** 도입.
5. **MAN은 checkpointing 켜도 OOM** → granularity 버그(아래 3.2). → **per-MAB** 수정.
6. **레시피가 RFDN baseline과 어긋나 있었음**(유효배치 4·warmup 없음·patience 7). → **RFDN 레시피로 정렬**(유효배치 2·warmup 5·patience 10).

---

## 2. `arch5b_fusion.py` (+144) — 항목별 이유

### 2.1 `SUPPORTED_SR_TYPES`에 hat/man 추가 + `_init_hat_sr()`/`_init_man_sr()`
- **왜**: 원래 rfdn/mamba/drct만 지원. HAT/MAN을 Arch5에 쓰려면 빌더 추가 필요. DRCT 패턴 재사용(셋 다 feature_channels=180이라 fusion 모듈 그대로 호환).

### 2.2 SR forward를 fp32 강제 (`with autocast(enabled=False)`)
- **왜**: HAT/DRCT(transformer)는 fp16 AMP에서 attention softmax overflow → NaN. SR은 (Phase2 frozen / 일부) 메모리 부담이 작아 fp32로 돌려도 됨. Fusion/YOLO는 필요시 fp16 유지 가능하게 분리.
- 코드 주석에 "2026-05-28 HAT 실패" 근거 명시.

### 2.3 `unfreeze_for_phase3()`에서 gradient checkpointing 자동 활성
- **왜**: Phase3 full-unfreeze fp32가 backward에서 OOM. `self.sr_type in ('hat','drct','man')`일 때 `enable_gradient_checkpointing()` 호출.
- **수학적으로 동일**(activation 재계산일 뿐) → 공정 비교 안 깨짐, 속도만 20-30%↓.

---

## 3. Backbone wrapper 변경 — 항목별 이유

### 3.1 HAT/MAN: `forward_features` / `forward_reconstruct` / `feature_channels` 추가
- **왜**: DRCTWrapper만 이 메서드들이 있었고 HAT/MANWrapper는 `forward()`만 있었음. Arch5 fusion은 중간 feature(`forward_features`)와 복원(`forward_reconstruct`)을 분리 호출하므로 필수.
- HAT: DRCT와 동일 패턴(embed_dim=180). MAN: `sub_mean→head→body(36 MAB)→body_t` 패턴, 원본이 `(sr,feature)` tuple 반환하던 것에서 feature를 노출.

### 3.2 `enable_gradient_checkpointing()` — **granularity가 핵심**
- HAT: arch `forward_features`의 `for layer in m.layers`(RHAG 6개)를 각각 `cp.checkpoint`로 감쌈.
- DRCT: 동일(RDG 6개).
- **MAN — 함정**: MAN은 `model.body = ModuleList[ResGroup 1개]`이고, **진짜 36개 MAB 블록은 ResGroup 내부**(`ResGroup.body`)에 있음.
  - 처음엔 ResGroup을 **통째로** checkpoint → backward 재계산 때 **36 MAB가 한꺼번에 부활** → 효과 0, 여전히 24GB OOM.
  - **수정**: wrapper의 `forward_features`에서 ResGroup.forward(`clone→36 MAB→body_t+residual`)를 복제하되 **개별 MAB마다 checkpoint** → **24GB OOM → 5.15GB** (검증됨).
  - **교훈**: checkpointing은 "재계산 단위(segment)"가 작아야 효과. 한 segment가 전체 블록을 담으면 무의미.

---

## 4. 학습 스크립트 (`train_arch5b_phase{2,3}_*_fullimg.py`)

### 4.1 완전 fp32
- `GradScaler(enabled=False)` + `autocast(enabled=False)`. **왜**: 위 NaN 이슈 + RFDN baseline이 fp32 → 공정 비교.

### 4.2 Phase 3 = RFDN 레시피 (이게 4090과 가장 큰 차이)
- `batch_size=2, grad_accum=1`(유효배치 2), `patience=10`, `warmup_epochs=5`.
- scheduler를 `CosineAnnealingLR` → **`LambdaLR`(5ep linear warmup→cosine)** 로 교체. per-group lr에 동일 배율 적용.
- per-group lr: SR×0.2, detector×0.6, fusion×1.0 (base 5e-5).
- **왜**: §02 문서 3.1 참조 (유효배치=lr 공정성, RFDN 정렬).

---

## 5. 안 바뀐 것 (중요)

- **`attention_fusion.py` 0 라인 변경** = Fusion 알고리즘(CrossAttn/CBAM/α_s init -2.0)은 4090과 **완전 동일**. → "구조"는 동일, "precision·메모리·레시피"만 CHEETAH에서 조정.
- third_party 원본(HAT/DRCT/MAN repo) 불변 — checkpointing은 런타임에 `forward_features` 교체로 적용.

---

## 6. 무결성/재현 체크리스트

- [ ] 4090 버전(`code/arch5/arch5b_fusion.py`)과 본 CHEETAH 버전 diff = **144 라인** (위 2절 항목으로 전부 설명됨 — 알고리즘 변경 아님).
- [ ] `attention_fusion.py` diff = **0** (동일 확인).
- [ ] checkpointing은 **결과 불변**(activation 재계산) — 수렴/수치에 영향 없음.
- [ ] Phase 3 레시피만 RFDN 정렬, Phase 2는 정렬 전 완료(문서에 명시).
- [ ] 결과(mAP)는 아직 없음 — eval 미실행.
