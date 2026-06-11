# Arch5B (SR Feature Fusion) — CHEETAH 서버 버전 (완전 fp32 + RFDN 레시피)

> 이 디렉토리는 **CHEETAH 서버(RTX A5000 ×2)** 에서 진행한 Arch5B 학습의 **실제 사용 코드 · 설정 · 방법론 · 해석**을 자기완결적으로 모은 것입니다.
> ⚠️ **결과(mAP) 미포함** — 방법론·코드·설정·전처리·데이터셋 중심. Phase 3 학습 진행 중(미완).
> 기준일: 2026-06-11 (KST).

---

## 0. 이게 뭔가 / 왜 따로 있나

- 본 repo의 `code/arch5/arch5b_fusion.py`, `code/shared/sr_backbones/*_wrapper.py` 등은 **4090 PC 버전**입니다.
- CHEETAH에서는 HAT/MAN을 새로 활성화하고 **완전 fp32 + RFDN 레시피 + gradient checkpointing**으로 다시 학습하면서 코드를 수정했습니다 → 4090 버전과 **다릅니다**.
- 기존 4090 파일을 덮어쓰지 않기 위해, CHEETAH에서 실제로 돌린 버전을 **이 하위 디렉토리에 그대로 복사**해 두었습니다. (`code/shared/fusion/attention_fusion.py`는 4090과 **완전 동일**이라 변경 없음.)

## 1. 디렉토리 구성

```
cheetah_fp32/
├── README.md                       ← 이 파일
├── arch5b_fusion.py                ← 메인 파이프라인 (4090 대비 +144 라인 수정)
├── attention_fusion.py             ← Fusion 모듈 (4090과 동일, α_s 정의 포함)
├── backbones/
│   ├── hat_wrapper.py              ← forward_features/reconstruct + checkpointing 추가
│   ├── man_wrapper.py              ← 동일 + per-MAB checkpointing
│   └── drct_wrapper.py             ← 동일 + RDG checkpointing
├── scripts/
│   └── train_arch5b_phase{2,3}_{hat,man,drct}_fullimg.py   ← 학습 스크립트 6개
├── configs/
│   └── arch5b_phase{2,3}_{hat,man,drct}.yaml + arch5b_fusion.yaml
└── docs/
    ├── 01_methodology.md           ← Arch5B 방법론 + 설계 이유
    ├── 02_settings_recipe.md       ← 전체 학습 설정(Phase2/3) + 각 설정 이유  ★핵심
    ├── 03_data_preprocessing.md    ← 데이터셋 + 전처리/데이터로딩 + 이유
    └── 04_code_changes_and_rationale.md  ← 4090→CHEETAH 변경점 + 제작 이유
```

## 2. backbone 현황 (2026-06-11, 결과 제외)

| backbone | type | Phase 2 (fp32) | Phase 3 (fp32) |
|---|---|---|---|
| HAT | Hybrid Attn transformer (39.6M) | ✅ 완료 | 🔵 진행중 |
| MAN | Multi-scale Attn CNN (27.6M) | ✅ 완료 | 🔵 진행중 |
| DRCT | Dense Residual transformer (33.0M) | ✅ 완료 | ⬜ 미실행 |
| RFDN | (baseline, 4060/4090) | — | — (여기 없음) |

## 3. 4090 자료와의 관계 (혼동 방지)

| 자료 | 위치 | 비고 |
|---|---|---|
| RFDN Arch5b 0.9102 | 4060/4090 | run 로그/eval은 handoff에 없음, **weights만** 보유(`weights/sr_finetuned/rfdn`, `weights/yolo_8s_rfdn`) |
| DRCT Arch5b **fp16** | 4090 | `iac_lab/runs/arch5b_phase{2,3}_drct_*_fp16_4090` (archive) |
| HAT/MAN/DRCT Arch5b **fp32** | **CHEETAH(여기)** | 본 디렉토리 대상 |

## 4. 실행 방법 (재현)

```bash
source ~/.venv/arch5b-py310-torch212-cu118/bin/activate   # torch 2.1.2+cu118
cd /home/jovyan/changmin/dark_vessel_research/handoff_cheetah_20260525/code
# Phase 2 (fusion warm-up)
CUDA_VISIBLE_DEVICES=0 python iac_lab/scripts/train_arch5b_phase2_hat_fullimg.py
# Phase 3 (joint, Phase 2 best.pt에서 자동 resume)
CUDA_VISIBLE_DEVICES=0 python iac_lab/scripts/train_arch5b_phase3_hat_fullimg.py
```
- 데이터: `smart_airbus_data`(HR 768²) + `smart_airbus_data_lr`(LR 192²), ×4 SR.
- 자세한 설정/이유는 `docs/02_settings_recipe.md`.

## 5. 주의

- **Weights는 repo에 안 들어감**(.gitignore `*.pt`) — 경로/크기/MD5만 `ARCH5_STATUS_CHEETAH.md` 참조.
- **결과 수치(mAP)는 아직 없음** — eval 미실행, 학습 진행 중.
- commit/push는 사용자가 직접 (외부 작업).
