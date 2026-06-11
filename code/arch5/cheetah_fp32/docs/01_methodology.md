# 01. Arch5B 방법론 + 설계 이유 (CHEETAH)

> 모든 내용은 **실제 코드 검증분**입니다 (`arch5b_fusion.py`, `attention_fusion.py`). 결과 수치 없음.

---

## 1. 한 줄 정의

Arch5B = **feature-level SR–detector fusion**. SR encoder의 *중간 feature*를 detector의 multi-scale pyramid(P3/P4/P5)에 주입하여, super-resolution 정보를 검출 단계에서 직접 활용한다.

- Arch0(cascade)/Arch2(soft-gate)/Arch4(adaptive)는 SR을 **이미지 수준**에서만 결합.
- Arch5B는 그 위에 **feature 수준 결합**을 추가 → 이것이 핵심 기여.

## 2. 전체 데이터 흐름 (코드 검증: `arch5b_fusion.py:forward`)

```
LR(192²) ─► SR encoder ─┬─► 중간 feature  F^sr  [B,180,H,W]   (forward_features)
                        └─► 복원 SR 이미지 (768²)             (forward_reconstruct)
                                   │
복원 SR 이미지 ─► YOLO backbone ─► pyramid {F^yolo_s}, s∈{P3,P4,P5}
                                   │            (detector_input='sr')
F^sr ─► MultiScaleAttentionFusion ─► 각 스케일에 주입 ─► {F'_s} ─► YOLO Detect head
```

⚠️ **중요(framing)**: `detector_input='sr'`(config·로그 검증). 즉 **detector는 SR 복원 이미지 위에서 동작**하고, Arch5B는 거기에 SR encoder의 **중간 feature를 추가로 융합**한다. "이미지를 안 쓰고 feature만"이 아니라 **"cascade(SR 이미지) 위에 feature-level 결합을 더한 것"**.

## 3. SR feature 추출 (코드 검증)

- `forward_features(LR)` → 단일 feature map `F^sr ∈ R^{C_sr×H×W}`. **C_sr=180** (HAT/DRCT/MAN 공통), RFDN=50.
- 같은 feature로부터 `forward_reconstruct(F^sr)`가 복원 SR 이미지 생성 (detector 입력 + SR loss용).
- 단일 feature map을 P3/P4/P5 **각 스케일로 채널정렬 + resize**하여 주입.

## 4. MultiScaleAttentionFusion (코드 검증: `attention_fusion.py`)

스케일별(SingleScaleFusion) 처리:

1. **채널 정렬 + resize**: `F^sr`를 1×1 Conv(+BN+ReLU)로 YOLO 채널({128,256,512})에 맞추고, 해당 스케일 공간 크기로 bilinear resize → `F̂^sr_s`.
2. **CrossAttention** (P4, P5만): SR=query, YOLO=key/value, multi-head(4), scaled dot-product `softmax(QᵀK/√d)V`. → SR 근거로 YOLO feature의 어느 위치를 강조할지 결정.
   - **P3 제외 이유**: P3는 공간 해상도가 커서 dense attention map이 메모리상 과도 → CBAM만 적용.
3. **CBAM**: channel attention → spatial attention 순차. (기성 모듈, 인용)
4. **결합 + 게이트**:
   ```
   A_s = CBAM(CrossAttn(F̂^sr_s, F^yolo_s))   (P4,P5)
       = CBAM(F̂^sr_s)                          (P3)
   F'_s = α_s · Conv3×3([A_s ; F^yolo_s]) + (1-α_s) · F^yolo_s
   α_s = σ(a_s)
   ```

## 5. 학습 게이트 α_s — 설계 이유 ★

- `self.alpha = nn.Parameter(...)` (스케일마다 1개, 총 3개) = **학습되는 모델 weight**(체크포인트 내장).
- **init: logit a_s = -2.0 → α_s = σ(-2.0) ≈ 0.12** (`attention_fusion.py`).
- **왜 logit/0.12로 시작하나** (코드 주석 "start near identity"):
  - 시작 시 `F'_s ≈ 0.88·F^yolo + 0.12·fused` → **사전학습된 detector를 거의 보존**하면서 fusion 기여를 12%만 부여.
  - fusion conv도 small-scale Kaiming(×0.1)로 초기화 → 초기에 **identity에 가깝게** 시작해 안정적 warm-up.
  - 이후 a_s가 end-to-end 학습되며 **스케일별로 SR 기여도를 스스로 조절**. (학습 후 0.12→0.23~0.41로 상승 = 모델이 SR feature를 실제로 채택)

## 6. 왜 feature-level인가 (기여의 논리)

- cascade는 SR **이미지**만 detector에 전달 → SR의 미세 디테일이 이미지 재합성·재인코딩을 거치며 손실 가능.
- Arch5B는 SR encoder의 **중간 표현(detail-enhanced feature)** 을 detector pyramid에 직접 주입 → 정보 손실 없이 검출에 활용.
- α_s 게이트로 "얼마나 섞을지"를 데이터가 결정 → 안전(사전학습 detector 보존)하면서 학습 가능.

## 7. 학습/구조 경계 (논문 III vs V)

- **III(구조)**: 위 fusion 구성, α_s(게이트, init σ(-2.0)=0.12).
- **V(학습)**: loss weight(λ_sr=0.3, λ_det=1.0), 2-phase 스케줄, per-group lr 등 → `02_settings_recipe.md` 참조.
- α_s **수렴값은 결과(V/Results)**, init은 구조(III).

> 참고: 기존 4090 문서 `docs/03_architectures/04_arch5b_fusion.md`도 동일 아키텍처를 다룸. 본 문서는 **코드 재검증 + detector_input='sr' framing + α_s 학습성**을 명확히 한 CHEETAH 버전.
