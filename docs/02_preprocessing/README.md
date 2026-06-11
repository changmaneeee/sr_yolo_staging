# 02. Preprocessing

학습/평가 시 데이터에 적용되는 모든 전처리 절차를 다룬다.
**모든 SR backbone(RFDN/DRCT/HAT/MAN)에서 동일한 전처리가 적용되어야 공정 비교가 성립**한다.

| 문서 | 내용 |
|:--|:--|
| [01_lr_generation.md](01_lr_generation.md) | HR → LR 생성 (4× downscale) |
| [02_normalization.md](02_normalization.md) | input_range 0-255 vs 0-1, SR backbone별 처리 |
| [03_dataloading.md](03_dataloading.md) | 학습 시 augmentation, 평가 시 변환 절차 |

---

## 검증 체크리스트

- [ ] LR 생성 방식이 단일하고 결정론적 (deterministic)
- [ ] 모든 SR backbone이 같은 input_range을 받는가
- [ ] 학습 augmentation이 모든 SR backbone에서 동일한가
- [ ] 평가 시 SR 결과가 동일한 방식으로 후처리 되는가
