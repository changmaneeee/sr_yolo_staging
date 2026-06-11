# 05-07. SR Backbone Fine-tune (Optional)

## 개요

대부분의 실험에서 SR backbone은 pretrained를 그대로 사용한다. 다만 일부 특수 실험에서 fine-tune 진행.

## DRCT crop64 fine-tune (검증 #36)

### 동기
DRCT의 Swin window 크기와 Arch4 crop 크기(LR 64×64 → HR 256×256) 사이의 불일치 검증.

### 절차
- Base: DRCT base weight
- 데이터: subset의 64×64 LR crop들
- Loss: SR L1 loss
- Epoch: ~10
- 저장: `weights/sr_finetuned/drct/best.pt` (이미 properbase에서 사용 중)

### 검증 결과
- crop64 fine-tune 후 Arch4 mAP@50 변화 측정 → 영향 미미

자세히: [docs/07_experiments/](../07_experiments/) (검증 #36)

## sr_finetuned 디렉토리

| SR | Path |
|:--|:--|
| DRCT | `weights/sr_finetuned/drct/best.pt` |
| HAT | `weights/sr_finetuned/hat/best.pt` |
| MAN | `weights/sr_finetuned/man/best.pt` |

**이 weights가 모든 properbase + NMS sweep에서 사용됨**.

## 검증 포인트

- [ ] Fine-tune의 목적이 명확 (작은 ROI 대응)
- [ ] Train set만 사용
- [ ] 평가 시 sr_finetuned weight가 일관되게 사용됨
