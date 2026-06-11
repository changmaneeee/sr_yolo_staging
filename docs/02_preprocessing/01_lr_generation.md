# 02-01. LR Image Generation

## 개요

HR(768×768) 이미지에서 LR(192×192) 이미지를 생성하는 방식은 **단일 결정론적 과정**이며, 모든 SR backbone이 같은 LR 입력을 받는다.

## 생성 방식

- **방법**: Bicubic downscale (PIL/cv2 표준)
- **Scale factor**: 4× (768 → 192)
- **색공간**: RGB (HR과 동일)
- **포맷**: JPEG (.jpg), HR과 동일 quality

### 생성 logic (참고용)
```python
from PIL import Image

hr = Image.open(hr_path).convert("RGB")  # 768×768
lr = hr.resize((192, 192), Image.BICUBIC)
lr.save(lr_path, quality=95)
```

### 정확한 LR 데이터셋 위치
- `/home/changmin/smart_airbus_data_lr/images/train/` (108,414장)
- `/home/changmin/smart_airbus_data_lr/images/val/` (28,884장)

### 생성 시점
2026-01-12 (LR dataset 초기 구축, MD5 `0087ca54` 기준의 RFDN weight도 같은 시점)

## 일관성 검증

### LR과 HR의 1:1 매칭
모든 LR 이미지는 HR과 정확히 같은 stem(파일명 prefix)을 갖는다.
```bash
# 매칭 확인
HR_STEMS=$(ls /home/changmin/smart_airbus_data/images/val/ | sort)
LR_STEMS=$(ls /home/changmin/smart_airbus_data_lr/images/val/ | sort)
diff <(echo "$HR_STEMS") <(echo "$LR_STEMS")
# 1장 차이만 있음 (LR이 1장 더 있지만 subset6418에서 무력화됨)
```

### 모든 SR 평가에서 동일 LR 사용
- RFDN, DRCT, HAT, MAN 모두 같은 `smart_airbus_data_lr/images/val/`을 입력으로 받는다.
- subset6418 평가 시에도 `subset6418_lr/images/val/`은 원본 LR을 단순 복사한 것.

## 주의: LR 생성 방식이 SR 학습/평가에 미치는 영향

본 연구는 **bicubic으로 만든 LR**을 SR 입력으로 사용한다. 만약 LR이 다른 방식(예: degradation model)으로 만들어졌다면 SR 결과가 달라질 수 있으나, **모든 SR backbone에 같은 LR이 들어가므로 backbone 간 비교는 공정**하다.

## 검증 포인트

- [ ] LR 생성에 bicubic + 4× scale이 사용되었는가
- [ ] 모든 SR backbone이 같은 LR 입력 받는가
- [ ] LR/HR stem 1:1 매칭되는가
- [ ] subset6418의 LR이 원본 LR의 단순 복사본인가 (재가공 없음)
