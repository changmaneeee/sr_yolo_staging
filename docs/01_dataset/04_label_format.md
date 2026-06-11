# 01-04. Label Format

## Format

YOLO 표준 format (Ultralytics 호환):

```
class_id x_center y_center width height
```

- `class_id`: 0 (ship, 단일 클래스)
- `x_center, y_center`: 박스 중심 좌표, 0~1 정규화
- `width, height`: 박스 폭/높이, 0~1 정규화
- 좌표 기준: 해당 이미지의 크기 (HR이면 768×768, LR이면 192×192)

## 실제 예시

`/home/changmin/smart_airbus_data/labels/val/0002756f7.txt`:
```
0 0.439453 0.067057 0.035156 0.040365
```

해석:
- class 0 (ship)
- 중심점 (0.4395, 0.0671) × 768 = (337.5, 51.5) 픽셀
- 폭 0.0352 × 768 = 27 픽셀
- 높이 0.0404 × 768 = 31 픽셀
- → 작은 ship 인스턴스 (이 dataset의 전형적 크기)

## 멀티 인스턴스

이미지 한 장에 여러 ship이 있으면 줄마다 한 인스턴스:
```
0 0.500 0.500 0.040 0.045
0 0.300 0.700 0.030 0.035
0 0.700 0.300 0.025 0.030
```

## HR ↔ LR Label 관계

LR과 HR은 같은 stem을 사용하며, **label 파일도 동일**.
- LR label: `/home/changmin/smart_airbus_data_lr/labels/val/{stem}.txt`
- HR label: `/home/changmin/smart_airbus_data/labels/val/{stem}.txt`
- 두 label은 0~1 정규화되어 있으므로 좌표값 자체는 같음 (해석 시 해당 이미지 크기에 맞춰 픽셀로 변환됨)

본 연구의 평가 코드는 **HR label만 ground truth로 사용**하며, LR 좌표는 SR 후 HR 공간으로 변환되어 비교됨.

## 클래스 정의

| ID | Name |
|:--|:--|
| 0 | ship |

`data.yaml`:
```yaml
nc: 1
names: ['ship']   # 또는 {0: 'ship'}
```

## 검증 포인트

- [ ] 모든 label이 5개 컬럼 (class + x + y + w + h)
- [ ] 좌표가 0~1 범위
- [ ] class_id가 모두 0
- [ ] subset6418의 모든 label 파일 non-empty
- [ ] HR/LR label이 같은 stem + 같은 내용
