# 01. Dataset

본 디렉토리는 SR-YOLO 연구에 사용된 데이터셋의 모든 정보를 담고 있다.
검사자는 다음 문서들을 순서대로 읽으며 데이터셋의 무결성을 검증한다.

| 문서 | 내용 |
|:--|:--|
| [01_source.md](01_source.md) | 데이터셋 출처, 다운로드 경로, 원본 정보 |
| [02_structure.md](02_structure.md) | 디렉토리 구조, 파일 개수, 이미지 사양 |
| [03_train_val_split.md](03_train_val_split.md) | Train/Val split 기준 |
| [04_label_format.md](04_label_format.md) | YOLO label format + 클래스 정의 |
| [05_subset_6418.md](05_subset_6418.md) | labeled_only subset 선정 기준 (메인 평가용) |

---

## 검증 체크리스트

검사자는 본 디렉토리 검토 후 다음을 확인:

- [ ] HR 이미지 위치, 개수, 해상도 명확
- [ ] LR 이미지 위치, 개수, 해상도 명확
- [ ] HR과 LR의 파일명/개수가 1:1 대응되는가
- [ ] Train/Val split 기준이 명시되어 있는가
- [ ] Label format이 standard YOLO format인가
- [ ] subset6418 선정 로직이 reproducible한가
- [ ] subset6418의 LR과 HR이 같은 stem(파일명 prefix)을 가지는가
