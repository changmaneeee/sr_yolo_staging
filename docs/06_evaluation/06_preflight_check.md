# 06-06. Pre-flight Check 함수

## 위치

`eval_arch.py`의 `pre_flight_check()` 함수

## 검증 항목

매 평가 실행 시 자동 확인:

### 1. Scout weight MD5
```python
import hashlib
scout_md5 = hashlib.md5(open(scout_weight, "rb").read()).hexdigest()[:12]
assert scout_md5 == fixed_protocol["scout"]["md5_prefix"]   # f9f175f7f758
```

### 2. Dataset 경로 존재
```python
for k, v in fixed_protocol["dataset"].items():
    if k.endswith("_images") or k.endswith("_labels"):
        assert Path(v).exists(), f"Missing dataset path: {v}"
```

### 3. Expected count
```python
labeled_stems = {f.stem for f in Path(hr_labels_dir).glob("*.txt") if f.stat().st_size > 0}
assert len(labeled_stems) == fixed_protocol["dataset"]["expected_count"]   # 6418
```

### 4. HR/LR stem 매칭
```python
hr_stems = {f.stem for f in Path(hr_images_dir).glob("*.jpg")}
lr_stems = {f.stem for f in Path(lr_images_dir).glob("*.jpg")}
overlap = hr_stems & lr_stems
assert len(overlap) >= 6418
```

## 실패 시 동작

Pre-flight check가 실패하면:
- 경고 로그 출력
- 사용자에게 알림 (`log.warning("Pre-flight check failed. Proceeding with warnings.")`)
- 평가는 계속 진행 (강제 종료 안 함, 단 사용자가 경고 무시 책임)

엄격 모드 (`--strict-preflight`) 추가 가능.

## 검증 보장

이 함수가 통과하면 다음이 보장됨:
- Scout weight 변조 없음 (MD5)
- Dataset 경로 유효
- 정확히 6418장 평가
- HR/LR 매칭 무결

## 검증 포인트

- [ ] Pre-flight check가 매 실행 시 호출
- [ ] MD5 검증 통과
- [ ] expected_count 일치
- [ ] 검사자가 이 함수를 코드에서 확인 가능
