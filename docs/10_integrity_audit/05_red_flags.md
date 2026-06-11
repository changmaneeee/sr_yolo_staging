# 10-05. Red Flags (검사자가 의심해야 할 항목)

검사자는 다음 항목들이 정말 보장되는지 의심하며 확인.

## 의심 1: 데이터셋 조작?

- HR/LR 이미지가 정말 같은 dataset인가? stem 매칭 100%?
- subset6418이 정말 라벨 있는 것만? size > 0 검증?
- 28883 vs 28884 차이가 정말 무해한가?

→ 확인 방법: `docs/01_dataset/02_structure.md`의 검증 명령 실행.

## 의심 2: 같은 model을 SR이라 부르는가?

- RFDN과 DRCT가 정말 다른 architecture인가?
- 4개 wrapper가 정말 별도 model 정의?

→ 확인 방법: 각 wrapper 코드 직접 확인 (`code/shared/sr_backbones/`).

## 의심 3: Sniper가 SR별로 정말 다른가?

- 각 SR의 Sniper weight 파일이 정말 다른가? (size, MD5)
- 같은 데이터로 학습된 것을 단순 복사한 게 아닌가?

→ 확인 방법:
```bash
md5sum /home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/20260609_070148_hardneg_drct_pb/weights/best.pt
md5sum /home/changmin/dark_vessel_sr_yolo/weights/yolo_sniper_hardneg/20260609_*_hardneg_hat_pb/weights/best.pt
# → 다른 MD5여야 함
```

## 의심 4: 평가 set의 일관성

- 정말 모든 실험이 같은 6418장에서 측정되었나?
- 평가 중 일부 이미지를 제외하지 않았는가?

→ 확인 방법: 각 결과 JSON의 `num_images: 6418` 확인.

## 의심 5: NMS 파라미터가 결과에 큰 영향?

- v2 best가 정말 grid search에서 정직하게 선정되었나?
- Cherry-picking 가능성?

→ 확인 방법:
- `iac_runs/nms_sweep_2stage/{sr}/stage2/summary.json` 전체 Top-10 확인
- 1위와 2위, 3위의 차이가 작은지 확인 (작으면 cherry-pick 위험)

## 의심 6: Run-to-run variance가 실제로 측정됐나?

- DRCT 5회 + RFDN 5회 측정 결과가 존재?
- 변동 폭이 정말 < 0.5pp?

→ 확인 방법: Task #63의 결과 JSON 확인.

## 의심 7: Arch5의 SOTA 결과가 정직?

- 0.9102가 정말 subset6418에서 측정?
- 다른 set (전체 28883)에서 측정한 것 아닌가?
- 학습 시 val 누설은 없었나?

→ 확인 방법: Arch5 eval 결과 JSON의 `num_images` 확인.

## 의심 8: 1차 결과(rfdn base, v1)가 정말 폐기됐나?

- 잘못된 결과가 paper에 들어가지 않는다는 보장?

→ 확인 방법:
- `docs/08_results/04_paper_numbers.md`에서 사용되는 수치가 모두 2차/v2 출처인지
- `iac_runs/nms_sweep_2stage_v1_missing_params/` 디렉토리는 백업으로만 존재

## 의심 9: 코드 수정 후 재실행 안 한 경우?

- fix 후에도 같은 weight로 평가?
- 수정 후 결과가 정말 새로 측정된 것?

→ 확인 방법: 각 결과 JSON의 timestamp 확인.

## 의심 10: 공정성 보장이 정말 코드 레벨?

- "공정 비교"라 주장하지만, 코드에서 실제로 같은 path를 거치는가?

→ 확인 방법:
- `eval_arch.py`의 분기점 확인
- 4 SR이 같은 함수 통과하는지

## 검사 보고서 권장 구조

```
# Integrity Audit Report

## Summary
- Overall: PASS / WARN / FAIL
- 주요 발견: ...

## Section-by-Section
- A. Dataset: PASS (all 8 items)
- B. Preprocessing: PASS (5/5)
- ...

## Red Flag Investigations
- Red flag 1: ... (결과)
- ...

## Recommendations
- ...

## Conclusion
- 본 연구는 paper 제출에 ... 함.
```
