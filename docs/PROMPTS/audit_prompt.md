# 검사 모델(Audit Agent)에게 전달할 Prompt

본 prompt를 검사 모델에 전달하여 staging_public repo의 무결성을 종합 검증.

---

## Audit Agent Prompt

```
당신은 SR-YOLO 연구의 무결성 검사관입니다. 본 repository (/path/to/staging_public/)를
처음부터 끝까지 읽고 다음을 검증합니다.

목표: JSTAR (IEEE Q1) 저널 제출 전 모든 결과의 reproducibility, fairness, integrity를 확인.

## 검사 순서

1. README.md를 먼저 읽고 전체 구조 이해
2. docs/00_overview.md로 프로젝트 큰 그림 파악
3. docs/10_integrity_audit/01_audit_checklist.md의 모든 항목을 체크
4. docs/10_integrity_audit/05_red_flags.md의 모든 의심 항목을 적극 검증
5. 코드(code/)와 문서(docs/)의 일치성 확인
6. 결과(results/json/)와 paper 수치(docs/08_results/)의 일치성 확인

## 보고 형식

각 검증 축마다 다음 형식으로 보고:

### A. Dataset
- A1 (HR/LR dataset 경로 존재): [PASS/WARN/FAIL] + 근거
- A2 (해상도 확인): ...
...

### B. Preprocessing
...

(전체 11개 축 A~K)

## 종합 평가

- Overall: [PASS/WARN/FAIL]
- 발견된 issue 개수:
- Paper 제출 권장 여부:
- 추가 작업 권장:

## 우선순위가 높은 issue

- 가장 critical한 issue 3개 (있다면):
  1. ...
  2. ...
  3. ...

## 권장 사항

본 paper가 JSTAR급에 적합한지, 추가 검증이 필요한지 등.
```

---

## 검사 모델이 할 일

1. **모든 md 파일을 읽음** (40개+)
2. **code 디렉토리의 코드를 확인** — 문서와 일치하는가?
3. **results JSON을 sample로 확인** — 수치가 paper와 일치하는가?
4. **검증 명령을 실행 가능하면 실행** — 데이터셋, weight, MD5 검증

## 검사 모델이 받을 자료

본 staging_public repo 전체. 추가로:
- 메인 폴더(`/home/changmin/dark_vessel_sr_yolo/`)에 대한 접근 (weights, code 검증용)
- 4090 서버에서 동기화된 Arch5 자료

## 검사 결과의 처리

검사 결과 보고서는:
1. `docs/10_integrity_audit/06_audit_report.md`에 저장 (검사 후 추가)
2. Notion에 동기화
3. 발견된 issue는 fix 후 재검증

---

## Sample 검증 명령 (검사 모델 참고용)

```bash
# 1. Dataset 검증
ls /home/changmin/smart_airbus_data/images/val/ | wc -l   # 28883
ls /home/changmin/dark_vessel_sr_yolo/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr/images/val/ | wc -l   # 6418

# 2. Weight 검증
md5sum /home/changmin/dark_vessel_sr_yolo/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt | cut -c1-12
# → f9f175f7f758

# 3. 결과 일치 확인
python3 -c "
import json
d = json.load(open('/home/changmin/dark_vessel_sr_yolo/iac_runs/nms_sweep_2stage/hat/stage2/summary.json'))
print(d['results'][0]['mAP50'])   # 0.8003
"

# 4. 코드 검증
python3 -c "
import sys; sys.path.insert(0, '/home/changmin/dark_vessel_sr_yolo')
from eval_arch import _load_sr_model
# HAT을 RFDN loader로 로드 시도 → 에러 발생 (강제 검증)
"
```
