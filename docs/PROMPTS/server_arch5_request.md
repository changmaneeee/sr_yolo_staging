# 서버 (4090) 동기화 요청 Prompt

본 prompt는 4090 서버에서 작업 중인 agent/Changmin에게 전달되어, Arch5 관련 모든 자료를 본 staging repo로 가져오기 위함.

---

## 서버에서 해야 할 작업

### 1. staging_public repo clone

```bash
cd /home/changmin/dark_vessel_sr_yolo
# git A의 .gitignore에 의해 staging_public이 무시되므로 별도 clone
git clone https://github.com/changmaneeee/sr_yolo_staging.git staging_public_server
cd staging_public_server
```

또는 기존 디렉토리에서 pull:
```bash
cd /home/changmin/dark_vessel_sr_yolo/staging_public_server
git pull origin main
```

### 2. Arch5 관련 자료 추가

다음을 staging_public_server의 해당 위치로 복사:

#### A. Arch5 Phase 2 학습 자료
- 학습 log: `iac_lab/runs/{timestamp}_arch5b_drct_phase2/train_log.txt`
  → 본 repo의 `results/json/arch5/phase2_drct_log.txt`
- 학습 config: `iac_lab/runs/{timestamp}_arch5b_drct_phase2/config.yaml`
  → `results/json/arch5/phase2_drct_config.yaml`
- Phase 2 best mAP eval JSON (subset6418)
  → `results/json/arch5/phase2_drct_eval.json`

#### B. Arch5 Phase 3 학습 자료
- 학습 log: `iac_lab/runs/{timestamp}_arch5b_drct_phase3/train_log.txt`
  → `results/json/arch5/phase3_drct_log.txt`
- 학습 config: `iac_lab/runs/{timestamp}_arch5b_drct_phase3/config.yaml`
  → `results/json/arch5/phase3_drct_config.yaml`
- Phase 3 best mAP eval JSON (subset6418)
  → `results/json/arch5/phase3_drct_eval.json`

#### C. Phase 3 checkpoint 경로
- 절대 경로 (서버 기준): `/path/to/arch5b_drct_phase3_best.pt`
- 파일 크기: (MB)
- MD5 prefix: (12자리)
- → `docs/09_reproducibility/03_weight_locations.md`의 Arch5 섹션에 추가

#### D. Arch5 코드 (서버에서 사용한 버전)
- `src/models/pipelines/arch5b_fusion.py` (서버 버전)
- `src/models/fusion/attention_fusion.py`
- 학습 스크립트: `scripts/train_arch5_phase2.py`, `scripts/train_arch5_phase3.py`
- → 본 repo의 `code/arch5/`에 복사

#### E. 학습된 α 값 (learnable α)
- Phase 3 best checkpoint에서 α 추출
- 값을 `docs/08_results/04_paper_numbers.md`에 추가
- α 시간변화 (learning curve) → `results/figures_data/alpha_evolution.csv` (이미 있음)

### 3. RFDN Arch5b SOTA (0.9102) 자료

기존 staging에 이미 일부 있으나, 서버에서 확인:
- Phase 3 checkpoint MD5
- 학습 hyperparameter
- Small vessel recall +84% 측정 방식

### 4. HAT / MAN Arch5b (만약 진행 중이라면)

- Phase 2/3 학습 status
- 진행 시 동일하게 자료 정리

### 5. 변경 사항 commit & push

```bash
cd /home/changmin/dark_vessel_sr_yolo/staging_public_server

# 추가된 파일들
git add results/json/arch5/
git add code/arch5/
git add docs/  # 업데이트된 weight 경로 등

git commit -m "Add Arch5b (RFDN/DRCT) Phase 2+3 training assets from 4090 server

- Phase 2 logs, configs, eval JSONs
- Phase 3 logs, configs, eval JSONs
- Arch5 code (server version)
- Updated weight paths and learnable alpha values"

git push origin main
```

### 6. 본 (4060) repo에 pull

서버에서 push 후 본 측에서:
```bash
cd /home/changmin/dark_vessel_sr_yolo/staging_public
git pull origin main
```

---

## 형식 통일

### 결과 JSON 형식
arch4_eval_ultralytics.py 형식과 일치:
```json
{
  "meta": {
    "time": "...",
    "arch": "arch5b_fusion",
    "num_images": 6418,
    "avg_ms_per_image": 12.0,
    "device": "cuda:0 (RTX 4090)"
  },
  "runs": [{
    "tag": "ARCH5",
    "results_dict": {
      "metrics/precision(B)": ...,
      "metrics/recall(B)": ...,
      "metrics/mAP50(B)": 0.9102,
      "metrics/mAP50-95(B)": ...
    }
  }]
}
```

### Code 위치
- `code/arch5/arch5b_fusion.py`
- `code/arch5/attention_fusion.py`
- `code/arch5/train_phase2.py`
- `code/arch5/train_phase3.py`

---

## 주의사항

1. **Train/Val 분리**: Phase 2/3 모두 train set만 사용했음을 명시 (val 누설 없음 확인)
2. **subset6418 평가**: Arch5 결과도 동일한 6418장에서 측정 (Pre-flight check 추가 권장)
3. **Code 검사용 표시**: 서버 버전과 본 repo 버전이 동일한지 diff 확인
4. **Weights는 본 repo에 안 들어감**: 경로만 명시
5. **Notion 동기화**: 변경 사항을 Notion에도 기록

---

## 동기화 완료 확인

본 staging repo에 다음이 추가되었으면 완료:
- [ ] `results/json/arch5/phase2_drct_*` 파일
- [ ] `results/json/arch5/phase3_drct_*` 파일
- [ ] `code/arch5/` 디렉토리 (서버 코드 복사)
- [ ] `docs/09_reproducibility/03_weight_locations.md`에 Arch5 checkpoint 경로 추가
- [ ] `docs/07_experiments/05_arch5_results.md`에 정확한 수치 반영
- [ ] (선택) HAT/MAN Arch5 자료
