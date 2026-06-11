#!/bin/bash
# =============================================================================
# Robust v2 → Sensitivity 자동 파이프라인 (JSTAR급)
# =============================================================================
# - 정밀한 v2 완료 확인 (JSON 무결성 + 값 검증)
# - tmux/프로세스 죽음 자동 감지 + 재시작
# - CUDA 에러 자동 retry
# - Pre-flight check 강화
# - 단계별 상세 로깅
# =============================================================================

# set -e 제거: 에러가 나도 계속 진행하며 복구하기 위함
set -uo pipefail

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
SWEEP_DIR="${PROJECT_ROOT}/iac_runs/nms_sweep_2stage"
SENS_DIR="${PROJECT_ROOT}/iac_runs/nms_sensitivity"

LOG_DIR="${PROJECT_ROOT}/iac_runs/robust_pipeline_logs"
mkdir -p "${LOG_DIR}"

STATE_FILE="${LOG_DIR}/state.txt"
MAIN_LOG="${LOG_DIR}/main_$(date +%Y%m%d_%H%M%S).log"

# ─── Logging helpers ───
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$MAIN_LOG"
}

set_state() {
  echo "$1" > "$STATE_FILE"
  log "STATE: $1"
}

# ─── v2 sweep 완료 정밀 검증 ───
verify_v2_complete() {
  local all_ok=true
  for sr in drct hat man rfdn; do
    local summary="${SWEEP_DIR}/${sr}/stage2/summary.json"
    if [ ! -f "$summary" ]; then
      return 1
    fi

    # JSON 무결성
    if ! python3 -c "import json; json.load(open('$summary'))" 2>/dev/null; then
      log "WARN: ${sr} summary.json corrupt"
      return 2
    fi

    # 결과 10개 모두 있는지
    local n_results=$(python3 -c "import json; d=json.load(open('$summary')); print(len(d.get('results',[])))" 2>/dev/null)
    if [ -z "$n_results" ] || [ "$n_results" -lt 10 ]; then
      log "WARN: ${sr} only ${n_results}/10 results"
      return 3
    fi

    # mAP50 값이 합리적 (0.5 ~ 1.0)
    local best_map=$(python3 -c "import json; d=json.load(open('$summary')); print(d['results'][0]['mAP50'])" 2>/dev/null)
    if (( $(echo "$best_map < 0.5 || $best_map > 1.0" | bc -l 2>/dev/null) )); then
      log "WARN: ${sr} best mAP50=${best_map} suspicious"
      return 4
    fi

    # 개별 stage2 JSON 10개 무결성
    for f in "${SWEEP_DIR}/${sr}/stage2"/*.json; do
      [[ "$f" == *summary* ]] && continue
      if ! python3 -c "import json; json.load(open('$f'))" 2>/dev/null; then
        log "WARN: ${sr} ${f} corrupt"
        return 5
      fi
    done
  done

  return 0
}

# ─── nms_sweep tmux 죽음 감지 + 재시작 ───
check_and_restart_sweep() {
  if tmux has-session -t nms_sweep 2>/dev/null; then
    return 0
  fi

  log "ALERT: nms_sweep tmux DIED. 재시작 시도..."

  # GPU 회복 대기
  sleep 10

  tmux new-session -d -s nms_sweep \
    "stdbuf -oL bash ${PROJECT_ROOT}/iac_runs/run_nms_sweep_2stage.sh 2>&1 | tee -a ${LOG_DIR}/sweep_restart.log"

  sleep 3
  if tmux has-session -t nms_sweep 2>/dev/null; then
    log "nms_sweep 재시작 성공"
    return 0
  else
    log "ERROR: nms_sweep 재시작 실패"
    return 1
  fi
}

# ─── Pre-flight check ───
preflight_check() {
  log "Pre-flight check 시작..."

  # weight 파일
  for f in \
    weights/rfdn_arch4/model_best.pt \
    weights/sr_finetuned/drct/best.pt \
    weights/sr_finetuned/hat/best.pt \
    weights/sr_finetuned/man/best.pt \
    weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/weights/best.pt \
    weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt; do
    if [ ! -f "${PROJECT_ROOT}/$f" ]; then
      log "ERROR: 필수 weight 없음: $f"
      return 1
    fi
  done
  log "  필수 weight OK"

  # properbase sniper 확인
  for sr in drct hat man; do
    local sn=$(find ${PROJECT_ROOT}/weights/yolo_sniper_hardneg/ -maxdepth 1 -name "*hardneg_${sr}_pb*" -type d 2>/dev/null | sort -r | head -1)
    if [ -z "$sn" ] || [ ! -f "${sn}/weights/best.pt" ]; then
      log "ERROR: ${sr} properbase sniper 없음"
      return 1
    fi
  done
  log "  properbase sniper OK"

  # 디스크 공간 (최소 5GB)
  local free_gb=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')
  if [ "$free_gb" -lt 5 ]; then
    log "ERROR: 디스크 공간 부족 (${free_gb}GB 남음)"
    return 1
  fi
  log "  디스크 공간 OK (${free_gb}GB)"

  # GPU 메모리
  local gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$gpu_free" ]; then
    log "WARN: GPU 정보 가져오기 실패 (계속 진행)"
  elif [ "$gpu_free" -lt 1000 ]; then
    log "WARN: GPU 메모리 적음 (${gpu_free}MB free)"
  else
    log "  GPU 메모리 OK (${gpu_free}MB free)"
  fi

  log "Pre-flight check 통과"
  return 0
}

# ─── sensitivity 실행 with retry ───
run_sensitivity_with_retry() {
  local max_retries=5
  local attempt=0

  while [ $attempt -lt $max_retries ]; do
    attempt=$((attempt + 1))
    log "Sensitivity 시도 ${attempt}/${max_retries}"

    set_state "sensitivity_running_attempt_${attempt}"

    # 별도 tmux에서 실행
    tmux kill-session -t sensitivity 2>/dev/null || true
    tmux new-session -d -s sensitivity \
      "stdbuf -oL bash ${PROJECT_ROOT}/iac_runs/run_nms_sensitivity.sh 2>&1 | tee ${LOG_DIR}/sensitivity_attempt_${attempt}.log"

    sleep 3
    if ! tmux has-session -t sensitivity 2>/dev/null; then
      log "ERROR: sensitivity tmux 시작 실패. 30초 대기 후 재시도"
      sleep 30
      continue
    fi

    log "sensitivity tmux 시작 완료, 진행 모니터링..."

    # 모니터링: tmux 살아있는지 5분마다 확인
    while true; do
      sleep 300

      if ! tmux has-session -t sensitivity 2>/dev/null; then
        log "ALERT: sensitivity tmux DIED. 진행 상태 확인 중..."
        break
      fi

      # 진행 카운트
      local total=$(ls "$SENS_DIR"/*/[mf]*.json "$SENS_DIR"/*/r*.json 2>/dev/null | wc -l)
      log "  진행 중: ${total}/48 evals 완료"
    done

    # 완료 여부 확인 (48 evals)
    local n_done=$(ls "$SENS_DIR"/*/*.json 2>/dev/null | grep -v summary | wc -l)
    if [ "$n_done" -ge 48 ]; then
      log "Sensitivity 완료! (${n_done}/48 evals)"
      set_state "sensitivity_complete"
      return 0
    else
      log "Sensitivity 중단됨 (${n_done}/48). 재시작..."
      # GPU 회복 대기
      sleep 30
    fi
  done

  log "ERROR: sensitivity ${max_retries}회 시도 모두 실패"
  set_state "sensitivity_failed"
  return 1
}

# ═════════════════════════════════════════════
# 메인 로직
# ═════════════════════════════════════════════

log "========================================"
log "Robust Pipeline 시작"
log "========================================"

set_state "waiting_v2"

# 1. v2 완료 폴링 (with tmux 자동 재시작)
POLL_INTERVAL=300  # 5분
CHECK_COUNT=0

while true; do
  CHECK_COUNT=$((CHECK_COUNT + 1))
  log "v2 polling check #${CHECK_COUNT}"

  # v2 sweep tmux 죽었는지 체크
  if ! tmux has-session -t nms_sweep 2>/dev/null; then
    log "WARN: nms_sweep tmux 없음. 완료된 건지 죽은 건지 확인..."

    # 완료된 건지 체크
    if verify_v2_complete; then
      log "v2 자연 종료 + 검증 통과"
      break
    fi

    # 죽은 거면 재시작
    if ! check_and_restart_sweep; then
      log "ERROR: 재시작 실패. 30초 대기 후 재시도"
      sleep 30
      continue
    fi
  fi

  # v2 완료 검증
  if verify_v2_complete; then
    log "v2 완료 + 검증 통과"
    break
  fi

  sleep $POLL_INTERVAL
done

set_state "v2_complete"

# 2. v2 결과 출력
log ""
log "============================================================"
log " v2 최종 결과"
log "============================================================"
python3 << 'EOF' 2>&1 | tee -a "$MAIN_LOG"
import json
sweep = "/home/changmin/dark_vessel_sr_yolo/iac_runs/nms_sweep_2stage"
properbase = {"RFDN": 0.8007, "DRCT": 0.7973, "HAT": 0.7905, "MAN": 0.7918}
fair = {"RFDN": 0.7731, "DRCT": 0.7806, "HAT": 0.7733, "MAN": 0.7720}

print(f"\n{'SR':<6} {'Fair':>8} {'Properbase':>11} {'v2 best':>9} {'Config':<30}")
print("-" * 70)
for sr in ["RFDN", "DRCT", "HAT", "MAN"]:
    s2 = json.load(open(f"{sweep}/{sr.lower()}/stage2/summary.json"))
    b = s2["results"][0]
    print(f"{sr:<6} {fair[sr]:>8.4f} {properbase[sr]:>11.4f} {b['mAP50']:>9.4f} {b['tag']:<30}")
EOF

# 3. Pre-flight check
log ""
if ! preflight_check; then
  log "ERROR: Pre-flight check 실패. 사용자 개입 필요"
  set_state "preflight_failed"
  exit 1
fi

# 4. Sensitivity with retry
log ""
log "============================================================"
log " Sensitivity Ablation 시작"
log "============================================================"

if run_sensitivity_with_retry; then
  log "전체 성공!"
else
  log "Sensitivity 실패. 사용자 개입 필요"
  exit 1
fi

# 5. 종합 분석
log ""
log "============================================================"
log " 최종 종합 분석"
log "============================================================"

python3 << 'EOF' 2>&1 | tee -a "$MAIN_LOG"
import json, glob
from pathlib import Path

sweep = Path("/home/changmin/dark_vessel_sr_yolo/iac_runs/nms_sweep_2stage")
sens = Path("/home/changmin/dark_vessel_sr_yolo/iac_runs/nms_sensitivity")

properbase = {"RFDN": 0.8007, "DRCT": 0.7973, "HAT": 0.7905, "MAN": 0.7918}
fair = {"RFDN": 0.7731, "DRCT": 0.7806, "HAT": 0.7733, "MAN": 0.7720}

print("\n=== v2 + Sensitivity 종합 ===\n")
print(f"{'SR':<6} {'Fair':>8} {'Properbase':>11} {'v2':>8} {'Sens Range':>15} {'Sens Min':>9} {'Sens Max':>9}")
print("-" * 90)

for sr in ["RFDN", "DRCT", "HAT", "MAN"]:
    s2 = json.load(open(sweep / sr.lower() / "stage2" / "summary.json"))
    v2_best = s2["results"][0]["mAP50"]

    # Sensitivity 결과
    sens_files = list((sens / sr.lower()).glob("*.json"))
    sens_vals = []
    for f in sens_files:
        if "summary" in f.name: continue
        try:
            d = json.load(open(f))
            sens_vals.append(d["runs"][0]["results_dict"]["metrics/mAP50(B)"])
        except: pass

    if sens_vals:
        s_min, s_max = min(sens_vals), max(sens_vals)
        s_range = f"{(s_max-s_min)*100:+.2f}pp"
    else:
        s_min, s_max, s_range = 0, 0, "N/A"

    print(f"{sr:<6} {fair[sr]:>8.4f} {properbase[sr]:>11.4f} {v2_best:>8.4f} {s_range:>15} {s_min:>9.4f} {s_max:>9.4f}")

# Ranking robustness
print("\nRanking 비교:")
v2_rank = {}
for sr in ["RFDN","DRCT","HAT","MAN"]:
    s2 = json.load(open(sweep / sr.lower() / "stage2" / "summary.json"))
    v2_rank[sr] = s2["results"][0]["mAP50"]

print(f"  Fair:       {' > '.join(f'{k}({v:.4f})' for k,v in sorted(fair.items(), key=lambda x:-x[1]))}")
print(f"  Properbase: {' > '.join(f'{k}({v:.4f})' for k,v in sorted(properbase.items(), key=lambda x:-x[1]))}")
print(f"  v2:         {' > '.join(f'{k}({v:.4f})' for k,v in sorted(v2_rank.items(), key=lambda x:-x[1]))}")
EOF

log ""
log "========================================"
log "Robust Pipeline 완료: $(date)"
log "========================================"

set_state "all_complete"
