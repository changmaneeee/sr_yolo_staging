#!/bin/bash
# =============================================================================
# Old Pipeline 재실행 (전용 SR detector를 base로 사용)
# Phase A(crop dump)는 재사용, Phase B+C+D+E만 재실행
#
# 변경 요소 (이전 대비):
#   BASE_SNIPER: yolo_8s_rfdn → 각 SR 전용 detector
#     DRCT: weights/yolo_8s_drct/weights/best.pt
#     HAT:  weights/yolo_8s_hat/best.pt
#     MAN:  weights/yolo_8s_man/best.pt
# =============================================================================
set -euo pipefail

source /home/changmin/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel_mamba

export PYTHONUNBUFFERED=1

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# === 공통 경로 ===
HELPER="${PROJECT_ROOT}/iac_jetson/arch4_overnight_helper.py"
ARCH4_PY="${PROJECT_ROOT}/src/models/pipelines/arch4_roi_awareNMS_ablation.py"
WIRING="${PROJECT_ROOT}/iac_jetson/arch4_wiring_check.py"
ARCH4_EVAL="${PROJECT_ROOT}/iac_scripts/arch4_eval_ultralytics.py"
MINE_HARDNEG="${PROJECT_ROOT}/iac_jetson/mine_sniper_hard_negatives.py"
BUILD_HARDNEG="${PROJECT_ROOT}/iac_jetson/build_sniper_hardneg_dataset.py"
TRAIN_SNIPER="${PROJECT_ROOT}/iac_jetson/train_sniper_crop_yolo.py"
NEW_SCOUT="${PROJECT_ROOT}/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt"

LR_VAL="/home/changmin/smart_airbus_data_lr/images/val"
HR_VAL="/home/changmin/smart_airbus_data/images/val"
HR_LABELS_VAL="/home/changmin/smart_airbus_data/labels/val"
HR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr_data.yaml"
LR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr_data.yaml"

extract_prf() {
  python "${HELPER}" extract --json "$1" --mode prf 2>/dev/null || echo "parse_failed"
}

# =============================================================================
# run_one_sr: 단일 SR에 대해 Phase B~E 실행
# 인자: $1=SR_NAME $2=SR_WEIGHTS $3=BASE_SNIPER $4=ARCH4_CONFIG $5=CROP_DATA
# =============================================================================
run_one_sr() {
  local SR_NAME="$1"
  local SR_WEIGHTS="$2"
  local BASE_SNIPER="$3"
  local ARCH4_CONFIG="$4"
  local CROP_DATA="$5"

  local RUN_DIR="${PROJECT_ROOT}/iac_runs/${TIMESTAMP}_${SR_NAME}_properbase_pipeline"
  mkdir -p "${RUN_DIR}"

  local SNIPER_OUT="${PROJECT_ROOT}/weights/yolo_sniper_${SR_NAME}_properbase"

  echo ""
  echo "################################################################"
  echo "  ${SR_NAME^^} Old Pipeline (proper base detector)"
  echo "  BASE_SNIPER: ${BASE_SNIPER}"
  echo "  SR_WEIGHTS:  ${SR_WEIGHTS}"
  echo "  CROP_DATA:   ${CROP_DATA}"
  echo "  RUN_DIR:     ${RUN_DIR}"
  echo "################################################################"

  # Pre-flight
  for req in "${SR_WEIGHTS}" "${BASE_SNIPER}" "${ARCH4_CONFIG}" "${CROP_DATA}/data.yaml"; do
    if [ ! -f "${req}" ]; then
      echo "ERROR: missing ${req}"
      return 1
    fi
  done

  run_arch4_direct() {
    local scout_w="$1"
    local sniper_w="$2"
    local out_json="$3"
    python "${WIRING}" \
      --project_root "${PROJECT_ROOT}" \
      --arch4_config "${ARCH4_CONFIG}" \
      --arch4_py "${ARCH4_PY}" \
      --lr_images_dir "${LR_VAL}" \
      --hr_images_dir "${HR_VAL}" \
      --hr_labels_dir "${HR_LABELS_VAL}" \
      --max_images 0 \
      --device cuda \
      --half \
      --modes sr \
      --sniper_imgsz_mode fixed \
      --sniper_imgsz_fixed 256 \
      --sr_weights "${SR_WEIGHTS}" \
      --yolo_weights_lr "${scout_w}" \
      --yolo_weights_hr "${sniper_w}" \
      --out_json "${out_json}"
  }

  # ── Phase B: crop-ft (50 epoch) ──
  echo ""
  echo "========================================"
  echo "[${SR_NAME}] PHASE B: Crop-FT (base=${BASE_SNIPER##*/})"
  echo "========================================"

  local CROPFT_NAME="${TIMESTAMP}_cropft_${SR_NAME}_pb"
  local CROPFT_DIR="${SNIPER_OUT}/${CROPFT_NAME}"
  python "${TRAIN_SNIPER}" \
    --data "${CROP_DATA}/data.yaml" \
    --base_weights "${BASE_SNIPER}" \
    --imgsz 256 \
    --epochs 50 \
    --batch 32 \
    --patience 15 \
    --optimizer AdamW \
    --lr0 0.001 \
    --lrf 0.01 \
    --warmup_epochs 3 \
    --project "${SNIPER_OUT}" \
    --name "${CROPFT_NAME}" \
    --device 0 \
    --workers 0 \
    --save_period 10 \
    --amp true \
    --cache ram

  local CROPFT_BEST="${CROPFT_DIR}/weights/best.pt"
  if [ ! -f "${CROPFT_BEST}" ]; then
    echo "ERROR: missing crop-ft best weights ${CROPFT_BEST}"
    return 1
  fi

  run_arch4_direct "${NEW_SCOUT}" "${CROPFT_BEST}" "${RUN_DIR}/phase_b_cropft_direct.json"
  echo "[${SR_NAME}] cropft: $(extract_prf "${RUN_DIR}/phase_b_cropft_direct.json")"

  # ── Phase C: hardneg FT (25 epoch) ──
  echo ""
  echo "========================================"
  echo "[${SR_NAME}] PHASE C: Hard-negative FT"
  echo "========================================"

  python "${MINE_HARDNEG}" \
    --dataset_root "${CROP_DATA}" \
    --split train \
    --weights "${CROPFT_BEST}" \
    --out_csv "${RUN_DIR}/phase_c_hardneg_manifest.csv" \
    --out_json "${RUN_DIR}/phase_c_hardneg_summary.json" \
    --device 0 \
    --imgsz 256 \
    --batch 32 \
    --conf 0.001 \
    --iou 0.45 \
    --max_det 50 \
    --hardneg_thresh 0.25

  python "${BUILD_HARDNEG}" \
    --base_dataset_root "${CROP_DATA}" \
    --hardneg_csv "${RUN_DIR}/phase_c_hardneg_manifest.csv" \
    --out_dir "${RUN_DIR}/phase_c_manifest_dataset" \
    --hardneg_thresh 0.25 \
    --target_negative_ratio 0.30 \
    --max_extra_repeats 4

  local HARDNEG_NAME="${TIMESTAMP}_hardneg_${SR_NAME}_pb"
  local HARDNEG_DIR="${PROJECT_ROOT}/weights/yolo_sniper_hardneg/${HARDNEG_NAME}"
  python "${TRAIN_SNIPER}" \
    --data "${RUN_DIR}/phase_c_manifest_dataset/data.yaml" \
    --base_weights "${CROPFT_BEST}" \
    --imgsz 256 \
    --epochs 25 \
    --batch 16 \
    --patience 10 \
    --optimizer AdamW \
    --lr0 0.0005 \
    --lrf 0.01 \
    --warmup_epochs 2 \
    --project "${PROJECT_ROOT}/weights/yolo_sniper_hardneg" \
    --name "${HARDNEG_NAME}" \
    --device 0 \
    --workers 0 \
    --save_period 10 \
    --amp false

  local HARDNEG_BEST="${HARDNEG_DIR}/weights/best.pt"
  if [ ! -f "${HARDNEG_BEST}" ]; then
    echo "ERROR: missing hardneg best weights ${HARDNEG_BEST}"
    return 1
  fi

  run_arch4_direct "${NEW_SCOUT}" "${HARDNEG_BEST}" "${RUN_DIR}/phase_c_hardneg_direct.json"
  echo "[${SR_NAME}] hardneg: $(extract_prf "${RUN_DIR}/phase_c_hardneg_direct.json")"

  # ── Phase D: interpolation ──
  echo ""
  echo "========================================"
  echo "[${SR_NAME}] PHASE D: Interpolation"
  echo "========================================"

  local INTERP_DIR="${PROJECT_ROOT}/weights/yolo_sniper_interp_${SR_NAME}_pb/${TIMESTAMP}"
  mkdir -p "${INTERP_DIR}"

  for ALPHA in 0.20 0.30 0.40 0.50 0.60 0.70; do
    local TAG="a${ALPHA//./}"
    local OUT_W="${INTERP_DIR}/interp_${TAG}.pt"
    python "${HELPER}" interpolate \
      --ckpt-a "${CROPFT_BEST}" \
      --ckpt-b "${HARDNEG_BEST}" \
      --alpha "${ALPHA}" \
      --out "${OUT_W}" >/dev/null

    run_arch4_direct "${NEW_SCOUT}" "${OUT_W}" "${RUN_DIR}/phase_d_interp_${TAG}.json"
    echo "[${SR_NAME}] interp_${TAG}: $(extract_prf "${RUN_DIR}/phase_d_interp_${TAG}.json")"
  done

  # ── Phase E: choose best + bonus ──
  echo ""
  echo "========================================"
  echo "[${SR_NAME}] PHASE E: Choose best + bonus"
  echo "========================================"

  local BEST_JSON
  BEST_JSON="$(python "${HELPER}" choose-best \
    --glob "${RUN_DIR}/phase_b_cropft_direct.json" \
    --glob "${RUN_DIR}/phase_c_hardneg_direct.json" \
    --glob "${RUN_DIR}/phase_d_interp_*.json")"
  local BEST_TAG="$(basename "${BEST_JSON}" .json)"

  local BEST_SNIPER
  if [[ "${BEST_TAG}" == "phase_b_cropft_direct" ]]; then
    BEST_SNIPER="${CROPFT_BEST}"
  elif [[ "${BEST_TAG}" == "phase_c_hardneg_direct" ]]; then
    BEST_SNIPER="${HARDNEG_BEST}"
  else
    local SUFFIX="${BEST_TAG#phase_d_}"
    BEST_SNIPER="${INTERP_DIR}/${SUFFIX}.pt"
    if [ ! -f "${BEST_SNIPER}" ]; then
      BEST_SNIPER="${INTERP_DIR}/interp_${SUFFIX#interp_}.pt"
    fi
  fi

  for BONUS in 0.00 0.03 0.05; do
    local BTAG="bonus_${BONUS//./}"
    local CFG_OUT="${RUN_DIR}/${BTAG}.yaml"
    python "${HELPER}" patch-config \
      --base "${ARCH4_CONFIG}" \
      --out "${CFG_OUT}" \
      --set "model.arch4.sniper_score_bonus=${BONUS}" >/dev/null

    python "${WIRING}" \
      --project_root "${PROJECT_ROOT}" \
      --arch4_config "${CFG_OUT}" \
      --arch4_py "${ARCH4_PY}" \
      --lr_images_dir "${LR_VAL}" \
      --hr_images_dir "${HR_VAL}" \
      --hr_labels_dir "${HR_LABELS_VAL}" \
      --max_images 0 \
      --device cuda \
      --half \
      --modes sr \
      --sniper_imgsz_mode fixed \
      --sniper_imgsz_fixed 256 \
      --sr_weights "${SR_WEIGHTS}" \
      --yolo_weights_lr "${NEW_SCOUT}" \
      --yolo_weights_hr "${BEST_SNIPER}" \
      --out_json "${RUN_DIR}/phase_e_${BTAG}.json"
    echo "[${SR_NAME}] ${BTAG}: $(extract_prf "${RUN_DIR}/phase_e_${BTAG}.json")"
  done

  local BEST_FINAL_JSON
  BEST_FINAL_JSON="$(python "${HELPER}" choose-best \
    --glob "${BEST_JSON}" \
    --glob "${RUN_DIR}/phase_e_bonus_*.json")"
  local BEST_FINAL_TAG="$(basename "${BEST_FINAL_JSON}" .json)"

  # Final mAP eval
  local FINAL_CFG
  if [[ "${BEST_FINAL_TAG}" == phase_e_bonus_* ]]; then
    FINAL_CFG="${RUN_DIR}/${BEST_FINAL_TAG#phase_e_}.yaml"
  else
    FINAL_CFG="${ARCH4_CONFIG}"
  fi

  python "${HELPER}" patch-config \
    --base "${FINAL_CFG}" \
    --out "${RUN_DIR}/final_map.yaml" \
    --set "model.yolo.weights_lr=${NEW_SCOUT}" \
    --set "model.yolo.weights_hr=${BEST_SNIPER}" >/dev/null

  python "${ARCH4_EVAL}" \
    --arch4_config "${RUN_DIR}/final_map.yaml" \
    --hr_data_yaml "${HR_DATA_YAML}" \
    --lr_data_yaml "${LR_DATA_YAML}" \
    --eval_space hr \
    --max_images 0 \
    --device cuda \
    --batch 1 \
    --out_json "${RUN_DIR}/final_map_result.json"

  echo ""
  echo "========================================"
  echo "[${SR_NAME}] FINAL SUMMARY"
  echo "========================================"
  echo "SR: ${SR_NAME}"
  echo "BASE_SNIPER: ${BASE_SNIPER}"
  echo "cropft:  $(extract_prf "${RUN_DIR}/phase_b_cropft_direct.json")"
  echo "hardneg: $(extract_prf "${RUN_DIR}/phase_c_hardneg_direct.json")"
  for f in "${RUN_DIR}"/phase_d_interp_*.json; do
    [ -f "$f" ] && echo "$(basename "$f" .json): $(extract_prf "$f")"
  done
  for f in "${RUN_DIR}"/phase_e_bonus_*.json; do
    [ -f "$f" ] && echo "$(basename "$f" .json): $(extract_prf "$f")"
  done
  echo "BEST: ${BEST_FINAL_TAG}"

  # mAP 출력
  python -c "
import json
d = json.load(open('${RUN_DIR}/final_map_result.json'))
r = d['runs'][0]['results_dict']
print(f'mAP50={r[\"metrics/mAP50(B)\"]:.4f} P={r[\"metrics/precision(B)\"]:.4f} R={r[\"metrics/recall(B)\"]:.4f}')
" 2>/dev/null

  echo "RUN_DIR: ${RUN_DIR}"
  echo "DONE: $(date)"
}


# =============================================================================
# MAIN: DRCT → HAT → MAN 순차 실행
# =============================================================================

echo "============================================================"
echo " Arch4 Old Pipeline — Proper Base Detector (Phase B~E only)"
echo " Started: $(date)"
echo " Phase A crops: reused from previous runs"
echo "============================================================"

# ── DRCT ── (SKIP: completed 2026-06-09 14:17, best F1=0.7663)
DRCT_EXIT=0
echo "[SKIP] DRCT already completed. Best bonus_000 F1=0.7663"

# ── HAT ──
run_one_sr "hat" \
  "${PROJECT_ROOT}/weights/sr_finetuned/hat/best.pt" \
  "${PROJECT_ROOT}/weights/yolo_8s_hat/best.pt" \
  "${PROJECT_ROOT}/iac_runs/arch4_hat_old_pipeline_config.yaml" \
  "${PROJECT_ROOT}/data/arch4_sniper_crops_hat_old"
HAT_EXIT=$?

# ── MAN ──
run_one_sr "man" \
  "${PROJECT_ROOT}/weights/sr_finetuned/man/best.pt" \
  "${PROJECT_ROOT}/weights/yolo_8s_man/best.pt" \
  "${PROJECT_ROOT}/iac_runs/arch4_man_old_pipeline_config.yaml" \
  "${PROJECT_ROOT}/data/arch4_sniper_crops_man_old"
MAN_EXIT=$?

echo ""
echo "============================================================"
echo " HAT+MAN DONE"
echo " DRCT exit: ${DRCT_EXIT}"
echo " HAT exit:  ${HAT_EXIT}"
echo " MAN exit:  ${MAN_EXIT}"
echo " Finished:  $(date)"
echo "============================================================"

# ================================================================
# DRCT mAP eval (크래시 복구) + 전체 비교 테이블
# ================================================================
echo ""
echo "============================================================"
echo " [추가] DRCT properbase mAP eval (크래시 복구)"
echo "============================================================"

DRCT_PB_DIR="${PROJECT_ROOT}/iac_runs/20260609_070148_drct_properbase_pipeline"

if [ -f "${DRCT_PB_DIR}/final_map.yaml" ] && [ ! -f "${DRCT_PB_DIR}/final_map_result.json" ]; then
  echo "Running DRCT mAP eval..."
  python "${ARCH4_EVAL}" \
    --arch4_config "${DRCT_PB_DIR}/final_map.yaml" \
    --hr_data_yaml "${HR_DATA_YAML}" \
    --lr_data_yaml "${LR_DATA_YAML}" \
    --eval_space hr \
    --max_images 0 \
    --device cuda \
    --batch 1 \
    --out_json "${DRCT_PB_DIR}/final_map_result.json"

  echo "DRCT mAP eval done."
  python -c "
import json
d = json.load(open('${DRCT_PB_DIR}/final_map_result.json'))
r = d['runs'][0]['results_dict']
print(f'DRCT properbase: mAP50={r[\"metrics/mAP50(B)\"]:.4f} P={r[\"metrics/precision(B)\"]:.4f} R={r[\"metrics/recall(B)\"]:.4f}')
" 2>/dev/null
else
  if [ -f "${DRCT_PB_DIR}/final_map_result.json" ]; then
    echo "DRCT mAP already exists, skipping."
  else
    echo "WARNING: DRCT final_map.yaml not found!"
  fi
fi

# ================================================================
# 전체 비교 테이블 생성
# ================================================================
echo ""
echo "============================================================"
echo " PROPERBASE vs RFDN-BASE 전체 비교"
echo "============================================================"

python3 << 'COMPARE_EOF'
import json, glob
from pathlib import Path

PROJECT = Path("/home/changmin/dark_vessel_sr_yolo")

# rfdn base (1차) 결과
rfdn_base = {
    "RFDN": {"mAP50": 0.8007, "src": "기존 완료 (2026-03-25)"},
    "DRCT": {"dir": "20260607_200010_drct_old_sniper_pipeline"},
    "HAT":  {"dir": "20260608_123941_hat_old_sniper_pipeline"},
    "MAN":  {"dir": "20260608_221354_man_old_sniper_pipeline"},
}

# properbase (2차) 결과
proper_base = {
    "DRCT": {"dir": "20260609_070148_drct_properbase_pipeline"},
}

# HAT/MAN properbase dirs (just created)
for d in sorted(Path(PROJECT / "iac_runs").glob("*_hat_properbase_pipeline")):
    proper_base["HAT"] = {"dir": d.name}
for d in sorted(Path(PROJECT / "iac_runs").glob("*_man_properbase_pipeline")):
    proper_base["MAN"] = {"dir": d.name}

def get_map50(run_dir_name):
    """Extract mAP50 from final_map_result.json"""
    f = PROJECT / "iac_runs" / run_dir_name / "final_map_result.json"
    if f.exists():
        d = json.load(open(f))
        r = d["runs"][0]["results_dict"]
        return r.get("metrics/mAP50(B)", None)
    # Try bonus_000_map.json as fallback
    f2 = PROJECT / "iac_runs" / run_dir_name / "phase_e_bonus_000_map.json"
    if f2.exists():
        d = json.load(open(f2))
        r = d["runs"][0]["results_dict"]
        return r.get("metrics/mAP50(B)", None)
    return None

def get_best_f1(run_dir_name):
    """Extract best F1 from phase_e results"""
    best_f1 = 0
    best_tag = ""
    for f in sorted(glob.glob(str(PROJECT / "iac_runs" / run_dir_name / "phase_e_bonus_*.json"))):
        d = json.load(open(f))
        r = d.get("results", {}).get("sr", {})
        f1 = r.get("f1_50_direct", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_tag = Path(f).stem
    return best_f1, best_tag

# Fair eval baselines
fair = {"RFDN": 0.7731, "DRCT": 0.7806, "HAT": 0.7733, "MAN": 0.7720}

print()
print(f"{'SR':<6} {'fair':>8} {'rfdn_base':>10} {'proper_base':>12} {'delta_rb':>9} {'delta_pb':>9}")
print("-" * 60)

for sr in ["RFDN", "DRCT", "HAT", "MAN"]:
    fair_val = fair[sr]

    # rfdn base mAP
    if sr == "RFDN":
        rb_map = 0.8007
    elif sr in rfdn_base and "dir" in rfdn_base[sr]:
        rb_map = get_map50(rfdn_base[sr]["dir"])
    else:
        rb_map = None

    # properbase mAP
    if sr == "RFDN":
        pb_map = 0.8007  # same (RFDN base = proper base for RFDN)
    elif sr in proper_base and "dir" in proper_base[sr]:
        pb_map = get_map50(proper_base[sr]["dir"])
    else:
        pb_map = None

    rb_str = f"{rb_map:.4f}" if rb_map else "---"
    pb_str = f"{pb_map:.4f}" if pb_map else "진행중"
    d_rb = f"+{(rb_map-fair_val)*100:.2f}pp" if rb_map else "---"
    d_pb = f"+{(pb_map-fair_val)*100:.2f}pp" if pb_map else "---"

    print(f"{sr:<6} {fair_val:>8.4f} {rb_str:>10} {pb_str:>12} {d_rb:>9} {d_pb:>9}")

print()
print("fair = eval_arch.py (fixed_protocol), rfdn_base = 1차 old pipeline (잘못된 base)")
print("proper_base = 2차 old pipeline (각 SR 전용 detector base)")
COMPARE_EOF

echo ""
echo "============================================================"
echo " Properbase + DRCT mAP 완료: $(date)"
echo " NMS Sweep 시작..."
echo "============================================================"

# ── NMS Sweep 자동 연계 ──
bash "${PROJECT_ROOT}/iac_runs/run_nms_sweep_properbase.sh"

echo ""
echo "============================================================"
echo " 전체 완료 (properbase + NMS sweep): $(date)"
echo "============================================================"
