#!/bin/bash
# =============================================================================
# NMS Sensitivity Ablation (JSTAR급 robustness 검증)
# =============================================================================
# 각 SR의 v2 best config에서 핵심 4개 파라미터를 ±변동
# → ranking이 robust한지 확인 (paper supplementary용)
#
# 4 SR × 4 params × 3 values = 48 evals × 6418장 ≈ ~10h
# =============================================================================
set -euo pipefail

source /home/changmin/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel_mamba

export PYTHONUNBUFFERED=1

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
SWEEP_DIR="${PROJECT_ROOT}/iac_runs/nms_sweep_2stage"
SENS_DIR="${PROJECT_ROOT}/iac_runs/nms_sensitivity"
mkdir -p "${SENS_DIR}"

EVAL_SCRIPT="${PROJECT_ROOT}/iac_scripts/arch4_eval_ultralytics.py"
HR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr_data.yaml"
LR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr_data.yaml"
SCOUT="${PROJECT_ROOT}/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt"

declare -A SR_WEIGHT
declare -A SNIPER_WEIGHT

SR_WEIGHT[rfdn]="${PROJECT_ROOT}/weights/rfdn_arch4/model_best.pt"
SR_WEIGHT[drct]="${PROJECT_ROOT}/weights/sr_finetuned/drct/best.pt"
SR_WEIGHT[hat]="${PROJECT_ROOT}/weights/sr_finetuned/hat/best.pt"
SR_WEIGHT[man]="${PROJECT_ROOT}/weights/sr_finetuned/man/best.pt"

SNIPER_WEIGHT[rfdn]="${PROJECT_ROOT}/weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/weights/best.pt"
SNIPER_WEIGHT[drct]=$(find ${PROJECT_ROOT}/weights/yolo_sniper_hardneg/ -maxdepth 1 -name "*hardneg_drct_pb*" -type d | sort -r | head -1)/weights/best.pt
SNIPER_WEIGHT[hat]=$(find ${PROJECT_ROOT}/weights/yolo_sniper_hardneg/ -maxdepth 1 -name "*hardneg_hat_pb*" -type d | sort -r | head -1)/weights/best.pt
SNIPER_WEIGHT[man]=$(find ${PROJECT_ROOT}/weights/yolo_sniper_hardneg/ -maxdepth 1 -name "*hardneg_man_pb*" -type d | sort -r | head -1)/weights/best.pt

echo "============================================================"
echo " NMS Sensitivity Ablation"
echo " Started: $(date)"
echo "============================================================"

# v2 best config 추출
declare -A BEST_P1
declare -A BEST_ROI

for sr in drct hat man rfdn; do
  s2="${SWEEP_DIR}/${sr}/stage2/summary.json"
  if [ ! -f "$s2" ]; then
    echo "ERROR: ${sr} v2 summary 없음. v2 완료 후 실행하세요."
    exit 1
  fi
  best_tag=$(python3 -c "import json; d=json.load(open('$s2')); print(d['results'][0]['tag'])")
  BEST_P1[$sr]=$(echo "$best_tag" | sed 's/p1\([^_]*\)_.*/\1/')
  BEST_ROI[$sr]=$(echo "$best_tag" | sed 's/.*roi\([^_]*\)_.*/\1/')
  echo "  ${sr} v2 best: p1=${BEST_P1[$sr]}, roi=${BEST_ROI[$sr]}"
done

# Sensitivity grid (각 파라미터 ±)
declare -A PARAM_VALUES
PARAM_VALUES[max_det]="2 3 5"
PARAM_VALUES[final_conf]="0.20 0.25 0.30"
PARAM_VALUES[roi_small_thresh]="16 32 48"
PARAM_VALUES[replace_margin]="0.05 0.10 0.20"

make_config() {
  local OUT="$1" SR="$2" P1="$3" ROI="$4"
  local FINAL_CONF="${5:-0.25}"
  local MAX_DET="${6:-3}"
  local ROI_SMALL="${7:-32.0}"
  local REPLACE_MARGIN="${8:-0.1}"

  cat > "$OUT" << YAMLEOF
data: {upscale_factor: 4}
model:
  sr:
    type: ${SR}
    weights: ${SR_WEIGHT[$SR]}
    rfdn: {nf: 50, num_modules: 4}
  yolo:
    weights_lr: ${SCOUT}
    weights_hr: ${SNIPER_WEIGHT[$SR]}
    classes: 1
    num_classes: 1
  arch4:
    pass1_conf: ${P1}
    high_conf: 0.45
    pass2_conf: 0.45
    final_conf: ${FINAL_CONF}
    sniper_conf: 0.001
    merge_iou: 0.5
    roi_expansion: ${ROI}
    crop_size_lr: 64
    batch_size_sr: 32
    scout_nms_iou: 0.5
    roi_merge_iou: 0.3
    roi_center_ratio: 0.35
    sniper_nms_iou: 0.45
    final_nms_iou: 0.5
    drop_uncertain_if_sniper_hits: true
    sniper_score_bonus: 0.0
    merge_policy: size_cond
    final_fusion_method: soft_nms
    soft_nms_sigma: 0.3
    roi_small_thresh: ${ROI_SMALL}
    roi_large_thresh: 96.0
    large_roi_score_thresh: 0.5
    sniper_replace_margin: ${REPLACE_MARGIN}
    sniper_max_det_per_crop: ${MAX_DET}
YAMLEOF
}

run_eval() {
  python "${EVAL_SCRIPT}" \
    --arch4_config "$1" --hr_data_yaml "${HR_DATA_YAML}" --lr_data_yaml "${LR_DATA_YAML}" \
    --eval_space hr --max_images 0 --device cuda --batch 1 --out_json "$2" 2>/dev/null
}

# ─── 각 SR × 각 파라미터 sensitivity ───
for sr in drct hat man rfdn; do
  SR_SENS_DIR="${SENS_DIR}/${sr}"
  mkdir -p "${SR_SENS_DIR}"

  p1="${BEST_P1[$sr]}"
  roi="${BEST_ROI[$sr]}"

  echo ""
  echo "════════════════════════════════════════════"
  echo "  ${sr^^} Sensitivity (best: p1=${p1}, roi=${roi})"
  echo "════════════════════════════════════════════"

  # max_det sensitivity (final_conf=0.25, roi_small=32, replace=0.1 고정)
  for md in 2 3 5; do
    TAG="md${md}"
    CFG="${SR_SENS_DIR}/${TAG}.yaml"
    OUT="${SR_SENS_DIR}/${TAG}.json"
    [ -f "$OUT" ] && { echo "  [max_det=${md}] skip"; continue; }
    make_config "$CFG" "$sr" "$p1" "$roi" "0.25" "$md" "32.0" "0.1"
    echo -n "  [max_det=${md}] ... "
    run_eval "$CFG" "$OUT"
    python3 -c "import json; d=json.load(open('$OUT')); print(f'mAP50={d[\"runs\"][0][\"results_dict\"][\"metrics/mAP50(B)\"]:.4f}')" 2>/dev/null
  done

  # final_conf sensitivity
  for fc in 0.20 0.25 0.30; do
    TAG="fc${fc}"
    CFG="${SR_SENS_DIR}/${TAG}.yaml"
    OUT="${SR_SENS_DIR}/${TAG}.json"
    [ -f "$OUT" ] && { echo "  [final_conf=${fc}] skip"; continue; }
    make_config "$CFG" "$sr" "$p1" "$roi" "$fc" "3" "32.0" "0.1"
    echo -n "  [final_conf=${fc}] ... "
    run_eval "$CFG" "$OUT"
    python3 -c "import json; d=json.load(open('$OUT')); print(f'mAP50={d[\"runs\"][0][\"results_dict\"][\"metrics/mAP50(B)\"]:.4f}')" 2>/dev/null
  done

  # roi_small_thresh sensitivity
  for rst in 16 32 48; do
    TAG="rst${rst}"
    CFG="${SR_SENS_DIR}/${TAG}.yaml"
    OUT="${SR_SENS_DIR}/${TAG}.json"
    [ -f "$OUT" ] && { echo "  [roi_small=${rst}] skip"; continue; }
    make_config "$CFG" "$sr" "$p1" "$roi" "0.25" "3" "${rst}.0" "0.1"
    echo -n "  [roi_small=${rst}] ... "
    run_eval "$CFG" "$OUT"
    python3 -c "import json; d=json.load(open('$OUT')); print(f'mAP50={d[\"runs\"][0][\"results_dict\"][\"metrics/mAP50(B)\"]:.4f}')" 2>/dev/null
  done

  # replace_margin sensitivity
  for rm in 0.05 0.10 0.20; do
    TAG="rm${rm}"
    CFG="${SR_SENS_DIR}/${TAG}.yaml"
    OUT="${SR_SENS_DIR}/${TAG}.json"
    [ -f "$OUT" ] && { echo "  [replace_margin=${rm}] skip"; continue; }
    make_config "$CFG" "$sr" "$p1" "$roi" "0.25" "3" "32.0" "$rm"
    echo -n "  [replace_margin=${rm}] ... "
    run_eval "$CFG" "$OUT"
    python3 -c "import json; d=json.load(open('$OUT')); print(f'mAP50={d[\"runs\"][0][\"results_dict\"][\"metrics/mAP50(B)\"]:.4f}')" 2>/dev/null
  done
done

# ─── Sensitivity 분석 + 최종 비교표 ───
echo ""
echo "============================================================"
echo " Sensitivity Analysis 결과"
echo "============================================================"

python3 << 'EOF'
import json, glob
from pathlib import Path

sens_dir = Path("/home/changmin/dark_vessel_sr_yolo/iac_runs/nms_sensitivity")
sweep_dir = Path("/home/changmin/dark_vessel_sr_yolo/iac_runs/nms_sweep_2stage")

# v2 best
v2_best = {}
for sr in ["drct", "hat", "man", "rfdn"]:
    s2 = sweep_dir / sr / "stage2" / "summary.json"
    if s2.exists():
        d = json.load(open(s2))
        v2_best[sr] = d["results"][0]["mAP50"]

print(f"\n{'SR':<6} {'v2 best':>10} {'Max Δ':>8} {'Range':>10}")
print("-" * 50)

for sr in ["rfdn", "drct", "hat", "man"]:
    results = {}
    for f in glob.glob(str(sens_dir / sr / "*.json")):
        try:
            d = json.load(open(f))
            r = d["runs"][0]["results_dict"]
            tag = Path(f).stem
            results[tag] = r["metrics/mAP50(B)"]
        except: pass

    if results:
        vals = list(results.values())
        max_delta = max(vals) - min(vals)
        print(f"{sr.upper():<6} {v2_best.get(sr,0):>10.4f} {max_delta:>+7.2%}pp {min(vals):.4f}~{max(vals):.4f}")

print()
print("Ranking robustness check:")
# For each parameter variation, check if v2 ranking is preserved
EOF

echo ""
echo "============================================================"
echo " 전체 완료: $(date)"
echo "============================================================"
