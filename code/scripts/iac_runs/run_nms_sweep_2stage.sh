#!/bin/bash
# =============================================================================
# Section IV NMS Sweep — 2-Stage (RFDN sweep과 동일 프로토콜)
# =============================================================================
# Stage 1: 200장 quick scan → 120 combos (~20s each → ~40min/SR)
# Stage 2: Top-10 → 6418장 full eval (~12min each → ~2h/SR)
#
# 순서: DRCT → HAT → (sigma 영향 확인) → MAN/RFDN
# =============================================================================
set -euo pipefail

source /home/changmin/miniconda3/etc/profile.d/conda.sh
conda activate dark_vessel_mamba

export PYTHONUNBUFFERED=1

PROJECT_ROOT="/home/changmin/dark_vessel_sr_yolo"
SWEEP_DIR="${PROJECT_ROOT}/iac_runs/nms_sweep_2stage"
mkdir -p "${SWEEP_DIR}"

EVAL_SCRIPT="${PROJECT_ROOT}/iac_scripts/arch4_eval_ultralytics.py"
HR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_hr_data.yaml"
LR_DATA_YAML="${PROJECT_ROOT}/iac_runs/20260316_arch024_fullval_rfdnyolo_db/subset6418_lr_data.yaml"
SCOUT="${PROJECT_ROOT}/weights/yolo_lr_improved/8s_aug_deadline_try_stage2/weights/best.pt"

TOP_K=10
STAGE1_IMAGES=200

# ─── SR별 weight 정의 ───
declare -A SR_WEIGHT
declare -A SNIPER_WEIGHT

SR_WEIGHT[rfdn]="${PROJECT_ROOT}/weights/rfdn_arch4/model_best.pt"
SR_WEIGHT[drct]="${PROJECT_ROOT}/weights/sr_finetuned/drct/best.pt"
SR_WEIGHT[hat]="${PROJECT_ROOT}/weights/sr_finetuned/hat/best.pt"
SR_WEIGHT[man]="${PROJECT_ROOT}/weights/sr_finetuned/man/best.pt"

# Properbase sniper
find_best_sniper() {
  local sr_name="$1"
  local sniper_dir=$(find ${PROJECT_ROOT}/weights/yolo_sniper_hardneg/ -maxdepth 1 -name "*hardneg_${sr_name}_pb*" -type d 2>/dev/null | sort -r | head -1)
  if [ -n "$sniper_dir" ] && [ -f "${sniper_dir}/weights/best.pt" ]; then
    echo "${sniper_dir}/weights/best.pt"
    return
  fi
  local pb_dir=$(find ${PROJECT_ROOT}/iac_runs/ -maxdepth 1 -name "*${sr_name}_properbase_pipeline" -type d 2>/dev/null | sort -r | head -1)
  if [ -n "$pb_dir" ] && [ -f "${pb_dir}/final_map.yaml" ]; then
    grep "weights_hr:" "${pb_dir}/final_map.yaml" | awk '{print $2}' | head -1
    return
  fi
  echo "NOT_FOUND"
}

SNIPER_WEIGHT[rfdn]="${PROJECT_ROOT}/weights/yolo_sniper_hardneg/20260325_023318_hardneg_newscout/weights/best.pt"
SNIPER_WEIGHT[drct]=$(find_best_sniper "drct")
SNIPER_WEIGHT[hat]=$(find_best_sniper "hat")
SNIPER_WEIGHT[man]=$(find_best_sniper "man")

# ─── Grid ───
PASS1_VALS=(0.005 0.0075 0.01 0.0125 0.015 0.0175 0.02 0.025)
ROI_VALS=(1.5 1.75 2.0 2.25 2.5)
SIGMA_FULL=(0.0 0.3 0.5)
SIGMA_FIXED=(0.3)

HIGH_CONF=0.45
FINAL_CONF=0.25
SNIPER_CONF=0.001
CROP_SIZE=64

echo "============================================================"
echo " NMS Sweep 2-Stage (Section IV)"
echo " Stage 1: ${STAGE1_IMAGES}장 × grid → Top-${TOP_K}"
echo " Stage 2: 6418장 × Top-${TOP_K}"
echo " Started: $(date)"
echo "============================================================"

# ─── Pre-flight ───
echo ""
echo "[Pre-flight] Weight 확인"
for sr in rfdn drct hat man; do
  sw="${SR_WEIGHT[$sr]}"
  sn="${SNIPER_WEIGHT[$sr]}"
  sw_ok=$([ -f "$sw" ] && echo "✅" || echo "❌")
  sn_ok=$([ -f "$sn" ] && echo "✅" || echo "❌")
  echo "  ${sr}: SR=${sw_ok} Sniper=${sn_ok}"
  [ "$sn_ok" = "❌" ] && echo "    → $sn"
done

# ─── eval 함수 ───
run_eval() {
  local CFG_FILE="$1"
  local OUT_JSON="$2"
  local MAX_IMG="$3"

  python "${EVAL_SCRIPT}" \
    --arch4_config "${CFG_FILE}" \
    --hr_data_yaml "${HR_DATA_YAML}" \
    --lr_data_yaml "${LR_DATA_YAML}" \
    --eval_space hr \
    --max_images "${MAX_IMG}" \
    --device cuda \
    --batch 1 \
    --out_json "${OUT_JSON}" 2>/dev/null
}

make_config() {
  local OUT="$1" SR="$2" P1="$3" ROI="$4" SIGMA="$5"
  local FUSION="soft_nms"
  [ "$SIGMA" = "0.0" ] && FUSION="nms"

  cat > "$OUT" << YAMLEOF
data:
  upscale_factor: 4
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
    high_conf: ${HIGH_CONF}
    pass2_conf: ${HIGH_CONF}
    final_conf: ${FINAL_CONF}
    sniper_conf: ${SNIPER_CONF}
    merge_iou: 0.5
    roi_expansion: ${ROI}
    crop_size_lr: ${CROP_SIZE}
    batch_size_sr: 32
    scout_nms_iou: 0.5
    roi_merge_iou: 0.3
    roi_center_ratio: 0.35
    sniper_nms_iou: 0.45
    final_nms_iou: 0.5
    drop_uncertain_if_sniper_hits: true
    sniper_score_bonus: 0.0
    merge_policy: size_cond
    final_fusion_method: ${FUSION}
    soft_nms_sigma: ${SIGMA}
    roi_small_thresh: 32.0
    roi_large_thresh: 96.0
    large_roi_score_thresh: 0.5
    sniper_replace_margin: 0.1
    sniper_max_det_per_crop: 3
YAMLEOF
}

# ─── 2-Stage sweep 함수 ───
run_2stage_sweep() {
  local SR_NAME="$1"
  shift
  local SIGMA_LIST=("$@")

  local SR_DIR="${SWEEP_DIR}/${SR_NAME}"
  local S1_DIR="${SR_DIR}/stage1"
  local S2_DIR="${SR_DIR}/stage2"
  mkdir -p "${S1_DIR}" "${S2_DIR}"

  local N_SIGMA=${#SIGMA_LIST[@]}
  local N_TOTAL=$((${#PASS1_VALS[@]} * ${#ROI_VALS[@]} * N_SIGMA))

  echo ""
  echo "════════════════════════════════════════════"
  echo "  ${SR_NAME^^} NMS Sweep 2-Stage"
  echo "  Stage 1: ${STAGE1_IMAGES}장 × ${N_TOTAL} combos"
  echo "  Stage 2: 6418장 × Top-${TOP_K}"
  echo "  Sniper: ${SNIPER_WEIGHT[$SR_NAME]}"
  echo "  Start:  $(date)"
  echo "════════════════════════════════════════════"

  # ── Stage 1: Quick scan ──
  echo ""
  echo "──── Stage 1: Quick Scan (${STAGE1_IMAGES}장) ────"

  local COUNT=0
  for pass1 in "${PASS1_VALS[@]}"; do
    for roi in "${ROI_VALS[@]}"; do
      for sigma in "${SIGMA_LIST[@]}"; do
        COUNT=$((COUNT + 1))
        local TAG="p1${pass1}_roi${roi}_s${sigma}"
        local CFG="${S1_DIR}/${TAG}.yaml"
        local OUT="${S1_DIR}/${TAG}.json"

        if [ -f "$OUT" ]; then
          echo "[${COUNT}/${N_TOTAL}] ${TAG} — skip"
          continue
        fi

        make_config "$CFG" "$SR_NAME" "$pass1" "$roi" "$sigma"
        echo -n "[${COUNT}/${N_TOTAL}] ${TAG} ... "

        run_eval "$CFG" "$OUT" "${STAGE1_IMAGES}"

        if [ -f "$OUT" ]; then
          python3 -c "
import json
d = json.load(open('${OUT}'))
r = d['runs'][0]['results_dict']
print(f'mAP50={r.get(\"metrics/mAP50(B)\",0):.4f} P={r.get(\"metrics/precision(B)\",0):.4f} R={r.get(\"metrics/recall(B)\",0):.4f}')
" 2>/dev/null
        else
          echo "FAIL"
        fi
      done
    done
  done

  # ── Stage 1 Summary + Top-K 선정 ──
  echo ""
  echo "──── Stage 1 → Top-${TOP_K} 선정 ────"

  python3 << S1_SUMMARY
import json, glob
from pathlib import Path

results = []
for f in sorted(glob.glob("${S1_DIR}/*.json")):
    try:
        d = json.load(open(f))
        r = d["runs"][0]["results_dict"]
        tag = Path(f).stem
        results.append({
            "tag": tag,
            "mAP50": r.get("metrics/mAP50(B)", 0),
            "precision": r.get("metrics/precision(B)", 0),
            "recall": r.get("metrics/recall(B)", 0),
            "mAP50_95": r.get("metrics/mAP50-95(B)", 0),
        })
    except:
        pass

results.sort(key=lambda x: -(x["mAP50"] or 0))

with open("${S1_DIR}/summary.json", "w") as f:
    json.dump({"sr": "${SR_NAME}", "stage": 1, "n_images": ${STAGE1_IMAGES}, "n_combos": len(results), "results": results}, f, indent=2)

print(f"Stage 1: {len(results)} combos 완료")
for i, r in enumerate(results[:${TOP_K}]):
    print(f"  Top-{i+1}: {r['tag']} mAP50={r['mAP50']:.4f}")

# Write top-K tags
with open("${S1_DIR}/top_tags.txt", "w") as f:
    for r in results[:${TOP_K}]:
        f.write(r["tag"] + "\n")
S1_SUMMARY

  # ── Stage 2: Full eval on Top-K ──
  echo ""
  echo "──── Stage 2: Full Eval (6418장 × Top-${TOP_K}) ────"

  local S2_COUNT=0
  while IFS= read -r TAG; do
    S2_COUNT=$((S2_COUNT + 1))
    local CFG="${S2_DIR}/${TAG}.yaml"
    local OUT="${S2_DIR}/${TAG}.json"

    if [ -f "$OUT" ]; then
      echo "[${S2_COUNT}/${TOP_K}] ${TAG} — skip"
      continue
    fi

    # NOTE: 이전 sweep 결과 reuse 안 함 (누락 파라미터 다름)

    # Parse tag → params
    local P1=$(echo "$TAG" | sed 's/p1\([^_]*\)_.*/\1/')
    local ROI=$(echo "$TAG" | sed 's/.*roi\([^_]*\)_.*/\1/')
    local SIGMA=$(echo "$TAG" | sed 's/.*s\(.*\)/\1/')

    make_config "$CFG" "$SR_NAME" "$P1" "$ROI" "$SIGMA"
    echo -n "[${S2_COUNT}/${TOP_K}] ${TAG} (6418장) ... "

    run_eval "$CFG" "$OUT" 0

    if [ -f "$OUT" ]; then
      python3 -c "
import json
d = json.load(open('${OUT}'))
r = d['runs'][0]['results_dict']
print(f'mAP50={r.get(\"metrics/mAP50(B)\",0):.4f} P={r.get(\"metrics/precision(B)\",0):.4f} R={r.get(\"metrics/recall(B)\",0):.4f}')
" 2>/dev/null
    else
      echo "FAIL"
    fi
  done < "${S1_DIR}/top_tags.txt"

  # ── Stage 2 Summary ──
  python3 << S2_SUMMARY
import json, glob
from pathlib import Path

results = []
for f in sorted(glob.glob("${S2_DIR}/*.json")):
    try:
        d = json.load(open(f))
        r = d["runs"][0]["results_dict"]
        tag = Path(f).stem
        results.append({
            "tag": tag,
            "mAP50": r.get("metrics/mAP50(B)", 0),
            "precision": r.get("metrics/precision(B)", 0),
            "recall": r.get("metrics/recall(B)", 0),
            "mAP50_95": r.get("metrics/mAP50-95(B)", 0),
        })
    except:
        pass

results.sort(key=lambda x: -(x["mAP50"] or 0))

with open("${S2_DIR}/summary.json", "w") as f:
    json.dump({"sr": "${SR_NAME}", "stage": 2, "n_images": 6418, "n_combos": len(results), "results": results}, f, indent=2)

print()
print(f"═══ ${SR_NAME^^} FINAL (Stage 2, 6418장) ═══")
for i, r in enumerate(results[:5]):
    print(f"  #{i+1}: {r['tag']} mAP50={r['mAP50']:.4f}")
S2_SUMMARY

  echo ""
  echo "  ${SR_NAME^^} 완료: $(date)"
}

# ═══════════════════════════════════════════════
# 실행 (v2: sigma exclude 이미 확인됨, 4개 SR 모두 40 combos)
# 누락 6개 파라미터 모두 포함:
#   - final_conf: 0.25 (properbase 일치)
#   - roi_small_thresh: 32.0
#   - roi_large_thresh: 96.0
#   - large_roi_score_thresh: 0.5
#   - sniper_replace_margin: 0.1
#   - sniper_max_det_per_crop: 3
# ═══════════════════════════════════════════════

run_2stage_sweep "drct" "${SIGMA_FIXED[@]}"
run_2stage_sweep "hat" "${SIGMA_FIXED[@]}"
run_2stage_sweep "man" "${SIGMA_FIXED[@]}"
run_2stage_sweep "rfdn" "${SIGMA_FIXED[@]}"

# Sigma 판단 스킵 (v1에서 이미 무의미 확인)
echo "" && echo "Sigma sweep 스킵 (v1에서 delta=0.0000 확인됨)"
SIGMA_DECISION="exclude"
echo "exclude" > "${SWEEP_DIR}/sigma_decision.txt"

if false; then  # v1 잔여 코드 비활성화
  echo "[Phase 3] Sigma negligible → sigma=0.3 only"
  run_2stage_sweep "man" "${SIGMA_FIXED[@]}"
  run_2stage_sweep "rfdn" "${SIGMA_FIXED[@]}"
fi

# ═══════════════════════════════════════════════
# 최종 비교
# ═══════════════════════════════════════════════
echo ""
echo "============================================================"
echo " FINAL: 고정 NMS vs SR별 Best NMS"
echo "============================================================"

python3 << FINAL_TABLE
import json
from pathlib import Path

sweep_dir = Path("${SWEEP_DIR}")

fixed_nms = {"RFDN": 0.7731, "DRCT": 0.7806, "HAT": 0.7733, "MAN": 0.7720}

print()
print(f"{'SR':<6} {'Fixed':>8} {'Best':>8} {'Config':<35} {'Delta':>8}")
print("-" * 70)

for sr in ["RFDN", "DRCT", "HAT", "MAN"]:
    s2 = sweep_dir / sr.lower() / "stage2" / "summary.json"
    if s2.exists():
        data = json.load(open(s2))
        if data["results"]:
            b = data["results"][0]
            d = (b["mAP50"] - fixed_nms[sr]) * 100
            print(f"{sr:<6} {fixed_nms[sr]:>8.4f} {b['mAP50']:>8.4f} {b['tag']:<35} {d:>+7.2f}pp")
        else:
            print(f"{sr:<6} {fixed_nms[sr]:>8.4f} {'---':>8}")
    else:
        print(f"{sr:<6} {fixed_nms[sr]:>8.4f} {'---':>8} {'(not run)':<35}")

print()
FINAL_TABLE

echo "============================================================"
echo " 전체 완료: $(date)"
echo " 결과: ${SWEEP_DIR}"
echo "============================================================"
