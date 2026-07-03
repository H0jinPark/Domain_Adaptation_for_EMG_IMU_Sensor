#!/usr/bin/env bash
# align-first target_join_epoch 민감도 스윕 (5 seed × 60 epoch, 값마다 순차 실행).
# 각 값 로그: logs/alignfirst_sweep/join<J>.log
# 결과 JSON/summary: results/Learnable_R/learnable_r_cdan_alignfirst_*join<J>*
#
# 실행(백그라운드):
#   nohup bash Learnable_R/run_alignfirst_sweep.sh > logs/alignfirst_sweep/driver.log 2>&1 &
# 진행 확인:
#   tail -f logs/alignfirst_sweep/join0.log      # 특정 값
#   tail -f logs/alignfirst_sweep/driver.log     # 전체 진행/완료 표시
set -uo pipefail

# ---- conda env DA 활성화 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DA

# ---- 프로젝트 루트로 이동 (이 스크립트는 Learnable_R/ 안에 있음) ----
cd "$(dirname "$0")/.." || exit 1

export MM_DATA_DIR=preprocessed_MM_raw_isotropic
export PYTHONUNBUFFERED=1
mkdir -p logs/alignfirst_sweep

# 스윕할 target_join_epoch 값. (0 = align-first 끔 = 원본 base 통제군)
JOINS=(0 5 10 15 20 30)

echo "SWEEP START $(date) | joins: ${JOINS[*]} | data: $MM_DATA_DIR"
for J in "${JOINS[@]}"; do
  echo "=== target_join_epoch=$J 시작 $(date) ==="
  if python -u Learnable_R/learnable_r_cdan_alignfirst_train.py \
        --multi_seed --epochs 60 --target_join_epoch "$J" --no_cm --tag "join$J" \
        > "logs/alignfirst_sweep/join$J.log" 2>&1; then
    echo "=== join$J 완료 $(date) ==="
  else
    echo "!!! join$J 실패 $(date) (logs/alignfirst_sweep/join$J.log 확인) — 다음 값 계속"
  fi
done

echo ""
echo "===== SWEEP DONE $(date) — 값별 요약 ====="
for J in "${JOINS[@]}"; do
  S="results/Learnable_R/learnable_r_cdan_alignfirst_join${J}_summary.txt"
  echo "--- join$J ---"
  if [ -f "$S" ]; then grep -E "Target|Std|Shift" "$S"; else echo "(요약 없음 — 실패/미완)"; fi
done
