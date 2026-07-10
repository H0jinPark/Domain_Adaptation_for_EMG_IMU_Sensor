#!/usr/bin/env bash
# IMU 단독 align-first 기하 prior loss ablation 스윕.
# 같은 align-first 세팅(target_join_epoch=10, CDAN on)에서 R 정렬 prior 만 바꾼다:
#   gravonly : L_gravity 만 (λ_g=1, λ_pca=0)
#   pcaonly  : L_pca 만    (λ_g=0, λ_pca=1)
#   both     : 둘 다 (대조군, 현재 기본 λ_g=1, λ_pca=1)
# 학습/모델 코드는 그대로 — learnable_r_cdan_alignfirst_train.py 의 --lambda_g/--lambda_pca 만 조정.
# 5 seed × 60 epoch, 조합마다 순차 실행.
#   각 조합 로그: logs/loss_ablation_sweep/<tag>.log
#   결과 JSON/summary: results/Learnable_R/learnable_r_cdan_alignfirst_*<tag>*
#
# 실행(백그라운드):
#   nohup bash Learnable_R/run_loss_ablation_sweep.sh > logs/loss_ablation_sweep/driver.log 2>&1 &
# 진행 확인:
#   tail -f logs/loss_ablation_sweep/gravonly.log   # 특정 조합
#   tail -f logs/loss_ablation_sweep/driver.log     # 전체 진행/완료 표시
set -uo pipefail

# ---- conda env DA 활성화 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DA

# ---- 프로젝트 루트로 이동 (이 스크립트는 Learnable_R/ 안에 있음) ----
cd "$(dirname "$0")/.." || exit 1

export MM_DATA_DIR=preprocessed_MM_raw_isotropic
export PYTHONUNBUFFERED=1
mkdir -p logs/loss_ablation_sweep

# 고정 세팅: IMU 단독 align-first 최적값(join10), CDAN on.
JOIN=10
EPOCHS=60

# "태그:λ_g:λ_pca" 조합 목록.
CONFIGS=(
  "gravonly:1.0:0.0"
  "pcaonly:0.0:1.0"
  "both:1.0:1.0"
)

echo "LOSS ABLATION START $(date) | join=$JOIN epochs=$EPOCHS | data: $MM_DATA_DIR"
for C in "${CONFIGS[@]}"; do
  IFS=":" read -r TAG LAM_G LAM_PCA <<< "$C"
  echo "=== $TAG (λ_g=$LAM_G λ_pca=$LAM_PCA) 시작 $(date) ==="
  if python -u Learnable_R/learnable_r_cdan_alignfirst_train.py \
        --multi_seed --epochs "$EPOCHS" --target_join_epoch "$JOIN" \
        --lambda_g "$LAM_G" --lambda_pca "$LAM_PCA" \
        --no_cm --tag "$TAG" \
        > "logs/loss_ablation_sweep/$TAG.log" 2>&1; then
    echo "=== $TAG 완료 $(date) ==="
  else
    echo "!!! $TAG 실패 $(date) (logs/loss_ablation_sweep/$TAG.log 확인) — 다음 조합 계속"
  fi
done

echo ""
echo "===== LOSS ABLATION DONE $(date) — 조합별 요약 ====="
for C in "${CONFIGS[@]}"; do
  IFS=":" read -r TAG LAM_G LAM_PCA <<< "$C"
  S="results/Learnable_R/learnable_r_cdan_alignfirst_${TAG}_summary.txt"
  echo "--- $TAG (λ_g=$LAM_G λ_pca=$LAM_PCA) ---"
  if [ -f "$S" ]; then grep -E "Target|Std|Shift" "$S"; else echo "(요약 없음 — 실패/미완)"; fi
done
