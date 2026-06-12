#!/usr/bin/env bash
# 전체 DA 매트릭스 일괄 실행 (16개)
#   순서:
#     1) hybrid_ssl pretrain   (multi-seed 0..4)
#     2) 5_channel/*_train.py  (no_da, Coral, MMD, DANN, CDAN)
#     3) Multimodal/*_train.py
#     4) pretrain_Multimodal/*_train.py  (pretrain_seed=0 weights 사용)
#
# 사용법:
#   nohup bash scripts/run_all_da.sh > logs/all_$(date +%Y%m%d_%H%M).out 2>&1 &
#   tail -f logs/all_*.out
#
# 실패해도 계속 진행. 각 모델별 개별 로그는 logs/<model>.log 에 저장.

set -u  # 미정의 변수 에러
# set -e 안 씀: 한 모델 실패해도 나머지 계속 돌리기

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/da_matrix_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

SEEDS="0 1 2 3 4"
EPOCHS=30
PRETRAIN_EPOCHS=50

# ANSI 컬러 (로그 파일에도 들어가지만 가독성 위해 유지)
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

run_one() {
  local tag="$1"; shift
  local logfile="$LOG_DIR/${tag}.log"
  local start; start=$(date +%s)

  echo -e "\n${YELLOW}=== [$(date +'%H:%M:%S')] START: ${tag} ===${NC}"
  echo "  cmd: $*"
  echo "  log: $logfile"

  if "$@" > "$logfile" 2>&1; then
    local elapsed=$(( $(date +%s) - start ))
    echo -e "${GREEN}=== [$(date +'%H:%M:%S')] DONE  ${tag}  (${elapsed}s) ===${NC}"
  else
    local elapsed=$(( $(date +%s) - start ))
    echo -e "${RED}=== [$(date +'%H:%M:%S')] FAIL  ${tag}  (${elapsed}s)  → check $logfile ===${NC}"
  fi
}

OVERALL_START=$(date +%s)
echo -e "${GREEN}Pipeline start: $(date)${NC}"
echo "Repo: $ROOT"
echo "Log dir: $LOG_DIR"
echo "Seeds: $SEEDS  |  DA epochs: $EPOCHS  |  Pretrain epochs: $PRETRAIN_EPOCHS"

# ---------- 1. Hybrid SSL Pretrain (16번째) ----------
run_one "00_pretrain_hybrid_ssl" \
  python pretrain_Multimodal/hybrid_ssl_train.py \
    --multi_seed --seeds $SEEDS --epochs $PRETRAIN_EPOCHS

# ---------- 2. 5_channel (5개) ----------
for METHOD in no_da Coral MMD DANN CDAN; do
  run_one "5ch_${METHOD}" \
    python 5_channel/${METHOD}_train.py \
      --multi_seed --seeds $SEEDS --epochs $EPOCHS --no_cm
done

# ---------- 3. Multimodal (5개) ----------
for METHOD in no_da Coral MMD DANN CDAN; do
  run_one "mm_${METHOD}" \
    python Multimodal/${METHOD}_train.py \
      --multi_seed --seeds $SEEDS --epochs $EPOCHS --no_cm
done

# ---------- 4. pretrain_Multimodal (5개, pretrain_seed=0 사용) ----------
for METHOD in no_da Coral MMD DANN CDAN; do
  run_one "pt_${METHOD}" \
    python pretrain_Multimodal/${METHOD}_train.py \
      --multi_seed --seeds $SEEDS --epochs $EPOCHS --no_cm --pretrain_seed 0
done

# ---------- 요약 ----------
TOTAL=$(( $(date +%s) - OVERALL_START ))
H=$(( TOTAL / 3600 )); M=$(( (TOTAL % 3600) / 60 ))
echo -e "\n${GREEN}=== Pipeline finished: $(date)  (총 ${H}h ${M}m) ===${NC}"
echo "개별 로그: $LOG_DIR"
echo ""
echo "성공/실패 요약:"
for f in "$LOG_DIR"/*.log; do
  name=$(basename "$f" .log)
  if grep -q "Traceback\|Error" "$f"; then
    echo -e "  ${RED}[FAIL]${NC} $name"
  else
    echo -e "  ${GREEN}[OK]${NC}   $name"
  fi
done
