#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 2026-08-04 Physics-informed CDAN 스윕 (EMG→IMU 디코더 + 재구성/jerk 손실)
#
# 무엇을 묻는가 —
#   EMG 인코더가 IMU 를 복원하도록 강제하면, 기기 지문(HF 노이즈) 대신 기기 무관한
#   운동 성분을 담게 되어 target 전이가 좋아지는가?
#   08-04 진단에서 EMG 도메인 갭(A-dist≈2.0)의 서명이 전부 고주파였고(SSC d=+2.53,
#   ZC +1.92, 250-450Hz 단독으로 기기 완전식별), 생리 대역 특징은 거의 안 움직였다.
#   즉 인코더가 붙잡을 수 있는 "쉬운" 기기 단서가 실재한다. IMU 복원은 그 단서로는
#   풀 수 없는 과제라, 표현을 운동학 쪽으로 미는 정규화가 된다.
#
#   L = L_cls + w_d·L_domain + λ_rec·L_rec + λ_jerk·L_jerk
#
#   두 보조 손실 다 짝꿍 IMU 만 쓰고 라벨을 안 본다 → target 배치에도 걸린다(기본).
#   그 기여를 분리하려고 그룹 D 에서 source 에만 거는 조건을 둔다.
#
# 참고 문헌 — Estimating IMU signals from surface EMG using physics-informed and
#   domain-adaptive neural networks (J. Electromyogr. Kinesiol., 2026).
#   EMG→IMU 회귀 + GRL 도메인 적응 + jerk 최소화. 그쪽은 과제(task) 간 도메인이고
#   여기는 기기(device) 간이다. jerk prior 자체는 minimum-jerk 모델(Flash & Hogan 1985).
#
# 데이터 — preprocessed_MM_pca (축 정렬 R_pca 적용, zscore).
#   재구성 타깃이 IMU 라 두 도메인이 같은 축 프레임이어야 한다. raw(R=I)는 쓰면 안 된다.
#   주의: zscore 는 축별 독립 정규화라 회전 구조·축간 크기비가 일부 깨진다. 물리
#   재구성에는 isotropic 이 더 맞지만 기존 표와의 정규화 조건을 맞추려 zscore 로 간다.
#   MM_DATA_DIR 만 바꾸면 preprocessed_MM_pca_isotropic 으로 재실행할 수 있다.
#
# λ 범위 근거 (physics_cdan_train.py --probe + 1에폭 스모크)
#   초기화 직후: L_cls=2.31  L_rec(huber)=0.53  L_jerk=0.163  (정답 IMU jerk=0.0047)
#   → 디코더 초기 출력이 실제보다 34배 튄다. λ_rec≈0.2~1.3 이 5~30% 기여.
#   λ_jerk=1.0 으로 1에폭 돌리면 jerk_ratio=0.158 — 예측이 실제보다 6배 매끄럽다(과평활).
#   그래서 λ_jerk 는 1.0 을 상한으로 두고 0.01 까지 로그 간격으로 훑는다.
#
# 잡 구성 (전부 seeds 0-4, 30ep, 원본 백본)
#   그룹 C  대조군 λ_rec=0, λ_jerk=0 (기존 CDAN 과 수치적으로 동일 경로)     1잡
#   그룹 A  λ_rec ∈ {0.1, 0.3, 1.0}, λ_jerk=0                             3잡
#   그룹 B  λ_rec=0.3 고정, λ_jerk ∈ {0.01, 0.1, 1.0}                     3잡
#   그룹 D  λ_rec=0.3, λ_jerk=0.1, --no_aux_on_target (target 기여 분리)    1잡
#   합계 8잡, 예상 약 3.5~4시간 (스모크 기준 1에폭 ≈ 10초, 잡당 ≈ 25~30분)
#
#   그룹 B 의 λ_rec=0.3 은 A 결과를 보기 전에 미리 고정한 값이다. A 에서 최적이
#   크게 다르게 나오면 B 는 그 값으로 다시 돌려야 한다.
#
# 실행:   nohup bash EMG_physics/run_physics_2026-08-04.sh > /dev/null 2>&1 &
# 진행:   tail -f logs/physics_2026-08-04/driver.log
# 이어하기: 결과 json 의 λ·aux 설정이 기대와 맞으면 건너뛴다. 전부 재실행은 REDO=1.
# ---------------------------------------------------------------------------
set -euo pipefail

# 이 스크립트는 EMG_physics/ 안에 있지만 학습은 루트 기준 경로를 쓴다.
cd "$(dirname "$0")/.."

# --- 이중 실행 방지 (07-27 에 17초 차로 두 번 떠서 결과 json 을 공유한 적 있음) ---
LOCK="/tmp/physics_2026-08-04.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "이미 같은 스윕이 돌고 있다 (lock: $LOCK). 중복 실행을 막고 종료한다." >&2
    exit 1
fi

# --- 학습은 conda DA 인터프리터로 (base 에는 numpy 가 없어 즉사한다) ---
PY="/home/user1/miniconda3/envs/DA/bin/python"
[ -x "$PY" ] || { echo "DA 파이썬이 없다: $PY" >&2; exit 1; }

SEEDS="0 1 2 3 4"
EPOCHS=30
DATA="preprocessed_MM_pca"
REDO="${REDO:-0}"

LOGDIR="logs/physics_2026-08-04"
RES="results/EMG_physics"
mkdir -p "$LOGDIR" "$RES"
DRIVER="$LOGDIR/driver.log"

TRAIN="EMG_physics/physics_cdan_train.py"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }
rule() { echo "==================================================================" | tee -a "$DRIVER"; }

{
  echo "###################################################################"
  echo "# Physics-informed CDAN 스윕  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# EMG→IMU 디코더 + 재구성/jerk 손실 | data=$DATA"
  echo "# seeds=$SEEDS | ${EPOCHS}ep | 8잡 | REDO=$REDO"
  echo "###################################################################"
} | tee -a "$DRIVER"

"$PY" -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
    | tee -a "$DRIVER"

START_TS=$SECONDS
DONE_N=0; FAIL_N=0; SKIP_N=0

# ---------------------------------------------------------------------------
# Phase 0 — 사전 준비물 + 모델 동치성 검증
#   λ=0 대조군이 기존 CDAN 과 같은 경로임을 여기서 못 박고 시작한다.
#   이게 깨지면 이 스윕의 모든 Δ 해석이 무너지므로 실패 시 즉시 중단한다.
# ---------------------------------------------------------------------------
rule
log "Phase 0 — 준비물 확인 + 기존 CDAN 과의 동치성 검증"
[ -f "$TRAIN" ] || { log "FAIL 학습 스크립트가 없다: $TRAIN"; exit 1; }
[ -f "$DATA/y_target_test.npy" ] || { log "FAIL 데이터 폴더가 없다: $DATA"; exit 1; }
"$PY" EMG_physics/physics_model.py 2>&1 | tee -a "$DRIVER" \
    || { log "FAIL 동치성 검증 실패 — λ=0 대조군이 성립하지 않는다"; exit 1; }

# λ 규모 프로브도 로그에 남겨 둔다 (나중에 λ 선택 근거를 되짚을 수 있게)
log "λ 규모 프로브 (기록용)"
env "MM_DATA_DIR=$DATA" "$PY" -u "$TRAIN" --probe > "$LOGDIR/probe.log" 2>&1 \
    && tail -20 "$LOGDIR/probe.log" | tee -a "$DRIVER" \
    || log "경고: 프로브 실패 (학습은 계속한다)"

# ---------------------------------------------------------------------------
# 공용 잡 실행기
#   결과 json 이 있어도 λ·aux 설정이 기대와 다르면 다시 돈다
#   (07-29 러너는 config 를 안 보고 스킵해서 설정을 바꿔도 조용히 건너뛰었다).
# ---------------------------------------------------------------------------
run_job() {
    local label="$1" tag="$2" lrec="$3" ljerk="$4" aux_tgt="$5"; shift 5
    local out="$RES/phys_cdan_result_${tag}.json"
    local logf="$LOGDIR/${tag}.log"
    rule
    if [ "$REDO" != "1" ] && [ -f "$out" ]; then
        local got
        got="$("$PY" -c "
import json;d=json.load(open('$out'))
print('%g|%g|%s' % (d.get('lambda_rec',-1), d.get('lambda_jerk',-1),
                    str(d.get('aux_on_target','?')).lower()))" 2>/dev/null || echo "?")"
        local want
        want="$("$PY" -c "print('%g|%g|%s' % ($lrec, $ljerk, '$aux_tgt'))")"
        if [ "$got" = "$want" ]; then
            log "SKIP   $label  (결과 있음, 설정 일치: $got)"
            SKIP_N=$((SKIP_N + 1)); return 0
        fi
        log "재실행 $label  (결과는 있으나 설정=$got, 기대=$want)"
    fi
    log "START  $label"
    echo "  cfg : lambda_rec=$lrec lambda_jerk=$ljerk aux_on_target=$aux_tgt $*" | tee -a "$DRIVER"
    echo "  log : $logf" | tee -a "$DRIVER"
    local t0=$SECONDS rc=0
    env "MM_DATA_DIR=$DATA" \
        "$PY" -u "$TRAIN" --multi_seed --seeds $SEEDS --epochs $EPOCHS \
              --lambda_rec "$lrec" --lambda_jerk "$ljerk" \
              --no_cm --tag "$tag" "$@" > "$logf" 2>&1 || rc=$?
    local mins=$(( (SECONDS - t0) / 60 ))
    if [ "$rc" -eq 0 ]; then
        DONE_N=$((DONE_N + 1)); log "DONE   $label  (${mins}분)"
    else
        FAIL_N=$((FAIL_N + 1)); log "FAIL   $label  (${mins}분, rc=$rc) — $logf 확인"
    fi
    # 한 잡이 죽어도 전체가 멈추지 않게 계속 간다
    return 0
}

# ---------------------------------------------------------------------------
# 그룹 C — 대조군 (물리 손실 없음)
# ---------------------------------------------------------------------------
rule; log "그룹 1/4 (C) — 대조군 λ_rec=0 λ_jerk=0"
run_job "C 대조군" "ctrl" 0 0 true

# ---------------------------------------------------------------------------
# 그룹 A — 재구성 손실만
# ---------------------------------------------------------------------------
rule; log "그룹 2/4 (A) — 재구성 손실만, λ_rec ∈ {0.1, 0.3, 1.0}"
for lr in 0.1 0.3 1.0; do
    run_job "A rec=$lr" "rec${lr}" "$lr" 0 true
done

# ---------------------------------------------------------------------------
# 그룹 B — 재구성 + jerk
# ---------------------------------------------------------------------------
rule; log "그룹 3/4 (B) — λ_rec=0.3 고정, λ_jerk ∈ {0.01, 0.1, 1.0}"
for lj in 0.01 0.1 1.0; do
    run_job "B rec=0.3 jerk=$lj" "rec0.3_jerk${lj}" 0.3 "$lj" true
done

# ---------------------------------------------------------------------------
# 그룹 D — target 보조손실 기여 분리
# ---------------------------------------------------------------------------
rule; log "그룹 4/4 (D) — source 에만 보조손실 (target 기여 분리)"
run_job "D rec=0.3 jerk=0.1 src만" "rec0.3_jerk0.1_srconly" 0.3 0.1 false --no_aux_on_target

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
rule
log "요약 (총 ${DONE_N}완료 / ${SKIP_N}스킵 / ${FAIL_N}실패, $(( (SECONDS - START_TS) / 60 ))분)"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$DRIVER"
import json, os
import numpy as np

RES = "results/EMG_physics"

def load(tag):
    p = os.path.join(RES, f"phys_cdan_result_{tag}.json")
    return json.load(open(p)) if os.path.isfile(p) else None

def cell(j, k):
    if not j or k not in j.get("mean", {}):
        return f"{'—':>14}"
    return f"{j['mean'][k]:>7.2f} ±{j['std'][k]:>5.2f}"

def num(j, k, fmt="{:>7.3f}"):
    if not j or k not in j.get("mean", {}):
        return f"{'—':>7}"
    return fmt.format(j["mean"][k])

def paired(a, b, k):
    if not (a and b):
        return []
    da = {r["seed"]: r[k] for r in a["results"] if k in r}
    db = {r["seed"]: r[k] for r in b["results"] if k in r}
    return [db[s] - da[s] for s in sorted(set(da) & set(db))]

def stat(d):
    if not d:
        return "—"
    m = float(np.mean(d))
    if len(d) < 2:
        return f"{m:+.2f}"
    sd = float(np.std(d, ddof=1))
    return f"{m:+.2f} ± {sd:.2f} (se {sd/len(d)**0.5:.2f}, n={len(d)})"

ctrl = load("ctrl")
rows = [
    ("대조군 (λ=0)",           "ctrl"),
    ("rec 0.1",                "rec0.1"),
    ("rec 0.3",                "rec0.3"),
    ("rec 1.0",                "rec1.0"),
    ("rec 0.3 + jerk 0.01",    "rec0.3_jerk0.01"),
    ("rec 0.3 + jerk 0.1",     "rec0.3_jerk0.1"),
    ("rec 0.3 + jerk 1.0",     "rec0.3_jerk1.0"),
    ("rec 0.3 + jerk 0.1 (src만)", "rec0.3_jerk0.1_srconly"),
]

print("\n=== Physics-informed CDAN — TEST 기준 ===")
print("  selection = target_val (oracle, 기존 멀티모달 CDAN 과 동일 규약)")
print(f"\n  {'조건':<26}{'Src-Test':>15}{'Tgt-Test':>15}{'recon_corr':>12}{'jerk_ratio':>12}")
for lab, tag in rows:
    j = load(tag)
    print(f"  {lab:<26}{cell(j,'source_test_acc')}{cell(j,'target_test_acc')}"
          f"{num(j,'final_recon_corr')}{num(j,'final_jerk_ratio')}")

print("\n  paired Δ vs 대조군 (같은 seed, Target Test):")
for lab, tag in rows[1:]:
    print(f"    {lab:<28} {stat(paired(ctrl, load(tag), 'target_test_acc'))}")

d_tgt = paired(load("rec0.3_jerk0.1_srconly"), load("rec0.3_jerk0.1"), "target_test_acc")
print(f"\n  target 보조손실의 순기여 (both − src만): {stat(d_tgt)}")

print("""
읽는 법
  · recon_corr 이 0 근처면 디코더가 IMU 를 전혀 못 배운 것이고, 그러면 물리 손실은
    그냥 잡음 정규화였을 뿐이라 "물리 정보를 넣었다"는 주장을 할 수 없다.
    1에폭 스모크에서 corr=0.365 였으니 학습은 된다. 30에폭 후 값을 확인할 것.
  · jerk_ratio = 예측 jerk / 실제 jerk.
      ≈1   적절
      <<1  과평활 — λ_jerk 가 너무 크다. λ_jerk=1.0 은 1에폭에 이미 0.158 이었다.
      >>1  jerk 항이 사실상 안 걸림
    ratio 가 0.2 밑인데 정확도가 올랐다면, 그건 물리적 타당성 덕이 아니라 단순히
    디코더 gradient 가 인코더를 규제한 효과일 가능성이 크다 — 해석에 주의.
  · target 순기여가 +면 "라벨 없는 target IMU 를 썼다"는 게 UDA 기여로 성립한다.
    0 근처면 보조손실은 그냥 표현 정규화이고 도메인 적응 주장은 약해진다.
  · 윈도우 90% 중첩 → 유효 표본수 명목의 약 1/10, 세션 1개 ≈ 1.37%p.
    Δ 가 1%p 안쪽이면 유의하다고 보기 어렵다.
  · Source Test 가 같이 떨어지면 보조손실이 분류 용량을 잡아먹은 것이다.
    Target 만 오르는 게 도메인 적응이고, 둘 다 내려가면 λ 가 과하다.""")
PYEOF

rule
log "SWEEP DONE  (완료 ${DONE_N} / 스킵 ${SKIP_N} / 실패 ${FAIL_N})"
