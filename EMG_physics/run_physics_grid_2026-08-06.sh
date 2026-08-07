#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 2026-08-06 Physics-informed CDAN — λ_rec × λ_jerk 6x6 전면 격자 (36잡)
#
# 무엇을 묻는가 —
#   08-04 스윕은 λ_rec ∈ {0.1,0.3,1.0} 과 λ_jerk ∈ {0.01,0.1,1.0} 을 **따로** 훑었고
#   (λ_rec=0.3 고정), 두 항의 상호작용을 못 봤다. 최고 조건이 rec0.3+jerk1.0 이었는데
#   그게 λ_rec=0.3 이라는 임의 선택에 얹힌 값이라 신뢰하기 어렵다. 여기서는 두 축을
#   같은 간격으로 전면 격자로 덮는다.
#
#   L = L_cls + w_d·L_domain + λ_rec·L_rec + λ_jerk·L_jerk
#   λ_rec, λ_jerk ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}  → 6x6 = 36잡
#
#   (0,0) 칸이 대조군이다. 격자 안에 있으므로 별도 대조군 잡이 필요 없다.
#
# 08-04 대비 바뀐 것 (해석할 때 반드시 감안할 것)
#   1) 융합단이 **nointerp** 다. IMU 인코더가 stride 1 + MaxPool 제거로 길이 500 을
#      그대로 내고, F.interpolate 없이 EMG map 과 concat 한다. 파라미터 수는 동일
#      (12,134,415) 이라 용량 교락은 없고, 차이는 순수하게 시간해상도/보간 효과다.
#      → (0,0) 칸과 08-04 대조군(phys_cdan_result_ctrl.json, Tgt 88.37)의 차이가
#        그대로 nointerp 효과다. 요약에서 자동으로 뽑는다.
#      → 대가: IMU 인코더 RF 가 약 10초에서 1.2초로 줄어든다.
#   2) 최종 진단(recon_corr 등)을 target test **전체**(54배치)로 잰다. 08-04 는
#      배치 하나였다. seed 간 상관 분석을 하려면 표본이 필요해서 바꿨다.
#
# 데이터 — preprocessed_MM_pca (R_pca 축 정렬 + 축별 z-score)
#   08-04 와 **같은 데이터**다. isotropic 을 쓰지 않는 이유:
#   이번 스윕에는 중력방향/주축 손실(orient/axes)이 없어서 isotropic 이 필요없고,
#   기존 isotropic 런들은 Source 가 오히려 낮았다(81.7~87.9 vs z-score 판 94.4).
#   재구성 타깃이 IMU 라 두 도메인이 같은 축 프레임이어야 하므로 raw(R=I)는 금지.
#
# λ 범위 근거
#   08-04 프로브: 초기화 직후 L_cls=2.31, L_rec(huber)=0.53, L_jerk=0.163.
#   λ_rec 1.0 이면 L_rec 기여가 대략 20~25%, λ_jerk 1.0 이면 jerk_ratio 가 0.339 까지
#   내려가 과평활 구간에 들어간다. 즉 1.0 은 두 항 모두 "이미 충분히 센" 상한이고,
#   0.2 간격이면 아래쪽(약한 정규화)을 촘촘히 덮는다.
#
# ⚠ 통계 경고 — 이 격자의 1등 칸을 그대로 믿으면 안 된다
#   seed 5개, 관측 σ≈2.2%p → paired Δ 의 se ≈ 0.97%p.
#   대조군 제외 35칸 중 **최댓값은 순전히 우연으로도 +2%p 근처**에 나온다. 이는
#   08-04 최고 효과(+1.75%p)와 같은 크기다. 따라서 이 스윕의 산출물은
#   "어느 칸이 이겼다"가 아니라 **응답면의 모양**(단조성/능선/상호작용 유무)이다.
#   후보가 좁혀지면 그 근방을 seed 12~15 로 재실행해서 확정할 것.
#   윈도우 90% 중첩 때문에 유효 표본수는 명목의 약 1/10, 세션 1개 ≈ 1.37%p 다.
#
# 소요 — 잡당 약 27분 (08-04 실측 26분 + nointerp 오버헤드 2.4%) x 36잡 ≈ **16시간**
#
# 실행:   nohup bash EMG_physics/run_physics_grid_2026-08-06.sh > /dev/null 2>&1 &
# 진행:   tail -f logs/physics_grid_2026-08-06/driver.log
# 이어하기: 결과 json 의 λ·aux·융합 설정이 기대와 맞으면 건너뛴다. 전부 재실행은 REDO=1.
# ---------------------------------------------------------------------------
set -euo pipefail

# 이 스크립트는 EMG_physics/ 안에 있지만 학습은 루트 기준 경로를 쓴다.
cd "$(dirname "$0")/.."

# --- 이중 실행 방지 (07-27 에 17초 차로 두 번 떠서 결과 json 을 공유한 적 있음) ---
LOCK="/tmp/physics_grid_2026-08-06.lock"
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
GRID="0 0.2 0.4 0.6 0.8 1.0"
REDO="${REDO:-0}"

LOGDIR="logs/physics_grid_2026-08-06"
RES="results/EMG_physics"
mkdir -p "$LOGDIR" "$RES"
DRIVER="$LOGDIR/driver.log"

TRAIN="EMG_physics/physics_cdan_train.py"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }
rule() { echo "==================================================================" | tee -a "$DRIVER"; }

{
  echo "###################################################################"
  echo "# Physics-informed CDAN λ_rec x λ_jerk 격자  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# data=$DATA | 융합=nointerp (보간 없음) | seeds=$SEEDS | ${EPOCHS}ep"
  echo "# 격자 {$GRID}^2 = 36잡 | 예상 약 16시간 | REDO=$REDO"
  echo "###################################################################"
} | tee -a "$DRIVER"

"$PY" -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
    | tee -a "$DRIVER"

START_TS=$SECONDS
DONE_N=0; FAIL_N=0; SKIP_N=0

# ---------------------------------------------------------------------------
# Phase 0 — 사전 준비물 + 모델 동치성 검증
#   (0,0) 대조군이 기존 CDAN 과 같은 경로임을 여기서 못 박고 시작한다. 이게 깨지면
#   이 스윕의 모든 Δ 해석이 무너지므로 실패 시 즉시 중단한다. nointerp 판은
#   Compact.nointerp_model.InterFusionCDAN 을 기준으로 동치성을 검증한다.
# ---------------------------------------------------------------------------
rule
log "Phase 0 — 준비물 확인 + 대응 CDAN 과의 동치성 검증 (두 융합 모드 모두)"
[ -f "$TRAIN" ] || { log "FAIL 학습 스크립트가 없다: $TRAIN"; exit 1; }
[ -f "$DATA/y_target_test.npy" ] || { log "FAIL 데이터 폴더가 없다: $DATA"; exit 1; }
"$PY" EMG_physics/physics_model.py 2>&1 | tee -a "$DRIVER" \
    || { log "FAIL 동치성 검증 실패 — (0,0) 대조군이 성립하지 않는다"; exit 1; }

# λ 규모 프로브도 로그에 남겨 둔다 (나중에 λ 선택 근거를 되짚을 수 있게)
log "λ 규모 프로브 (기록용)"
env "MM_DATA_DIR=$DATA" "$PY" -u "$TRAIN" --probe > "$LOGDIR/probe.log" 2>&1 \
    && tail -22 "$LOGDIR/probe.log" | tee -a "$DRIVER" \
    || log "경고: 프로브 실패 (학습은 계속한다)"

# ---------------------------------------------------------------------------
# 공용 잡 실행기
#   결과 json 이 있어도 λ·aux·융합 설정이 기대와 다르면 다시 돈다
#   (07-29 러너는 config 를 안 보고 스킵해서 설정을 바꿔도 조용히 건너뛰었다).
#   태그에 grid_ 접두사를 붙여 08-04 산출물(ctrl, rec0.3_jerk1.0 …)과 절대 안 겹치게 한다.
# ---------------------------------------------------------------------------
run_job() {
    local label="$1" tag="$2" lrec="$3" ljerk="$4"
    local out="$RES/phys_cdan_result_${tag}.json"
    local logf="$LOGDIR/${tag}.log"
    rule
    if [ "$REDO" != "1" ] && [ -f "$out" ]; then
        local got
        got="$("$PY" -c "
import json;d=json.load(open('$out'))
print('%g|%g|%s|%s' % (d.get('lambda_rec',-1), d.get('lambda_jerk',-1),
                       str(d.get('aux_on_target','?')).lower(),
                       d.get('imu_fusion','?')))" 2>/dev/null || echo "?")"
        local want
        want="$("$PY" -c "print('%g|%g|true|nointerp' % ($lrec, $ljerk))")"
        if [ "$got" = "$want" ]; then
            log "SKIP   $label  (결과 있음, 설정 일치: $got)"
            SKIP_N=$((SKIP_N + 1)); return 0
        fi
        log "재실행 $label  (결과는 있으나 설정=$got, 기대=$want)"
    fi
    log "START  $label"
    echo "  cfg : lambda_rec=$lrec lambda_jerk=$ljerk fusion=nointerp data=$DATA" | tee -a "$DRIVER"
    echo "  log : $logf" | tee -a "$DRIVER"
    local t0=$SECONDS rc=0
    env "MM_DATA_DIR=$DATA" \
        "$PY" -u "$TRAIN" --multi_seed --seeds $SEEDS --epochs $EPOCHS \
              --lambda_rec "$lrec" --lambda_jerk "$ljerk" \
              --no_cm --tag "$tag" > "$logf" 2>&1 || rc=$?
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
# 격자 실행
#   λ_rec 을 바깥 루프로 둔다 — 중간에 끊겨도 "λ_jerk 축 한 줄"이 통째로 남아
#   부분 결과만으로도 한 축의 응답을 읽을 수 있다.
# ---------------------------------------------------------------------------
N=0
for lr in $GRID; do
    rule; log "행 λ_rec=$lr  (λ_jerk 6칸)"
    for lj in $GRID; do
        N=$((N + 1))
        run_job "[$N/36] rec=$lr jerk=$lj" "grid_r${lr}_j${lj}" "$lr" "$lj"
    done
done

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
rule
log "요약 (총 ${DONE_N}완료 / ${SKIP_N}스킵 / ${FAIL_N}실패, $(( (SECONDS - START_TS) / 60 ))분)"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$DRIVER"
import json, os
import numpy as np

RES = "results/EMG_physics"
GRID = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
# f-string 안에 백슬래시를 못 넣는다(Python 3.10) — 헤더는 밖에서 만든다.
HDR = "{:<14}".format("λ_rec\\λ_jerk")

def load(tag):
    p = os.path.join(RES, f"phys_cdan_result_{tag}.json")
    return json.load(open(p)) if os.path.isfile(p) else None

def gcell(lr, lj):
    return load(f"grid_r{lr}_j{lj}")

def mean_of(j, k):
    return j["mean"][k] if j and k in j.get("mean", {}) else None

def paired(a, b, k="target_test_acc"):
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
    return f"{m:+.2f} ± {sd:.2f} (se {sd/len(d)**0.5:.2f}, t={m/(sd/len(d)**0.5):+.2f}, n={len(d)})"

def matrix(key, fmt="{:>8.2f}"):
    print("\n  " + HDR + "".join(f"{lj:>9}" for lj in GRID))
    for lr in GRID:
        row = f"  {lr:<14}"
        for lj in GRID:
            v = mean_of(gcell(lr, lj), key)
            row += f"{'—':>9}" if v is None else fmt.format(v)
        print(row)

ctrl = gcell("0", "0")

print("\n=== λ_rec x λ_jerk 격자 — TEST 기준 ===")
print("  data=preprocessed_MM_pca | 융합=nointerp | selection = target_val (oracle)")
print("  (0,0) 칸이 대조군이다.")

print("\n[Target Test 정확도 %]"); matrix("target_test_acc")
print("\n[Source Test 정확도 %]"); matrix("source_test_acc")
print("\n[recon_corr — 0 근처면 디코더가 IMU 를 전혀 못 배운 것]")
matrix("final_recon_corr", "{:>8.3f}")
print("\n[jerk_ratio — 예측jerk/실제jerk, <<1 이면 과평활]")
matrix("final_jerk_ratio", "{:>8.3f}")

print("\n[paired Δ vs (0,0) 대조군, Target Test %p — 같은 seed]")
matrix_done = False
if ctrl:
    print("\n  " + HDR + "".join(f"{lj:>9}" for lj in GRID))
    for lr in GRID:
        row = f"  {lr:<14}"
        for lj in GRID:
            d = paired(ctrl, gcell(lr, lj))
            row += f"{'—':>9}" if not d else f"{np.mean(d):>+9.2f}"
        print(row)
    matrix_done = True

    # 상위 5칸만 통계까지
    cells = []
    for lr in GRID:
        for lj in GRID:
            if lr == "0" and lj == "0":
                continue
            d = paired(ctrl, gcell(lr, lj))
            if d:
                cells.append((float(np.mean(d)), lr, lj, d))
    cells.sort(reverse=True)
    print("\n  상위 5칸 (paired t 포함):")
    for m, lr, lj, d in cells[:5]:
        print(f"    rec={lr:<4} jerk={lj:<4} {stat(d)}")
    if cells:
        print("\n  하위 3칸:")
        for m, lr, lj, d in cells[-3:]:
            print(f"    rec={lr:<4} jerk={lj:<4} {stat(d)}")
else:
    print("  (0,0) 대조군 결과가 없어 Δ 를 못 낸다.")

# nointerp 효과 — 08-04 대조군(interp, 같은 데이터·같은 λ=0)과의 paired 비교
old = load("ctrl")
if old and ctrl:
    print("\n[곁가지] nointerp 효과 = (0,0) − 08-04 대조군(interp, 같은 데이터·λ=0)")
    print(f"  Target Test : {stat(paired(old, ctrl))}")
    print(f"  Source Test : {stat(paired(old, ctrl, 'source_test_acc'))}")
    print("  두 팔은 융합단만 다르고 파라미터 수가 같다(12,134,415) — 용량 교락 없음.")

print("""
읽는 법 — 이 격자는 "1등 칸 찾기"가 아니라 "응답면 모양 보기"다
  · ⚠ seed 5개, paired Δ 의 se ≈ 1%p 다. 대조군 제외 35칸의 **최댓값은 우연만으로도
    +2%p 근처**에 나온다 (08-04 최고 효과 +1.75%p 와 같은 크기). 상위 칸의 t 값이
    2.776(n=5 양측 0.05)을 넘지 못하면 "이겼다"고 쓰지 말 것.
  · 볼 것은 (1) λ_rec 축을 따라 단조성이 있는가 (2) λ_jerk 가 λ_rec 과 상호작용하는가
    (3) 능선이 대각선인가 축평행인가. 이건 개별 칸의 잡음에 덜 휘둘린다.
  · recon_corr 이 0 근처면 디코더가 아무것도 못 배운 것이고, 그러면 그 칸의 이득은
    "물리 정보"가 아니라 잡음 정규화다.
  · 2026-08-06 재분석: λ 를 고정했을 때 r(recon_corr, target acc) = −0.50 (p=0.006).
    **복원을 잘한 런일수록 전이가 나빴다.** 격자에서도 이 관계가 재현되는지 확인할 것 —
    재현되면 λ_rec 을 키우는 방향은 죽은 방향이다.
  · jerk_ratio 가 0.2 밑인데 정확도가 올랐다면 물리적 타당성 덕이 아니라 jerk 항이
    복원 충실도를 깎는 capacity limiter 로 작동한 것으로 읽어야 한다.
  · Source 가 같이 떨어지면 λ 가 분류 용량을 잡아먹은 것이다. Target 만 오르는 게
    도메인 적응이고, 둘 다 내려가면 λ 가 과하다.
  · 윈도우 90% 중첩 → 유효 표본수는 명목의 약 1/10, 세션 1개 ≈ 1.37%p.""")
PYEOF

rule
log "SWEEP DONE  (완료 ${DONE_N} / 스킵 ${SKIP_N} / 실패 ${FAIL_N})"
