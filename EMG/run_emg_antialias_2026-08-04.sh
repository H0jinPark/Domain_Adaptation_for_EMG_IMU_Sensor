#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 2026-08-04 EMG 안티에일리어싱 A/B — 두 가지를 동시에 처리한다
#
# (1) 우리 전처리가 만들어내는 도메인 격차를 없앤다
#     현재 `df.resample('1ms').mean()` 은 1ms bin 평균 = 사실상 2-tap 이동평균이라
#     안티에일리어싱 필터 구실을 못 한다. 그래서 >500Hz 성분이 20-450Hz 안으로 접힌다.
#     이게 도메인 간 비대칭인 게 문제다 (08-04 측정, 네이티브 레이트 PSD):
#         samsung1 : >500Hz 전력 / 20-450Hz 전력 = 0.07% (biceps) / 0.05% (triceps)
#         samsung2 : 같은 비율이 12.52% / 5.88%
#     즉 target 에만 있는 고주파를 우리 손으로 대역 안에 넣고 있다. 네이티브에서
#     450Hz LPF 를 먼저 걸면 HF 격차의 15~26% 가 사라진다 (SSC d 2.51→2.12,
#     HF% Δ 4.72→3.49). 나머지 75~85% 는 samsung2 자체의 대역 내 노이즈 바닥이라
#     이걸로는 안 없어진다 — 그래서 이 실험은 "전이가 오를 것"을 기대하는 게 아니라
#     **우리 몫의 아티팩트를 닫는 것**이 목적이다. 전이가 안 올라도 성공이다.
#
# (2) EMG-only baseline 을 현행 규약으로 재생성한다
#     results/EMG/emg_baseline_summary.txt (85.91% → 29.71%, shift 56.21%p) 는
#     2026-06-25 자 = 7:1.5:1.5 3분할 **이전** 이라 무효다. test 규약으로 다시 뜬다.
#     selection 은 source val (source-only 라 target 라벨을 안 봄 → oracle 아님),
#     보고는 target_test_acc.
#
# 잡 구성 (EMG-only source-only, 원본 백본, seeds 0-4, 30ep)
#   전처리 2팔 : preprocessed_MM_raw_aaoff (대조) / preprocessed_MM_raw_aa (처리)
#   학습   2잡 : 각 팔에 대해 multi_seed 1회씩
#   합계 2잡, 예상 40~60분
#
# 왜 대조군을 기존 preprocessed_MM_raw 로 안 쓰나 —
#   그 폴더는 옛 코드로 만들어져 sessions_*.npy 가 없다. AA 말고 다른 것도 달라졌을
#   수 있어 A/B 가 오염된다. 두 팔 모두 **같은 코드·같은 manifest** 로 새로 뜬다.
#   분할은 preprocessed_MM_raw/split_manifest.json 을 재사용하므로 기존 표와 호환된다.
#
# 실행:   nohup bash EMG/run_emg_antialias_2026-08-04.sh > /dev/null 2>&1 &
# 진행:   tail -f logs/emg_antialias_2026-08-04/driver.log
# 이어하기: 결과 json 이 있고 preproc.emg_antialias 가 기대와 맞으면 건너뛴다. 전부 재실행은 REDO=1.
# ---------------------------------------------------------------------------
set -euo pipefail

# 이 스크립트는 EMG/ 안에 있지만 data_preprocess_MM.py 는 루트에 있으므로 루트로 이동한다.
cd "$(dirname "$0")/.."

# --- 이중 실행 방지 (07-27 에 17초 차로 두 번 떠서 결과 json 을 공유한 적 있음) ---
LOCK="/tmp/emg_antialias_2026-08-04.lock"
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
REDO="${REDO:-0}"

LOGDIR="logs/emg_antialias_2026-08-04"
EMG_RES="results/EMG"
mkdir -p "$LOGDIR" "$EMG_RES"
DRIVER="$LOGDIR/driver.log"

TRAIN="EMG/emg_baseline_train.py"
MANIFEST="preprocessed_MM_raw/split_manifest.json"
DIR_OFF="preprocessed_MM_raw_aaoff"
DIR_ON="preprocessed_MM_raw_aa"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER"; }
rule() { echo "==================================================================" | tee -a "$DRIVER"; }

{
  echo "###################################################################"
  echo "# EMG 안티에일리어싱 A/B + baseline 재생성  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# aaoff(대조) vs aa(네이티브 450Hz LPF) | seeds=$SEEDS | ${EPOCHS}ep | REDO=$REDO"
  echo "###################################################################"
} | tee -a "$DRIVER"

"$PY" -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
    | tee -a "$DRIVER"

START_TS=$SECONDS
DONE_N=0; FAIL_N=0; SKIP_N=0

# ---------------------------------------------------------------------------
# Phase 0 — 사전 준비물
# ---------------------------------------------------------------------------
rule
log "Phase 0 — 사전 준비물 확인"
[ -f "$MANIFEST" ] || { log "FAIL 기준 분할 manifest 가 없다: $MANIFEST"; exit 1; }
[ -f "$TRAIN" ]    || { log "FAIL 학습 스크립트가 없다: $TRAIN"; exit 1; }
log "OK  기준 분할=$MANIFEST"

# ---------------------------------------------------------------------------
# Phase 1 — 전처리 2팔 생성 (EMG 안티에일리어싱만 다르고 나머지 전부 동일)
# ---------------------------------------------------------------------------
rule
log "Phase 1 — 전처리 2팔 생성 (분할·R·IMU정규화 동일, EMG AA 만 다름)"
make_arm() {
    local out="$1"; shift
    if [ "$REDO" != "1" ] && [ -f "$out/y_target_test.npy" ] && [ -f "$out/preproc_config.json" ]; then
        log "SKIP   전처리 $out (이미 있음)"; return 0
    fi
    log "전처리 → $out   (extra: $*)"
    "$PY" -u data_preprocess_MM.py --method raw --split_manifest "$MANIFEST" \
          --out "$out" "$@" > "$LOGDIR/prep_$(basename "$out").log" 2>&1 \
        || { log "FAIL   전처리 $out — $LOGDIR/prep_$(basename "$out").log 확인"; return 1; }
    log "DONE   전처리 $out"
}
make_arm "$DIR_OFF"
make_arm "$DIR_ON" --emg_antialias --emg_aa_cutoff 450

# --- 두 팔이 정말 AA 하나만 다른지 검증 -------------------------------------
# 핵심 검사: AA 는 EMG 컬럼만 건드리므로 **IMU 배열은 비트 단위로 같아야 한다.**
# IMU 가 다르면 AA 외의 무언가가 같이 변한 것이고, A/B 해석이 무너진다.
log "두 팔 무결성 검사 (분할 동일 / IMU 동일 / EMG 만 다름)"
"$PY" - "$DIR_OFF" "$DIR_ON" <<'PYEOF' 2>&1 | tee -a "$DRIVER"
import json, os, sys
import numpy as np

off, on = sys.argv[1], sys.argv[2]
ok = True

def fail(msg):
    global ok
    print(f"  [FAIL] {msg}"); ok = False

# 1) 세션 분할이 완전히 같은가
try:
    a = json.load(open(os.path.join(off, "split_manifest.json")))
    b = json.load(open(os.path.join(on,  "split_manifest.json")))
    ka = {d: [sorted(a[d][s]) for s in ("train", "val", "test")] for d in ("source", "target")}
    kb = {d: [sorted(b[d][s]) for s in ("train", "val", "test")] for d in ("source", "target")}
    if ka == kb:
        print(f"  [OK]   세션 분할 동일  source={[len(x) for x in ka['source']]} "
              f"target={[len(x) for x in ka['target']]}")
    else:
        fail("세션 분할이 두 팔에서 다르다 — A/B 비교 불가")
except Exception as e:
    fail(f"split_manifest 읽기 실패: {e}")

# 2) preproc_config 가 emg_antialias 에서만 다른가
try:
    ca = json.load(open(os.path.join(off, "preproc_config.json")))
    cb = json.load(open(os.path.join(on,  "preproc_config.json")))
    diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    if ca.get("emg_antialias") is not False or cb.get("emg_antialias") is not True:
        fail(f"AA 플래그가 기대와 다르다: off={ca.get('emg_antialias')} on={cb.get('emg_antialias')}")
    elif diff <= {"emg_antialias", "emg_aa_cutoff"}:
        print(f"  [OK]   preproc_config 차이 = {sorted(diff)} (AA 관련만)  "
              f"cutoff={cb.get('emg_aa_cutoff')}Hz")
    else:
        fail(f"AA 외의 설정도 다르다: {sorted(diff)}")
except Exception as e:
    fail(f"preproc_config 읽기 실패: {e}")

# 3) IMU 는 비트 단위 동일, EMG 는 달라야 한다
for pre in ("train", "val", "test", "target_train", "target_val", "target_test"):
    try:
        ia = np.load(os.path.join(off, f"X_imu_{pre}.npy"), mmap_mode="r")
        ib = np.load(os.path.join(on,  f"X_imu_{pre}.npy"), mmap_mode="r")
        if ia.shape != ib.shape:
            fail(f"IMU {pre}: shape 불일치 {ia.shape} vs {ib.shape}")
        elif not np.array_equal(np.asarray(ia), np.asarray(ib)):
            fail(f"IMU {pre}: AA 는 EMG 만 건드려야 하는데 IMU 가 바뀌었다")
        ya = np.load(os.path.join(off, f"y_{pre}.npy"), allow_pickle=True)
        yb = np.load(os.path.join(on,  f"y_{pre}.npy"), allow_pickle=True)
        if not np.array_equal(ya, yb):
            fail(f"y {pre}: 라벨이 다르다")
        ea = np.load(os.path.join(off, f"X_emg_{pre}.npy"), mmap_mode="r")
        eb = np.load(os.path.join(on,  f"X_emg_{pre}.npy"), mmap_mode="r")
        if ea.shape != eb.shape:
            fail(f"EMG {pre}: shape 불일치 {ea.shape} vs {eb.shape}")
        else:
            n = min(512, ea.shape[0])
            d = float(np.abs(np.asarray(ea[:n], np.float64) - np.asarray(eb[:n], np.float64)).mean())
            if d == 0.0:
                fail(f"EMG {pre}: 두 팔이 완전히 같다 — AA 가 실제로 안 걸렸다")
            else:
                print(f"  [OK]   {pre:<13} IMU/라벨 동일, EMG 평균절대차 {d:.5f}")
    except Exception as e:
        fail(f"{pre} 배열 검사 실패: {e}")

sys.exit(0 if ok else 1)
PYEOF

# ---------------------------------------------------------------------------
# 공용 잡 실행기
#   결과 json 이 있어도 preproc.emg_antialias 가 기대와 다르면 다시 돈다
#   (07-29 러너는 config 를 안 보고 스킵해서 설정을 바꿔도 조용히 건너뛰었다).
# ---------------------------------------------------------------------------
run_job() {
    local label="$1" out="$2" logf="$3" want_aa="$4" datadir="$5" tag="$6"
    rule
    if [ "$REDO" != "1" ] && [ -f "$out" ]; then
        local got
        got="$("$PY" -c "import json;d=json.load(open('$out'));print(str(d.get('preproc',{}).get('emg_antialias','?')).lower())" 2>/dev/null || echo "?")"
        if [ "$got" = "$want_aa" ]; then
            log "SKIP   $label  (결과 있음, emg_antialias=$got 일치)"
            SKIP_N=$((SKIP_N + 1)); return 0
        fi
        log "재실행 $label  (결과는 있으나 emg_antialias=$got, 기대=$want_aa)"
    fi
    log "START  $label"
    echo "  data: $datadir" | tee -a "$DRIVER"
    echo "  log : $logf"    | tee -a "$DRIVER"
    local t0=$SECONDS rc=0
    env "MM_DATA_DIR=$datadir" \
        "$PY" -u "$TRAIN" --multi_seed --seeds $SEEDS --epochs $EPOCHS \
              --tag "$tag" > "$logf" 2>&1 || rc=$?
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
# Phase 2 — EMG-only baseline 2잡
#   혼동 행렬은 그대로 저장한다(--no_cm 안 씀). 2잡뿐이라 비용이 작고,
#   06-25 자 옛 CM 이 무효라 test 기준 CM 을 새로 확보해 두는 편이 낫다.
# ---------------------------------------------------------------------------
rule; log "Phase 2 — EMG-only source-only baseline (aaoff / aa)"
run_job "EMG baseline | AA off (대조)" \
        "$EMG_RES/emg_baseline_result_aaoff.json" \
        "$LOGDIR/emg_aaoff.log" "false" "$DIR_OFF" "aaoff"
run_job "EMG baseline | AA on 450Hz" \
        "$EMG_RES/emg_baseline_result_aa.json" \
        "$LOGDIR/emg_aa.log" "true" "$DIR_ON" "aa"

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
rule
log "요약 (총 ${DONE_N}완료 / ${SKIP_N}스킵 / ${FAIL_N}실패, $(( (SECONDS - START_TS) / 60 ))분)"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$DRIVER"
import json, os
import numpy as np

def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None

off = load("results/EMG/emg_baseline_result_aaoff.json")
on  = load("results/EMG/emg_baseline_result_aa.json")

def cell(j, k):
    if not j or k not in j.get("mean", {}):
        return f"{'—':>14}"
    return f"{j['mean'][k]:>7.2f} ±{j['std'][k]:>5.2f}"

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

print("\n=== EMG-only source-only baseline — 안티에일리어싱 A/B (TEST 기준) ===")
print("  selection = source val (leakage-free) | 보고 = target_test_acc")
print(f"\n  {'arm':<16}{'Src-Test':>15}{'Tgt-Test':>15}{'Test-Shift':>15}")
for lab, j in [("AA off (대조)", off), ("AA on 450Hz", on)]:
    print(f"  {lab:<16}{cell(j,'source_test_acc')}{cell(j,'target_test_acc')}{cell(j,'test_shift')}")

print("\n  paired Δ (같은 seed, AA on − AA off):")
for k, lab in [("source_test_acc", "Source Test"),
               ("target_test_acc", "Target Test"),
               ("test_shift",      "Test Shift ")]:
    print(f"    {lab:<14} {stat(paired(off, on, k))}")

print("""
읽는 법
  · 이 실험의 1차 목적은 "우리 전처리가 만든 아티팩트 제거"다. Target Test 가 안 올라도
    실패가 아니다 — 08-04 진단에서 에일리어싱은 HF 격차의 15~26% 뿐이고, 나머지는
    samsung2 자체의 대역 내 노이즈 바닥이라 LPF 로 없앨 수 있는 성질이 아니었다.
  · Δ 가 유의하게 +면 덤으로 전이도 얻은 것이고, 0 근처면 "아티팩트는 닫았고 갭은
    하드웨어 고유였다"는 결론이 선다. 어느 쪽이든 이후 실험은 AA on 을 기본으로 간다.
  · Δ 가 유의하게 −면 그건 뜻밖이므로 해석 전에 Phase 1 무결성 검사 로그부터 다시 볼 것.
  · 윈도우 90% 중첩 → 유효 표본수 명목의 약 1/10, 세션 1개 ≈ 1.37%p.
    Δ 가 1%p 안쪽이면 유의하다고 보기 어렵다.
  · Source Test 는 거의 안 변해야 정상이다(같은 도메인, 450Hz 위는 원래 비어 있음).
    Source 가 크게 떨어지면 LPF 가 신호를 깎은 것이므로 cutoff 를 다시 볼 것.

  주의 — results/EMG/emg_baseline_summary.txt (85.91% → 29.71%) 는 2026-06-25 자
  구 2분할이라 위 표와 **비교 금지**. 태그가 없어 파일명이 겹치지 않으니 남아는 있다.""")
PYEOF

rule
log "SWEEP DONE  (완료 ${DONE_N} / 스킵 ${SKIP_N} / 실패 ${FAIL_N})"
