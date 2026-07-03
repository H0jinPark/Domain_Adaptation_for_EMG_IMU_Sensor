# 연구 보고서 — R 학습 궤적 안정화 3종 실험 (파라미터화 · LR 타이밍)
**날짜:** 2026-07-03
**대상:** Samsung1(source)→Samsung2(target), IMU 단독, 10종 운동, JOINT CDAN + PCA/gravity prior
**한 줄:** seed42 는 Target 77% 로 잘 되는데 다른 seed 는 R 이 늦게 수렴해 붕괴한다 — 이 "R 학습 타이밍" 문제를 세 갈래(LR 워밍업 · 쿼터니언 · 자유행렬+정칙화)로 각각 별도 파일에 구현했다.

---

## 1. 문제 정의 (왜 이 실험인가)

- **증상:** 최고 조합(`pca_grav_cdan_joint`)이 seed42 에서 Target **77.2%**. 그러나 multi-seed(gravity+CDAN, `free_g1`)는 **66.6 ± 5.4%**, seed2 는 **58%로 붕괴**.
- **진단(학습 양상 비교):** R 에는 사실상 **정답에 가까운 방향**이 존재하고, 모든 seed 가 **결국엔** 그 R 로 수렴한다. 그런데 비-42 seed 에선 **R 이 정해지는 속도가 인코더·판별기 학습보다 느리다.** R 이 한 번 **반대 방향으로 갔다가 제자리로 돌아오는데**, 그 사이 인코더/판별기가 이미 **잘못 정렬된 신호에 적응**해버려 나쁜 국소해에 갇힌다.
- **따라서 레버는 "R 이 인코더보다 먼저·안정적으로 정답 방향을 잡게" 하는 것.** 두 축으로 공략: **(A) 학습 타이밍**(인코더를 늦춤) 과 **(B) R 파라미터화**(초반 방향전환을 쉽게).

---

## 2. 공통 구조 (중복 제거)

세 실험이 손실·데이터·CDAN·평가 규약을 공유하므로 공통 로직을 한 모듈로 묶고, 세 아이디어는 **각각 실행 가능한 별도 entry 파일**로 두었다(요청대로 하나씩 돌릴 수 있음).

- **`Learnable_R/learnable_r_cdan_common.py`** — `run_epoch`/`train`/`evaluate_r`/`write_result_json`/`run_and_summarize`/`add_common_args`.
  - 원본 `learnable_r_cdan_train.py` 대비 일반화한 곳만:
    1. R 의 leaf 파라미터를 파라미터화와 무관하게 `next(model.r.parameters())` 로 잡음 (so3=`w`(3), quat=`q`(4), matrix=`M`(3×3)) → freeze 판정·∂R gradient 진단 공용.
    2. 손실에 `λ_rot·L_rot` 항 추가(자유행렬 전용, so3/quat 는 λ_rot=0).
    3. 인코더/판별기 LR 워밍업 훅(아이디어 1).
- **모델 `learnable_r_model.py` 추가분:**
  - `LearnableRQuat` (단위 쿼터니언), `LearnableRMatrix` (자유 3×3), `build_learnable_r(r_param, init)` 팩토리.
  - `rotation_reg_loss(R) = ‖RᵀR−I‖²_F + (det R−1)²`.
  - `LearnableRCDAN(..., r_param="so3"|"quat"|"matrix")` — 기본 so3(기존 스크립트 하위호환).

세 entry 파일은 `run_and_summarize(args, name, r_param, ...)` 만 호출하고 아이디어별 인자만 추가한다.

---

## 3. 세 실험 구현

### 3.1 아이디어 1 — 인코더/판별기 LR 워밍업 (파일 `learnable_r_cdan_lrsched_train.py`, R=SO(3) 유지)

- **아이디어:** R 에게 선두를 준다. 인코더/판별기(optimizer group 0)의 lr 을 초반 `--enc_warmup_epochs` 동안 `--enc_lr_floor`→1.0 **선형 워밍업**, R lr(group 1)은 **처음부터 풀강도**. 초반엔 R 이 물리 prior(PCA/gravity)로 정답 방향을 먼저 잡고, 인코더/판별기는 그 뒤 **이미 옳게 정렬된** 신호에 적응.
- **구현:** cosine 스케줄 값 위에 그 에폭만 배수를 곱한다. 다음 `scheduler.step()` 이 `base_lr` 로 재계산하므로 배수가 누적되지 않음(워밍업이 그 에폭에만 국소 적용).
  ```python
  if enc_warmup_epochs > 0 and epoch < enc_warmup_epochs:
      warm = enc_lr_floor + (1 - enc_lr_floor) * (epoch + 1) / enc_warmup_epochs
      optimizer.param_groups[0]["lr"] *= warm   # group 0 = 인코더/판별기만
  ```
- **R 파라미터화는 원본 SO(3)(so3_exp) 그대로** → 순수하게 "학습 타이밍" 효과만 격리.
- **기본값:** `--enc_warmup_epochs 10 --enc_lr_floor 0.1`. (인코더/R 상대속도 자체는 원래 `--lr/--r_lr` 로도 조절되지만, 이 파일이 더하는 건 "초반에만 늦췄다 복귀"하는 **타이밍**이다.)

### 3.2 아이디어 2a — 단위 쿼터니언 (파일 `learnable_r_cdan_quat_train.py`)

- **아이디어:** R 의 **표현**을 바꿔 초반 방향전환을 쉽게. 원본 so(3) 행렬지수는 회전각 θ→0 근처(모든 R 이 항등에서 출발 = 정확히 이 영역)에서 sin θ/θ·(1−cos θ)/θ² 항 때문에 gradient 가 왜곡·정체되기 쉽다.
- **구현:** `q=(w,x,y,z)` nn.Parameter, forward 마다 `q/‖q‖` 정규화 후 회전행렬로 변환.
  - init `q=(1,0,0,0)` → R=I 에서 출발(원본과 동일 출발점).
  - **파라미터 3→4** 개(최적화 여유), 항등 부근 **특이점 없음**(매끄러운 landscape) 가설.
  - 정규화로 **det=+1·RᵀR=I 구조 보장** → `λ_rot=0`(정칙화 불필요).
- 손실·데이터·CDAN 규약은 원본과 **완전히 동일**, R 표현만 다름. 스모크: 섭동에도 det=+1·‖RᵀR−I‖=1e-7, ∂L_pca/∂q norm 정상.

### 3.3 아이디어 2b — 자유 3×3 행렬 + 정칙화 손실 (파일 `learnable_r_cdan_matrix_train.py`)

- **아이디어:** 회전을 **구조로 강제하지 않는다.** R=M 자유행렬(9성분, 유클리드 공간 자유이동)이라 초반 재정렬이 쉽다는 가설. proper rotation 은 **손실**로 부드럽게 당김 → **모든 물리 prior(회전·중력·PCA)를 손실 항으로 통일** 표현.
- **구현:** `M=nn.Parameter(eye(3))`, `R=M` 그대로.
  - `L_rot = ‖MᵀM−I‖²_F + (det M−1)²`, 가중치 `--lambda_rot`(기본 1.0).
  - `‖MᵀM−I‖²` = 축 직교·단위, `(det−1)²` = 반사 아닌 proper rotation·핸디드니스.
- **대가:** λ_rot 가 약하면 det 붕괴·비회전해로 샐 수 있어 튜닝 필요(그래서 `--lambda_rot` 노출). 스모크: init 은 정확히 회전(L_rot=0), 섭동 시 det=1.17·‖RᵀR−I‖=1.86 로 벗어나며 L_rot=3.5 로 당김 확인.

---

## 4. 실행 방법 (하나씩)

공통 데이터: `MM_DATA_DIR=preprocessed_MM_raw_isotropic`, 권장 `--epochs 60 --multi_seed`(seed 0~4 로 재현성 직접 검증이 목적). env `DA`, 유저가 직접 실행.

```bash
# 아이디어 1 — 인코더/판별기 LR 워밍업 (SO(3))
MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_lrsched_train.py \
    --multi_seed --epochs 60 --enc_warmup_epochs 10 --enc_lr_floor 0.1 --tag warmup10

# 아이디어 2a — 쿼터니언
MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_quat_train.py \
    --multi_seed --epochs 60 --tag quat

# 아이디어 2b — 자유행렬 + rotation reg
MM_DATA_DIR=preprocessed_MM_raw_isotropic python Learnable_R/learnable_r_cdan_matrix_train.py \
    --multi_seed --epochs 60 --lambda_rot 1.0 --tag matrix
```

**판정 기준:** 목표는 최고치 갱신이 아니라 **재현성**. baseline = SO(3) 원본 `free_g1` Target **66.6 ± 5.4%**(seed2 58% 붕괴). 각 실험에서 (1) **std 가 줄고** (2) **seed2 붕괴가 사라지는지**, 그리고 로그의 **∠fromI / ∂R gradient 궤적**으로 R 이 초반에 정답 방향을 더 빨리·안정적으로 잡는지 본다.

---

## 5. 산출물

- 신규: `learnable_r_cdan_common.py`(공통), `learnable_r_cdan_lrsched_train.py`(1), `learnable_r_cdan_quat_train.py`(2a), `learnable_r_cdan_matrix_train.py`(2b).
- 모델 추가: `LearnableRQuat`·`LearnableRMatrix`·`build_learnable_r`·`rotation_reg_loss`, `LearnableRCDAN(r_param=...)`.
- 결과 저장: `results/Learnable_R/{name}_result_{tag}.json`, `{name}{tag}_summary.txt`, R 은 `results/R_matrices/R_{name}{tag}_seed*.npy`.
- 원본 `learnable_r_cdan_train.py` 는 그대로 보존(비교 baseline).

## 6. 다음 단계
1. 세 실험 multi-seed 실행 → baseline(66.6±5.4%) 대비 std·seed2 붕괴 비교.
2. 가장 안정적인 파라미터화를 고르고, 필요시 아이디어 1(LR 워밍업)과 **결합**(직교 개선이라 함께 적용 가능).
3. R 궤적 로그(∠fromI, ∂R gradient)로 "R 이 인코더보다 먼저 수렴" 가설 직접 확인.
