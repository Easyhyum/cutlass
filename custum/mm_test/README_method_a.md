# Method A (V9 spatial SM ramp) — quick reference

## 빌드 (한 번만)

```bash
cd /workspace/custum
METHOD_A_RAMP_V9=1 python setup_bf16_sm80_streamk.py build_ext --inplace
```

`METHOD_A_RAMP_V9=1` 이 빠지면 V9 device 코드가 컴파일 안 됨 → baseline only.

## 실험 실행

```bash
cd /workspace/custum/mm_test
./run_eval.sh                                      # default: GPU 3, 100 bursts, full sweep, gap 600ms
GPU=0 ./run_eval.sh                                # 다른 GPU
MM_N_BURSTS=30 ./run_eval.sh                       # 빠른 sweep (10분)
MM_V9_STARTS=70,80,90 MM_V9_STEPS_NS=500,5000 ./run_eval.sh   # 부분 sweep
MM_OP=qkv_proj ./run_eval.sh                       # 다른 layer
TAG_PREFIX=myexp ./run_eval.sh                     # 파일 이름 prefix
```

자동으로 plot 생성됨. 출력:
- `logs/v9_<TS>_segments.csv` — 측정 raw
- `logs/v9_<TS>_segments_with_power.csv` — 각 burst의 power 통계
- `logs/v9_<TS>_gpu<N>_power.csv` — nvidia-smi 50ms log
- `logs/v9_<TS>_timeline.png`
- `logs/v9_<TS>_summary.png`
- `logs/v9_<TS>_analysis.txt`

## 환경 변수 전체 목록

| env var | default | 설명 |
|---|---|---|
| `GPU` | 3 | 사용할 GPU index |
| `MM_OP` | down_proj | layer (qkv_proj/o_proj/gate_up_proj/down_proj/lm_head) |
| `MM_M` | 8192 | batch×seq dimension |
| `MM_N_BURSTS` | 100 | bursts per config |
| `MM_M_KERNELS` | 150 | kernels per burst |
| `MM_BURST_GAP_MS` | 600 | burst 사이 idle gap (clock 회복용) |
| `MM_CFG_GAP_MS` | 500 | config 사이 추가 gap |
| `MM_GLOBAL_WARMUP_MS` | 3000 | 시작 시 warmup |
| `MM_V9_STARTS` | 60,65,70,75,80,85,90,95,100 | start_pct sweep |
| `MM_V9_STEPS_NS` | 500,2000,5000,10000,20000,50000 | step_ns sweep |
| `TAG_PREFIX` | v9 | 출력 파일 이름 prefix |

## 기존 데이터로 plot만 다시 생성

```bash
./make_plots.sh v9_20260522_043405                 # tag만 주면 자동으로 segments/power 찾음
./make_plots.sh gap_recovery_20260522_042613
```

## 다른 보조 스크립트

- `run_gap_recovery.sh` — gap_ms 변화에 따른 clock 회복 측정
- `plot_s100_zoom.py` — s100 baseline의 burst-idle toggling 자세히 보기
- `plot_gap_recovery.py` — gap recovery zoom plot

## V9 mechanism 빠른 reminder

```
host:                threshold = n_sms × start_pct / 100
                     prime_ramp_v9(threshold, step_ns)  # one-shot
                     또는 gemm_streamk(..., v9_smid_threshold=..., v9_step_ns=...)

device (operator() 진입 시 1회):
  if (smid >= threshold) {
      slot = smid - threshold
      __nanosleep(slot * step_ns)   // graduated delay per SM
  }
  # 그 후 baseline mainloop
```

- start_pct = 70 → 132 SMs 즉시 활성, 56 SMs는 1 by 1 graduated 시작
- step_ns = 500 → SMs 사이 0.5μs 간격, max delay 27.5μs (1.3% kernel)
- step_ns = 5000 → 2.5μs 간격, max delay 137.5μs (6.5% kernel)
- step_ns ≥ 10000 → max delay > kernel time, kernel이 delay-bound

## Config 이름 규칙

`s<start_pct>_step<step_ns>`

예:
- `s60_step500` → start_pct=60, step_ns=500
- `s100_step*` → ramp 비활성 (baseline)


## Run M Sweep
cd /workspace/custum/mm_test
MM_N_BURSTS=50 MM_BURST_GAP_MS=500 MM_CFG_GAP_MS=1500 \
  TAG=msweep_$(date +%H%M%S) ./run_M_sweep.sh