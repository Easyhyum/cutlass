# test_torch_chunked_ol — torch-level M-chunking with tail OVERLAP

Same sweep as `test_torch_chunked` but **`pipe` mode uses 2 alternating MMA
streams** so chunk N+1 can start on stream B while chunk N is still finishing
on stream A.  Because CUDA only overlaps when SMs are free, this gives a
small **tail-overlap** (chunk N's last partial wave's idle SMs are picked up
by chunk N+1's first wave) without forcing full-power concurrent execution.

## Pipe mode design

```
chunk i → streams[i % 2]            (i.e. alternating)
copy of chunk i → s_copy stream      (CUTLASS only; cublas uses out= → no copy)
```

| measurement | torch_chunked (strict) pipe | torch_chunked_ol pipe |
|---|---|---|
| MMAs on s_mma stream | strictly sequential, never overlap | **alternate streams → tail-overlap** |
| copy on s_copy | parallel with next MMA | parallel with next MMAs (any stream) |
| TFLOPS recovery | none (pipe ≈ seq) | **+ tail-latency reclaimed** |
| power | seq-like | slightly higher (tail SMs filled) |

## Files

```
test_torch_chunked_ol/
├── README.md
├── eval_torch_chunked.py    pipe mode = 2-stream alternating
├── plot_chunked.py
├── plot_timeline_v9.py
├── run.sh                   tag prefix tchunk_ol_
└── watch_plot.sh
```

## Run (GPU 0 by default — keeps memory rule)

```bash
cd /workspace/custum/mm_test/test_torch_chunked_ol
./run.sh
```

Output in `logs/tchunk_ol_<TS>/`.

Compare with `test_torch_chunked/logs/tchunk_065255/` to see whether tail-
overlap recovers TF closer to BASE.
