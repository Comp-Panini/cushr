# cushr_gpu — Week 4/5 GPU decoder

The GPU half of cuSHR. Week 4 implements the single-best (K=1) Viterbi
relaxation and proves it is **bit-identical** to the CPU `TopKDecoder` at K=1.
Week 5 adds kernel unit tests, a profiling playbook, and the K>1 design doc.

## Files

| File | Role |
|------|------|
| `gpu_lattice.cuh` | Device-side lattice view: reverse CSR + precomputed edge scores + K=1 outputs (`best_score`, `best_parent`). |
| `viterbi_k1.cu` | The kernels: `init_best` (base case) and `viterbi_relax_level` (one warp per destination node, strided in-edge scan, `__shfl_down_sync` argmax with the CPU tie-break). |
| `host_driver.cu` | Loads `.npz` via the CPU loader, precomputes per-edge scores with the same `EdgeScorer`, runs the per-sentence level sweep, reconstructs paths, and compares to the CPU decoder. |
| `tests/test_viterbi_k1.cu` | Week-5 edge-case unit tests (empty / single-node / single-edge / high-branching / tie-break). |
| `Makefile` | Builds both binaries with `nvcc -arch=sm_80`. |
| `PROFILING.md` | Nsight Systems / Compute playbook + results template. |
| `kbest_design.md` | Week-6 warp-level bitonic top-K design (for advisor sign-off). |

## Build & run (lab A100 box, CUDA 12.x)

```bash
cd cushr_gpu
make                       # builds cushr_viterbi_k1
./cushr_viterbi_k1 ../data/cushr_data_full_with_gold.npz --scorer log_linear
```

For the `log_linear` scorer, pass the same weights/bias you used on the CPU side
(`--weights w0,w1,... --bias b`); with none given it falls back to all-ones so
the binary still runs as a smoke test. `--limit N` controls how many sentences
are checked (default 1000).

Run the unit tests:

```bash
make test                  # builds + runs cushr_viterbi_tests
```

## Week-4 acceptance criteria

- **Correctness:** `BIT-IDENTICAL: N/N sentences match CPU` for at least 1000
  sentences. The check compares the sink's `best_score` as raw 32-bit ints (no
  epsilon) and demands the reconstructed node list equals the CPU's top-1 path.
- **Latency:** 50–200 µs/sentence for the level sweep. This is intentionally
  slow — one kernel launch per topological level means we are **launch-overhead
  bound**, which is exactly what Week 8 batching fixes.
- **Profile:** capture an Nsight report (see `PROFILING.md`) confirming the time
  is dominated by launch/sync overhead, not memory or compute.

## Why bit-identical is achievable

The only floating-point operation that affects the result is `best_score[u] +
edge_score`. The edge scores are computed **once on the host with the very same
`EdgeScorer`** the CPU decoder uses, then shipped to the device — so the inputs
are identical bit-for-bit. A max-reduction over distinct floats is
order-independent, and the kernel resolves score ties to the smaller parent
index, mirroring the CPU `EntryGreater` rule. Hence identical winners.
