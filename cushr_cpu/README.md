# cushr_cpu

CPU reference decoder for the cuSHR project (Week 3 deliverable).

This is the **correctness oracle** that every later GPU kernel is checked
against. The data structures and tie-break rules in this library are the
canonical specification — the GPU kernels in weeks 4, 6, 8 must produce
matching output triple-by-triple.

## What's in here

```
cushr_cpu/
├── include/cushr/        public headers
│   ├── lattice.hpp       Lattice class, CSR + reverse-CSR storage
│   ├── scorer.hpp        EdgeScorer interface + 3 implementations
│   ├── decoder.hpp       TopKDecoder (the main artifact of week 3)
│   ├── metrics.hpp       Word-level F1, Perfect Match
│   └── json_io.hpp       Golden-output JSON writer
├── src/                  implementations
├── apps/evaluate.cpp     end-to-end CLI: load .npz, decode, score, dump JSON
├── tests/                doctest unit tests
└── CMakeLists.txt
```

## Build

Requires CMake ≥ 3.16, a C++17 compiler, and zlib. `cnpy` (for `.npz`) and
`doctest` are pulled in via `FetchContent`, so the only system dep is zlib.

```bash
cd cushr_cpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Use

Once Week 2's ingest has produced `cushr_data.npz`:

```bash
./build/cushr_evaluate cushr_data.npz \
    --scorer log_linear \
    --weights 0.7,-0.2,0.5,...   # one weight per feat_dim \
    --bias 0.0 \
    --K 10 \
    --golden golden_outputs.json \
    --golden-n 500
```

Expected output (on SIGHUM-test, per the project plan):

- Top-1 word-level F1 ≈ 85% with a hand-tuned `LogLinearScorer`
- ≈ 100 sentences/sec single-threaded
- `golden_outputs.json` containing top-10 paths for 500 sentences. This is
  the regression artifact for Week 4.

## The data structure (why these choices)

Each node `v` stores a `std::vector<Entry>` of at most `K` entries:

```cpp
struct Entry {
    float score;
    int   parent_node;   // -1 for source
    int   parent_rank;   // -1 for source
};
```

We do **not** materialise full `vector<int>` paths at every node. Paths are
reconstructed only at the sentence sink, by walking backpointers
(`reconstruct(sink, rank)`).

**Why backpointers, not full paths?** Two reasons:

1. The GPU kernel in Week 6 stores `(score, parent, rank)` triples — that's
   the spec in the project plan. CPU/GPU comparison is direct, no
   translation layer needed.
2. Memory is `O(N · K)` instead of `O(N · K · L)` where `L` is path length.
   At `K = 32` and average `L = 20`, that's a 20× win that you also need
   on the GPU.

**Tie-break:** when two entries tie on `score`, ordering is by
`(parent_node, parent_rank)` ascending. This is deterministic, mirror-able
on the GPU, and is the same rule used by `EntryGreater` in `decoder.cpp`.

## The candidate-merge algorithm (collect-and-sort)

For each destination node `v`:

1. For each incoming edge `e = (u, v)`, for each `r ∈ 0..K-1` in `topk[u]`:
   push `(topk[u][r].score + score(e), u, r)` into a scratch buffer.
2. `std::partial_sort` the buffer; keep the first `min(K, buf.size())`.

`std::partial_sort` is `O(C log K)` where `C = in_degree(v) · K`. On SIGHUM
in-degrees are 2–10 and `K ≤ 32`, so `C ≤ 320`. A more sophisticated
k-way-merge would be asymptotically tighter but invisible at this scale and
much harder to read. The reference oracle should be obvious correct.

## What's intentionally not here

- **Parallelism.** Single-threaded. The whole point is to have one place
  whose output is unambiguous.
- **Performance tuning.** No SIMD, no cache-blocking. Throughput target is
  100 sent/s, which we hit comfortably.
- **Lattice mutation.** Loader produces immutable `Lattice` objects.

## Things to check before Week 4

- [ ] All tests pass: `ctest --test-dir build --output-on-failure`
- [ ] `cushr_evaluate` runs end-to-end on Week 2's `.npz`
- [ ] `golden_outputs.json` exists and is byte-stable across runs
  (re-run the harness twice, diff the files — should be identical)
- [ ] Top-1 F1 in a sensible range (≈ 85% with `LogLinearScorer`)
- [ ] Throughput ≈ 100 sentences/sec
