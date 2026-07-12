# cushr_gpu/Makefile
#
# Builds the Week-4 K=1 GPU Viterbi binary and the Week-5 kernel unit tests.
#
# Toolchain assumptions (the lab A100 box):
#   - CUDA Toolkit 12.x (nvcc on PATH)
#   - A100 => compute capability 8.0 => -arch=sm_80
#   - g++ for the host C++ that nvcc dispatches to
#
# We REUSE the CPU library sources (Lattice loader, scorers, decoder, cnpy) so
# the GPU run loads the .npz and scores edges with the exact same code as the
# oracle. nvcc compiles those .cpp files as host code automatically.

NVCC      ?= nvcc
ARCH      ?= sm_80
# -O2 host opt, C++17 to match cushr_cpu, and the include dirs for the CPU lib.
NVCCFLAGS  = -std=c++17 -O2 -arch=$(ARCH) \
             -I. -I../cushr_cpu/include -I../cushr_cpu/vendor
LDLIBS     = -lz   # cnpy needs zlib

# CPU library sources we link against (host-compiled by nvcc).
CPU_DIR  = ../cushr_cpu
CPU_SRC  = $(CPU_DIR)/src/lattice.cpp $(CPU_DIR)/src/scorer.cpp \
           $(CPU_DIR)/src/decoder.cpp $(CPU_DIR)/src/metrics.cpp \
           $(CPU_DIR)/vendor/cnpy.cpp

# Device + host driver sources for the main binary.
GPU_SRC  = viterbi_k1.cu host_driver.cu

# Device + test sources for the unit-test binary.
TEST_SRC = viterbi_k1.cu tests/test_viterbi_k1.cu

.PHONY: all test test-kbest bench clean

all: cushr_viterbi_k1

# Main binary: K=1 GPU decode + bit-identical comparison to CPU.
cushr_viterbi_k1: $(GPU_SRC) $(CPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(GPU_SRC) $(CPU_SRC) $(LDLIBS)

# Week-5 unit tests (synthetic in-memory lattices, edge cases).
cushr_viterbi_tests: $(TEST_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(TEST_SRC) $(LDLIBS)

test: cushr_viterbi_tests
	./cushr_viterbi_tests

# ---- Week-6 K-best (K2) --------------------------------------------------
# K-best decode driver: runs the warp-level k-best merge over every sentence
# and checks score-equivalence against the Week-3 CPU TopKDecoder.
KBEST_SRC = kbest_merge_warp.cu host_driver_kbest.cu
cushr_kbest: $(KBEST_SRC) $(CPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(KBEST_SRC) $(CPU_SRC) $(LDLIBS)

# K-best unit tests (synthetic lattices, no CPU lib / npz needed).
KBEST_TEST_SRC = kbest_merge_warp.cu tests/test_kbest_merge.cu
cushr_kbest_tests: $(KBEST_TEST_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(KBEST_TEST_SRC) $(LDLIBS)

test-kbest: cushr_kbest_tests
	./cushr_kbest_tests

# ---- Week-7 benchmarking (CP-4) ------------------------------------------
# The actual runs happen on the cluster via slurm (nvcc/ncu are not local):
#   sbatch bench_kbest.slurm      # full-dataset timing+recall -> kbest_bench.csv
#   sbatch profile_kbest.slurm    # Nsight Compute            -> ncu_kbest_K*.csv
# Then, anywhere with Python:
#   python make_benchmark_md.py   # -> KBEST_BENCHMARK.md + plots
bench: cushr_kbest
	@echo "Built cushr_kbest. Now submit the benchmark jobs:"
	@echo "  sbatch bench_kbest.slurm"
	@echo "  sbatch profile_kbest.slurm"
	@echo "then: python make_benchmark_md.py"

clean:
	rm -f cushr_viterbi_k1 cushr_viterbi_tests cushr_kbest cushr_kbest_tests
