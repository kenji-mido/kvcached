# AMD GPU (ROCm) Porting Status

Last updated: 2026-03-14 (test quality session)

## Current Status: Validated on AMD Instinct MI300X — vLLM & SGLang

All HIP VMM APIs verified on real hardware via `hipMemGetInfo` physical GPU memory measurement.
Both vLLM and SGLang integration tested end-to-end via official ROCm Docker images.
Elastic memory (hipMemMap/hipMemUnmap) confirmed: per-page +8MB/-8MB delta measured, linear scaling verified.

## Completed

- [x] `gpu_compat.hpp` — CUDA/HIP abstraction layer (`#ifdef USE_ROCM`)
- [x] `gpu_utils.hpp` — GPU-agnostic error handling (replaces `cuda_utils.hpp`)
- [x] `constants.hpp` — `kGPUAllocGranularity` for dynamic alignment
- [x] `page.hpp` / `page.cpp` — type/API migration to `gpu_*` aliases
- [x] `ftensor.cpp` — type/API migration, dynamic alignment, `hipMemAddressReserve` type cast fix
- [x] `allocator.cpp` — `init_gpu_()`, VMM check skip on ROCm, dynamic granularity, contiguous layout size fix
- [x] `allocator.hpp` — `init_cuda_()` → `init_gpu_()` rename
- [x] `setup.py` — ROCm detection, `CppExtension` + `-DUSE_ROCM` + `amdhip64`, ROCm include/lib paths
- [x] `kvcached/utils.py` — page size validation relaxed on ROCm
- [x] CUDA regression — build passes, pre-commit passes, CPU tests pass
- [x] Test infrastructure — `conftest.py` markers, `test_gpu_compat.py`, `test_rocm_vmm.py`
- [x] E2E elastic memory benchmark — `tests/test_elastic_memory_e2e.py` (vLLM, kvcached vs baseline)
- [x] E2E elastic memory benchmark — `tests/test_sglang_e2e.py` (SGLang, kvcached vs baseline)
- [x] Documentation — `docs/amd-gpu-testing-guide.md`, `docs/amd-gpu-status.md`
- [x] Scripts — `setup_amd_dev.sh` (Docker-based: pulls official ROCm images, runs all tests)

## Validated on Hardware

### Build & Unit Tests

| Test | Result |
|------|--------|
| C++ extension build (`-DUSE_ROCM`, links `libamdhip64.so.7`) | PASS |
| CPU-only tests (8 tests) | PASS |
| GPU compat tests (15 tests) | PASS |
| ROCm VMM tests (7 tests: init, granularity, create, map/unmap, GPU memory, linear delta, lifecycle) | PASS |

### HIP VMM API Hardware Verification

Each API verified by observing `hipMemGetInfo()` GPU memory changes on MI300X:

| HIP API | Operation | GPU Memory Delta | Status |
|---------|-----------|-----------------|--------|
| `hipMemGetAllocationGranularity` | Query allocation granularity | N/A | PASS |
| `hipMemAddressReserve` | Reserve virtual address space | -152 MB | PASS |
| `hipMemCreate` | Allocate physical GPU memory | (included in map) | PASS |
| `hipMemMap` | Map physical → virtual (1 page) | -8 MB | PASS |
| `hipMemSetAccess` | Set read/write access | N/A | PASS |
| GPU read/write | `tensor[0] = 42` on mapped memory | N/A | PASS |
| `hipMemUnmap` | Unmap physical memory | +8 MB freed | PASS |
| `hipMemRelease` | Release physical allocation | (included in unmap) | PASS |
| `hipMemAddressFree` | Free virtual address | (at shutdown) | PASS |

Binary verification: `vmm_ops.so` links to `libamdhip64.so.7`, `libhsa-runtime64.so.1` — 11 HIP VMM symbols resolved.

### vLLM Integration (E2E)

Tested with both source build and official Docker image (`vllm/vllm-openai-rocm:latest`).

| Test | Result |
|------|--------|
| vLLM v0.17.1 ROCm Docker | PASS |
| ROCm platform detection | PASS |
| Autopatch (5/6 patches applied, v0.9+ path) | PASS |
| Inference with `facebook/opt-125m` | PASS — correct text generated |
| Elastic memory benchmark (7 checks) | ALL PASS |

### SGLang Integration (E2E)

Tested with official Docker image (`lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x`).

| Test | Result |
|------|--------|
| SGLang v0.5.9 ROCm Docker | PASS |
| Autopatch (4/4 patches applied) | PASS |
| Inference with `facebook/opt-125m` | PASS — correct text generated |
| Elastic memory benchmark (7 checks) | ALL PASS |
| kvcached IPC (kvctl) memory tracking | PASS — Virtual/Physical/Used/Prealloc visible |

SGLang elastic benchmark results (64 prompts × 512 tokens, `MAX_RESERVED_PAGES=2`):
- Physical peak: 912 MB (0.93% of 97,648 MB virtual)
- hipMemUnmap freed: 768 MB after completion
- GPU memory correlated with Physical changes

Note: SGLang on bare-metal (pip install) is not supported by SGLang itself.
The official Docker image is the recommended and tested deployment method.

### Elastic Memory Benchmark (`test_elastic_memory_e2e.py` / `test_sglang_e2e.py`)

Verified with `MAX_RESERVED_PAGES=2` to force `hipMemUnmap`.
Both vLLM and SGLang pass all 7 checks:

| Check | What it proves | Result |
|-------|---------------|--------|
| Physical << Virtual at idle (< 5%) | Not pre-allocated | PASS |
| Large batch peak > Small batch peak | On-demand growth | PASS |
| GPU memory << Virtual KV limit (< 10%) | No upfront allocation | PASS |
| Used pages varied during inference | alloc/free cycle works | PASS |
| Physical shrank after completion | hipMemUnmap confirmed | PASS |
| Pages re-mapped on new requests | Page reuse works | PASS |
| GPU memory correlates with Physical growth | Real hardware, not bookkeeping | PASS |

Baseline comparison (vanilla vLLM/SGLang without kvcached):
- Engines allocate KV cache memory upfront (50%+ of GPU)
- GPU memory stays flat during all phases — no elasticity
- 3/3 baseline checks PASS (confirms NON-elastic behavior)

### Physical GPU Memory Verification (`test_rocm_vmm.py`)

Measured via `hipMemGetInfo` (`torch.cuda.mem_get_info`), not mocks:

```
Map page 0 (offset=0MB):  482 MB → 490 MB  (+8 MB)
Map page 1 (offset=8MB):  490 MB → 498 MB  (+8 MB)
Map page 2 (offset=16MB): 498 MB → 506 MB  (+8 MB)
Map page 3 (offset=24MB): 506 MB → 514 MB  (+8 MB)
Unmap all:                 514 MB → 482 MB  (-32 MB, exact baseline return)
```

Each compound page = `page_size × num_layers × num_kv_buffers` = 2MB × 2 × 2 = **8 MB**.
Delta is exactly +8 MB per map, -8 MB per unmap, linear scaling confirmed.

## Known Issues (Resolved)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `hip/hip_runtime.h: No such file` | `setup.py` missing ROCm include path | Added `/opt/rocm/include` to `include_dirs` |
| `hipMemAddressReserve` type mismatch | HIP takes `void*`, CUDA takes `CUdeviceptr` (integer) | Added `reinterpret_cast<gpu_devptr_t>()` |
| `test_kv_tensor_alloc_free` segfault | `total_kv_size < compound_page_size` in contiguous layout | Round up `total_kv_size` to multiple of `compound_page_size` |
| `hipDeviceAttributeVMMSupported` = 0 | ROCm bug | Skipped via `#ifndef USE_ROCM` |
| SGLang `sgl-kernel` bare-metal | Wheel not published for ROCm 7.1 | Use official Docker image instead |

## Next Steps

### Ready for Upstream Contribution

1. **PR to `kenji-mido/kvcached` main branch** — merge `feature/amd-gpu-support`
2. **Upstream to original kvcached repo** — the 3 C++ fixes + setup.py change are minimal and safe

### Test Quality Improvements (Completed)

1. **`test_gpu_compat.py` — ROCm detection tests replaced** ✅
   - Was: inline `torch.version.hip is not None` (tautological — only tested Python itself)
   - Now: imports and tests the actual `_is_rocm()` function from `kvcached.utils`,
     plus a consistency check against `setup.py`'s `IS_ROCM`
   - Added: `test_is_rocm_matches_runtime` (no mocks — tests real environment)
   - Added: `test_setup_py_is_rocm_consistent` (subprocess check)
2. **`test_gpu_compat.py` — error cases expanded** ✅
   - Added: `test_page_size_rocm_rejects_non_1mb_aligned` (ROCm accepts 3MB)
   - Added: `test_page_size_rejects_float_string` (e.g. "2.5")
3. **`test_rocm_vmm.py` — physical GPU memory verification added** ✅
   - **Bug found and fixed**: tests were passing `[0, 1, 2, 3]` as page indices, but
     `map_to_kv_tensors` expects byte offsets (multiples of `compound_page_size = 8 MB`).
     All small integers mapped to `page_id = N / 8MB = 0`, so only page 0 was ever mapped.
     Tests passed coincidentally because 1 page = 8 MB matched the (wrong) threshold.
   - Fixed: tests now pass correct byte offsets `[0, 8MB, 16MB, 24MB]`
   - Added: `test_map_increases_gpu_memory` — 4 pages → +32 MB, unmap → -32 MB
   - Added: `test_map_per_page_delta_is_linear` — maps pages one at a time,
     asserts exactly +8 MB per page via `hipMemGetInfo` (no tolerance, exact match)
   - Added: `test_map_unmap_cycle_returns_memory` — full lifecycle:
     map → write → read → unmap (exact baseline return) → remap (same delta)
   - Added: `tests/diag_gpu_memory.py` — standalone diagnostic that prints raw
     `hipMemGetInfo` values at every step for manual verification
4. **`test_elastic_memory_e2e.py` / `test_sglang_e2e.py` — thresholds documented and dynamic** ✅
   - All magic numbers now have inline comments explaining their rationale
   - Baseline thresholds derived from `gpu_total_bytes()` (portable across GPU sizes)
   - Added Check 7: "GPU memory correlates with Physical growth" — ensures VMM
     operations affect real hardware, not just bookkeeping counters
5. **`setup_amd_dev.sh` — pipeline exit code bug fixed** ✅
   - Changed `$?` → `${PIPESTATUS[0]}` for all 4 Docker pipeline commands
   - Summary already checks all 4 results (VLLM_ELASTIC, VLLM_BASELINE,
     SGLANG_ELASTIC, SGLANG_BASELINE)

### Remaining Work

1. **Multi-GPU testing** — not yet tested with tensor parallelism on ROCm
2. **Larger model testing** — validated with opt-125m; test with larger models under memory pressure
3. **CI/CD** — add ROCm CI pipeline (requires AMD GPU runner or Docker-in-Docker)

## Test Environments

### Bare-metal (vLLM source build)

- GPU: AMD Instinct MI300X VF (192 GB, gfx942)
- ROCm: 7.1.0
- PyTorch: 2.10.0+rocm7.1
- vLLM: 0.17.1+rocm710 (source build)
- Python: 3.12.3
- OS: Ubuntu 24.04 (Linux 6.8.0)

### Docker — vLLM

- Image: `vllm/vllm-openai-rocm:latest`
- PyTorch: 2.9.1+rocm7.0
- vLLM: 0.17.1

### Docker — SGLang

- Image: `lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312`
- PyTorch: 2.9.1+rocm7.2
- SGLang: 0.5.9

## Branch Info

- Branch: `feature/amd-gpu-support`
- Base: `main` (`bd0b8ed`)
- Remote: `kenji-mido/kvcached`

## Timeline

- KubeCon Japan 2026: 7/29-30
- CFP deadline: 3/29
- Target: validated on MI300X before CFP submission ✅
