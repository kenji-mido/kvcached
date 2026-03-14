# AMD GPU (ROCm) Porting Status

Last updated: 2026-03-14

## Current Status: Validated on AMD Instinct MI300X (ROCm 7.1)

All HIP VMM APIs verified on real hardware. vLLM integration tested end-to-end.

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
- [x] E2E elastic memory benchmark — `tests/test_elastic_memory_e2e.py` (kvcached vs baseline)
- [x] Documentation — `docs/amd-gpu-testing-guide.md`, `docs/amd-gpu-status.md`
- [x] Scripts — `setup_amd_dev.sh` (full environment: kvcached + vLLM ROCm source build)

## Validated on Hardware

### Build & Unit Tests

| Test | Result |
|------|--------|
| C++ extension build (`-DUSE_ROCM`, links `libamdhip64.so.7`) | PASS |
| CPU-only tests (8 tests) | PASS |
| GPU compat tests (5 tests) | PASS |
| ROCm VMM smoke tests (3 tests: init, granularity, alloc/map/unmap) | PASS |

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
| Elastic memory benchmark (6 checks) | ALL PASS |

### SGLang Integration (E2E)

Tested with official Docker image (`lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x`).

| Test | Result |
|------|--------|
| SGLang v0.5.9 ROCm Docker | PASS |
| Autopatch (4/4 patches applied) | PASS |
| Inference with `facebook/opt-125m` | PASS — correct text generated |
| kvcached IPC (kvctl) memory tracking | PASS — Virtual/Physical/Used/Prealloc visible |

Note: SGLang on bare-metal (pip install) is not supported by SGLang itself.
The official Docker image is the recommended and tested deployment method.

### Elastic Memory Benchmark (`test_elastic_memory_e2e.py`)

Verified with `MAX_RESERVED_PAGES=2` to force `hipMemUnmap`:

| Check | Result |
|-------|--------|
| Physical << Virtual at idle (0.05%) | PASS |
| Physical grows on-demand during inference (48 → 960 MB) | PASS |
| Physical shrinks after completion (960 → 96 MB, hipMemUnmap) | PASS |
| GPU memory correlates with Physical changes | PASS |
| Used pages cycle (alloc → free) | PASS |
| Pages re-mapped on new requests | PASS |

Baseline comparison (vanilla vLLM without kvcached):
- vLLM allocates 98,498 MB upfront (50.2% of 192 GB)
- GPU memory stays flat during all phases — no elasticity

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

### Remaining Work

3. **Multi-GPU testing** — not yet tested with tensor parallelism on ROCm
4. **Larger model testing** — validated with opt-125m; test with larger models under memory pressure
5. **CI/CD** — add ROCm CI pipeline (requires AMD GPU runner)

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
