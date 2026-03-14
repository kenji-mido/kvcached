# AMD GPU (ROCm) Porting Status

Last updated: 2026-03-14

## Current Status: Code Complete, Awaiting AMD GPU Validation

## Completed

- [x] `gpu_compat.hpp` — CUDA/HIP abstraction layer (`#ifdef USE_ROCM`)
- [x] `gpu_utils.hpp` — GPU-agnostic error handling (replaces `cuda_utils.hpp`)
- [x] `constants.hpp` — `kGPUAllocGranularity` for dynamic alignment
- [x] `page.hpp` / `page.cpp` — type/API migration to `gpu_*` aliases
- [x] `ftensor.cpp` — type/API migration, dynamic alignment
- [x] `allocator.cpp` — `init_gpu_()`, VMM check skip on ROCm, dynamic granularity
- [x] `allocator.hpp` — `init_cuda_()` → `init_gpu_()` rename
- [x] `setup.py` — ROCm detection, `CppExtension` + `-DUSE_ROCM` + `amdhip64`
- [x] `kvcached/utils.py` — page size validation relaxed on ROCm
- [x] CUDA regression — build passes, pre-commit passes, CPU tests pass
- [x] Test infrastructure — `conftest.py` markers, `test_gpu_compat.py`, `test_rocm_vmm.py`
- [x] Documentation — `docs/amd-gpu-testing-guide.md`
- [x] Scripts — `scripts/setup_amd_dev.sh`, `scripts/setup_rocm_env.sh`, `scripts/run_amd_tests.sh`

## Next: AMD GPU Validation (MI300X)

### Phase 1: Build & Smoke (estimated: 30 min)

1. Transfer `setup_amd_dev.sh` to MI300X machine and run it
2. Verify `-DUSE_ROCM` build succeeds (links against `amdhip64`)
3. Run `bash scripts/run_amd_tests.sh --quick` (CPU-only tiers)
4. Run `bash scripts/run_amd_tests.sh --gpu` (GPU tiers)

### Phase 2: Core Functionality (estimated: 1-2 hours)

5. `test_rocm_vmm.py` — init/shutdown, granularity query, tensor alloc/free
6. `test_paged_allocator_aliasing.py` — VMM mapping correctness
7. `test_kvcache_manager.py` — alloc/free/resize/trim

### Phase 3: Integration (estimated: 2-4 hours)

8. vLLM ROCm integration test:
   ```bash
   pip install vllm  # ROCm build
   ENABLE_KVCACHED=true python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3-0.6B --no-enable-prefix-caching
   curl http://localhost:8000/v1/completions -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello"}'
   ```
9. SGLang ROCm integration test (if available)
10. Multi-model controller test (`controller/`)

### Phase 4: Stress & Edge Cases

11. Granularity edge cases (4096B vs 2MB page sizes)
12. Large model test (memory pressure)
13. Multi-GPU test (if available)

## Known Risks

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| `hipDeviceAttributeVMMSupported` = 0 | Init fails | `#ifndef USE_ROCM` skip | Done |
| `hipGetErrorString` signature | Build error | Wrapper in `gpu_compat.hpp` | Done |
| Granularity 4096B vs 2MB | Alignment error | Dynamic query + Python relaxation | Done |
| `kStartAddr` VA validity on MI300X | Mapping fails | VA space likely sufficient | Untested |
| `hipMemAllocationProp` layout | Struct mismatch | Designated initializers | Untested |
| ROCm vLLM/SGLang compatibility | Patching fails | Need ROCm-specific patches? | Unknown |

## Branch Info

- Branch: `feature/amd-gpu-support`
- Base: `main` (`bd0b8ed`)
- Remote: `kenji-mido/kvcached`
- Files changed: 16 (+679 / -97)

## Test Environment Requirements

- AMD Instinct GPU (MI300X recommended)
- ROCm 6.3+ (7.1.0 recommended)
- PyTorch ROCm build
- Python 3.10+

## Timeline

- KubeCon Japan 2026: 7/29-30
- CFP deadline: 3/29
- Target: validate on MI300X before CFP submission
