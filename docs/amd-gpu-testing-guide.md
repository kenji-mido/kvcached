# AMD GPU (ROCm) Testing Guide

This document describes how to verify the HIP/ROCm port of kvcached on an AMD Instinct GPU (e.g. MI300X).

## What was done

The codebase now supports both NVIDIA CUDA and AMD ROCm via `#ifdef USE_ROCM` conditional compilation. Key changes:

| Component | Change |
|-----------|--------|
| `csrc/inc/gpu_compat.hpp` | **New.** Abstraction layer: type aliases, function macros, constant macros for CUDA↔HIP |
| `csrc/inc/gpu_utils.hpp` | **Replaces** `cuda_utils.hpp`. Error-handling macros now use `gpu_result_t`/`GPU_SUCCESS` |
| `csrc/inc/constants.hpp` | Added `kGPUAllocGranularity` (dynamic, queried at runtime) |
| `csrc/inc/page.hpp` | `CUdevice` → `gpu_device_t`, `CUmemGenericAllocationHandle` → `gpu_mem_handle_t` |
| `csrc/inc/allocator.hpp` | `init_cuda_()` → `init_gpu_()`, removed direct `<cuda_runtime.h>` include |
| `csrc/page.cpp` | All CUDA VMM calls → `gpu*` macros |
| `csrc/ftensor.cpp` | Same; hardcoded 2MB alignment → `kGPUAllocGranularity` |
| `csrc/allocator.cpp` | `init_gpu_()`: VMM support check skipped on ROCm (known bug), granularity queried dynamically |
| `setup.py` | ROCm: `CppExtension` + `-DUSE_ROCM` + `amdhip64`; CUDA: `CUDAExtension` + `cuda` (unchanged) |
| `kvcached/utils.py` | `_get_page_size()` relaxed to 1MB alignment on ROCm |
| `tests/conftest.py` | pytest markers: `requires_cuda`, `requires_rocm`, `requires_gpu` |
| `tests/test_gpu_compat.py` | Build + detection + page-size validation (runs on any machine) |
| `tests/test_rocm_vmm.py` | ROCm VMM smoke tests (skipped without AMD GPU) |

## Prerequisites

- AMD Instinct GPU (MI300X recommended)
- ROCm 6.3+ (7.1.0 recommended)
- PyTorch ROCm build (`torch.version.hip` must not be `None`)
- Python 3.10+

## Build on ROCm

```bash
# In a ROCm environment with PyTorch ROCm installed:
pip install -e . --no-build-isolation --no-cache-dir

# Verify USE_ROCM was set:
python -c "import kvcached.vmm_ops; print('vmm_ops loaded successfully')"
```

## Test Plan

### 1. Build verification

```bash
pip install -e . --no-build-isolation --no-cache-dir
# Should complete without errors. Verify the extension links against amdhip64.
```

### 2. Unit tests (no GPU needed — already verified on CUDA)

```bash
pytest tests/test_gpu_compat.py tests/test_shm_info_tracker.py -v
```

### 3. ROCm VMM smoke tests (requires AMD GPU)

```bash
pytest tests/test_rocm_vmm.py -v
```

This runs:
- `test_init_kvcached_rocm` — init/shutdown cycle
- `test_granularity_query` — verifies granularity query succeeds
- `test_kv_tensor_alloc_free` — create → map → unmap → shutdown

### 4. Full existing test suite on ROCm

```bash
pytest tests/ -v -k "not requires_cuda"
```

### 5. KVCacheManager integration test

```bash
pytest tests/test_kvcache_manager.py -v
pytest tests/test_paged_allocator_aliasing.py -v
```

### 6. vLLM/SGLang integration (manual)

```bash
# With ROCm vLLM:
ENABLE_KVCACHED=true python -m vllm.entrypoints.openai.api_server \
    --model <model> --no-enable-prefix-caching

# With ROCm SGLang:
ENABLE_KVCACHED=true python -m sglang.launch_server \
    --model <model> --disable-radix-cache
```

## Known Issues / Risks

| Issue | Status | Mitigation |
|-------|--------|------------|
| `hipDeviceAttributeVMMSupported` returns 0 | ROCm bug | Skipped via `#ifndef USE_ROCM` in `init_gpu_()` |
| `hipGetErrorString` has different signature | Handled | `gpu_get_drv_error_string()` wrapper in `gpu_compat.hpp` |
| Granularity may differ (4096B vs 2MB) | Handled | `kGPUAllocGranularity` queried dynamically, Python validation relaxed |
| `kStartAddr` (0x1f0'000'000'000) VA validity | Untested on MI300X | MI300X VA space should be sufficient; make configurable if needed |
| `hipMemAllocationProp` struct layout | Untested | Using designated initializers for safety |

## CI Integration

```bash
# CUDA CI (existing)
pytest tests/ -k "not requires_rocm"

# ROCm CI (new)
pytest tests/ -k "not requires_cuda"

# CPU-only (no GPU)
pytest tests/test_gpu_compat.py tests/test_shm_info_tracker.py
```
