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
| `csrc/ftensor.cpp` | Same; `hipMemAddressReserve` type cast fix for HIP `void*` arg |
| `csrc/allocator.cpp` | `init_gpu_()`: VMM check skipped on ROCm; contiguous layout size alignment fix |
| `setup.py` | ROCm: `CppExtension` + `-DUSE_ROCM` + `amdhip64` + ROCm include/lib paths |
| `kvcached/utils.py` | `_get_page_size()` relaxed to 1MB alignment on ROCm |
| `tests/conftest.py` | pytest markers: `requires_cuda`, `requires_rocm`, `requires_gpu` |
| `tests/test_gpu_compat.py` | Build + detection + page-size validation (runs on any machine) |
| `tests/test_rocm_vmm.py` | ROCm VMM smoke tests (skipped without AMD GPU) |
| `tests/test_elastic_memory_e2e.py` | E2E elastic memory benchmark with vLLM (kvcached vs baseline) |

## Prerequisites

- AMD Instinct GPU (MI300X recommended)
- ROCm 7.1+ (tested on 7.1.0)
- PyTorch 2.10.0+rocm7.1 (`torch.version.hip` must not be `None`)
- Python 3.10+

## Quick Start

Use the self-contained setup script:

```bash
scp setup_amd_dev.sh user@<AMD-GPU-HOST>:~/
ssh user@<AMD-GPU-HOST> bash setup_amd_dev.sh
```

This installs everything: ROCm PyTorch, kvcached, vLLM (source build), and runs all tests including the E2E elastic memory benchmark.

## Build on ROCm

```bash
pip install -e . --no-build-isolation --no-cache-dir

# Verify USE_ROCM was set and linked against amdhip64:
python -c "import kvcached.vmm_ops; print('vmm_ops loaded successfully')"
ldd $(python -c "import kvcached.vmm_ops; print(kvcached.vmm_ops.__file__)") | grep hip
```

## Test Plan

### 1. Build verification

```bash
pip install -e . --no-build-isolation --no-cache-dir
# Should complete without errors. Verify the extension links against amdhip64.
```

### 2. Unit tests (no GPU needed)

```bash
pytest tests/test_gpu_compat.py tests/test_shm_info_tracker.py -v
```

`test_gpu_compat.py` の内容:

- `test_rocm_detection_sets_is_rocm_true/false` — `torch.version.hip` による ROCm 検出ロジック
- `test_page_size_default` — デフォルト 2MB
- `test_page_size_validation_cuda_valid` — CUDA: 2MB, 4MB を受理
- `test_page_size_validation_rocm_valid` — ROCm: 1MB, 2MB を受理
- `test_page_size_cuda_rejects_1mb/3mb` — CUDA: 2MB 非倍数を拒否
- `test_page_size_rejects_zero/negative/non_integer` — 異常値を拒否

### 3. ROCm VMM テスト (AMD GPU 必須)

```bash
pytest tests/test_rocm_vmm.py -v
```

kvcached のコア VMM 動作を検証するテスト:

- `test_init_kvcached_rocm` — init/shutdown。init 直後に `kv_tensors_created() == False` をアサート
- `test_granularity_query` — `hipMemGetAllocationGranularity` で granularity 取得が成功すること
- `test_kv_tensor_create` — `create_kv_tensors` がGPU テンソル (`is_cuda`, 正しい dtype, `numel > 0`) を返すこと
- `test_kv_tensor_map_unmap` — map 後にGPU メモリへ書き込み→読み出しで値が一致すること。マッピング失敗時はセグフォルトまたは値不一致で FAIL

### 4. Full test suite on ROCm

```bash
pytest tests/ -v -k "not requires_cuda"
```

### 5. Elastic memory E2E benchmark (requires AMD GPU + vLLM)

```bash
# kvcached enabled — verify elastic hipMemMap/hipMemUnmap
python tests/test_elastic_memory_e2e.py

# Baseline comparison — vanilla vLLM (pre-allocates all KV memory)
python tests/test_elastic_memory_e2e.py --baseline
```

The benchmark:
- Loads `facebook/opt-125m` on GPU
- Runs 5 phases: idle → small batch → large batch → cooldown → re-grow
- Monitors KV cache memory via kvcached IPC (Used/Prealloc/Physical/Free/Virtual)
- Monitors GPU memory via `hipMemGetInfo` (through `torch.cuda.mem_get_info`)
- Prints timeline with bar chart showing memory changes
- Verifies 6 checks (elastic growth, shrinkage, hipMemUnmap, re-map)

### 6. vLLM/SGLang integration via Docker (recommended)

```bash
# vLLM — official ROCm Docker image
docker run --rm --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G \
    --security-opt seccomp=unconfined \
    -v /path/to/kvcached:/kvcached \
    vllm/vllm-openai-rocm:latest \
    -c 'cd /kvcached && pip install -e . --no-build-isolation -q && \
        python tests/test_elastic_memory_e2e.py'

# SGLang — official ROCm Docker image
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G \
    --security-opt seccomp=unconfined \
    -v /path/to/kvcached:/kvcached \
    lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312 \
    bash -c 'cd /kvcached && pip install -e . --no-build-isolation -q && \
        python tests/test_sglang_e2e.py'

# Baseline comparison (vanilla vLLM, no kvcached)
docker run --rm --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G \
    --security-opt seccomp=unconfined \
    -v /path/to/kvcached:/kvcached \
    vllm/vllm-openai-rocm:latest \
    -c 'cd /kvcached && pip install -e . --no-build-isolation -q && \
        python tests/test_elastic_memory_e2e.py --baseline'
```

## Test Runner Script

全テストをまとめて実行する場合:

```bash
bash scripts/run_amd_tests.sh           # 全 tier
bash scripts/run_amd_tests.sh --quick   # Tier 1-2 のみ (GPU 不要)
bash scripts/run_amd_tests.sh --gpu     # Tier 3-4 のみ (GPU 必須)
```

| Tier | 内容 | GPU 必須 |
|------|------|---------|
| 1a | 共有メモリ (CPU テスト) | No |
| 1b | ビルド・検出ロジック | No |
| 2 | pre-commit (lint, format) | No |
| 3a | ROCm VMM スモーク | Yes |
| 3b | ページアライアシング | Yes |
| 4 | KVCacheManager 統合 | Yes |

実行後、PASS/FAIL/SKIP のサマリーが表示されます。

## Bugs Fixed During Validation

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `hip/hip_runtime.h: No such file` | `setup.py` didn't add `/opt/rocm/include` | Added ROCm include/lib paths |
| `hipMemAddressReserve` invalid conversion | HIP 4th arg is `void*`, CUDA is integer | `reinterpret_cast<gpu_devptr_t>()` |
| `test_kv_tensor_alloc_free` segfault | Virtual size < compound page size | Round up `total_kv_size` to multiple of `compound_page_size` |

## Known Issues / Risks

| Issue | Status | Mitigation |
|-------|--------|------------|
| `hipDeviceAttributeVMMSupported` returns 0 | ROCm bug | Skipped via `#ifndef USE_ROCM` in `init_gpu_()` |
| `hipGetErrorString` has different signature | Handled | `gpu_get_drv_error_string()` wrapper in `gpu_compat.hpp` |
| Granularity may differ (4096B vs 2MB) | Handled | `kGPUAllocGranularity` queried dynamically, Python validation relaxed |
| `kStartAddr` (0x1f0'000'000'000) VA validity | Validated on MI300X | Works correctly |
| `hipMemAllocationProp` struct layout | Validated | Designated initializers work correctly |
| SGLang `sgl-kernel` ROCm 7.1 | Not available | SGLang integration deferred |

## CI Integration

```bash
# CUDA CI (existing)
pytest tests/ -k "not requires_rocm"

# ROCm CI (new)
pytest tests/ -k "not requires_cuda"

# CPU-only (no GPU)
pytest tests/test_gpu_compat.py tests/test_shm_info_tracker.py

# E2E elastic memory (requires GPU + vLLM)
python tests/test_elastic_memory_e2e.py
```
