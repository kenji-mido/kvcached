# CLAUDE.md - AI Assistant Guide for kvcached

This document provides guidance for AI assistants working with the kvcached codebase.

## Project Overview

**kvcached** (KV cache daemon) is a KV cache library for LLM serving/training on shared GPUs. It brings OS-style virtual memory abstraction to LLM systems, enabling elastic and demand-driven KV cache allocation to improve GPU utilization under dynamic workloads.

- **License**: Apache-2.0
- **Python**: 3.9 - 3.12
- **Supported Engines**: SGLang (v0.5.3+) and vLLM (v0.11.0+)
- **Repository**: https://github.com/ovg-project/kvcached

## Repository Structure

```
kvcached/
├── kvcached/                 # Main Python package
│   ├── __init__.py
│   ├── autopatch.py          # Auto-patching entry point
│   ├── kv_cache_manager.py   # Core memory management
│   ├── page_allocator.py     # Physical page allocation
│   ├── mem_info_tracker.py   # Memory tracking via shared memory
│   ├── tp_ipc_util.py        # Tensor parallelism IPC utilities
│   ├── locks.py              # Synchronization primitives
│   ├── utils.py              # Shared utilities
│   ├── cli/                  # Command-line tools (kvctl, kvtop)
│   │   ├── kvctl.py          # Memory control CLI
│   │   └── kvtop.py          # Memory monitoring tool
│   └── integration/          # Engine integrations
│       ├── patch_base.py     # Base patching infrastructure
│       ├── version_utils.py  # Version detection
│       ├── sglang/           # SGLang-specific patches
│       │   ├── autopatch.py
│       │   ├── interfaces.py
│       │   └── patches.py
│       └── vllm/             # vLLM-specific patches
│           ├── autopatch.py
│           ├── interfaces.py
│           └── patches.py
├── csrc/                     # C++/CUDA source code
│   ├── allocator.cpp         # GPU virtual memory allocator
│   ├── ftensor.cpp           # Tensor operations
│   ├── page.cpp              # Page management
│   ├── torch_bindings.cpp    # PyTorch bindings
│   └── inc/                  # C++ headers
│       ├── constants.hpp     # Constants (kStartAddr, page IDs)
│       ├── allocator.hpp
│       ├── cuda_utils.hpp
│       └── impl/             # Implementation details
├── controller/               # Multi-LLM controller & router
├── engine_integration/       # Engine patches and scripts
│   ├── patches/
│   └── scripts/
├── examples/                 # Usage examples
│   ├── 01_simple_two_models/
│   ├── 02_memory_control/
│   ├── 03_model_router_sleep/
│   ├── 04_inference_and_finetune/
│   ├── 05_multi_agents/
│   ├── 06_serverless_serving/
│   └── 07_inference_and_diffusion/
├── benchmarks/               # Performance benchmarks
│   ├── simple_bench/
│   ├── bench_kvcached_overhead/
│   ├── bench_latency_benefit/
│   ├── bench_map_parallelism/
│   ├── bench_tp_ipc/
│   ├── bench_vmm/
│   └── gsm8k/
├── tests/                    # Unit tests
├── tools/                    # Development tools
│   ├── addlicense.sh         # SPDX header tool
│   ├── dev_copy_pth.py       # Copy .pth for dev installs
│   └── mypy.sh               # Type checking
├── docker/                   # Docker configurations
├── pyproject.toml            # Project metadata and dependencies
├── setup.py                  # Build configuration
└── kvcached_autopatch.pth    # Python site-packages autopatch hook
```

## Build and Installation

### From Source (Development)
```bash
# Ensure PyTorch is installed first
pip install torch>=2.6.0

# Install in editable mode
pip install -e . --no-build-isolation --no-cache-dir

# Copy autopatch .pth file to site-packages
python tools/dev_copy_pth.py
```

### From PyPI
```bash
pip install kvcached --no-build-isolation
```

### Verify Installation
```bash
python -c "import kvcached; from kvcached import vmm_ops; print('kvcached imported successfully')"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ENABLE_KVCACHED=true` | Enable kvcached functionality |
| `KVCACHED_AUTOPATCH=1` | Enable automatic engine patching |
| `KVCACHED_CONTIGUOUS_LAYOUT=true` | Use contiguous tensor layout |
| `CUDA_HOME` | CUDA installation path |

## Running Tests

Tests require a GPU and are run with pytest:
```bash
pytest tests/
```

Key test files:
- `tests/test_kvcache_manager.py` - Core KV cache manager tests
- `tests/test_shm_info_tracker.py` - Shared memory tracker tests
- `tests/test_sleep_manager.py` - Sleep/wakeup functionality tests
- `tests/test_traffic_monitor.py` - Traffic monitoring tests

## Code Style and Linting

### Pre-commit Hooks
The project uses pre-commit for code quality. Set up with:
```bash
pip install pre-commit
pre-commit install
```

Run all checks:
```bash
pre-commit run --all-files
```

### Formatting and Linting Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **ruff** | Python linting and formatting | `pyproject.toml` (line-length: 100) |
| **isort** | Import sorting | profile: black, line_length: 100 |
| **mypy** | Type checking | `pyproject.toml` |
| **clang-format** | C++/CUDA formatting | `.clang-format` (BasedOnStyle: LLVM) |
| **codespell** | Spell checking | - |
| **pymarkdown** | Markdown linting | - |
| **actionlint** | GitHub Actions linting | - |

### SPDX License Headers
All source files require SPDX license headers:
```python
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
```

For C++/CUDA:
```cpp
// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
```

## Architecture Patterns

### Autopatch System
kvcached uses runtime patching to integrate with vLLM and SGLang:
1. `kvcached_autopatch.pth` triggers `kvcached.autopatch` on Python startup
2. `when_imported` hooks detect when engines are loaded
3. Patches are applied to engine classes for elastic memory support

### Memory Management Hierarchy
1. **KVCacheManager** - High-level block allocation/deallocation
2. **PageAllocator** - Physical page management with pre-allocation
3. **vmm_ops** (C++) - GPU virtual memory operations via CUDA

### Engine Integration Pattern
Each engine integration follows this structure:
- `autopatch.py` - Registers `when_imported` hooks
- `interfaces.py` - Public API for the engine
- `patches.py` - Monkey patches for engine classes

## Key APIs

### CLI Tools
- `kvctl` - Memory control CLI for setting limits
- `kvtop` - Real-time memory monitoring

### Python APIs
```python
from kvcached.kv_cache_manager import KVCacheManager
from kvcached.integration.vllm.interfaces import init_kvcached, alloc_kv_cache
from kvcached.integration.sglang.interfaces import init_kvcached, alloc_kv_cache
```

## Important Notes for Development

### Prefix Caching Limitation
kvcached does not support prefix caching. Always use:
- vLLM: `--no-enable-prefix-caching`
- SGLang: `--disable-radix-cache`

### WSL2 Compatibility
For WSL2, the `kStartAddr` in `csrc/inc/constants.hpp` is set to 8GB (0x2'000'000'00) for compatibility with WSL2's virtual address space limitations.

### Tensor Parallelism
kvcached supports tensor parallelism via IPC coordination in `tp_ipc_util.py`.

### Thread Safety
The `KVCacheManager` uses `synchronized` decorators and supports both sync and async scheduling modes.

## Common Development Tasks

### Adding a New Engine Integration
1. Create directory under `kvcached/integration/<engine>/`
2. Implement `autopatch.py`, `interfaces.py`, `patches.py`
3. Register hooks in `kvcached/autopatch.py`

### Modifying C++ Code
1. Edit files in `csrc/`
2. Rebuild with `pip install -e . --no-build-isolation --no-cache-dir --force-reinstall`

### Running Benchmarks
```bash
cd benchmarks/simple_bench
./start_server.sh [sglang|vllm] --venv-path $VENV_PATH --model <model>
./start_client.sh [sglang|vllm] --venv-path $VENV_PATH --model <model>
```

## Docker Development

Pre-built images:
```bash
docker pull ghcr.io/ovg-project/kvcached-sglang:latest
docker pull ghcr.io/ovg-project/kvcached-vllm:latest
docker pull ghcr.io/ovg-project/kvcached-dev:latest
```

Build locally:
```bash
docker build -f docker/Dockerfile.vllm -t vllm-kvcached .
docker build -f docker/Dockerfile.sglang -t sglang-kvcached .
```

## Git Workflow

- Main development happens on feature branches
- All commits should pass pre-commit hooks
- CI runs pre-commit and mypy on Python 3.9-3.12
