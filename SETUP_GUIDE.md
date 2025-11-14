# kvcached Setup Guide for WSL2

This guide provides step-by-step instructions for running kvcached with vLLM and SGLang on WSL2 environment.

## Prerequisites

- WSL2 with CUDA 12.8 installed at `/usr/local/cuda-12.8`
- Python 3.x
- NVIDIA GPU with drivers installed
- `nvidia-smi` working

## Initial Setup (One-Time)

### Step 1: Create vLLM Virtual Environment

```bash
# Create project directory
mkdir -p ~/Work/vllm_wsl2_rtx4080
cd ~/Work/vllm_wsl2_rtx4080

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and wheel
pip install -U pip wheel

# Install PyTorch with CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify PyTorch installation
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY

# Install vLLM
pip install vllm

# Optional: Install additional dependencies
pip install autoawq huggingface_hub
```

### Step 2: Install kvcached in vLLM Environment

```bash
# Set WSL-specific environment variables (required for compilation)
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Navigate to kvcached directory
cd ~/Work/kvcached/kvcached

# Install kvcached in editable mode
pip install -e . --no-build-isolation --no-cache-dir

# Install required dependencies
pip install wrapt posix_ipc

# Copy autopatch .pth file to venv
python tools/dev_copy_pth.py

# Verify kvcached installation
python -c "import kvcached; from kvcached import vmm_ops; print('kvcached imported successfully')"
```

**Expected output:**
```
kvcached imported successfully
```

### Step 3: Create SGLang Virtual Environment

```bash
# Navigate to kvcached directory
cd ~/Work/kvcached/kvcached

# Create virtual environment for SGLang
python3 -m venv vllm-venv
source vllm-venv/bin/activate

# Upgrade pip and wheel
pip install -U pip wheel

# Set WSL-specific environment variables
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Install PyTorch with CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install SGLang
pip install "sglang[all]"

# Install kvcached in editable mode
pip install -e . --no-build-isolation --no-cache-dir

# Install required dependencies
pip install wrapt posix_ipc

# Copy autopatch .pth file
python tools/dev_copy_pth.py

# Verify installation
python -c "import kvcached; from kvcached import vmm_ops; print('kvcached imported successfully')"
```

### Step 4: Modify kvcached for WSL2 Compatibility

**IMPORTANT:** Before using kvcached on WSL2, modify the virtual address start location:

Edit `csrc/inc/constants.hpp`:

```cpp
// Change from default (33.8TB) to 8GB for WSL2
static constexpr size_t kStartAddr = 0x2'000'000'00;  // 8GB (WSL2 compatibility)
```

After modifying, rebuild kvcached in both environments:

```bash
# For vLLM environment
cd ~/Work/vllm_wsl2_rtx4080
source venv/bin/activate
cd ~/Work/kvcached/kvcached
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
pip install -e . --no-build-isolation --no-cache-dir --force-reinstall

# For SGLang environment
cd ~/Work/kvcached/kvcached
source vllm-venv/bin/activate
pip install -e . --no-build-isolation --no-cache-dir --force-reinstall
```

## Environment Variables (Required for Both)

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

**Important:** The WSL library path (`/usr/lib/wsl/lib`) is critical for CUDA operations in WSL2.

## Option 1: SGLang + kvcached (Fully Working)

### Start Server

```bash
# Set environment variables
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Activate SGLang venv
source /home/kenji/Work/kvcached/kvcached/vllm-venv/bin/activate

# Start server
python3 -m sglang.launch_server \
  --model openai-community/gpt2 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --trust-remote-code \
  --port 30000 \
  --host 0.0.0.0
```

### Test API

```bash
curl -s -X POST http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-community/gpt2",
    "prompt": "Once upon a time",
    "max_tokens": 20,
    "temperature": 0
  }'
```

### Expected Output

Server logs should show:
```
[kvcached][INFO] Successfully patched sglang: elastic_allocator, elastic_memory_pool, scheduler_memory_leak
[kvcached][INFO] Init kvcached KV cache allocator: num_layers=12, mem_size_per_layer=363MB, total_mem_size=8735MB
[kvcached][INFO] VirtualKV Cache is allocated. #tokens: 248479, K size: 7.99 GB, V size: 7.99 GB
[kvcached][INFO] Physical KV Cache limits by --mem-fraction-static: #tokens: 248479, K size: 4.27 GB, V size: 4.27 GB
INFO:     Application startup complete.
```

API should return JSON with completion text.

### Stop Server

```bash
pkill -f "sglang.launch_server"
```

## Option 2: vLLM + kvcached (Fully Working)

### Start Server

```bash
# Set environment variables
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Activate vLLM venv
source /home/kenji/Work/vllm_wsl2_rtx4080/venv/bin/activate

# Start server
vllm serve openai-community/gpt2 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.3 \
  --port 8000 \
  --host 0.0.0.0 \
  --disable-frontend-multiprocessing \
  --no-enable-prefix-caching
```

**Important:** The `--no-enable-prefix-caching` flag is required because kvcached does not support prefix caching.

### Test API

```bash
curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-community/gpt2",
    "prompt": "Once upon a time",
    "max_tokens": 20,
    "temperature": 0
  }'
```

### Expected Output

Server logs should show:
```
[kvcached][INFO] Successfully patched vllm: elastic_block_pool, engine_core, gpu_model_runner, gpu_worker, kv_cache_coordinator
[kvcached][INFO] Init kvcached KV cache allocator: num_layers=12, mem_size_per_layer=173MB, total_mem_size=4158MB
INFO: Using FlashInfer for top-p & top-k sampling.
INFO: Application startup complete.
```

All 6 patches should be applied successfully, and kvcached KV cache allocator should be initialized.

### Stop Server

```bash
pkill -f "vllm serve"
```

## WSL2-Specific Requirements

### 1. Constants Modification

The file `csrc/inc/constants.hpp` must have `kStartAddr` set to 8GB for WSL2:

```cpp
static constexpr size_t kStartAddr = 0x2'000'000'00;  // 8GB (WSL2 compatibility)
```

### 2. CUDA Graph Limitation

SGLang requires `--disable-cuda-graph` to avoid "requires grad" errors during CUDA graph capture in WSL2.

### 3. Library Paths

WSL2 requires explicit library paths:
- `LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH`
- `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`

Without these, you'll get "Could not find CUDA lib directory" errors.

### 4. Frontend Multiprocessing (vLLM only)

vLLM requires `--disable-frontend-multiprocessing` flag, otherwise API requests will timeout.

### 5. Prefix Caching (vLLM only)

vLLM 0.11.0 enables prefix caching by default, which is not supported by kvcached. Use `--no-enable-prefix-caching` flag to disable it.

## Troubleshooting

### Problem: API requests timeout

**Solution:** For vLLM, add `--disable-frontend-multiprocessing` flag.

### Problem: "Failed to patch kv_cache_coordinator" warning

**Solution:** Add `--no-enable-prefix-caching` flag to disable prefix caching, which is not supported by kvcached.

### Problem: "Could not find CUDA lib directory"

**Solution:** Ensure `LIBRARY_PATH` and `LD_LIBRARY_PATH` include `/usr/lib/wsl/lib`.

### Problem: FlashInfer compilation errors

**Solution:**
1. Clear FlashInfer cache: `rm -rf ~/.cache/flashinfer`
2. Ensure `CUDA_HOME=/usr/local/cuda-12.8`
3. Ensure WSL library paths are set

### Problem: CUDA graph capture fails with "requires grad" error

**Solution:** Add `--disable-cuda-graph` flag (SGLang only).

### Problem: OOM during initialization

**Solution:** Verify `kStartAddr` in `csrc/inc/constants.hpp` is set to 8GB (0x2'000'000'00).

## Health Check Commands

### Check if server is listening

```bash
# SGLang
netstat -tuln | grep 30000

# vLLM
netstat -tuln | grep 8000
```

### Check server health endpoint

```bash
# SGLang
curl http://127.0.0.1:30000/health

# vLLM
curl http://127.0.0.1:8000/health
```

### Check GPU memory usage

```bash
nvidia-smi
```

### Check kvcached IPC segments

```bash
kvctl list
```

## Running in Background

### Start server in background

```bash
# SGLang
nohup python3 -m sglang.launch_server \
  --model openai-community/gpt2 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --trust-remote-code \
  --port 30000 \
  --host 0.0.0.0 \
  > /tmp/sglang.log 2>&1 &

# vLLM
nohup vllm serve openai-community/gpt2 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.3 \
  --port 8000 \
  --host 0.0.0.0 \
  --disable-frontend-multiprocessing \
  --no-enable-prefix-caching \
  > /tmp/vllm.log 2>&1 &
```

### Monitor logs

```bash
# SGLang
tail -f /tmp/sglang.log

# vLLM
tail -f /tmp/vllm.log
```

## Summary

- **SGLang**: Fully compatible with kvcached, all elastic memory features working
- **vLLM**: Fully compatible with kvcached when using `--no-enable-prefix-caching` flag
- **Both engines**: Provide full kvcached functionality on WSL2 with proper configuration

## Verified Versions

- vLLM: 0.11.0 (tested)
- SGLang: 0.5.5.post2 (tested)
- PyTorch: 2.5.1+cu121
- CUDA: 12.8 (WSL2)
