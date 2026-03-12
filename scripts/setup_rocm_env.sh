#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# ROCm environment setup script for kvcached AMD GPU testing.
#
# Usage:
#   bash scripts/setup_rocm_env.sh [--venv DIR] [--pytorch-version VER] [--skip-venv]
#
# Prerequisites:
#   - AMD Instinct GPU (MI300X recommended)
#   - ROCm 6.3+ installed (https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html)
#   - Python 3.10+
#
# What this script does:
#   1. Verify ROCm installation (rocm-smi, hipcc)
#   2. Create a Python venv (or use existing)
#   3. Install PyTorch ROCm build
#   4. Install kvcached in editable mode with -DUSE_ROCM
#   5. Run basic smoke tests

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KVCACHED_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_DIR="${KVCACHED_DIR}/venv-rocm"
PYTORCH_VERSION=""  # empty = latest stable
SKIP_VENV=false

# ─── Colors ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$(tput bold); RESET=$(tput sgr0)
    GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); RED=$(tput setaf 1)
    CYAN=$(tput setaf 6)
else
    BOLD=""; RESET=""; GREEN=""; YELLOW=""; RED=""; CYAN=""
fi

info()  { echo "${BOLD}${GREEN}[INFO]${RESET} $*"; }
warn()  { echo "${BOLD}${YELLOW}[WARN]${RESET} $*"; }
error() { echo "${BOLD}${RED}[ERROR]${RESET} $*" >&2; }
step()  { echo ""; echo "${BOLD}${CYAN}==> $*${RESET}"; }

# ─── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)        VENV_DIR="$2"; shift 2 ;;
        --pytorch-version) PYTORCH_VERSION="$2"; shift 2 ;;
        --skip-venv)   SKIP_VENV=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--venv DIR] [--pytorch-version VER] [--skip-venv]"
            echo ""
            echo "Options:"
            echo "  --venv DIR           Python venv directory (default: venv-rocm)"
            echo "  --pytorch-version V  PyTorch version (default: latest stable)"
            echo "  --skip-venv          Skip venv creation, use current Python"
            exit 0 ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Step 1: Verify ROCm ────────────────────────────────────────────────────
step "Step 1/5: Verifying ROCm installation"

if ! command -v rocm-smi &>/dev/null; then
    error "rocm-smi not found. Please install ROCm first."
    error "See: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html"
    exit 1
fi

if ! command -v hipcc &>/dev/null; then
    error "hipcc not found. ROCm dev tools may not be installed."
    exit 1
fi

ROCM_VERSION=$(rocm-smi --version 2>/dev/null | head -1 || echo "unknown")
info "ROCm version: ${ROCM_VERSION}"

info "Detected GPUs:"
rocm-smi --showproductname 2>/dev/null | head -20 || rocm-smi -i 2>/dev/null | head -20

# ─── Step 2: Python venv ────────────────────────────────────────────────────
step "Step 2/5: Setting up Python environment"

if [[ "$SKIP_VENV" = true ]]; then
    info "Skipping venv creation (--skip-venv)"
    PYTHON="python3"
    PIP="pip3"
else
    if [[ -d "$VENV_DIR" ]]; then
        info "Using existing venv: ${VENV_DIR}"
    else
        info "Creating venv: ${VENV_DIR}"
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PYTHON="python"
    PIP="pip"
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
info "Python: ${PYTHON_VERSION}"

$PIP install --upgrade pip setuptools wheel -q

# ─── Step 3: Install PyTorch ROCm ───────────────────────────────────────────
step "Step 3/5: Installing PyTorch (ROCm build)"

# Check if PyTorch ROCm is already installed
EXISTING_HIP=$($PYTHON -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "none")
if [[ "$EXISTING_HIP" != "none" && "$EXISTING_HIP" != "None" ]]; then
    info "PyTorch ROCm already installed (HIP: ${EXISTING_HIP})"
else
    if [[ -n "$PYTORCH_VERSION" ]]; then
        info "Installing PyTorch ${PYTORCH_VERSION} for ROCm..."
        $PIP install "torch==${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/rocm6.3
    else
        info "Installing latest stable PyTorch for ROCm..."
        $PIP install torch --index-url https://download.pytorch.org/whl/rocm6.3
    fi
fi

# Verify
$PYTHON -c "
import torch
assert torch.version.hip is not None, 'PyTorch is not a ROCm build!'
print(f'  PyTorch {torch.__version__} (HIP {torch.version.hip})')
print(f'  CUDA/HIP available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
"

# ─── Step 4: Install kvcached ───────────────────────────────────────────────
step "Step 4/5: Building and installing kvcached (ROCm)"

cd "$KVCACHED_DIR"

# Install dependencies
$PIP install numpy posix_ipc wrapt -q

# Build with ROCm support
info "Running: pip install -e . --no-build-isolation --no-cache-dir"
$PIP install -e . --no-build-isolation --no-cache-dir

# Verify the extension loaded
$PYTHON -c "
import kvcached.vmm_ops
print('  kvcached.vmm_ops loaded successfully')
"

# Install test dependencies
$PIP install pytest -q

# ─── Step 5: Smoke test ─────────────────────────────────────────────────────
step "Step 5/5: Running smoke tests"

info "CPU-only tests (no GPU required):"
$PYTHON -m pytest tests/test_shm_info_tracker.py -v --tb=short 2>&1 | tail -15

info ""
info "GPU compat tests (build + detection):"
$PYTHON -m pytest tests/test_gpu_compat.py -v --tb=short 2>&1 | tail -10

info ""
info "ROCm VMM smoke tests:"
$PYTHON -m pytest tests/test_rocm_vmm.py -v --tb=short 2>&1 | tail -10

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}${GREEN}============================================${RESET}"
echo "${BOLD}${GREEN}  ROCm environment setup complete!${RESET}"
echo "${BOLD}${GREEN}============================================${RESET}"
echo ""
info "Venv:     ${VENV_DIR}"
info "kvcached: ${KVCACHED_DIR}"
echo ""
info "Next steps:"
echo "  1. Run full test suite:    pytest tests/ -v -k 'not requires_cuda'"
echo "  2. KVCacheManager test:    pytest tests/test_kvcache_manager.py -v"
echo "  3. vLLM integration:       ENABLE_KVCACHED=true python -m vllm.entrypoints.openai.api_server \\"
echo "                               --model <model> --no-enable-prefix-caching"
echo ""
echo "  See docs/amd-gpu-testing-guide.md for full details."
