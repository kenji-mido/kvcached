#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# kvcached AMD GPU Development Environment Setup
#
# Self-contained script: scp to an AMD GPU machine, run, and get a fully
# working dev environment with kvcached + vLLM built for ROCm.
#
# Usage:
#   scp scripts/setup_amd_dev.sh user@<AMD-GPU-HOST>:~/
#   ssh user@<AMD-GPU-HOST> bash setup_amd_dev.sh
#
# Afterwards:
#   ssh user@<AMD-GPU-HOST>
#   gh auth login
#   source ~/setup_env.sh
#   cd ~/kvcached
#   pytest tests/ -v -k "not requires_cuda"
#
set -uo pipefail

TOTAL_STEPS=10
step=0
log() { step=$((step+1)); echo ""; echo "[$step/$TOTAL_STEPS] $1"; echo "---"; }

REPO_URL="https://github.com/kenji-mido/kvcached.git"
BRANCH="feature/amd-gpu-support"
WORK_DIR="$HOME/kvcached"
VENV_DIR="$HOME/venv"
VLLM_DIR="$HOME/vllm-source"
VLLM_VERSION="v0.17.1"

echo "============================================"
echo "  kvcached AMD GPU Dev Environment Setup"
echo "============================================"
echo "  Branch: $BRANCH"
echo "  Work dir: $WORK_DIR"
echo "  vLLM: $VLLM_VERSION (source build)"

# --- 1. Base packages ---
log "Installing base packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq 2>&1 | grep -v "^W:" || true
    sudo apt-get install -y -qq build-essential cmake curl git jq ninja-build 2>&1 \
        || echo "  WARNING: some base packages failed to install"
    sudo apt-get install -y -qq python3-venv python3-dev 2>&1 \
        || sudo apt-get install -y -qq python3.10-venv python3.10-dev 2>&1 \
        || echo "  WARNING: python3-venv not installed"
elif command -v dnf &> /dev/null; then
    sudo dnf install -y gcc gcc-c++ cmake curl git jq ninja-build python3-devel 2>&1 \
        || echo "  WARNING: some base packages failed to install"
else
    echo "  WARNING: unsupported package manager. Install build tools manually."
fi

# --- 2. ROCm detection ---
log "Detecting ROCm..."
ROCM_VERSION=""
if [ -f /opt/rocm/.info/version ]; then
    ROCM_VERSION=$(head -1 /opt/rocm/.info/version | cut -d- -f1)
    echo "  ROCm $ROCM_VERSION detected"
elif command -v rocminfo &> /dev/null; then
    ROCM_VERSION=$(rocminfo 2>/dev/null | grep -oP 'ROCm.*?(\d+\.\d+)' | head -1 | grep -oP '\d+\.\d+' || echo "")
    echo "  ROCm detected (version: ${ROCM_VERSION:-unknown})"
else
    echo "  ERROR: ROCm not found. Cannot continue."
    echo "  Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html"
    exit 1
fi

ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)
ROCM_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f2)

# GPU check
if command -v rocm-smi &> /dev/null; then
    echo ""
    rocm-smi --showproductname 2>/dev/null | head -10 || rocm-smi -i 2>/dev/null | head -10 || true
elif command -v amd-smi &> /dev/null; then
    amd-smi static 2>/dev/null | head -10 || true
fi

# --- 3. GitHub CLI ---
log "Installing GitHub CLI..."
if ! command -v gh &> /dev/null; then
    if command -v apt-get &> /dev/null; then
        (type -p wget >/dev/null || sudo apt-get install -y wget) && \
        sudo mkdir -p -m 755 /etc/apt/keyrings && \
        out=$(mktemp) && wget -qO "$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg && \
        sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg < "$out" > /dev/null && \
        sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
        sudo apt-get update -qq 2>&1 | grep -v "^W:" || true && \
        sudo apt-get install -y -qq gh 2>&1 || echo "  WARNING: gh install failed"
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y gh 2>&1 || echo "  WARNING: gh install failed"
    else
        echo "  WARNING: install gh manually: https://github.com/cli/cli#installation"
    fi
else
    echo "  gh already installed: $(gh --version | head -1)"
fi

# --- 4. Claude Code ---
log "Installing Claude Code..."
if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash \
        || echo "  WARNING: Claude Code install failed. Install manually: https://claude.ai/install.sh"
else
    echo "  Claude Code already installed."
fi

# --- 5. Python venv + PyTorch (ROCm) ---
log "Setting up Python + PyTorch (ROCm)..."
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR" || { echo "  ERROR: venv creation failed"; exit 1; }
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel -q

# Determine PyTorch index URL based on ROCm version
if [ "${ROCM_MAJOR:-6}" = "7" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_MAJOR}.${ROCM_MINOR}"
    TORCH_VERSION="2.10.0"
    echo "  ROCm ${ROCM_MAJOR}.${ROCM_MINOR}: Installing PyTorch ${TORCH_VERSION}..."
else
    TORCH_INDEX="https://download.pytorch.org/whl/rocm6.3"
    TORCH_VERSION="2.6.0"
    echo "  ROCm 6.x: Installing PyTorch ${TORCH_VERSION}..."
fi

# Check if correct PyTorch ROCm is already installed
EXISTING_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
EXISTING_HIP=$(python3 -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "none")
if [[ "$EXISTING_TORCH" == "${TORCH_VERSION}+"* ]] && [ "$EXISTING_HIP" != "none" ] && [ "$EXISTING_HIP" != "None" ]; then
    echo "  PyTorch ROCm already installed ($EXISTING_TORCH, HIP: $EXISTING_HIP)"
else
    pip install -q "torch==${TORCH_VERSION}" "torchvision" "torchaudio" \
        --index-url "$TORCH_INDEX" 2>&1 \
        || echo "  WARNING: PyTorch install failed"
fi

# Verify PyTorch
python3 -c "
import torch
assert torch.version.hip is not None, 'PyTorch is not a ROCm build!'
print(f'  PyTorch {torch.__version__} (HIP {torch.version.hip})')
print(f'  GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
" || { echo "  ERROR: PyTorch ROCm verification failed"; exit 1; }

# --- 6. Clone & build kvcached ---
log "Cloning and building kvcached..."
if [ -d "$WORK_DIR" ]; then
    echo "  $WORK_DIR already exists, updating..."
    cd "$WORK_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
    git checkout "$BRANCH"
fi

# Install Python dependencies
pip install -q numpy posix_ipc wrapt pytest pre-commit setuptools-scm packaging 2>&1

# Build kvcached with ROCm
echo "  Building kvcached (this may take a few minutes)..."
pip install -e . --no-build-isolation --no-cache-dir 2>&1 | tail -5

# Verify
python3 -c "import kvcached.vmm_ops; print('  kvcached.vmm_ops loaded successfully')" \
    || { echo "  ERROR: kvcached build failed"; exit 1; }

# --- 7. Build vLLM from source for ROCm ---
log "Building vLLM ${VLLM_VERSION} from source for ROCm..."

# Install amdsmi (required for vLLM ROCm platform detection)
pip install -q amdsmi 2>&1 || echo "  WARNING: amdsmi install failed"

if [ -d "$VLLM_DIR" ]; then
    echo "  $VLLM_DIR already exists, updating..."
    cd "$VLLM_DIR"
    git fetch origin tag "$VLLM_VERSION" --depth 1 2>&1 || true
    git checkout "$VLLM_VERSION" 2>&1 || true
else
    git clone --depth 1 --branch "$VLLM_VERSION" https://github.com/vllm-project/vllm.git "$VLLM_DIR" 2>&1
    cd "$VLLM_DIR"
fi

# Install vLLM common dependencies (without pulling CUDA torch)
echo "  Installing vLLM dependencies..."
pip install -q --no-deps pyyaml opencv-python-headless 2>&1
pip install -q aiohttp openai pydantic 'transformers>=4.56.0' 'tokenizers>=0.21.1' \
    'fastapi[standard]' msgspec pyzmq gguf 'mistral_common[image]' \
    compressed-tensors depyf cloudpickle watchfiles python-json-logger \
    pybase64 cbor2 ijson setproctitle openai-harmony mcp grpcio grpcio-reflection \
    'opentelemetry-sdk>=1.27.0' 'opentelemetry-api>=1.27.0' 'opentelemetry-exporter-otlp>=1.27.0' \
    'opentelemetry-semantic-conventions-ai>=0.4.1' kaldi-native-fbank \
    'xgrammar==0.1.29' 'outlines_core==0.2.11' 'llguidance>=1.3.0,<1.4.0' \
    'lm-format-enforcer==0.11.3' 'lark==1.2.2' 'diskcache==5.6.3' \
    partial-json-parser filelock blake3 py-cpuinfo einops anthropic \
    model-hosting-container-standards ninja ipython orjson 2>&1 | tail -3

# Build vLLM for ROCm
echo "  Building vLLM for ROCm (this will take several minutes)..."
export VLLM_TARGET_DEVICE=rocm
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export MAX_JOBS=4
rm -rf build/ dist/ *.egg-info

pip install -e . --no-build-isolation --no-deps 2>&1 | tail -5

# Verify
VLLM_INSTALLED=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "FAILED")
if [ "$VLLM_INSTALLED" != "FAILED" ]; then
    echo "  vLLM $VLLM_INSTALLED installed successfully"
else
    echo "  WARNING: vLLM build failed. Integration tests will not work."
fi

# Verify ROCm platform detection
python3 -c "
from vllm.platforms import current_platform
assert current_platform.is_rocm(), 'vLLM did not detect ROCm!'
print(f'  vLLM platform: {type(current_platform).__name__}')
" 2>/dev/null || echo "  WARNING: vLLM ROCm platform detection failed"

# --- 8. Environment config file ---
log "Creating environment setup file..."
cat > ~/setup_env.sh << ENVEOF
#!/bin/bash
# Load environment with: source ~/setup_env.sh
export PATH="\$HOME/.local/bin:\$HOME/.claude/bin:/opt/rocm/bin:\$PATH"
export ROCM_HOME=/opt/rocm
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

echo "Environment loaded:"
echo "  Python:   \$(python3 --version 2>/dev/null || echo 'N/A')"
echo "  PyTorch:  \$(python3 -c 'import torch; print(f"{torch.__version__} (HIP {torch.version.hip})")' 2>/dev/null || echo 'N/A')"
echo "  ROCm:     \$(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')"
echo "  GPU:      \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "  kvcached: $WORK_DIR"
echo "  vLLM:     \$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'N/A')"
echo "  Branch:   \$(cd $WORK_DIR && git branch --show-current 2>/dev/null || echo 'N/A')"
ENVEOF
chmod +x ~/setup_env.sh

if ! grep -q "setup_env.sh" ~/.bashrc 2>/dev/null; then
    echo '[ -f ~/setup_env.sh ] && source ~/setup_env.sh' >> ~/.bashrc
fi

# --- 9. Smoke tests ---
log "Running smoke tests..."
cd "$WORK_DIR"

echo "  CPU-only tests:"
python3 -m pytest tests/test_shm_info_tracker.py -v --tb=short 2>&1 | tail -12

echo ""
echo "  GPU compat tests:"
python3 -m pytest tests/test_gpu_compat.py -v --tb=short 2>&1 | tail -8

echo ""
echo "  ROCm VMM smoke tests:"
python3 -m pytest tests/test_rocm_vmm.py -v --tb=short 2>&1 | tail -8

# --- 10. vLLM + kvcached elastic memory E2E test ---
log "Running vLLM + kvcached elastic memory E2E test..."
echo "  This verifies hipMemMap/hipMemUnmap elastic behavior end-to-end."
echo "  (Loads facebook/opt-125m, runs inference, observes physical memory changes)"
echo ""
cd "$WORK_DIR"
ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
    timeout 300 python3 tests/test_elastic_memory_e2e.py 2>&1 \
    | grep -v "^(EngineCore\|^INFO\|^WARNING\|^Processed\|^Rendering\|^\[kvcached\]\|triton_kernels\|/root/vllm\|SyntaxWarning\|resource_tracker"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Authenticate GitHub (for pushes):"
echo "     gh auth login"
echo ""
echo "  2. Load environment (auto-loaded on next login):"
echo "     source ~/setup_env.sh"
echo ""
echo "  3. Run full test suite:"
echo "     cd $WORK_DIR"
echo "     pytest tests/ -v -k 'not requires_cuda'"
echo ""
echo "  4. Elastic memory benchmark:"
echo "     cd $WORK_DIR"
echo "     python tests/test_elastic_memory_e2e.py"
echo ""
echo "  5. vLLM integration (with kvcached autopatch):"
echo "     ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 python -m vllm.entrypoints.openai.api_server \\"
echo "       --model <model> --no-enable-prefix-caching --enforce-eager"
echo ""
echo "  6. Use Claude Code for further development:"
echo "     cd $WORK_DIR && claude"
echo ""
echo "  See docs/amd-gpu-testing-guide.md for full details."
