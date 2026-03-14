#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# kvcached AMD GPU Development Environment Setup
#
# Self-contained script: scp to an AMD GPU machine, run, and get a fully
# working dev environment with kvcached built for ROCm.
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

TOTAL_STEPS=8
step=0
log() { step=$((step+1)); echo ""; echo "[$step/$TOTAL_STEPS] $1"; echo "---"; }

REPO_URL="https://github.com/kenji-mido/kvcached.git"
BRANCH="feature/amd-gpu-support"
WORK_DIR="$HOME/kvcached"
VENV_DIR="$HOME/venv"

echo "============================================"
echo "  kvcached AMD GPU Dev Environment Setup"
echo "============================================"
echo "  Branch: $BRANCH"
echo "  Work dir: $WORK_DIR"

# --- 1. Base packages ---
log "Installing base packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq 2>&1 | grep -v "^W:" || true
    sudo apt-get install -y -qq build-essential cmake curl git jq 2>&1 \
        || echo "  WARNING: some base packages failed to install"
    sudo apt-get install -y -qq python3-venv python3-dev 2>&1 \
        || sudo apt-get install -y -qq python3.10-venv python3.10-dev 2>&1 \
        || echo "  WARNING: python3-venv not installed"
elif command -v dnf &> /dev/null; then
    sudo dnf install -y gcc gcc-c++ cmake curl git jq python3-devel 2>&1 \
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
    echo "  WARNING: ROCm not found. GPU tests will not work."
    echo "  Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html"
fi

ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)

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

# Check if PyTorch ROCm is already installed
EXISTING_HIP=$(python3 -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "none")
if [ "$EXISTING_HIP" != "none" ] && [ "$EXISTING_HIP" != "None" ]; then
    echo "  PyTorch ROCm already installed (HIP: $EXISTING_HIP)"
else
    if [ "${ROCM_MAJOR:-6}" = "7" ]; then
        echo "  ROCm 7.x: Installing PyTorch nightly for rocm7.0..."
        pip install -q --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.0 2>&1 \
            || echo "  WARNING: PyTorch nightly install failed"
    else
        echo "  ROCm 6.x: Installing PyTorch stable for rocm6.3..."
        pip install -q torch --index-url https://download.pytorch.org/whl/rocm6.3 2>&1 \
            || echo "  WARNING: PyTorch install failed"
    fi
fi

# Verify PyTorch
python3 -c "
import torch
assert torch.version.hip is not None, 'PyTorch is not a ROCm build!'
print(f'  PyTorch {torch.__version__} (HIP {torch.version.hip})')
print(f'  GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
" || echo "  WARNING: PyTorch ROCm verification failed"

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
pip install -q numpy posix_ipc wrapt pytest pre-commit 2>&1

# Build kvcached with ROCm
echo "  Building kvcached (this may take a few minutes)..."
pip install -e . --no-build-isolation --no-cache-dir 2>&1 | tail -5

# Verify
python3 -c "import kvcached.vmm_ops; print('  kvcached.vmm_ops loaded successfully')" \
    || echo "  WARNING: kvcached build verification failed"

# --- 7. Environment config file ---
log "Creating environment setup file..."
cat > ~/setup_env.sh << ENVEOF
#!/bin/bash
# Load environment with: source ~/setup_env.sh
export PATH="\$HOME/.local/bin:\$HOME/.claude/bin:/opt/rocm/bin:\$PATH"
export ROCM_HOME=/opt/rocm
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

echo "Environment loaded:"
echo "  Python:   \$(python3 --version 2>/dev/null || echo 'N/A')"
echo "  PyTorch:  \$(python3 -c 'import torch; print(f"{torch.__version__} (HIP {torch.version.hip})")' 2>/dev/null || echo 'N/A')"
echo "  ROCm:     \$(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')"
echo "  GPU:      \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "  kvcached: $WORK_DIR"
echo "  Branch:   \$(cd $WORK_DIR && git branch --show-current 2>/dev/null || echo 'N/A')"
ENVEOF
chmod +x ~/setup_env.sh

if ! grep -q "setup_env.sh" ~/.bashrc 2>/dev/null; then
    echo '[ -f ~/setup_env.sh ] && source ~/setup_env.sh' >> ~/.bashrc
fi

# --- 8. Smoke tests ---
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
echo "  4. vLLM integration test:"
echo "     ENABLE_KVCACHED=true python -m vllm.entrypoints.openai.api_server \\"
echo "       --model <model> --no-enable-prefix-caching"
echo ""
echo "  5. Use Claude Code for further development:"
echo "     cd $WORK_DIR && claude"
echo ""
echo "  See docs/amd-gpu-testing-guide.md for full details."
