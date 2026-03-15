#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# kvcached AMD GPU Development & Test Environment Setup (Docker-based)
#
# Prerequisites:
#   - AMD GPU with ROCm driver installed on host
#   - Docker with GPU access (--device=/dev/kfd --device=/dev/dri)
#
# Usage:
#   bash setup_amd_dev.sh          # Full setup + all tests
#   bash setup_amd_dev.sh --test   # Tests only (skip clone)
#
set -uo pipefail

# NOTE: We use PIPESTATUS[0] below to capture the Docker exit code
# from pipelines like `docker run ... | grep -v ...`, since $? would
# capture grep's exit code instead.

REPO_URL="https://github.com/midokura/kvcached.git"
BRANCH="feature/amd-upstream-rebase"
WORK_DIR="${KVCACHED_DIR:-$HOME/kvcached}"

VLLM_IMAGE="vllm/vllm-openai-rocm:latest"
SGLANG_IMAGE="lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312"

DOCKER_GPU_ARGS="--device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined"

step=0
log() { step=$((step+1)); echo ""; echo "[$step] $1"; echo "---"; }

echo "============================================"
echo "  kvcached AMD GPU Setup (Docker-based)"
echo "============================================"
echo "  Branch:  $BRANCH"
echo "  Work:    $WORK_DIR"
echo "  vLLM:    $VLLM_IMAGE"
echo "  SGLang:  $SGLANG_IMAGE"

# ── 1. Prerequisites check ───────────────────────────────────────────────
log "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "  ERROR: Docker is not installed."
    exit 1
fi
echo "  Docker: $(docker --version | head -1)"

if [ ! -e /dev/kfd ]; then
    echo "  ERROR: /dev/kfd not found. ROCm driver is not installed."
    exit 1
fi
echo "  ROCm driver: OK (/dev/kfd exists)"

if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname 2>/dev/null | grep "Card Series" | head -1 | sed 's/^/  /'
fi

# ── 1.5. GitHub CLI ─────────────────────────────────────────────────────
log "Installing GitHub CLI..."
if ! command -v gh &> /dev/null; then
    (type -p wget >/dev/null || (apt-get update -qq && apt-get install -y -qq wget)) && \
    mkdir -p -m 755 /etc/apt/keyrings && \
    out=$(mktemp) && wget -qO "$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg && \
    cat "$out" | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update -qq 2>&1 | grep -v "^W:" || true && \
    apt-get install -y -qq gh 2>&1 || echo "  WARNING: gh install failed"
else
    echo "  gh already installed: $(gh --version | head -1)"
fi

# ── 1.6. Claude Code ────────────────────────────────────────────────────
log "Installing Claude Code..."
export PATH="$HOME/.local/bin:$HOME/.claude/bin:$PATH"
if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash \
        || echo "  WARNING: Claude Code install failed. Install manually."
fi
if command -v claude &> /dev/null; then
    echo "  Claude Code: $(claude --version 2>/dev/null || echo 'installed')"
else
    echo "  WARNING: claude not found in PATH after install."
fi

# ── 2. Clone kvcached ────────────────────────────────────────────────────
if [ "${1:-}" != "--test" ]; then
    log "Cloning kvcached..."
    if [ -d "$WORK_DIR" ]; then
        echo "  $WORK_DIR exists, updating..."
        cd "$WORK_DIR"
        git fetch origin
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
    else
        git clone "$REPO_URL" "$WORK_DIR"
        cd "$WORK_DIR"
        git checkout "$BRANCH"
    fi
fi
cd "$WORK_DIR"
echo "  kvcached: $(git log --oneline -1)"

# ── 3. Pull Docker images ────────────────────────────────────────────────
log "Pulling Docker images..."
docker pull "$VLLM_IMAGE" 2>&1 | tail -1
docker pull "$SGLANG_IMAGE" 2>&1 | tail -1

# ── 4. Build + unit tests ────────────────────────────────────────────────
log "Build + unit tests (vLLM image)..."
docker run --rm --entrypoint bash \
    $DOCKER_GPU_ARGS \
    -v "$WORK_DIR:/kvcached" \
    "$VLLM_IMAGE" \
    -c '
cd /kvcached
echo "=== Build ==="
pip install -e . --no-build-isolation -q 2>&1 | tail -1
python3 -c "import kvcached.vmm_ops; print(\"kvcached.vmm_ops loaded OK\")"
echo ""
echo "=== Unit tests ==="
python3 -m pytest tests/test_shm_info_tracker.py tests/test_gpu_compat.py tests/test_rocm_vmm.py -v --tb=short 2>&1 | tail -20
'

# ── 5. vLLM elastic memory benchmark ─────────────────────────────────────
log "vLLM + kvcached elastic memory benchmark..."
docker run --rm --entrypoint bash \
    $DOCKER_GPU_ARGS \
    -v "$WORK_DIR:/kvcached" \
    "$VLLM_IMAGE" \
    -c 'cd /kvcached && pip install -e . --no-build-isolation -q 2>&1 | tail -1 && python3 tests/test_elastic_memory_e2e.py 2>&1' \
    | grep -v "^(EngineCore\|^INFO\|^WARNING\|^Processed\|^Rendering\|^\[kvcached\]\|triton_kernels\|SyntaxWarning\|resource_tracker"
VLLM_ELASTIC=${PIPESTATUS[0]}

# ── 6. vLLM baseline (no kvcached) ───────────────────────────────────────
log "vLLM baseline (no kvcached)..."
docker run --rm --entrypoint bash \
    $DOCKER_GPU_ARGS \
    -v "$WORK_DIR:/kvcached" \
    "$VLLM_IMAGE" \
    -c 'cd /kvcached && pip install -e . --no-build-isolation -q 2>&1 | tail -1 && python3 tests/test_elastic_memory_e2e.py --baseline 2>&1' \
    | grep -v "^(EngineCore\|^INFO\|^WARNING\|^Processed\|^Rendering\|triton_kernels\|SyntaxWarning\|resource_tracker"
VLLM_BASELINE=${PIPESTATUS[0]}

# ── 7. SGLang elastic memory benchmark ───────────────────────────────────
log "SGLang + kvcached elastic memory benchmark..."
docker run --rm \
    $DOCKER_GPU_ARGS \
    -v "$WORK_DIR:/kvcached" \
    "$SGLANG_IMAGE" \
    bash -c 'cd /kvcached && pip install -e . --no-build-isolation -q 2>&1 | tail -1 && python3 tests/test_sglang_e2e.py 2>&1' \
    | grep -v "^\[aiter\]\|^INFO:aiter\|^\[Gloo\]\|^Loading pt\|^Capturing\|clang.*option\|^failed to\|type hints\|compile_template\|start build\|finish build\|import \["
SGLANG_ELASTIC=${PIPESTATUS[0]}

# ── 8. SGLang baseline (no kvcached) ─────────────────────────────────────
log "SGLang baseline (no kvcached)..."
docker run --rm \
    $DOCKER_GPU_ARGS \
    -v "$WORK_DIR:/kvcached" \
    "$SGLANG_IMAGE" \
    bash -c 'cd /kvcached && pip install -e . --no-build-isolation -q 2>&1 | tail -1 && python3 tests/test_sglang_e2e.py --baseline 2>&1' \
    | grep -v "^\[aiter\]\|^INFO:aiter\|^\[Gloo\]\|^Loading pt\|^Capturing\|clang.*option\|^failed to\|type hints\|compile_template\|start build\|finish build\|import \["
SGLANG_BASELINE=${PIPESTATUS[0]}

# ── 9. Environment config ─────────────────────────────────────────────────
log "Creating environment setup file..."
cat > ~/setup_env.sh << 'ENVEOF'
#!/bin/bash
# Load environment with: source ~/setup_env.sh
export PATH="$HOME/.local/bin:$HOME/.claude/bin:/opt/rocm/bin:$PATH"
export ROCM_HOME=/opt/rocm

echo "Environment loaded:"
echo "  ROCm:   $(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')"
echo "  Docker: $(docker --version 2>/dev/null || echo 'N/A')"
echo "  GPU:    $(rocm-smi --showproductname 2>/dev/null | grep 'Card Series' | head -1 | sed 's/.*: //' || echo 'N/A')"
echo "  gh:     $(gh --version 2>/dev/null | head -1 || echo 'N/A')"
echo "  claude: $(claude --version 2>/dev/null || echo 'N/A')"
ENVEOF
chmod +x ~/setup_env.sh

if ! grep -q "setup_env.sh" ~/.bashrc 2>/dev/null; then
    echo '[ -f ~/setup_env.sh ] && source ~/setup_env.sh' >> ~/.bashrc
fi

# ── 10. Summary ───────────────────────────────────────────────────────────
log "Summary"

echo ""
echo "============================================"
echo "  Results"
echo "============================================"
echo ""
echo "  vLLM + kvcached (elastic):     $([ $VLLM_ELASTIC -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  vLLM baseline (no kvcached):   $([ $VLLM_BASELINE -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  SGLang + kvcached (elastic):   $([ $SGLANG_ELASTIC -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  SGLang baseline (no kvcached): $([ $SGLANG_BASELINE -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo ""
echo "  To re-run individual tests:"
echo ""
echo "  # vLLM elastic (kvcached)"
echo "  docker run --rm --entrypoint bash \\"
echo "    $DOCKER_GPU_ARGS -v $WORK_DIR:/kvcached \\"
echo "    $VLLM_IMAGE \\"
echo "    -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_elastic_memory_e2e.py'"
echo ""
echo "  # vLLM baseline (no kvcached)"
echo "  docker run --rm --entrypoint bash \\"
echo "    $DOCKER_GPU_ARGS -v $WORK_DIR:/kvcached \\"
echo "    $VLLM_IMAGE \\"
echo "    -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_elastic_memory_e2e.py --baseline'"
echo ""
echo "  # SGLang elastic (kvcached)"
echo "  docker run --rm \\"
echo "    $DOCKER_GPU_ARGS -v $WORK_DIR:/kvcached \\"
echo "    $SGLANG_IMAGE \\"
echo "    bash -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_sglang_e2e.py'"
echo ""
echo "  # SGLang baseline (no kvcached)"
echo "  docker run --rm \\"
echo "    $DOCKER_GPU_ARGS -v $WORK_DIR:/kvcached \\"
echo "    $SGLANG_IMAGE \\"
echo "    bash -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_sglang_e2e.py --baseline'"
echo ""

echo ""
echo "  Next steps:"
echo ""
echo "  1. Load environment (auto-loaded on next login):"
echo "     source ~/setup_env.sh"
echo ""
echo "  2. Authenticate GitHub (for pushes):"
echo "     gh auth login"
echo ""
echo "  3. Use Claude Code for development:"
echo "     cd $WORK_DIR && claude"
echo ""

if [ $VLLM_ELASTIC -eq 0 ] && [ $VLLM_BASELINE -eq 0 ] && [ $SGLANG_ELASTIC -eq 0 ] && [ $SGLANG_BASELINE -eq 0 ]; then
    echo "  All tests PASSED."
    exit 0
else
    echo "  Some tests FAILED."
    exit 1
fi
