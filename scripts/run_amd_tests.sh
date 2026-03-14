#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# kvcached AMD GPU Test Runner (Docker-based)
#
# All tests run inside official ROCm Docker containers for reproducibility.
# Requires: AMD GPU with ROCm driver, Docker with GPU access.
#
# Usage:
#   bash scripts/run_amd_tests.sh           # Run all tiers
#   bash scripts/run_amd_tests.sh --quick   # Tier 1-2 only (unit tests)
#   bash scripts/run_amd_tests.sh --gpu     # Tier 3-5 only (GPU + E2E)
#   bash scripts/run_amd_tests.sh --e2e     # Tier 5 only (E2E benchmarks)
#
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KVCACHED_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

VLLM_IMAGE="vllm/vllm-openai-rocm:latest"
SGLANG_IMAGE="lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312"
DOCKER_GPU_ARGS="--device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined"

# ─── Colors ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$(tput bold); RESET=$(tput sgr0)
    GREEN=$(tput setaf 2); RED=$(tput setaf 1); YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6)
else
    BOLD=""; RESET=""; GREEN=""; RED=""; YELLOW=""; CYAN=""
fi

# ─── Parse args ─────────────────────────────────────────────────────────────
RUN_UNIT=true
RUN_GPU=true
RUN_E2E=true
case "${1:-all}" in
    --quick) RUN_GPU=false; RUN_E2E=false ;;
    --gpu)   RUN_UNIT=false ;;
    --e2e)   RUN_UNIT=false; RUN_GPU=false ;;
    all|"")  ;;
    -h|--help)
        echo "Usage: $0 [--quick|--gpu|--e2e|all]"
        echo "  --quick  Tier 1-2 only (unit tests, no heavy GPU)"
        echo "  --gpu    Tier 3-5 (GPU VMM + E2E benchmarks)"
        echo "  --e2e    Tier 5 only (E2E benchmarks: vLLM + SGLang)"
        echo "  all      All tiers (default)"
        exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
esac

# ─── State ──────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
RESULTS=()

run_tier() {
    local tier="$1" desc="$2"
    shift 2
    echo ""
    echo "${BOLD}${CYAN}[$tier] $desc${RESET}"
    echo "  ---"

    local output rc
    output=$("$@" 2>&1) && rc=0 || rc=$?

    # Extract summary line (pytest or custom)
    local summary
    summary=$(echo "$output" | grep -E '(passed|failed|error|skipped|ALL CHECKS|SOME CHECKS)' | tail -1)

    if [[ $rc -eq 0 ]]; then
        echo "  ${GREEN}PASS${RESET}  $summary"
        PASS=$((PASS+1))
        RESULTS+=("${GREEN}PASS${RESET}  $tier  $desc")
    elif [[ $rc -eq 5 ]]; then
        echo "  ${YELLOW}SKIP${RESET}  no tests collected"
        SKIP=$((SKIP+1))
        RESULTS+=("${YELLOW}SKIP${RESET}  $tier  $desc")
    else
        echo "  ${RED}FAIL${RESET}  $summary"
        echo "$output" | tail -20 | sed 's/^/  /'
        FAIL=$((FAIL+1))
        RESULTS+=("${RED}FAIL${RESET}  $tier  $desc")
    fi
}

# Helper: run a command inside the vLLM Docker container
docker_vllm() {
    docker run --rm --entrypoint bash \
        $DOCKER_GPU_ARGS \
        -v "$KVCACHED_DIR:/kvcached" \
        "$VLLM_IMAGE" \
        -c "cd /kvcached && pip install -e . --no-build-isolation -q 2>&1 | tail -1 && $1"
}

# Helper: run a command inside the SGLang Docker container
docker_sglang() {
    docker run --rm \
        $DOCKER_GPU_ARGS \
        -v "$KVCACHED_DIR:/kvcached" \
        "$SGLANG_IMAGE" \
        bash -c "cd /kvcached && pip install -e . --no-build-isolation -q 2>&1 | tail -1 && $1"
}

# ─── Header ─────────────────────────────────────────────────────────────────
echo "${BOLD}============================================${RESET}"
echo "${BOLD}  kvcached AMD GPU Test Runner (Docker)${RESET}"
echo "${BOLD}============================================${RESET}"
echo "  kvcached:  $KVCACHED_DIR"
echo "  vLLM:      $VLLM_IMAGE"
echo "  SGLang:    $SGLANG_IMAGE"

# ─── Tier 1: Unit tests (inside Docker, but CPU-only logic) ────────────────
if [[ "$RUN_UNIT" = true ]]; then
    run_tier "Tier 1a" "Shared memory tracker (CPU)" \
        docker_vllm "python3 -m pytest tests/test_shm_info_tracker.py -v --tb=short"

    run_tier "Tier 1b" "Build & ROCm detection logic" \
        docker_vllm "python3 -m pytest tests/test_gpu_compat.py -v --tb=short"
fi

# ─── Tier 3: GPU VMM smoke tests ───────────────────────────────────────────
if [[ "$RUN_GPU" = true ]]; then
    run_tier "Tier 3a" "ROCm VMM smoke (init/map/unmap + GPU memory verification)" \
        docker_vllm "python3 -m pytest tests/test_rocm_vmm.py -v --tb=short"

    run_tier "Tier 3b" "Paged allocator aliasing (VMM correctness)" \
        docker_vllm "python3 tests/test_paged_allocator_aliasing.py"

    run_tier "Tier 4" "KVCacheManager (alloc/free/resize/trim)" \
        docker_vllm "python3 -m pytest tests/test_kvcache_manager.py -v --tb=short"
fi

# ─── Tier 5: E2E benchmarks (vLLM + SGLang) ───────────────────────────────
if [[ "$RUN_E2E" = true ]]; then
    run_tier "Tier 5a" "vLLM + kvcached elastic memory (E2E)" \
        docker_vllm "python3 tests/test_elastic_memory_e2e.py"

    run_tier "Tier 5b" "vLLM baseline — no kvcached (E2E)" \
        docker_vllm "python3 tests/test_elastic_memory_e2e.py --baseline"

    run_tier "Tier 5c" "SGLang + kvcached elastic memory (E2E)" \
        docker_sglang "python3 tests/test_sglang_e2e.py"

    run_tier "Tier 5d" "SGLang baseline — no kvcached (E2E)" \
        docker_sglang "python3 tests/test_sglang_e2e.py --baseline"
fi

# ─── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}============================================${RESET}"
echo "${BOLD}  Summary${RESET}"
echo "${BOLD}============================================${RESET}"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
TOTAL=$((PASS+FAIL+SKIP))
echo "  ${GREEN}$PASS passed${RESET}, ${RED}$FAIL failed${RESET}, ${YELLOW}$SKIP skipped${RESET} (total: $TOTAL tiers)"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "  ${RED}Some tests failed. See output above for details.${RESET}"
    exit 1
else
    echo ""
    echo "  ${GREEN}All tests passed!${RESET}"
    exit 0
fi
