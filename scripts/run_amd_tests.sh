#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# kvcached AMD GPU Test Runner
#
# Runs all tests relevant to the ROCm port, categorized by tier.
# Outputs a summary with pass/fail/skip counts.
#
# Usage:
#   bash scripts/run_amd_tests.sh           # Run all tiers
#   bash scripts/run_amd_tests.sh --quick   # Tier 1-2 only (no GPU needed)
#   bash scripts/run_amd_tests.sh --gpu     # Tier 3-4 only (GPU required)
#
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KVCACHED_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$KVCACHED_DIR"

# ─── Colors ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$(tput bold); RESET=$(tput sgr0)
    GREEN=$(tput setaf 2); RED=$(tput setaf 1); YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6)
else
    BOLD=""; RESET=""; GREEN=""; RED=""; YELLOW=""; CYAN=""
fi

# ─── Parse args ─────────────────────────────────────────────────────────────
RUN_CPU=true
RUN_GPU=true
case "${1:-all}" in
    --quick) RUN_GPU=false ;;
    --gpu)   RUN_CPU=false ;;
    all|"")  ;;
    -h|--help)
        echo "Usage: $0 [--quick|--gpu|all]"
        echo "  --quick  Tier 1-2 only (no GPU needed)"
        echo "  --gpu    Tier 3-4 only (GPU required)"
        echo "  all      All tiers (default)"
        exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
esac

# ─── State ──────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
RESULTS=()

run_tier() {
    local tier="$1" desc="$2" cmd="$3"
    echo ""
    echo "${BOLD}${CYAN}[$tier] $desc${RESET}"
    echo "  Command: $cmd"
    echo "  ---"

    local output rc
    output=$(eval "$cmd" 2>&1) && rc=0 || rc=$?

    # Extract pytest summary line
    local summary
    summary=$(echo "$output" | grep -E '(passed|failed|error|skipped)' | tail -1)

    if [[ $rc -eq 0 ]]; then
        echo "  ${GREEN}PASS${RESET}  $summary"
        PASS=$((PASS+1))
        RESULTS+=("${GREEN}PASS${RESET}  $tier  $desc")
    elif [[ $rc -eq 5 ]]; then
        # pytest exit code 5 = no tests collected
        echo "  ${YELLOW}SKIP${RESET}  no tests collected"
        SKIP=$((SKIP+1))
        RESULTS+=("${YELLOW}SKIP${RESET}  $tier  $desc")
    else
        echo "  ${RED}FAIL${RESET}  $summary"
        # Show last 15 lines for debugging
        echo "$output" | tail -15 | sed 's/^/  /'
        FAIL=$((FAIL+1))
        RESULTS+=("${RED}FAIL${RESET}  $tier  $desc")
    fi
}

# ─── Header ─────────────────────────────────────────────────────────────────
echo "${BOLD}============================================${RESET}"
echo "${BOLD}  kvcached AMD GPU Test Runner${RESET}"
echo "${BOLD}============================================${RESET}"

# Detect environment
python3 -c "
import torch
hip = torch.version.hip
gpu = 'N/A'
try: gpu = torch.cuda.get_device_name(0)
except: pass
print(f'  PyTorch {torch.__version__}  HIP: {hip}  GPU: {gpu}')
" 2>/dev/null || echo "  PyTorch: not available"

# ─── Tier 1: CPU-only unit tests (no GPU) ────────────────────────────────
if [[ "$RUN_CPU" = true ]]; then
    run_tier "Tier 1a" "Shared memory tracker (CPU)" \
        "python3 -m pytest tests/test_shm_info_tracker.py -v --tb=short -q"

    run_tier "Tier 1b" "Build & ROCm detection logic (CPU)" \
        "python3 -m pytest tests/test_gpu_compat.py -v --tb=short -q"

# ─── Tier 2: pre-commit (no GPU) ─────────────────────────────────────────
    run_tier "Tier 2" "pre-commit (lint, format, headers)" \
        "pre-commit run --all-files"
fi

# ─── Tier 3: GPU VMM smoke tests ─────────────────────────────────────────
if [[ "$RUN_GPU" = true ]]; then
    run_tier "Tier 3a" "ROCm VMM smoke (init/map/unmap)" \
        "python3 -m pytest tests/test_rocm_vmm.py -v --tb=short -q"

    run_tier "Tier 3b" "Paged allocator aliasing (VMM correctness)" \
        "python3 -m pytest tests/test_paged_allocator_aliasing.py -v --tb=short -q"

# ─── Tier 4: kvcached core integration ───────────────────────────────────
    run_tier "Tier 4" "KVCacheManager (alloc/free/resize/trim)" \
        "python3 -m pytest tests/test_kvcache_manager.py -v --tb=short -q"
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
