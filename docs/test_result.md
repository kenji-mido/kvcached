root@rocm-7-1-software-gpu-mi300x1-192gb-devcloud-atl1:~# ./setup_amd_dev.sh
============================================
  kvcached AMD GPU Setup (Docker-based)
============================================
  Branch:  feature/amd-upstream-rebase
  Work:    /root/kvcached
  vLLM:    vllm/vllm-openai-rocm:latest
  SGLang:  lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312

[1] Checking prerequisites...
---
  Docker: Docker version 29.0.2, build 8108357
  ROCm driver: OK (/dev/kfd exists)
  GPU[0]                : Card Series:          AMD Instinct MI300X VF

[2] Installing GitHub CLI...
---
  gh already installed: gh version 2.88.1 (2026-03-12)

[3] Installing Claude Code...
---
  Claude Code: 2.1.76 (Claude Code)

[4] Cloning kvcached...
---
  /root/kvcached exists, updating...
A       tests/test_elastic_memory_e2e.py
A       tests/test_sglang_e2e.py
Already on 'feature/amd-upstream-rebase'
Your branch is up to date with 'origin/feature/amd-upstream-rebase'.
From https://github.com/kenji-mido/kvcached
 * branch            feature/amd-upstream-rebase -> FETCH_HEAD
Already up to date.
  kvcached: be2fd6e feat: add scripts and docs from MI300X-validated branch

[5] Pulling Docker images...
---
docker.io/vllm/vllm-openai-rocm:latest
docker.io/lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312

[6] Build + unit tests (vLLM image)...
---
=== Build ===
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
kvcached.vmm_ops loaded OK

=== Unit tests ===
tests/test_gpu_compat.py::test_setup_py_is_rocm_consistent PASSED        [ 43%]
tests/test_gpu_compat.py::test_page_size_default PASSED                  [ 46%]
tests/test_gpu_compat.py::test_page_size_validation_cuda_valid PASSED    [ 50%]
tests/test_gpu_compat.py::test_page_size_validation_rocm_valid PASSED    [ 53%]
tests/test_gpu_compat.py::test_page_size_cuda_rejects_1mb PASSED         [ 56%]
tests/test_gpu_compat.py::test_page_size_cuda_rejects_3mb PASSED         [ 60%]
tests/test_gpu_compat.py::test_page_size_rocm_accepts_3mb PASSED         [ 63%]
tests/test_gpu_compat.py::test_page_size_rejects_zero PASSED             [ 66%]
tests/test_gpu_compat.py::test_page_size_rejects_negative PASSED         [ 70%]
tests/test_gpu_compat.py::test_page_size_rejects_non_integer PASSED      [ 73%]
tests/test_gpu_compat.py::test_page_size_rejects_float_string PASSED     [ 76%]
tests/test_rocm_vmm.py::test_init_kvcached_rocm PASSED                   [ 80%]
tests/test_rocm_vmm.py::test_granularity_query PASSED                    [ 83%]
tests/test_rocm_vmm.py::test_kv_tensor_create PASSED                     [ 86%]
tests/test_rocm_vmm.py::test_kv_tensor_map_unmap PASSED                  [ 90%]
tests/test_rocm_vmm.py::test_map_increases_gpu_memory PASSED             [ 93%]
tests/test_rocm_vmm.py::test_map_per_page_delta_is_linear PASSED         [ 96%]
tests/test_rocm_vmm.py::test_map_unmap_cycle_returns_memory PASSED       [100%]

============================== 30 passed in 3.33s ==============================

[7] vLLM + kvcached elastic memory benchmark...
---
  "(e.g. 'abcdabcdabcd...' or '\emoji \emoji \emoji ...'). This feature "
  "(e.g. 'abcdabcdabcd...' or '\emoji \emoji \emoji ...'). This feature "
================================================================================
  kvcached Elastic Memory E2E Benchmark  [kvcached ENABLED]
================================================================================
  MAX_RESERVED_PAGES = 2
  MIN_RESERVED_PAGES = 1
  GPU before model: 330 MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 — Load model (facebook/opt-125m)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After model load (idle)
  │
  │  Virtual (address space) :     96,632 MB
  │  ┌─ Physical (GPU RAM)   :         48 MB  (0.05% of Virtual)
  │  │  ├ Used    (active KV) :          0 MB
  │  │  └ Prealloc (reserved) :         48 MB
  │  Free (unmapped)          :     96,584 MB
  │  GPU total used           :      1,956 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(0 MB)  ▒=Prealloc(48 MB)
  └

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 2 — Small batch  (4 prompts × 128 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 538.63it/s]
Processed prompts: 100%|██████████| 4/4 [00:02<00:00,  1.69it/s, est. speed input: 13.49 toks/s, output: 72.08 toks/s]

  ┌─ After Small batch
  │
  │  Virtual (address space) :     96,632 MB
  │  ┌─ Physical (GPU RAM)   :         96 MB  (0.10% of Virtual)
  │  │  ├ Used    (active KV) :          0 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     96,536 MB
  │  GPU total used           :      2,036 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(0 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 2 — Small batch ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │       0       48       48    96,584    96,632 │   1,956 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░
    0.2s │      48       48       96    96,536    96,632 │   2,012 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.4s │      48       48       96    96,536    96,632 │   2,012 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.6s │      48       48       96    96,536    96,632 │   2,014 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.8s │      48       48       96    96,536    96,632 │   2,022 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.0s │      48       48       96    96,536    96,632 │   2,026 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.2s │      48       48       96    96,536    96,632 │   2,028 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.4s │      48       48       96    96,536    96,632 │   2,030 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.6s │      48       48       96    96,536    96,632 │   2,032 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.8s │      48       48       96    96,536    96,632 │   2,032 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.0s │      48       48       96    96,536    96,632 │   2,034 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.2s │      48       48       96    96,536    96,632 │   2,036 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.4s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.6s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.8s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.0s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.2s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.4s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.6s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.8s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    4.0s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    4.2s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    4.3s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 3 — Large batch  (64 prompts × 512 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: exceed MAX_RESERVED_PAGES=2 → trigger hipMemUnmap
Rendering prompts: 100%|██████████| 64/64 [00:00<00:00, 6791.71it/s]
Processed prompts: 100%|██████████| 64/64 [00:04<00:00, 15.93it/s, est. speed input: 334.47 toks/s, output: 6837.50 toks/s]

  ┌─ After Large batch
  │
  │  Virtual (address space) :     96,632 MB
  │  ┌─ Physical (GPU RAM)   :         96 MB  (0.10% of Virtual)
  │  │  ├ Used    (active KV) :          0 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     96,536 MB
  │  GPU total used           :      2,038 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(0 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 3 — Large batch ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │       0       96       96    96,536    96,632 │   2,036 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    0.3s │      96       48      144    96,488    96,632 │   2,086 │ ███▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    0.6s │     192       48      240    96,392    96,632 │   2,182 │ ██████▒░░░░░░░░░░░░░░░░░░░░░░░
    0.9s │     288       48      336    96,296    96,632 │   2,278 │ █████████▒░░░░░░░░░░░░░░░░░░░░
    1.2s │     336       48      384    96,248    96,632 │   2,326 │ ██████████▒░░░░░░░░░░░░░░░░░░░
    1.5s │     384       48      432    96,200    96,632 │   2,374 │ ████████████▒░░░░░░░░░░░░░░░░░
    1.8s │     480       48      528    96,104    96,632 │   2,470 │ ███████████████▒░░░░░░░░░░░░░░
    2.1s │     528       48      576    96,056    96,632 │   2,518 │ ████████████████▒░░░░░░░░░░░░░
    2.4s │     624       48      672    95,960    96,632 │   2,614 │ ███████████████████▒░░░░░░░░░░
    2.7s │     720       48      768    95,864    96,632 │   2,710 │ ██████████████████████▒░░░░░░░
    3.0s │     768       48      816    95,816    96,632 │   2,758 │ ████████████████████████▒░░░░░
    3.3s │     816       48      864    95,768    96,632 │   2,806 │ █████████████████████████▒░░░░
    3.6s │     864       48      912    95,720    96,632 │   2,854 │ ███████████████████████████▒░░
    3.9s │     912       48      960    95,672    96,632 │   2,902 │ ████████████████████████████▒░
    4.2s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    4.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    4.8s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    5.1s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    5.4s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    5.7s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░
    6.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 4 — Cooldown  (4 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  generate() is synchronous — blocks are already freed.
  This phase confirms no further changes occur.

  ┌─ After Cooldown
  │
  │  Virtual (address space) :     96,632 MB
  │  ┌─ Physical (GPU RAM)   :         96 MB  (0.10% of Virtual)
  │  │  ├ Used    (active KV) :          0 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     96,536 MB
  │  GPU total used           :      2,038 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(0 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 4 — Cooldown ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 5 — Re-grow  (16 prompts × 256 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rendering prompts: 100%|██████████| 16/16 [00:00<00:00, 6912.74it/s]
Processed prompts: 100%|██████████| 16/16 [00:01<00:00, 10.00it/s, est. speed input: 70.02 toks/s, output: 547.67 toks/s]

  ┌─ After Re-grow
  │
  │  Virtual (address space) :     96,632 MB
  │  ┌─ Physical (GPU RAM)   :         96 MB  (0.10% of Virtual)
  │  │  ├ Used    (active KV) :          0 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     96,536 MB
  │  GPU total used           :      2,038 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(0 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 5 — Re-grow ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.1s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.2s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.3s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.4s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.5s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.6s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.7s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.8s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.9s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.0s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.1s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.2s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.3s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.4s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.5s │      48       48       96    96,536    96,632 │   2,038 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.6s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.7s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.8s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.9s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.1s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.2s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.3s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.4s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.6s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.7s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.8s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.9s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.0s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.1s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.2s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.3s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.4s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.5s │       0       96       96    96,536    96,632 │   2,038 │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

================================================================================
  VERIFICATION  [kvcached ENABLED]
================================================================================
  [PASS] Physical << Virtual at idle  (elastic, not pre-allocated)
         Physical 48 MB = 0.05% of Virtual 96,632 MB  [threshold: < 5%]
  [PASS] Large batch peak > small batch peak  (on-demand growth)
         small 96 MB, large 960 MB
  [PASS] GPU memory << Virtual KV limit  (no upfront allocation)
         GPU 1,956 MB vs Virtual 96,632 MB  [threshold: GPU < 10% of Virtual]
  [PASS] Used pages varied during inference  (alloc/free cycle)
         used: min=0 MB, max=912 MB
  [PASS] Physical shrank after completion  (hipMemUnmap confirmed)
         peak 960 MB → after 96 MB  (freed 864 MB)
  [PASS] Used pages re-appeared on new requests  (pages re-mapped)
         re-grow peak used 48 MB (physical 96 MB)
  [PASS] GPU memory correlates with Physical growth  (not just bookkeeping)
         GPU delta: +946 MB, Physical delta: +912 MB during large batch

  ALL CHECKS PASSED — Elastic memory (hipMemMap / hipMemUnmap) confirmed.
================================================================================

[8] vLLM baseline (no kvcached)...
---
  "(e.g. 'abcdabcdabcd...' or '\emoji \emoji \emoji ...'). This feature "
  "(e.g. 'abcdabcdabcd...' or '\emoji \emoji \emoji ...'). This feature "
================================================================================
  kvcached Elastic Memory E2E Benchmark  [BASELINE (vanilla vLLM)]
================================================================================
  kvcached is DISABLED — vLLM will pre-allocate all KV cache memory.
  GPU before model: 330 MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 — Load model (facebook/opt-125m)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After model load (idle)
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :     98,508 MB  (50.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 2 — Small batch  (4 prompts × 128 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rendering prompts: 100%|██████████| 4/4 [00:00<00:00, 516.46it/s]
Processed prompts: 100%|██████████| 4/4 [00:02<00:00,  1.84it/s, est. speed input: 14.74 toks/s, output: 86.14 toks/s]

  ┌─ After Small batch
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :     98,538 MB  (50.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 2 — Small batch ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │     98,508 │ █████████████████████████████░
    0.2s │     98,516 │ █████████████████████████████░
    0.4s │     98,516 │ █████████████████████████████░
    0.6s │     98,524 │ █████████████████████████████░
    0.8s │     98,524 │ █████████████████████████████░
    1.0s │     98,526 │ █████████████████████████████░
    1.2s │     98,528 │ █████████████████████████████░
    1.4s │     98,534 │ █████████████████████████████░
    1.6s │     98,534 │ █████████████████████████████░
    1.8s │     98,536 │ █████████████████████████████░
    2.0s │     98,538 │ ██████████████████████████████
    2.2s │     98,538 │ ██████████████████████████████
    2.4s │     98,538 │ ██████████████████████████████
    2.6s │     98,538 │ ██████████████████████████████
    2.8s │     98,538 │ ██████████████████████████████
    3.0s │     98,538 │ ██████████████████████████████
    3.2s │     98,538 │ ██████████████████████████████
    3.4s │     98,538 │ ██████████████████████████████
    3.6s │     98,538 │ ██████████████████████████████
    3.8s │     98,538 │ ██████████████████████████████
    4.0s │     98,538 │ ██████████████████████████████
    4.1s │     98,538 │ ██████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 3 — Large batch  (64 prompts × 512 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rendering prompts: 100%|██████████| 64/64 [00:00<00:00, 6859.22it/s]
Processed prompts: 100%|██████████| 64/64 [00:03<00:00, 17.00it/s, est. speed input: 356.93 toks/s, output: 7329.32 toks/s]

  ┌─ After Large batch
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :     98,540 MB  (50.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 3 — Large batch ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │     98,538 │ █████████████████████████████░
    0.2s │     98,538 │ █████████████████████████████░
    0.4s │     98,540 │ ██████████████████████████████
    0.6s │     98,540 │ ██████████████████████████████
    0.8s │     98,540 │ ██████████████████████████████
    1.0s │     98,540 │ ██████████████████████████████
    1.2s │     98,540 │ ██████████████████████████████
    1.4s │     98,540 │ ██████████████████████████████
    1.6s │     98,540 │ ██████████████████████████████
    1.8s │     98,540 │ ██████████████████████████████
    2.0s │     98,540 │ ██████████████████████████████
    2.2s │     98,540 │ ██████████████████████████████
    2.4s │     98,540 │ ██████████████████████████████
    2.6s │     98,540 │ ██████████████████████████████
    2.8s │     98,540 │ ██████████████████████████████
    3.0s │     98,540 │ ██████████████████████████████
    3.2s │     98,540 │ ██████████████████████████████
    3.4s │     98,540 │ ██████████████████████████████
    3.6s │     98,540 │ ██████████████████████████████
    3.8s │     98,540 │ ██████████████████████████████
    4.0s │     98,540 │ ██████████████████████████████
    4.2s │     98,540 │ ██████████████████████████████
    4.4s │     98,540 │ ██████████████████████████████
    4.6s │     98,540 │ ██████████████████████████████
    4.8s │     98,540 │ ██████████████████████████████
    5.0s │     98,540 │ ██████████████████████████████
    5.2s │     98,540 │ ██████████████████████████████
    5.4s │     98,540 │ ██████████████████████████████
    5.6s │     98,540 │ ██████████████████████████████
    5.7s │     98,540 │ ██████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 4 — Cooldown  (4 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  vLLM pre-allocated all KV memory — nothing changes.

  ┌─ After Cooldown
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :     98,540 MB  (50.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 4 — Cooldown ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │     98,540 │ ██████████████████████████████
    0.5s │     98,540 │ ██████████████████████████████
    1.0s │     98,540 │ ██████████████████████████████
    1.5s │     98,540 │ ██████████████████████████████
    2.0s │     98,540 │ ██████████████████████████████
    2.5s │     98,540 │ ██████████████████████████████
    3.0s │     98,540 │ ██████████████████████████████
    3.5s │     98,540 │ ██████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 5 — Re-grow  (16 prompts × 256 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rendering prompts: 100%|██████████| 16/16 [00:00<00:00, 7224.55it/s]
Processed prompts: 100%|██████████| 16/16 [00:01<00:00,  9.76it/s, est. speed input: 68.35 toks/s, output: 1085.03 toks/s]

  ┌─ After Re-grow
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :     98,540 MB  (50.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 5 — Re-grow ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │     98,540 │ ██████████████████████████████
    0.1s │     98,540 │ ██████████████████████████████
    0.2s │     98,540 │ ██████████████████████████████
    0.3s │     98,540 │ ██████████████████████████████
    0.4s │     98,540 │ ██████████████████████████████
    0.5s │     98,540 │ ██████████████████████████████
    0.6s │     98,540 │ ██████████████████████████████
    0.7s │     98,540 │ ██████████████████████████████
    0.8s │     98,540 │ ██████████████████████████████
    0.9s │     98,540 │ ██████████████████████████████
    1.0s │     98,540 │ ██████████████████████████████
    1.1s │     98,540 │ ██████████████████████████████
    1.2s │     98,540 │ ██████████████████████████████
    1.3s │     98,540 │ ██████████████████████████████
    1.4s │     98,540 │ ██████████████████████████████
    1.5s │     98,540 │ ██████████████████████████████
    1.6s │     98,540 │ ██████████████████████████████
    1.7s │     98,540 │ ██████████████████████████████
    1.8s │     98,540 │ ██████████████████████████████
    1.9s │     98,540 │ ██████████████████████████████
    2.0s │     98,540 │ ██████████████████████████████
    2.1s │     98,540 │ ██████████████████████████████
    2.2s │     98,540 │ ██████████████████████████████
    2.3s │     98,540 │ ██████████████████████████████
    2.4s │     98,540 │ ██████████████████████████████
    2.5s │     98,540 │ ██████████████████████████████
    2.6s │     98,540 │ ██████████████████████████████
    2.7s │     98,540 │ ██████████████████████████████
    2.8s │     98,540 │ ██████████████████████████████
    2.9s │     98,540 │ ██████████████████████████████
    3.0s │     98,540 │ ██████████████████████████████
    3.1s │     98,540 │ ██████████████████████████████
    3.2s │     98,540 │ ██████████████████████████████
    3.3s │     98,540 │ ██████████████████████████████
    3.4s │     98,540 │ ██████████████████████████████
    3.5s │     98,540 │ ██████████████████████████████
    3.6s │     98,540 │ ██████████████████████████████

================================================================================
  VERIFICATION  [BASELINE (vanilla vLLM)]
================================================================================
  [PASS] GPU memory is FLAT during inference  (pre-allocated, not elastic)
         GPU range: 98,538 – 98,540 MB  (delta 2 MB, threshold: < 196 MB [0.1% of 196,288 MB GPU])
  [PASS] GPU memory unchanged after requests complete  (no free/unmap)
         idle 98,508 MB → after 98,540 MB  (delta 32 MB, threshold: < 196 MB)
  [PASS] Large upfront KV allocation visible in GPU  (not elastic)
         GPU before model: 330 MB → after load: 98,508 MB  (KV estimate: 98,178 MB, threshold: > 589 MB)

  ALL CHECKS PASSED — Confirmed: vanilla vLLM is NOT elastic.
  GPU memory is allocated upfront and stays constant.
================================================================================
[rank0]:[W315 05:30:57.168227513 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

[9] SGLang + kvcached elastic memory benchmark...
---
[kvcached][INFO][2026-03-15 05:31:10][patch_base.py:98] Applying 4 patches for sglang
[kvcached][INFO][2026-03-15 05:31:10][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:10][patches.py:54] Elastic allocators patched (TokenToKVPool + PagedTokenToKVPool)
[kvcached][INFO][2026-03-15 05:31:15][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:15][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:15][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:15][patch_base.py:178] Successfully patched sglang: elastic_allocator, elastic_memory_pool, elastic_mla_memory_pool, scheduler_memory_leak
================================================================================
  SGLang + kvcached Elastic Memory Benchmark  [kvcached ENABLED]
================================================================================
  MAX_RESERVED_PAGES = 2
  MIN_RESERVED_PAGES = 1
  GPU before model: 330 MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 — Load model (facebook/opt-125m)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[kvcached][INFO][2026-03-15 05:31:17][patch_base.py:98] Applying 4 patches for sglang
[kvcached][INFO][2026-03-15 05:31:17][patch_base.py:98] Applying 4 patches for sglang
[kvcached][INFO][2026-03-15 05:31:17][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:17][patches.py:54] Elastic allocators patched (TokenToKVPool + PagedTokenToKVPool)
[kvcached][INFO][2026-03-15 05:31:17][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:17][patches.py:54] Elastic allocators patched (TokenToKVPool + PagedTokenToKVPool)
[kvcached][INFO][2026-03-15 05:31:20][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:20][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:21][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:21][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:21][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:21][patch_base.py:178] Successfully patched sglang: elastic_allocator, elastic_memory_pool, elastic_mla_memory_pool, scheduler_memory_leak
[kvcached][INFO][2026-03-15 05:31:21][version_utils.py:189] Detected sglang version: 0.5.9.dev20260312+ga57a44739
[kvcached][INFO][2026-03-15 05:31:21][patch_base.py:178] Successfully patched sglang: elastic_allocator, elastic_memory_pool, elastic_mla_memory_pool, scheduler_memory_leak
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.53it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.52it/s]

[kvcached][INFO][2026-03-15 05:31:24][page_allocator.py:155] Init kvcached KV cache allocator: num_layers=12, mem_size_per_layer=4068MB, total_mem_size=97648MB, page_size=2MB, tp_size=1, async_sched=True, contiguous_layout=True, enable_prealloc=True
[kvcached][INFO][2026-03-15 05:31:24][patches.py:400] VirtualKV Cache is allocated. #tokens: 2777543, K size: 95.84 GB, V size: 95.84 GB
[kvcached][INFO][2026-03-15 05:31:24][patches.py:404] Physical KV Cache limits by --mem-fraction-static: #tokens: 2777543, K size: 47.68 GB, V size: 47.68 GB
[kvcached][INFO][2026-03-15 05:31:24][patches.py:85] [kvcached] ElasticTokenToKVPoolAllocator in use: size=2777543 (page_size=1 path)
[2026-03-15 05:31:49] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.
Capturing batches (bs=1 avail_mem=187.29 GB): 100%|██████████| 52/52 [00:34<00:00,  1.50it/s]
[2026-03-15 05:32:08] Current hipcc not support: -mllvm -amdgpu-coerce-illegal-types=1, skip it.

  ┌─ After model load (idle)
  │
  │  Virtual (address space) :     97,648 MB
  │  ┌─ Physical (GPU RAM)   :         96 MB  (0.10% of Virtual)
  │  │  ├ Used    (active KV) :         48 MB
  │  │  └ Prealloc (reserved) :         48 MB
  │  Free (unmapped)          :     97,552 MB
  │  GPU total used           :      4,514 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [██████████████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(48 MB)  ▒=Prealloc(48 MB)
  └

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 2 — Small batch  (4 prompts × 128 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After Small batch
  │
  │  Virtual (address space) :     97,648 MB
  │  ┌─ Physical (GPU RAM)   :         96 MB  (0.10% of Virtual)
  │  │  ├ Used    (active KV) :         48 MB
  │  │  └ Prealloc (reserved) :         48 MB
  │  Free (unmapped)          :     97,552 MB
  │  GPU total used           :      4,726 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [██████████████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(48 MB)  ▒=Prealloc(48 MB)
  └

  ── Phase 2 — Small batch ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │      48       48       96    97,552    97,648 │   4,514 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.3s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    6.6s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    9.9s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   13.2s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   16.5s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   19.9s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   23.2s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   26.5s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   29.8s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   33.1s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   36.4s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   39.7s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   43.0s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   46.3s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   49.6s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   53.0s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   56.3s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   59.6s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   62.9s │      48       48       96    97,552    97,648 │   4,640 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   66.2s │      48       48       96    97,552    97,648 │   4,726 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
   67.9s │      48       48       96    97,552    97,648 │   4,726 │ ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 3 — Large batch  (64 prompts × 512 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal: exceed MAX_RESERVED_PAGES=2 → trigger hipMemUnmap

  ┌─ After Large batch
  │
  │  Virtual (address space) :     97,648 MB
  │  ┌─ Physical (GPU RAM)   :        144 MB  (0.15% of Virtual)
  │  │  ├ Used    (active KV) :         48 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     97,504 MB
  │  GPU total used           :      4,798 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(48 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 3 — Large batch ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │      48       48       96    97,552    97,648 │   4,726 │ █▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    0.1s │     144       48      192    97,456    97,648 │   4,846 │ █████▒░░░░░░░░░░░░░░░░░░░░░░░░
    0.2s │     240       48      288    97,360    97,648 │   4,942 │ ████████▒░░░░░░░░░░░░░░░░░░░░░
    0.3s │     336       48      384    97,264    97,648 │   5,038 │ ███████████▒░░░░░░░░░░░░░░░░░░
    0.4s │     432       48      480    97,168    97,648 │   5,134 │ ███████████████▒░░░░░░░░░░░░░░
    0.5s │     528       48      576    97,072    97,648 │   5,230 │ ██████████████████▒░░░░░░░░░░░
    0.6s │     624       48      672    96,976    97,648 │   5,326 │ █████████████████████▒░░░░░░░░
    0.7s │     720       48      768    96,880    97,648 │   5,422 │ █████████████████████████▒░░░░
    0.8s │     816       48      864    96,784    97,648 │   5,518 │ ████████████████████████████▒░
    0.9s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.0s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.1s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.2s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.3s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.4s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.5s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.6s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.7s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.8s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    1.9s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.0s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.1s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.2s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.3s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.4s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.5s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.6s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.7s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░
    2.8s │      48       96      144    97,504    97,648 │   4,798 │ █▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 4 — Cooldown  (4 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  generate() is synchronous — blocks are already freed.
  This phase confirms no further changes occur.

  ┌─ After Cooldown
  │
  │  Virtual (address space) :     97,648 MB
  │  ┌─ Physical (GPU RAM)   :        144 MB  (0.15% of Virtual)
  │  │  ├ Used    (active KV) :         48 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     97,504 MB
  │  GPU total used           :      4,798 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(48 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 4 — Cooldown ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.5s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.5s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.5s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    3.5s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 5 — Re-grow  (16 prompts × 256 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After Re-grow
  │
  │  Virtual (address space) :     97,648 MB
  │  ┌─ Physical (GPU RAM)   :        144 MB  (0.15% of Virtual)
  │  │  ├ Used    (active KV) :         48 MB
  │  │  └ Prealloc (reserved) :         96 MB
  │  Free (unmapped)          :     97,504 MB
  │  GPU total used           :      4,798 MB
  │
  │  Virtual scale:
  │  [▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   ▓=Physical  ░=Free(unmapped)
  │
  │  Physical breakdown:
  │  [████████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒]
  │   █=Used(48 MB)  ▒=Prealloc(96 MB)
  └

  ── Phase 5 — Re-grow ──
         │ ── KV cache (MB) ──────────────────────────     │ GPU(MB)  │ Physical bar (█=Used ▒=Prealloc ░=Free)
      t  │    Used Prealloc Physical      Free   Virtual │     GPU │
  ───────┼────────────────────────────────────────────────┼────────────────────────────────
    0.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.1s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.2s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.3s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.4s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.5s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.6s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.7s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.8s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    0.9s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.1s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.2s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.3s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.4s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.5s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.6s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.7s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.8s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    1.9s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.0s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.1s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.2s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    2.3s │      48       96      144    97,504    97,648 │   4,798 │ ██████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

================================================================================
  VERIFICATION  [kvcached ENABLED]
================================================================================
  [PASS] Physical << Virtual at idle  (elastic, not pre-allocated)
         Physical 96 MB = 0.10% of Virtual 97,648 MB  [threshold: < 5%]
  [PASS] Large batch peak > small batch peak  (on-demand growth)
         small 96 MB, large 864 MB
  [PASS] GPU memory << Virtual KV limit  (no upfront allocation)
         GPU 4,514 MB vs Virtual 97,648 MB  [threshold: GPU < 10% of Virtual]
  [PASS] Used pages varied during inference  (alloc/free cycle)
         used: min=48 MB, max=816 MB
  [PASS] Physical shrank after completion  (hipMemUnmap confirmed)
         peak 864 MB → after 144 MB  (freed 720 MB)
  [PASS] Used pages re-appeared on new requests  (pages re-mapped)
         re-grow peak used 48 MB (physical 144 MB)
  [PASS] GPU memory correlates with Physical growth  (not just bookkeeping)
         GPU delta: +1,004 MB, Physical delta: +768 MB during large batch

  ALL CHECKS PASSED — Elastic memory (hipMemMap / hipMemUnmap) confirmed.
================================================================================

[10] SGLang baseline (no kvcached)...
---
================================================================================
  SGLang + kvcached Elastic Memory Benchmark  [BASELINE (vanilla SGLang)]
================================================================================
  kvcached is DISABLED — SGLang will pre-allocate all KV cache memory.
  GPU before model: 330 MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 — Load model (facebook/opt-125m)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/opt/venv/lib/python3.10/site-packages/apex/transformer/functional/fused_rope.py:49: UserWarning: Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0
  warnings.warn("Aiter backend is selected for fused RoPE. This has lower precision. To disable aiter, export USE_ROCM_AITER_ROPE_BACKEND=0", UserWarning)
Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]

[2026-03-15 05:34:19] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.
Capturing batches (bs=1 avail_mem=92.01 GB): 100%|██████████| 52/52 [00:39<00:00,  1.33it/s]
[2026-03-15 05:34:36] Current hipcc not support: -mllvm -amdgpu-coerce-illegal-types=1, skip it.

  ┌─ After model load (idle)
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :    102,076 MB  (52.0% of 196,288 MB)
  │
  │  GPU scale:
  │  [███████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 2 — Small batch  (4 prompts × 128 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After Small batch
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :    102,330 MB  (52.1% of 196,288 MB)
  │
  │  GPU scale:
  │  [███████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 2 — Small batch ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │    102,076 │ █████████████████████████████░
    3.4s │    102,200 │ █████████████████████████████░
    6.8s │    102,200 │ █████████████████████████████░
   10.2s │    102,200 │ █████████████████████████████░
   13.6s │    102,200 │ █████████████████████████████░
   17.0s │    102,200 │ █████████████████████████████░
   20.4s │    102,200 │ █████████████████████████████░
   23.8s │    102,200 │ █████████████████████████████░
   27.2s │    102,200 │ █████████████████████████████░
   30.7s │    102,200 │ █████████████████████████████░
   34.1s │    102,200 │ █████████████████████████████░
   37.5s │    102,200 │ █████████████████████████████░
   40.9s │    102,200 │ █████████████████████████████░
   44.3s │    102,200 │ █████████████████████████████░
   47.7s │    102,200 │ █████████████████████████████░
   51.1s │    102,200 │ █████████████████████████████░
   54.5s │    102,200 │ █████████████████████████████░
   57.9s │    102,200 │ █████████████████████████████░
   61.3s │    102,200 │ █████████████████████████████░
   64.7s │    102,214 │ █████████████████████████████░
   68.1s │    102,330 │ ██████████████████████████████
   68.9s │    102,330 │ ██████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 3 — Large batch  (64 prompts × 512 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After Large batch
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :    102,376 MB  (52.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [███████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 3 — Large batch ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │    102,330 │ █████████████████████████████░
    0.1s │    102,330 │ █████████████████████████████░
    0.2s │    102,330 │ █████████████████████████████░
    0.3s │    102,354 │ █████████████████████████████░
    0.4s │    102,354 │ █████████████████████████████░
    0.5s │    102,376 │ ██████████████████████████████
    0.6s │    102,376 │ ██████████████████████████████
    0.7s │    102,376 │ ██████████████████████████████
    0.8s │    102,376 │ ██████████████████████████████
    0.9s │    102,376 │ ██████████████████████████████
    1.0s │    102,376 │ ██████████████████████████████
    1.1s │    102,376 │ ██████████████████████████████
    1.2s │    102,376 │ ██████████████████████████████
    1.3s │    102,376 │ ██████████████████████████████
    1.4s │    102,376 │ ██████████████████████████████
    1.5s │    102,376 │ ██████████████████████████████
    1.6s │    102,376 │ ██████████████████████████████
    1.7s │    102,376 │ ██████████████████████████████
    1.8s │    102,376 │ ██████████████████████████████
    1.9s │    102,376 │ ██████████████████████████████
    2.0s │    102,376 │ ██████████████████████████████
    2.1s │    102,376 │ ██████████████████████████████
    2.2s │    102,376 │ ██████████████████████████████
    2.3s │    102,376 │ ██████████████████████████████
    2.4s │    102,376 │ ██████████████████████████████
    2.5s │    102,376 │ ██████████████████████████████
    2.6s │    102,376 │ ██████████████████████████████
    2.7s │    102,376 │ ██████████████████████████████
    2.8s │    102,376 │ ██████████████████████████████
    2.9s │    102,376 │ ██████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 4 — Cooldown  (4 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SGLang pre-allocated all KV memory — nothing changes.

  ┌─ After Cooldown
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :    102,376 MB  (52.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [███████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 4 — Cooldown ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │    102,376 │ ██████████████████████████████
    0.5s │    102,376 │ ██████████████████████████████
    1.0s │    102,376 │ ██████████████████████████████
    1.5s │    102,376 │ ██████████████████████████████
    2.0s │    102,376 │ ██████████████████████████████
    2.5s │    102,376 │ ██████████████████████████████
    3.0s │    102,376 │ ██████████████████████████████
    3.5s │    102,376 │ ██████████████████████████████
    4.0s │    102,376 │ ██████████████████████████████

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 5 — Re-grow  (16 prompts × 256 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ After Re-grow
  │
  │  KV cache breakdown      :        N/A  (kvcached disabled)
  │  GPU total used           :    102,376 MB  (52.2% of 196,288 MB)
  │
  │  GPU scale:
  │  [███████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  │   █=Used  ░=Free
  └

  ── Phase 5 — Re-grow ──
         │ GPU(MB)     │ GPU bar (█=Used ░=Free)
      t  │        GPU │
  ───────┼────────────┼────────────────────────────────
    0.0s │    102,376 │ ██████████████████████████████
    0.1s │    102,376 │ ██████████████████████████████
    0.2s │    102,376 │ ██████████████████████████████
    0.3s │    102,376 │ ██████████████████████████████
    0.4s │    102,376 │ ██████████████████████████████
    0.5s │    102,376 │ ██████████████████████████████
    0.6s │    102,376 │ ██████████████████████████████
    0.7s │    102,376 │ ██████████████████████████████
    0.8s │    102,376 │ ██████████████████████████████
    0.9s │    102,376 │ ██████████████████████████████
    1.0s │    102,376 │ ██████████████████████████████
    1.1s │    102,376 │ ██████████████████████████████
    1.2s │    102,376 │ ██████████████████████████████
    1.3s │    102,376 │ ██████████████████████████████
    1.4s │    102,376 │ ██████████████████████████████
    1.5s │    102,376 │ ██████████████████████████████
    1.6s │    102,376 │ ██████████████████████████████
    1.7s │    102,376 │ ██████████████████████████████
    1.8s │    102,376 │ ██████████████████████████████
    1.9s │    102,376 │ ██████████████████████████████
    2.0s │    102,376 │ ██████████████████████████████
    2.1s │    102,376 │ ██████████████████████████████
    2.2s │    102,376 │ ██████████████████████████████

================================================================================
  VERIFICATION  [BASELINE (vanilla SGLang)]
================================================================================
  [PASS] GPU memory is FLAT during inference  (pre-allocated, not elastic)
         GPU range: 102,330 – 102,376 MB  (delta 46 MB, threshold: < 589 MB [0.3% of 196,288 MB GPU])
  [PASS] GPU memory roughly stable after requests  (no elastic free/unmap)
         idle 102,076 MB → after 102,376 MB  (delta 300 MB, threshold: < 589 MB)
  [PASS] Large upfront KV allocation visible in GPU  (not elastic)
         GPU before model: 330 MB → after load: 102,076 MB  (KV estimate: 101,746 MB, threshold: > 589 MB)

  ALL CHECKS PASSED — Confirmed: vanilla SGLang is NOT elastic.
================================================================================

[11] Creating environment setup file...
---

[12] Summary
---

============================================
  Results
============================================

  vLLM + kvcached (elastic):     PASS
  vLLM baseline (no kvcached):   PASS
  SGLang + kvcached (elastic):   PASS
  SGLang baseline (no kvcached): PASS

  To re-run individual tests:

  # vLLM elastic (kvcached)
  docker run --rm --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined -v /root/kvcached:/kvcached \
    vllm/vllm-openai-rocm:latest \
    -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_elastic_memory_e2e.py'

  # vLLM baseline (no kvcached)
  docker run --rm --entrypoint bash \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined -v /root/kvcached:/kvcached \
    vllm/vllm-openai-rocm:latest \
    -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_elastic_memory_e2e.py --baseline'

  # SGLang elastic (kvcached)
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined -v /root/kvcached:/kvcached \
    lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312 \
    bash -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_sglang_e2e.py'

  # SGLang baseline (no kvcached)
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined -v /root/kvcached:/kvcached \
    lmsysorg/sglang-daily:v0.5.9-rocm720-mi30x-20260312 \
    bash -c 'cd /kvcached && pip install -e . --no-build-isolation && python tests/test_sglang_e2e.py --baseline'


  Next steps:

  1. Load environment (auto-loaded on next login):
     source ~/setup_env.sh

  2. Authenticate GitHub (for pushes):
     gh auth login

  3. Use Claude Code for development:
     cd /root/kvcached && claude

  All tests PASSED.


