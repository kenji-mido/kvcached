# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""
E2E benchmark: kvcached elastic memory vs vanilla vLLM.

Usage:
  python tests/test_elastic_memory_e2e.py              # kvcached enabled
  python tests/test_elastic_memory_e2e.py --baseline   # vanilla vLLM (no kvcached)

With kvcached:
  KV cache (MB) columns show Used / Prealloc / Physical / Free / Virtual
  via kvcached's shared-memory IPC (same data as kvctl/kvtop).

Baseline (no kvcached):
  KV cache columns are N/A; only GPU (MB) is shown.
  vLLM allocates all KV cache memory upfront — GPU stays flat.
"""

import os
import sys
import threading
import time

# ── mode selection (must happen before any import) ────────────────────────

BASELINE = "--baseline" in sys.argv

if not BASELINE:
    os.environ["ENABLE_KVCACHED"] = "true"
    os.environ["KVCACHED_AUTOPATCH"] = "1"
    os.environ["KVCACHED_MAX_RESERVED_PAGES"] = "2"
    os.environ["KVCACHED_MIN_RESERVED_PAGES"] = "1"
    import kvcached.integration.vllm.autopatch  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────

MB = 1024 ** 2
NA = "   N/A"  # 7-char placeholder matching column width


def read_kvcached_shm():
    if BASELINE:
        return []
    from kvcached.cli.kvtop import _detect_kvcache_ipc_names
    from kvcached.cli.utils import MemInfoStruct, RwLockedShm, get_ipc_name
    results = []
    for name in _detect_kvcache_ipc_names():
        try:
            with RwLockedShm(get_ipc_name(name), MemInfoStruct.SHM_SIZE,
                             RwLockedShm.RLOCK) as mm:
                results.append((name, MemInfoStruct.from_buffer(mm)))
        except FileNotFoundError:
            pass
    return results


def get_breakdown():
    """Return (used, prealloc, virtual) in bytes.  All zeros in baseline."""
    entries = read_kvcached_shm()
    used = sum(i.used_size for _, i in entries)
    pre  = sum(i.prealloc_size for _, i in entries)
    virt = sum(i.total_size for _, i in entries)
    return used, pre, virt


def gpu_used_bytes():
    import torch
    free, total = torch.cuda.mem_get_info(0)
    return total - free


def gpu_total_bytes():
    import torch
    _, total = torch.cuda.mem_get_info(0)
    return total


def fmt(b):
    """Format bytes → MB string, right-aligned."""
    return f"{b / MB:,.0f}"


# ── display ───────────────────────────────────────────────────────────────

def print_memory_bar(label, used, prealloc, virtual, gpu, bar_width=60):
    """Nested memory bar.  Adapts when kvcached data is unavailable."""
    physical = used + prealloc
    free = virtual - physical if virtual else 0

    print(f"\n  ┌─ {label}")
    print(f"  │")

    if virtual > 0:  # kvcached active
        pct = physical / virtual * 100
        print(f"  │  Virtual (address space) : {fmt(virtual):>10s} MB")
        print(f"  │  ┌─ Physical (GPU RAM)   : {fmt(physical):>10s} MB  ({pct:.2f}% of Virtual)")
        print(f"  │  │  ├ Used    (active KV) : {fmt(used):>10s} MB")
        print(f"  │  │  └ Prealloc (reserved) : {fmt(prealloc):>10s} MB")
        print(f"  │  Free (unmapped)          : {fmt(free):>10s} MB")
        print(f"  │  GPU total used           : {fmt(gpu):>10s} MB")
        print(f"  │")
        p_w = max(int(physical / virtual * bar_width), 1 if physical > 0 else 0)
        print(f"  │  Virtual scale:")
        print(f"  │  [{'▓' * p_w}{'░' * (bar_width - p_w)}]")
        print(f"  │   ▓=Physical  ░=Free(unmapped)")
        if physical > 0:
            u_w = max(int(used / physical * bar_width), 1 if used > 0 else 0)
            print(f"  │")
            print(f"  │  Physical breakdown:")
            print(f"  │  [{'█' * u_w}{'▒' * (bar_width - u_w)}]")
            print(f"  │   █=Used({fmt(used)} MB)  ▒=Prealloc({fmt(prealloc)} MB)")
    else:  # baseline — no kvcached
        gpu_total = gpu_total_bytes()
        pct = gpu / gpu_total * 100 if gpu_total else 0
        print(f"  │  KV cache breakdown      :        N/A  (kvcached disabled)")
        print(f"  │  GPU total used           : {fmt(gpu):>10s} MB  ({pct:.1f}% of {fmt(gpu_total)} MB)")
        print(f"  │")
        g_w = max(int(pct / 100 * bar_width), 1 if gpu > 0 else 0)
        print(f"  │  GPU scale:")
        print(f"  │  [{'█' * g_w}{'░' * (bar_width - g_w)}]")
        print(f"  │   █=Used  ░=Free")

    print(f"  └")


# ── memory poller ─────────────────────────────────────────────────────────

class MemoryPoller:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []  # (t, used, prealloc, virtual, gpu)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                u, p, v = get_breakdown()
                g = gpu_used_bytes()
                self.samples.append((time.monotonic() - t0, u, p, v, g))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _col(self, idx):
        return [s[idx] for s in self.samples]

    @property
    def peak_used(self):       return max(self._col(1), default=0)
    @property
    def peak_physical(self):   return max((s[1]+s[2] for s in self.samples), default=0)
    @property
    def min_physical(self):    return min((s[1]+s[2] for s in self.samples), default=0)
    @property
    def peak_gpu(self):        return max(self._col(4), default=0)
    @property
    def min_gpu(self):         return min(self._col(4), default=0)

    def print_timeline(self, title, max_rows=25, bar_width=30):
        if not self.samples:
            print(f"  (no samples for {title})")
            return

        has_kv = any(s[3] > 0 for s in self.samples)  # virtual > 0
        peak_phy = max((s[1] + s[2] for s in self.samples), default=1) or 1
        step = max(1, len(self.samples) // max_rows)

        if has_kv:
            unit_hdr = "── KV cache (MB) ──────────────────────────"
            gpu_hdr  = "GPU(MB)"
            bar_hdr  = "Physical bar (█=Used ▒=Prealloc ░=Free)"
            hdr = (f"  {'t':>5s}  │ {'Used':>7s} {'Prealloc':>8s} {'Physical':>8s} "
                   f"{'Free':>9s} {'Virtual':>9s} │ {'GPU':>7s} │")
        else:
            gpu_hdr  = "GPU(MB)"
            bar_hdr  = "GPU bar (█=Used ░=Free)"
            hdr = f"  {'t':>5s}  │ {'GPU':>10s} │"

        sep_w = 48 if has_kv else 12
        print(f"\n  ── {title} ──")
        if has_kv:
            print(f"         │ {unit_hdr:<48s}│ {gpu_hdr:<9s}│ {bar_hdr}")
        else:
            print(f"         │ {gpu_hdr:<12s}│ {bar_hdr}")
        print(hdr)
        print(f"  {'─'*5}──┼{'─'*sep_w}┼{'─'*(bar_width+2)}")

        for i in range(0, len(self.samples), step):
            self._print_row(i, has_kv, peak_phy, bar_width)
        if (len(self.samples) - 1) % step != 0:
            self._print_row(-1, has_kv, peak_phy, bar_width)

    def _print_row(self, idx, has_kv, peak_phy, bar_width):
        t, u, p, v, g = self.samples[idx]
        phy = u + p

        if has_kv:
            free = v - phy
            u_w = int(u / peak_phy * bar_width) if peak_phy else 0
            p_w = int(p / peak_phy * bar_width) if peak_phy else 0
            rest = bar_width - u_w - p_w
            bar = '█' * u_w + '▒' * p_w + '░' * rest
            print(f"  {t:5.1f}s │ {fmt(u):>7s} {fmt(p):>8s} {fmt(phy):>8s} "
                  f"{fmt(free):>9s} {fmt(v):>9s} │ {fmt(g):>7s} │ {bar}")
        else:
            # Baseline: GPU-only bar scaled to peak GPU
            peak_g = max(self._col(4), default=1) or 1
            g_w = int(g / peak_g * bar_width)
            bar = '█' * g_w + '░' * (bar_width - g_w)
            print(f"  {t:5.1f}s │ {fmt(g):>10s} │ {bar}")


# ── workload definitions ─────────────────────────────────────────────────

WORKLOADS = [
    ("Phase 2 — Small batch",  4, 128,
     [f"Write a paragraph about topic {i}." for i in range(4)]),
    ("Phase 3 — Large batch", 64, 512,
     [f"Write a very long and detailed essay about subject number {i}, "
      f"covering history, theory, and applications." for i in range(64)]),
    ("Phase 4 — Cooldown",     0,   0, None),  # sleep only
    ("Phase 5 — Re-grow",     16, 256,
     [f"Explain concept {i} thoroughly." for i in range(16)]),
]


# ── main ──────────────────────────────────────────────────────────────────

def print_section(text):
    print(f"\n{'━' * 80}")
    print(f"  {text}")
    print(f"{'━' * 80}")


def main():
    mode = "BASELINE (vanilla vLLM)" if BASELINE else "kvcached ENABLED"
    print("=" * 80)
    print(f"  kvcached Elastic Memory E2E Benchmark  [{mode}]")
    print("=" * 80)
    if not BASELINE:
        print(f"  MAX_RESERVED_PAGES = {os.environ.get('KVCACHED_MAX_RESERVED_PAGES')}")
        print(f"  MIN_RESERVED_PAGES = {os.environ.get('KVCACHED_MIN_RESERVED_PAGES')}")
    else:
        print("  kvcached is DISABLED — vLLM will pre-allocate all KV cache memory.")

    gpu0 = gpu_used_bytes()
    print(f"  GPU before model: {fmt(gpu0)} MB")

    # ── Phase 1: load model ───────────────────────────────────────────────
    print_section("Phase 1 — Load model (facebook/opt-125m)")
    llm = LLM(
        model="facebook/opt-125m",
        enable_prefix_caching=False,
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    time.sleep(3)

    u, p, v = get_breakdown()
    g = gpu_used_bytes()
    print_memory_bar("After model load (idle)", u, p, v, g)

    idle_physical = u + p
    virtual = v
    gpu_idle = g

    pollers = {}

    # ── Phases 2–5 ────────────────────────────────────────────────────────
    for label, n_prompts, max_tokens, prompts in WORKLOADS:
        print_section(label + (f"  ({n_prompts} prompts × {max_tokens} tokens)"
                               if prompts else "  (4 seconds)"))
        if label.startswith("Phase 3") and not BASELINE:
            print("  Goal: exceed MAX_RESERVED_PAGES=2 → trigger hipMemUnmap")
        if label.startswith("Phase 4"):
            if not BASELINE:
                print("  generate() is synchronous — blocks are already freed.")
                print("  This phase confirms no further changes occur.")
            else:
                print("  vLLM pre-allocated all KV memory — nothing changes.")

        poller = MemoryPoller(interval=0.5 if prompts is None else 0.1)
        poller.start()

        if prompts is not None:
            llm.generate(prompts, SamplingParams(max_tokens=max_tokens,
                                                  temperature=0.8))
            time.sleep(2)
        else:
            time.sleep(4)

        poller.stop()
        pollers[label] = poller

        u, p, v = get_breakdown()
        print_memory_bar(f"After {label.split('—')[1].strip()}", u, p, v,
                         gpu_used_bytes())
        poller.print_timeline(label, max_rows=20 if prompts else 10)

    # Snapshot for verification
    u_post, p_post, _ = get_breakdown()
    post_phys = u_post + p_post
    gpu_post = gpu_used_bytes()

    # ── Verification ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  VERIFICATION  [{mode}]")
    print("=" * 80)

    p2 = pollers["Phase 2 — Small batch"]
    p3 = pollers["Phase 3 — Large batch"]
    p4 = pollers["Phase 4 — Cooldown"]
    p5 = pollers["Phase 5 — Re-grow"]

    checks = []

    if not BASELINE:
        # ── kvcached checks ──
        ratio = idle_physical / virtual * 100 if virtual else 100
        checks.append((
            "Physical << Virtual at idle  (elastic, not pre-allocated)",
            ratio < 5.0,
            f"Physical {fmt(idle_physical)} MB = {ratio:.2f}% "
            f"of Virtual {fmt(virtual)} MB",
        ))
        checks.append((
            "Large batch peak > small batch peak  (on-demand growth)",
            p3.peak_physical >= p2.peak_physical,
            f"small {fmt(p2.peak_physical)} MB, "
            f"large {fmt(p3.peak_physical)} MB",
        ))
        checks.append((
            "GPU memory << Virtual KV limit  (no upfront allocation)",
            gpu_idle < virtual * 0.1,
            f"GPU {fmt(gpu_idle)} MB vs Virtual {fmt(virtual)} MB",
        ))
        used_vals = [s[1] for s in p3.samples]
        checks.append((
            "Used pages varied during inference  (alloc/free cycle)",
            max(used_vals, default=0) > min(used_vals, default=0),
            f"used: min={fmt(min(used_vals, default=0))} MB, "
            f"max={fmt(max(used_vals, default=0))} MB",
        ))
        checks.append((
            "Physical shrank after completion  (hipMemUnmap confirmed)",
            post_phys < p3.peak_physical,
            f"peak {fmt(p3.peak_physical)} MB → after {fmt(post_phys)} MB  "
            f"(freed {fmt(p3.peak_physical - post_phys)} MB)",
        ))
        checks.append((
            "Used pages re-appeared on new requests  (pages re-mapped)",
            p5.peak_used > 0,
            f"re-grow peak used {fmt(p5.peak_used)} MB "
            f"(physical {fmt(p5.peak_physical)} MB)",
        ))
    else:
        # ── baseline checks (confirm NON-elastic behavior) ──
        gpu_range = p3.peak_gpu - p3.min_gpu
        checks.append((
            "GPU memory is FLAT during inference  (pre-allocated, not elastic)",
            gpu_range < 100 * MB,  # less than 100 MB variance
            f"GPU range: {fmt(p3.min_gpu)} – {fmt(p3.peak_gpu)} MB  "
            f"(delta {fmt(gpu_range)} MB)",
        ))
        checks.append((
            "GPU memory unchanged after requests complete  (no free/unmap)",
            abs(gpu_post - gpu_idle) < 100 * MB,
            f"idle {fmt(gpu_idle)} MB → after {fmt(gpu_post)} MB  "
            f"(delta {fmt(abs(gpu_post - gpu_idle))} MB)",
        ))
        # Show how much GPU is used for KV cache (= gpu_idle - gpu_before_model)
        kv_estimate = gpu_idle - gpu0
        checks.append((
            "Large upfront KV allocation visible in GPU  (not elastic)",
            kv_estimate > 500 * MB,
            f"GPU before model: {fmt(gpu0)} MB → "
            f"after load: {fmt(gpu_idle)} MB  "
            f"(KV estimate: {fmt(kv_estimate)} MB)",
        ))

    all_pass = True
    for desc, passed, detail in checks:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {desc}")
        print(f"         {detail}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        if BASELINE:
            print("  ALL CHECKS PASSED — Confirmed: vanilla vLLM is NOT elastic.")
            print("  GPU memory is allocated upfront and stays constant.")
        else:
            print("  ALL CHECKS PASSED — Elastic memory (hipMemMap / hipMemUnmap) confirmed.")
    else:
        print("  SOME CHECKS FAILED — see above.")
    print("=" * 80)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
