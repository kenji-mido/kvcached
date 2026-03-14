# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""
SGLang + kvcached E2E integration test.

Usage (inside SGLang ROCm Docker container):
  pip install -e /kvcached --no-build-isolation
  python /kvcached/tests/test_sglang_e2e.py
"""

import os
import sys

os.environ["ENABLE_KVCACHED"] = "true"
os.environ["KVCACHED_AUTOPATCH"] = "1"

import kvcached.integration.sglang.autopatch  # noqa: E402

from sglang import Engine  # noqa: E402


def main():
    print("=" * 70)
    print("  SGLang + kvcached E2E Integration Test")
    print("=" * 70)

    print("\nLoading model facebook/opt-125m...")
    engine = Engine(
        model_path="facebook/opt-125m",
        mem_fraction_static=0.5,
        disable_radix_cache=True,
    )

    print("Running inference...")
    result = engine.generate(
        prompt="Hello, my name is",
        sampling_params={"max_new_tokens": 30, "temperature": 0},
    )
    text = result["text"]
    print(f"Output: {text}")
    assert len(text) > 0, "Empty output!"

    # Check kvcached memory via IPC
    from kvcached.cli.kvtop import _detect_kvcache_ipc_names
    from kvcached.cli.utils import MemInfoStruct, RwLockedShm, get_ipc_name

    names = _detect_kvcache_ipc_names()
    print(f"\nkvcached IPC segments: {names}")
    for name in names:
        try:
            with RwLockedShm(get_ipc_name(name), MemInfoStruct.SHM_SIZE,
                             RwLockedShm.RLOCK) as mm:
                info = MemInfoStruct.from_buffer(mm)
                virt = info.total_size // 1024 // 1024
                used = info.used_size // 1024 // 1024
                pre = info.prealloc_size // 1024 // 1024
                phys = used + pre
                free = virt - phys
                print(f"  ┌─ {name}")
                print(f"  │  Virtual:  {virt:>8,d} MB")
                print(f"  │  Physical: {phys:>8,d} MB  ({phys/virt*100:.2f}%)")
                print(f"  │    Used:   {used:>8,d} MB")
                print(f"  │    Prealloc:{pre:>7,d} MB")
                print(f"  │  Free:     {free:>8,d} MB")
                print(f"  └")
        except Exception as e:
            print(f"  {name}: error: {e}")

    engine.shutdown()

    print("\n" + "=" * 70)
    print("  SUCCESS: SGLang + kvcached E2E test passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
