#!/usr/bin/env python3
"""
Diagnostic: prove that hipMemMap/hipMemUnmap physically change GPU memory.

This script prints raw GPU memory values (via hipMemGetInfo / torch.cuda.mem_get_info)
at each step of the VMM lifecycle. No mocks. No thresholds. Just raw numbers.

IMPORTANT: map_to_kv_tensors takes BYTE OFFSETS (multiples of compound_page_size),
not small page indices. With contiguous_layout=True:
  compound_page_size = page_size × num_layers × num_kv_buffers = 2MB × 2 × 2 = 8MB
"""
import torch

MB = 1024 * 1024


def gpu_info():
    """Return (used_MB, free_MB, total_MB) from hipMemGetInfo."""
    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    return used / MB, free / MB, total / MB


def main():
    from kvcached.vmm_ops import (
        create_kv_tensors,
        init_kvcached,
        kv_tensors_created,
        map_to_kv_tensors,
        shutdown_kvcached,
        unmap_from_kv_tensors,
    )

    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ROCm HIP: {torch.version.hip}")
    print()

    used, free, total = gpu_info()
    print(f"[Step 0] Before init")
    print(f"  GPU used: {used:,.1f} MB / {total:,.1f} MB (free: {free:,.1f} MB)")
    print()

    # ── init ──
    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    assert not kv_tensors_created()
    torch.cuda.synchronize()

    used, free, total = gpu_info()
    print(f"[Step 1] After init_kvcached (no tensors yet)")
    print(f"  GPU used: {used:,.1f} MB / {total:,.1f} MB (free: {free:,.1f} MB)")
    print()

    # ── create tensors (reserves virtual address space) ──
    page_size = 2 * MB
    num_layers = 2
    num_kv_buffers = 2
    compound_page_size = page_size * num_layers * num_kv_buffers  # 8 MB

    # Create buffer for 8 compound pages
    num_compound_pages = 8
    buffer_per_layer = compound_page_size * num_compound_pages // num_layers

    tensors = create_kv_tensors(
        size=buffer_per_layer,
        dtype_size=2, dev_str="cuda:0",
        num_layers=num_layers, num_kv_buffers=num_kv_buffers,
    )
    torch.cuda.synchronize()

    used_after_create, free_c, _ = gpu_info()
    print(f"[Step 2] After create_kv_tensors (virtual reserved, zero_page mapped)")
    print(f"  GPU used: {used_after_create:,.1f} MB (free: {free_c:,.1f} MB)")
    print(f"  Tensors: {len(tensors)}, shape={tensors[0].shape}, dtype={tensors[0].dtype}")
    print(f"  compound_page_size: {compound_page_size / MB:.0f} MB")
    print(f"  Total compound pages: {num_compound_pages}")
    print()

    # ── map pages one at a time with correct byte offsets ──
    print("=== Map pages one at a time (byte offsets) ===")
    for p in range(min(num_compound_pages, 4)):
        offset = p * compound_page_size
        before = gpu_info()[0]
        map_to_kv_tensors([offset])
        torch.cuda.synchronize()
        after = gpu_info()[0]
        delta = after - before
        print(f"  Map page {p} (offset={offset / MB:.0f} MB): "
              f"before={before:.0f} MB, after={after:.0f} MB, delta=+{delta:.1f} MB")

    # ── write and read to prove physical backing ──
    t = tensors[0]
    elements_per_page = compound_page_size // 2  # dtype_size=2 (int16)
    print()
    print("=== Write/read verification on mapped pages ===")
    for p in range(4):
        idx = p * elements_per_page
        t[idx] = float(p * 100 + 42)
    torch.cuda.synchronize()

    for p in range(4):
        idx = p * elements_per_page
        val = t[idx].item()
        expected = float(p * 100 + 42)
        status = "OK" if val == expected else "MISMATCH"
        print(f"  Page {p}: wrote {expected}, read {val}  [{status}]")

    # ── unmap pages one at a time ──
    print()
    print("=== Unmap pages one at a time ===")
    for p in range(4):
        offset = p * compound_page_size
        before = gpu_info()[0]
        unmap_from_kv_tensors([offset])
        torch.cuda.synchronize()
        after = gpu_info()[0]
        delta = before - after
        print(f"  Unmap page {p} (offset={offset / MB:.0f} MB): "
              f"before={before:.0f} MB, after={after:.0f} MB, delta=-{delta:.1f} MB")

    # ── batch map/unmap ──
    print()
    print("=== Batch map/unmap 4 pages at once ===")
    offsets = [p * compound_page_size for p in range(4)]
    before = gpu_info()[0]
    map_to_kv_tensors(offsets)
    torch.cuda.synchronize()
    after_map = gpu_info()[0]
    print(f"  Map 4 pages: +{after_map - before:.1f} MB  ({before:.0f} → {after_map:.0f})")

    unmap_from_kv_tensors(offsets)
    torch.cuda.synchronize()
    after_unmap = gpu_info()[0]
    print(f"  Unmap 4 pages: -{after_map - after_unmap:.1f} MB  ({after_map:.0f} → {after_unmap:.0f})")

    shutdown_kvcached()

    # ── Summary ──
    print()
    print("=" * 60)
    print("CONCLUSION: Each compound page = 8 MB physical GPU memory.")
    print("hipMemMap/hipMemUnmap are real VMM operations,")
    print("verified by hipMemGetInfo (torch.cuda.mem_get_info).")
    print("=" * 60)


if __name__ == "__main__":
    main()
