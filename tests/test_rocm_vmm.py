# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import gpu_used_bytes

requires_rocm = pytest.mark.skipif(
    torch.version.hip is None,
    reason="Requires ROCm/HIP",
)

MB = 1024 * 1024
PAGE_SIZE = 2 * MB
NUM_LAYERS = 2
NUM_KV_BUFFERS = 2
COMPOUND_PAGE_SIZE = PAGE_SIZE * NUM_LAYERS * NUM_KV_BUFFERS  # 8 MB
DTYPE_SIZE = 2  # int16


def _create_buffer(num_pages):
    """Init kvcached and create KV tensors sized for num_pages compound pages."""
    from kvcached.vmm_ops import create_kv_tensors, init_kvcached

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    buffer_per_layer = COMPOUND_PAGE_SIZE * num_pages // NUM_LAYERS
    tensors = create_kv_tensors(
        size=buffer_per_layer, dtype_size=DTYPE_SIZE, dev_str="cuda:0",
        num_layers=NUM_LAYERS, num_kv_buffers=NUM_KV_BUFFERS,
    )
    return tensors


@requires_rocm
def test_init_kvcached_rocm():
    """kvcached initializes and reports tensors not yet created."""
    from kvcached.vmm_ops import init_kvcached, kv_tensors_created, shutdown_kvcached

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    assert not kv_tensors_created(), "KV tensors should not exist right after init"
    shutdown_kvcached()


@requires_rocm
def test_granularity_query():
    """ROCm granularity is queried and init succeeds with default 2MB page size."""
    from kvcached.vmm_ops import init_kvcached, kv_tensors_created, shutdown_kvcached

    # Default page_size=0 means 2MB. If granularity query fails or
    # page size is not a multiple of granularity, init_gpu_() will abort.
    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    assert not kv_tensors_created()
    shutdown_kvcached()


@requires_rocm
def test_kv_tensor_create():
    """ROCm: create_kv_tensors returns valid GPU tensors."""
    from kvcached.vmm_ops import kv_tensors_created, shutdown_kvcached

    tensors = _create_buffer(num_pages=1)
    assert kv_tensors_created(), "KV tensors should exist after create_kv_tensors"
    assert len(tensors) > 0, "create_kv_tensors should return at least one tensor"

    t = tensors[0]
    assert t.is_cuda, "Tensor should be on GPU"
    assert t.dtype == torch.int16, "Tensor dtype should match requested dtype_size=2 (int16)"
    assert t.numel() > 0, "Tensor should have elements"

    shutdown_kvcached()


@requires_rocm
def test_kv_tensor_map_unmap():
    """ROCm: map and unmap work correctly — mapped memory is writable.

    Uses byte offset 0 (the first compound page). With contiguous_layout,
    this maps COMPOUND_PAGE_SIZE bytes of physical GPU memory.
    """
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    tensors = _create_buffer(num_pages=1)

    result = map_to_kv_tensors([0])
    assert result is True, "map_to_kv_tensors should return True on success"

    # Write to mapped memory — this would segfault if mapping failed
    t = tensors[0]
    t[0] = 42.0
    torch.cuda.synchronize()
    val = t[0].item()
    assert val == 42.0, f"Written value should be readable: expected 42.0, got {val}"

    result = unmap_from_kv_tensors([0])
    assert result is True, "unmap_from_kv_tensors should return True on success"

    shutdown_kvcached()


@requires_rocm
def test_map_increases_gpu_memory():
    """hipMemMap physically backs virtual pages — GPU memory must increase.

    IMPORTANT: map_to_kv_tensors takes BYTE OFFSETS (multiples of
    COMPOUND_PAGE_SIZE), not small page indices.
    """
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    num_pages = 4
    _create_buffer(num_pages)

    torch.cuda.synchronize()
    gpu_before = gpu_used_bytes()

    offsets = [p * COMPOUND_PAGE_SIZE for p in range(num_pages)]
    assert map_to_kv_tensors(offsets) is True

    torch.cuda.synchronize()
    gpu_after = gpu_used_bytes()

    expected = num_pages * COMPOUND_PAGE_SIZE
    map_delta = gpu_after - gpu_before
    assert map_delta >= expected, (
        f"hipMemMap should consume physical GPU memory: "
        f"before={gpu_before / MB:.1f} MB, after={gpu_after / MB:.1f} MB, "
        f"delta={map_delta / MB:.1f} MB, expected >= {expected / MB:.1f} MB"
    )

    # Unmap and verify memory is returned
    assert unmap_from_kv_tensors(offsets) is True

    torch.cuda.synchronize()
    gpu_after_unmap = gpu_used_bytes()

    unmap_delta = gpu_after - gpu_after_unmap
    assert unmap_delta >= expected, (
        f"hipMemUnmap should free physical GPU memory: "
        f"after_map={gpu_after / MB:.1f} MB, after_unmap={gpu_after_unmap / MB:.1f} MB, "
        f"delta={unmap_delta / MB:.1f} MB, expected >= {expected / MB:.1f} MB"
    )

    shutdown_kvcached()


@requires_rocm
def test_map_per_page_delta_is_linear():
    """Each compound page adds exactly COMPOUND_PAGE_SIZE bytes to GPU.

    Map pages one at a time and verify that hipMemGetInfo reports a
    consistent per-page delta — the fundamental elastic property.
    """
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    num_pages = 4
    _create_buffer(num_pages)

    torch.cuda.synchronize()

    # Map one page at a time, record GPU memory at each step.
    gpu_readings = [gpu_used_bytes()]
    for p in range(num_pages):
        map_to_kv_tensors([p * COMPOUND_PAGE_SIZE])
        torch.cuda.synchronize()
        gpu_readings.append(gpu_used_bytes())

    # Verify each step increased by COMPOUND_PAGE_SIZE (8 MB).
    for i in range(num_pages):
        delta = gpu_readings[i + 1] - gpu_readings[i]
        assert delta == COMPOUND_PAGE_SIZE, (
            f"Page {i}: GPU delta={delta / MB:.1f} MB, "
            f"expected {COMPOUND_PAGE_SIZE / MB:.1f} MB. "
            f"GPU readings: {[r / MB for r in gpu_readings]}"
        )

    # Unmap one page at a time, verify symmetric decrease.
    for p in range(num_pages):
        before = gpu_used_bytes()
        unmap_from_kv_tensors([p * COMPOUND_PAGE_SIZE])
        torch.cuda.synchronize()
        after = gpu_used_bytes()
        delta = before - after
        assert delta == COMPOUND_PAGE_SIZE, (
            f"Unmap page {p}: GPU delta=-{delta / MB:.1f} MB, "
            f"expected -{COMPOUND_PAGE_SIZE / MB:.1f} MB"
        )

    shutdown_kvcached()


@requires_rocm
def test_map_unmap_cycle_returns_memory():
    """Map->write->read->unmap->remap cycle: GPU memory returns to baseline.

    Verifies the full elastic lifecycle:
    1. Map pages -> GPU memory increases by COMPOUND_PAGE_SIZE per page
    2. Write and read data on mapped pages -> data integrity
    3. Unmap pages -> GPU memory returns to baseline
    4. Remap the same pages -> GPU memory increases again
    """
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    num_pages = 2
    tensors = _create_buffer(num_pages)

    torch.cuda.synchronize()
    gpu_baseline = gpu_used_bytes()

    # ── Cycle 1: map, write, verify, unmap ──
    offsets = [p * COMPOUND_PAGE_SIZE for p in range(num_pages)]
    assert map_to_kv_tensors(offsets) is True
    torch.cuda.synchronize()

    gpu_mapped = gpu_used_bytes()
    expected_delta = num_pages * COMPOUND_PAGE_SIZE
    assert gpu_mapped - gpu_baseline == expected_delta, (
        f"Map {num_pages} pages: delta={(gpu_mapped - gpu_baseline) / MB:.1f} MB, "
        f"expected {expected_delta / MB:.1f} MB"
    )

    # Write unique values per compound page and read back
    t = tensors[0]
    elements_per_page = COMPOUND_PAGE_SIZE // DTYPE_SIZE
    for p in range(num_pages):
        t[p * elements_per_page] = float(p * 1000 + 42)
    torch.cuda.synchronize()

    for p in range(num_pages):
        val = t[p * elements_per_page].item()
        expected_val = float(p * 1000 + 42)
        assert val == expected_val, (
            f"Page {p} write/read failed: expected {expected_val}, got {val}"
        )

    # Unmap — GPU memory must return to baseline exactly
    assert unmap_from_kv_tensors(offsets) is True
    torch.cuda.synchronize()
    gpu_unmapped = gpu_used_bytes()

    assert gpu_unmapped == gpu_baseline, (
        f"After unmap, GPU memory should return to baseline: "
        f"baseline={gpu_baseline / MB:.1f} MB, "
        f"after_unmap={gpu_unmapped / MB:.1f} MB"
    )

    # ── Cycle 2: remap same pages ──
    assert map_to_kv_tensors(offsets) is True
    torch.cuda.synchronize()
    gpu_remapped = gpu_used_bytes()

    assert gpu_remapped - gpu_unmapped == expected_delta, (
        f"Remap {num_pages} pages: delta={(gpu_remapped - gpu_unmapped) / MB:.1f} MB, "
        f"expected {expected_delta / MB:.1f} MB"
    )

    assert unmap_from_kv_tensors(offsets) is True
    shutdown_kvcached()
