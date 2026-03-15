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
    """ROCm: map and unmap work correctly — mapped memory is writable."""
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    tensors = _create_buffer(num_pages=1)

    result = map_to_kv_tensors([0])
    assert result is True, "map_to_kv_tensors should return True on success"

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
    """hipMemMap physically backs virtual pages — GPU memory must increase."""
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
        f"delta={map_delta / MB:.1f} MB, expected >= {expected / MB:.1f} MB"
    )

    assert unmap_from_kv_tensors(offsets) is True
    torch.cuda.synchronize()
    gpu_after_unmap = gpu_used_bytes()

    unmap_delta = gpu_after - gpu_after_unmap
    assert unmap_delta >= expected, (
        f"hipMemUnmap should free physical GPU memory: "
        f"delta={unmap_delta / MB:.1f} MB, expected >= {expected / MB:.1f} MB"
    )

    shutdown_kvcached()


@requires_rocm
def test_map_per_page_delta_is_linear():
    """Each compound page adds exactly COMPOUND_PAGE_SIZE bytes to GPU."""
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    num_pages = 4
    _create_buffer(num_pages)

    torch.cuda.synchronize()
    gpu_readings = [gpu_used_bytes()]
    for p in range(num_pages):
        map_to_kv_tensors([p * COMPOUND_PAGE_SIZE])
        torch.cuda.synchronize()
        gpu_readings.append(gpu_used_bytes())

    for i in range(num_pages):
        delta = gpu_readings[i + 1] - gpu_readings[i]
        assert delta == COMPOUND_PAGE_SIZE, (
            f"Page {i}: GPU delta={delta / MB:.1f} MB, "
            f"expected {COMPOUND_PAGE_SIZE / MB:.1f} MB"
        )

    for p in range(num_pages):
        before = gpu_used_bytes()
        unmap_from_kv_tensors([p * COMPOUND_PAGE_SIZE])
        torch.cuda.synchronize()
        after = gpu_used_bytes()
        assert before - after == COMPOUND_PAGE_SIZE, (
            f"Unmap page {p}: delta=-{(before - after) / MB:.1f} MB, "
            f"expected -{COMPOUND_PAGE_SIZE / MB:.1f} MB"
        )

    shutdown_kvcached()


@requires_rocm
def test_map_unmap_cycle_returns_memory():
    """Map->write->read->unmap->remap cycle: GPU memory returns to baseline."""
    from kvcached.vmm_ops import map_to_kv_tensors, shutdown_kvcached, unmap_from_kv_tensors

    num_pages = 2
    tensors = _create_buffer(num_pages)

    torch.cuda.synchronize()
    gpu_baseline = gpu_used_bytes()

    offsets = [p * COMPOUND_PAGE_SIZE for p in range(num_pages)]
    assert map_to_kv_tensors(offsets) is True
    torch.cuda.synchronize()

    expected_delta = num_pages * COMPOUND_PAGE_SIZE
    gpu_mapped = gpu_used_bytes()
    assert gpu_mapped - gpu_baseline == expected_delta

    # Write unique values and read back
    t = tensors[0]
    elements_per_page = COMPOUND_PAGE_SIZE // DTYPE_SIZE
    for p in range(num_pages):
        t[p * elements_per_page] = float(p * 1000 + 42)
    torch.cuda.synchronize()

    for p in range(num_pages):
        val = t[p * elements_per_page].item()
        assert val == float(p * 1000 + 42), f"Page {p} write/read failed"

    # Unmap — memory must return to baseline
    assert unmap_from_kv_tensors(offsets) is True
    torch.cuda.synchronize()
    assert gpu_used_bytes() == gpu_baseline, "GPU memory should return to baseline after unmap"

    # Remap — memory must increase again
    assert map_to_kv_tensors(offsets) is True
    torch.cuda.synchronize()
    assert gpu_used_bytes() - gpu_baseline == expected_delta, "Remap should consume same memory"

    assert unmap_from_kv_tensors(offsets) is True
    shutdown_kvcached()
