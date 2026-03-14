# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

requires_rocm = pytest.mark.skipif(
    torch.version.hip is None,
    reason="Requires ROCm/HIP",
)


@requires_rocm
def test_init_kvcached_rocm():
    """kvcached initializes and reports tensors not yet created."""
    from kvcached.vmm_ops import init_kvcached, kv_tensors_created, shutdown_kvcached

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    # After init, no KV tensors should exist yet
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
    from kvcached.vmm_ops import (
        create_kv_tensors,
        init_kvcached,
        kv_tensors_created,
        shutdown_kvcached,
    )

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    assert not kv_tensors_created()

    # page_size default = 2MB, dtype_size=2 (float16)
    page_size = 2 * 1024 * 1024
    num_layers = 2
    num_kv_buffers = 2
    tensors = create_kv_tensors(
        size=page_size, dtype_size=2, dev_str="cuda:0",
        num_layers=num_layers, num_kv_buffers=num_kv_buffers,
    )
    assert kv_tensors_created(), "KV tensors should exist after create_kv_tensors"
    assert len(tensors) > 0, "create_kv_tensors should return at least one tensor"

    # Verify tensor is on GPU and has expected properties
    t = tensors[0]
    assert t.is_cuda, "Tensor should be on GPU"
    assert t.dtype == torch.int16, "Tensor dtype should match requested dtype_size=2 (int16)"
    assert t.numel() > 0, "Tensor should have elements"

    shutdown_kvcached()


@requires_rocm
def test_kv_tensor_map_unmap():
    """ROCm: map and unmap work correctly — mapped memory is writable."""
    from kvcached.vmm_ops import (
        create_kv_tensors,
        init_kvcached,
        map_to_kv_tensors,
        shutdown_kvcached,
        unmap_from_kv_tensors,
    )

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)

    page_size = 2 * 1024 * 1024
    num_layers = 2
    num_kv_buffers = 2
    tensors = create_kv_tensors(
        size=page_size, dtype_size=2, dev_str="cuda:0",
        num_layers=num_layers, num_kv_buffers=num_kv_buffers,
    )

    # Map page at offset 0
    result = map_to_kv_tensors([0])
    assert result is True, "map_to_kv_tensors should return True on success"

    # Write to mapped memory — this would segfault if mapping failed
    t = tensors[0]
    t[0] = 42.0
    torch.cuda.synchronize()
    val = t[0].item()
    assert val == 42.0, f"Written value should be readable: expected 42.0, got {val}"

    # Unmap
    result = unmap_from_kv_tensors([0])
    assert result is True, "unmap_from_kv_tensors should return True on success"

    shutdown_kvcached()
