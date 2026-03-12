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
    """kvcached initializes on ROCm without crash."""
    from kvcached.vmm_ops import init_kvcached, shutdown_kvcached

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    shutdown_kvcached()


@requires_rocm
def test_granularity_query():
    """ROCm granularity is queried successfully."""
    from kvcached.vmm_ops import init_kvcached, shutdown_kvcached

    # Should not crash even with default 2MB page size
    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    shutdown_kvcached()


@requires_rocm
def test_kv_tensor_alloc_free():
    """ROCm: allocate and free KV tensors."""
    from kvcached.vmm_ops import (
        create_kv_tensors,
        init_kvcached,
        map_to_kv_tensors,
        shutdown_kvcached,
        unmap_from_kv_tensors,
    )

    init_kvcached(dev_str="cuda:0", page_size=0, contiguous_layout=True)
    create_kv_tensors(
        size=1024, dtype_size=2, dev_str="cuda:0", num_layers=2, num_kv_buffers=2
    )
    # map and unmap
    map_to_kv_tensors([0, 1])
    unmap_from_kv_tensors([0, 1])
    shutdown_kvcached()
