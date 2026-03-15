# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: test requires NVIDIA CUDA GPU")
    config.addinivalue_line("markers", "requires_rocm: test requires AMD ROCm GPU")
    config.addinivalue_line("markers", "requires_gpu: test requires any GPU")


@pytest.fixture(scope="session")
def gpu_backend():
    if torch.version.hip is not None:
        return "rocm"
    elif torch.cuda.is_available():
        return "cuda"
    return None


def gpu_used_bytes(device=0):
    """Return GPU memory currently used (total - free) via mem_get_info."""
    free, total = torch.cuda.mem_get_info(device)
    return total - free


def gpu_total_bytes(device=0):
    """Return total GPU memory via mem_get_info."""
    _, total = torch.cuda.mem_get_info(device)
    return total
