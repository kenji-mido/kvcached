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
