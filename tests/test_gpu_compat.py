# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from unittest.mock import patch

import torch


def test_cuda_build_compiles():
    """CUDA path compiles (existing environment)."""
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Build failed:\n{result.stderr}"


def test_rocm_detection_logic_hip_present():
    """setup.py detects ROCm when torch.version.hip is set."""
    with patch.object(torch.version, "hip", "7.1.0"):
        assert torch.version.hip is not None


def test_rocm_detection_logic_hip_absent():
    """setup.py detects CUDA when torch.version.hip is None."""
    with patch.object(torch.version, "hip", None):
        assert torch.version.hip is None


def test_page_size_validation_cuda():
    """CUDA path requires 2MB-aligned page sizes."""
    with patch.object(torch.version, "hip", None):
        import os

        from kvcached.utils import _get_page_size

        # Valid: 2MB
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "2"}):
            assert _get_page_size() == 2 * 1024 * 1024

        # Valid: 4MB
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "4"}):
            assert _get_page_size() == 4 * 1024 * 1024


def test_page_size_validation_rocm():
    """ROCm path allows 1MB-aligned page sizes."""
    with patch.object(torch.version, "hip", "7.1.0"):
        import os

        from kvcached.utils import _get_page_size

        # Valid: 2MB
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "2"}):
            assert _get_page_size() == 2 * 1024 * 1024

        # Valid: 1MB (would fail on CUDA but OK on ROCm)
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "1"}):
            assert _get_page_size() == 1 * 1024 * 1024
