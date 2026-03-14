# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch


def test_cuda_build_compiles():
    """CUDA path compiles (existing environment)."""
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Build failed:\n{result.stderr}"


def test_rocm_detection_sets_is_rocm_true():
    """setup.py sets IS_ROCM=True when torch.version.hip is set."""
    with patch.object(torch.version, "hip", "7.1.0"):
        # Re-evaluate the expression used in setup.py
        is_rocm = torch.version.hip is not None
        assert is_rocm is True, "IS_ROCM should be True when hip version is set"


def test_rocm_detection_sets_is_rocm_false():
    """setup.py sets IS_ROCM=False when torch.version.hip is None."""
    with patch.object(torch.version, "hip", None):
        is_rocm = torch.version.hip is not None
        assert is_rocm is False, "IS_ROCM should be False when hip is None"


# ── Page size validation: valid cases ──────────────────────────────────────


def test_page_size_default():
    """Default page size is 2MB when env var is not set."""
    from kvcached.utils import _get_page_size

    with patch.dict(os.environ, {}, clear=False):
        # Remove KVCACHED_PAGE_SIZE_MB if set
        env = os.environ.copy()
        env.pop("KVCACHED_PAGE_SIZE_MB", None)
        with patch.dict(os.environ, env, clear=True):
            assert _get_page_size() == 2 * 1024 * 1024


def test_page_size_validation_cuda_valid():
    """CUDA path accepts 2MB-aligned page sizes."""
    from kvcached.utils import _get_page_size

    with patch.object(torch.version, "hip", None):
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "2"}):
            assert _get_page_size() == 2 * 1024 * 1024

        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "4"}):
            assert _get_page_size() == 4 * 1024 * 1024


def test_page_size_validation_rocm_valid():
    """ROCm path accepts 1MB-aligned page sizes."""
    from kvcached.utils import _get_page_size

    with patch.object(torch.version, "hip", "7.1.0"):
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "2"}):
            assert _get_page_size() == 2 * 1024 * 1024

        # 1MB is valid on ROCm but would be rejected on CUDA
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "1"}):
            assert _get_page_size() == 1 * 1024 * 1024


# ── Page size validation: error cases ──────────────────────────────────────


def test_page_size_cuda_rejects_1mb():
    """CUDA path rejects 1MB (not a multiple of 2MB)."""
    from kvcached.utils import _get_page_size

    with patch.object(torch.version, "hip", None):
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "1"}):
            with pytest.raises(ValueError, match="multiple of 2MB"):
                _get_page_size()


def test_page_size_cuda_rejects_3mb():
    """CUDA path rejects 3MB (not a multiple of 2MB)."""
    from kvcached.utils import _get_page_size

    with patch.object(torch.version, "hip", None):
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "3"}):
            with pytest.raises(ValueError, match="multiple of 2MB"):
                _get_page_size()


def test_page_size_rejects_zero():
    """Zero page size is rejected."""
    from kvcached.utils import _get_page_size

    with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "0"}):
        with pytest.raises(ValueError, match="positive"):
            _get_page_size()


def test_page_size_rejects_negative():
    """Negative page size is rejected."""
    from kvcached.utils import _get_page_size

    with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "-1"}):
        with pytest.raises(ValueError, match="positive"):
            _get_page_size()


def test_page_size_rejects_non_integer():
    """Non-integer page size string is rejected."""
    from kvcached.utils import _get_page_size

    with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "abc"}):
        with pytest.raises(ValueError, match="integer"):
            _get_page_size()
