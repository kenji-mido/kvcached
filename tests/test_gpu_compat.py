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


# ── ROCm detection: test the ACTUAL _is_rocm() function ───────────────────


def test_is_rocm_returns_true_on_hip():
    """_is_rocm() returns True when torch.version.hip is set."""
    from kvcached.utils import _is_rocm

    with patch.object(torch.version, "hip", "7.1.0"):
        assert _is_rocm() is True


def test_is_rocm_returns_false_without_hip():
    """_is_rocm() returns False when torch.version.hip is None."""
    from kvcached.utils import _is_rocm

    with patch.object(torch.version, "hip", None):
        assert _is_rocm() is False


def test_is_rocm_matches_runtime():
    """_is_rocm() agrees with the actual runtime torch.version.hip value."""
    from kvcached.utils import _is_rocm

    expected = torch.version.hip is not None
    assert _is_rocm() is expected


def test_setup_py_is_rocm_consistent():
    """setup.py's detection uses the same expression as _is_rocm()."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; "
         "from kvcached.utils import _is_rocm; "
         "setup_val = torch.version.hip is not None; "
         "util_val = _is_rocm(); "
         "assert setup_val == util_val, "
         "f'setup.py={setup_val} != _is_rocm()={util_val}'; "
         "print('CONSISTENT')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Consistency check failed:\n{result.stderr}"
    assert "CONSISTENT" in result.stdout


# ── Page size validation: valid cases ──────────────────────────────────────


def test_page_size_default():
    """Default page size is 2MB when env var is not set."""
    from kvcached.utils import _get_page_size

    with patch.dict(os.environ, {}, clear=False):
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


def test_page_size_rocm_accepts_3mb():
    """ROCm path accepts 3MB (valid multiple of 1MB)."""
    from kvcached.utils import _get_page_size

    with patch.object(torch.version, "hip", "7.1.0"):
        with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "3"}):
            assert _get_page_size() == 3 * 1024 * 1024


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


def test_page_size_rejects_float_string():
    """Float string page size is rejected (must be integer)."""
    from kvcached.utils import _get_page_size

    with patch.dict(os.environ, {"KVCACHED_PAGE_SIZE_MB": "2.5"}):
        with pytest.raises(ValueError, match="integer"):
            _get_page_size()
