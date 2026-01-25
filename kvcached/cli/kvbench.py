# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark monitor TUI for KVcached.

Displays real-time metrics:
- GPU memory usage
- KVCache usage per IPC segment
- Per-model metrics from vLLM/SGLang (requests, tokens, latency)

Usage:
    kvbench --endpoints localhost:12346,localhost:12347
    kvbench --endpoints localhost:12346 --ipc CONTROLLER --output bench.csv
"""

import argparse
import csv
import curses
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError

from kvcached.cli.utils import (
    SHM_DIR,
    MemInfoStruct,
    RwLockedShm,
    _format_size,
    get_ipc_name,
)


@dataclass
class ModelMetrics:
    """Metrics collected from a vLLM/SGLang endpoint."""

    endpoint: str
    model_name: str = ""
    requests_running: int = 0
    requests_waiting: int = 0
    kv_cache_usage_perc: float = 0.0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    request_success_total: int = 0
    itl_sum: float = 0.0
    itl_count: int = 0
    e2e_latency_sum: float = 0.0
    e2e_latency_count: int = 0
    ttft_sum: float = 0.0
    ttft_count: int = 0
    reachable: bool = False
    error: str = ""

    # For rate calculations (previous values for delta)
    prev_prompt_tokens: int = 0
    prev_generation_tokens: int = 0
    prev_request_success: int = 0
    prev_ttft_sum: float = 0.0
    prev_ttft_count: int = 0
    prev_itl_sum: float = 0.0
    prev_itl_count: int = 0
    prev_e2e_sum: float = 0.0
    prev_e2e_count: int = 0
    prev_timestamp: float = 0.0

    # Initial values (for session delta)
    initial_prompt_tokens: int = -1  # -1 means not set
    initial_generation_tokens: int = -1
    initial_request_success: int = -1

    # Calculated rates
    prompt_tokens_rate: float = 0.0
    generation_tokens_rate: float = 0.0
    request_rate: float = 0.0

    # Recent averages (based on delta)
    recent_ttft_ms: float = 0.0
    recent_itl_ms: float = 0.0
    recent_e2e_ms: float = 0.0

    # Session delta (from start of monitoring)
    session_prompt_tokens: int = 0
    session_generation_tokens: int = 0
    session_request_success: int = 0

    @property
    def itl_avg_ms(self) -> float:
        """Average inter-token latency in milliseconds."""
        if self.itl_count > 0:
            return (self.itl_sum / self.itl_count) * 1000
        return 0.0

    @property
    def e2e_avg_ms(self) -> float:
        """Average end-to-end latency in milliseconds."""
        if self.e2e_latency_count > 0:
            return (self.e2e_latency_sum / self.e2e_latency_count) * 1000
        return 0.0

    @property
    def ttft_avg_ms(self) -> float:
        """Average time to first token in milliseconds."""
        if self.ttft_count > 0:
            return (self.ttft_sum / self.ttft_count) * 1000
        return 0.0


@dataclass
class KVCacheInfo:
    """KVCache info from shared memory segment."""

    ipc_name: str
    total_size: int = 0
    used_size: int = 0
    prealloc_size: int = 0
    reachable: bool = False


@dataclass
class GPUInfo:
    """GPU memory info."""

    total_bytes: int = 0
    used_bytes: int = 0
    available: bool = False


def _detect_kvcache_ipc_names() -> List[str]:
    """Detect all shared memory segments that look like KVCacheManager info."""
    candidates: List[str] = []
    try:
        for fname in os.listdir(SHM_DIR):
            path = os.path.join(SHM_DIR, fname)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size != MemInfoStruct.SHM_SIZE:
                continue
            try:
                with RwLockedShm(fname, MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK) as mm:
                    total_size = MemInfoStruct.from_buffer(mm).total_size
                    if total_size <= 0:
                        continue
                candidates.append(fname)
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return sorted(candidates)


def _get_kvcache_info(ipc_name: str) -> KVCacheInfo:
    """Get KVCache info from shared memory."""
    info = KVCacheInfo(ipc_name=ipc_name)
    try:
        with RwLockedShm(
            get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK
        ) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            info.total_size = int(mem_info.total_size)
            info.used_size = int(mem_info.used_size)
            info.prealloc_size = int(mem_info.prealloc_size)
            info.reachable = True
    except Exception:
        pass
    return info


def _get_gpu_info() -> GPUInfo:
    """Get GPU memory info via torch."""
    info = GPUInfo()
    try:
        import torch

        if torch.cuda.is_available():
            avail, total = torch.cuda.mem_get_info()
            info.total_bytes = total
            info.used_bytes = total - avail
            info.available = True
    except Exception:
        pass
    return info


def _parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus text format into a dict of metric_name -> value."""
    metrics: Dict[str, float] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: metric_name{labels} value
        # or: metric_name value
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+(.+)$", line)
        if match:
            name = match.group(1)
            try:
                value = float(match.group(2))
                metrics[name] = value
            except ValueError:
                pass
    return metrics


def _fetch_model_metrics(endpoint: str, prev: Optional[ModelMetrics] = None) -> ModelMetrics:
    """Fetch metrics from a vLLM/SGLang endpoint."""
    metrics = ModelMetrics(endpoint=endpoint)
    if prev:
        metrics.prev_prompt_tokens = prev.prompt_tokens_total
        metrics.prev_generation_tokens = prev.generation_tokens_total
        metrics.prev_request_success = prev.request_success_total
        metrics.prev_ttft_sum = prev.ttft_sum
        metrics.prev_ttft_count = prev.ttft_count
        metrics.prev_itl_sum = prev.itl_sum
        metrics.prev_itl_count = prev.itl_count
        metrics.prev_e2e_sum = prev.e2e_latency_sum
        metrics.prev_e2e_count = prev.e2e_latency_count
        metrics.prev_timestamp = prev.prev_timestamp if prev.prev_timestamp else time.time()
        # Carry over initial values
        metrics.initial_prompt_tokens = prev.initial_prompt_tokens
        metrics.initial_generation_tokens = prev.initial_generation_tokens
        metrics.initial_request_success = prev.initial_request_success

    url = f"http://{endpoint}/metrics"
    try:
        with urlopen(url, timeout=2) as resp:
            text = resp.read().decode("utf-8")
        prom = _parse_prometheus_metrics(text)
        metrics.reachable = True

        # Extract metrics (find keys that match patterns)
        for key, value in prom.items():
            if "model_name=" in key:
                # Extract model name
                match = re.search(r'model_name="([^"]+)"', key)
                if match and not metrics.model_name:
                    metrics.model_name = match.group(1)

            if "num_requests_running" in key:
                metrics.requests_running = int(value)
            elif "num_requests_waiting" in key:
                metrics.requests_waiting = int(value)
            elif "kv_cache_usage_perc" in key:
                metrics.kv_cache_usage_perc = value * 100
            elif "prompt_tokens_total" in key and "request_" not in key:
                metrics.prompt_tokens_total = int(value)
            elif "generation_tokens_total" in key and "request_" not in key:
                metrics.generation_tokens_total = int(value)
            elif "request_success_total" in key:
                metrics.request_success_total += int(value)
            elif "inter_token_latency_seconds_sum" in key:
                metrics.itl_sum = value
            elif "inter_token_latency_seconds_count" in key:
                metrics.itl_count = int(value)
            elif "e2e_request_latency_seconds_sum" in key:
                metrics.e2e_latency_sum = value
            elif "e2e_request_latency_seconds_count" in key:
                metrics.e2e_latency_count = int(value)
            elif "time_to_first_token_seconds_sum" in key:
                metrics.ttft_sum = value
            elif "time_to_first_token_seconds_count" in key:
                metrics.ttft_count = int(value)

        # Calculate rates
        now = time.time()
        if metrics.prev_timestamp > 0:
            dt = now - metrics.prev_timestamp
            if dt > 0:
                metrics.prompt_tokens_rate = (
                    metrics.prompt_tokens_total - metrics.prev_prompt_tokens
                ) / dt
                metrics.generation_tokens_rate = (
                    metrics.generation_tokens_total - metrics.prev_generation_tokens
                ) / dt
                metrics.request_rate = (
                    metrics.request_success_total - metrics.prev_request_success
                ) / dt

        # Calculate recent TTFT (delta-based)
        delta_ttft_count = metrics.ttft_count - metrics.prev_ttft_count
        if delta_ttft_count > 0:
            delta_ttft_sum = metrics.ttft_sum - metrics.prev_ttft_sum
            metrics.recent_ttft_ms = (delta_ttft_sum / delta_ttft_count) * 1000
        else:
            # No new requests, keep previous value or use cumulative average
            metrics.recent_ttft_ms = prev.recent_ttft_ms if prev and prev.recent_ttft_ms > 0 else metrics.ttft_avg_ms

        # Calculate recent ITL (delta-based)
        delta_itl_count = metrics.itl_count - metrics.prev_itl_count
        if delta_itl_count > 0:
            delta_itl_sum = metrics.itl_sum - metrics.prev_itl_sum
            metrics.recent_itl_ms = (delta_itl_sum / delta_itl_count) * 1000
        else:
            metrics.recent_itl_ms = prev.recent_itl_ms if prev and prev.recent_itl_ms > 0 else metrics.itl_avg_ms

        # Calculate recent E2E latency (delta-based)
        delta_e2e_count = metrics.e2e_latency_count - metrics.prev_e2e_count
        if delta_e2e_count > 0:
            delta_e2e_sum = metrics.e2e_latency_sum - metrics.prev_e2e_sum
            metrics.recent_e2e_ms = (delta_e2e_sum / delta_e2e_count) * 1000
        else:
            metrics.recent_e2e_ms = prev.recent_e2e_ms if prev and prev.recent_e2e_ms > 0 else metrics.e2e_avg_ms

        # Set initial values on first successful fetch
        if metrics.initial_prompt_tokens < 0:
            metrics.initial_prompt_tokens = metrics.prompt_tokens_total
            metrics.initial_generation_tokens = metrics.generation_tokens_total
            metrics.initial_request_success = metrics.request_success_total

        # Calculate session delta (from start of monitoring)
        metrics.session_prompt_tokens = metrics.prompt_tokens_total - metrics.initial_prompt_tokens
        metrics.session_generation_tokens = metrics.generation_tokens_total - metrics.initial_generation_tokens
        metrics.session_request_success = metrics.request_success_total - metrics.initial_request_success

        metrics.prev_timestamp = now

    except URLError as e:
        metrics.error = str(e.reason)
    except Exception as e:
        metrics.error = str(e)

    return metrics


def _draw_bar(width: int, percent: float, use_colors: bool) -> Tuple[str, int]:
    """Create a progress bar string and color pair."""
    filled = int(width * percent / 100)
    bar = "#" * filled + "-" * (width - filled)
    if percent < 50:
        color_pair = 1  # green
    elif percent < 80:
        color_pair = 2  # yellow
    else:
        color_pair = 3  # red
    return bar, color_pair


class TimeSeriesBuffer:
    """Buffer for time series data."""

    def __init__(self, max_points: int = 60):
        self.max_points = max_points
        self.gpu_usage: Deque[float] = deque(maxlen=max_points)
        self.kvcache_usage: Deque[float] = deque(maxlen=max_points)
        self.gen_tokens_rate: Deque[float] = deque(maxlen=max_points)
        self.request_rate: Deque[float] = deque(maxlen=max_points)
        self.ttft_ms: Deque[float] = deque(maxlen=max_points)
        self.timestamps: Deque[float] = deque(maxlen=max_points)

    def add_sample(
        self,
        gpu_pct: float,
        kv_pct: float,
        gen_rate: float,
        req_rate: float,
        ttft_ms: float = 0.0,
    ):
        """Add a sample to the buffer."""
        self.gpu_usage.append(gpu_pct)
        self.kvcache_usage.append(kv_pct)
        self.gen_tokens_rate.append(gen_rate)
        self.request_rate.append(req_rate)
        self.ttft_ms.append(ttft_ms)
        self.timestamps.append(time.time())


def _draw_ascii_graph(
    stdscr,
    row: int,
    col: int,
    width: int,
    height: int,
    data: Deque[float],
    label: str,
    unit: str,
    color_pair: int,
    use_colors: bool,
    fixed_max: Optional[float] = None,
) -> int:
    """Draw an ASCII time series graph.

    Args:
        fixed_max: If set, use this as the maximum Y-axis value (e.g., 100 for percentages)

    Returns the number of rows used.
    """
    if height < 3 or width < 20:
        return 0

    screen_height, screen_width = stdscr.getmaxyx()

    # Ensure we don't exceed screen bounds
    if row + height >= screen_height or col + width >= screen_width:
        return 0

    # Calculate graph dimensions
    label_width = 8  # Space for Y-axis labels
    graph_width = min(width - label_width - 2, len(data)) if data else width - label_width - 2
    graph_height = height - 2  # Leave space for X-axis and label

    if graph_width < 5 or graph_height < 2:
        return 0

    # Get data range
    data_list = list(data)[-graph_width:] if data else []
    if not data_list:
        data_list = [0.0]

    if fixed_max is not None:
        max_val = fixed_max
    else:
        max_val = max(max(data_list), 1.0)  # Avoid division by zero
    min_val = 0.0

    # Draw title
    title = f"{label} ({unit})"
    try:
        stdscr.addstr(row, col, title[:width], curses.color_pair(color_pair) | curses.A_BOLD if use_colors else curses.A_BOLD)
    except curses.error:
        pass
    row += 1

    # Draw graph area
    graph_chars = [" ", ".", ":", "-", "=", "+", "*", "#"]

    for y in range(graph_height):
        # Y-axis label
        y_val = max_val - (y / (graph_height - 1)) * (max_val - min_val) if graph_height > 1 else max_val
        y_label = f"{y_val:>6.1f} |"
        try:
            stdscr.addstr(row + y, col, y_label[:label_width], curses.A_DIM)
        except curses.error:
            pass

        # Graph points
        for x, val in enumerate(data_list):
            if x >= graph_width:
                break
            # Calculate vertical position
            if max_val > min_val:
                normalized = (val - min_val) / (max_val - min_val)
            else:
                normalized = 0

            # Which row should this point appear in?
            point_row = int((1 - normalized) * (graph_height - 1))

            char_col = col + label_width + x
            if char_col >= screen_width:
                break

            if point_row == y:
                # Draw point
                try:
                    stdscr.addstr(row + y, char_col, "*", curses.color_pair(color_pair) if use_colors else 0)
                except curses.error:
                    pass
            elif point_row < y:
                # Draw vertical line below point
                try:
                    stdscr.addstr(row + y, char_col, "|", curses.color_pair(color_pair) | curses.A_DIM if use_colors else curses.A_DIM)
                except curses.error:
                    pass

    # Draw X-axis
    x_axis_row = row + graph_height
    x_axis = " " * label_width + "+" + "-" * min(graph_width, width - label_width - 1) + ">"
    try:
        stdscr.addstr(x_axis_row, col, x_axis[:width], curses.A_DIM)
    except curses.error:
        pass

    # X-axis label (time)
    time_label = f"{' ' * label_width}0{' ' * (graph_width // 2 - 1)}time (s){' ' * (graph_width // 2 - 5)}{len(data_list)}"
    try:
        stdscr.addstr(x_axis_row + 1, col, time_label[:width], curses.A_DIM)
    except curses.error:
        pass

    return height


class BenchmarkMonitor:
    """Benchmark monitor with curses TUI."""

    def __init__(
        self,
        endpoints: List[str],
        ipc_names: Optional[List[str]] = None,
        output_file: Optional[Path] = None,
        refresh_rate: float = 1.0,
        graph_history: int = 60,
    ):
        self.endpoints = endpoints
        self.ipc_names = ipc_names
        self.output_file = output_file
        self.refresh_rate = refresh_rate
        self.graph_history = graph_history
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.csv_writer = None
        self.csv_file = None
        self.time_series = TimeSeriesBuffer(max_points=graph_history)

    def _init_csv(self):
        """Initialize CSV output file."""
        if self.output_file:
            self.csv_file = open(self.output_file, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                [
                    "timestamp",
                    "endpoint",
                    "model_name",
                    # Request counts (session delta)
                    "requests_running",
                    "requests_waiting",
                    "session_request_done",
                    # Token counts (session delta)
                    "session_prompt_tokens",
                    "session_generation_tokens",
                    # Rates
                    "prompt_tokens_rate",
                    "generation_tokens_rate",
                    "request_rate",
                    # Latency (recent/delta-based)
                    "recent_ttft_ms",
                    "recent_itl_ms",
                    "recent_e2e_ms",
                    # KVCache (from vLLM metrics)
                    "kv_cache_usage_perc",
                    # KVCache (from shared memory)
                    "kvcache_used_bytes",
                    "kvcache_prealloc_bytes",
                    "kvcache_total_bytes",
                    # GPU memory
                    "gpu_used_bytes",
                    "gpu_total_bytes",
                    "gpu_usage_perc",
                ]
            )

    def _write_csv_row(
        self,
        metrics: ModelMetrics,
        gpu_info: GPUInfo,
        kvcache_info: Optional[KVCacheInfo] = None,
    ):
        """Write a row to CSV."""
        if self.csv_writer:
            gpu_pct = (gpu_info.used_bytes / gpu_info.total_bytes * 100) if gpu_info.total_bytes else 0
            self.csv_writer.writerow(
                [
                    datetime.now().isoformat(),
                    metrics.endpoint,
                    metrics.model_name,
                    # Request counts
                    metrics.requests_running,
                    metrics.requests_waiting,
                    metrics.session_request_success,
                    # Token counts
                    metrics.session_prompt_tokens,
                    metrics.session_generation_tokens,
                    # Rates
                    f"{metrics.prompt_tokens_rate:.2f}",
                    f"{metrics.generation_tokens_rate:.2f}",
                    f"{metrics.request_rate:.2f}",
                    # Latency (recent/delta-based)
                    f"{metrics.recent_ttft_ms:.2f}",
                    f"{metrics.recent_itl_ms:.2f}",
                    f"{metrics.recent_e2e_ms:.2f}",
                    # KVCache (from vLLM metrics)
                    f"{metrics.kv_cache_usage_perc:.2f}",
                    # KVCache (from shared memory)
                    kvcache_info.used_size if kvcache_info and kvcache_info.reachable else 0,
                    kvcache_info.prealloc_size if kvcache_info and kvcache_info.reachable else 0,
                    kvcache_info.total_size if kvcache_info and kvcache_info.reachable else 0,
                    # GPU memory
                    gpu_info.used_bytes,
                    gpu_info.total_bytes,
                    f"{gpu_pct:.2f}",
                ]
            )
            self.csv_file.flush()

    def _close_csv(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()

    def run(self):
        """Run the monitor with curses UI."""
        try:
            curses.wrapper(self._curses_main)
        finally:
            self._close_csv()

    def _curses_main(self, stdscr):
        """Main curses loop."""
        curses.curs_set(0)
        stdscr.nodelay(True)

        # Initialize colors
        use_colors = False
        if curses.has_colors():
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, curses.COLOR_RED, -1)
                curses.init_pair(4, curses.COLOR_CYAN, -1)
                curses.init_pair(5, curses.COLOR_MAGENTA, -1)
                use_colors = True
            except Exception:
                pass

        # Show loading message
        stdscr.erase()
        stdscr.addstr(0, 0, "Initializing... please wait", curses.A_DIM)
        stdscr.refresh()

        # Initialize CSV
        self._init_csv()

        # Pre-load torch
        try:
            import torch

            torch.cuda.is_available()
        except Exception:
            pass

        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()

            header_attr = curses.color_pair(4) | curses.A_BOLD if use_colors else curses.A_BOLD
            section_attr = curses.color_pair(5) | curses.A_BOLD if use_colors else curses.A_BOLD

            row = 0

            # Title
            title = "KVcached Benchmark Monitor"
            stdscr.addstr(row, 0, title, header_attr)
            stdscr.addstr(row, len(title) + 2, datetime.now().strftime("%H:%M:%S"), curses.A_DIM)
            row += 2

            # GPU Memory
            gpu_info = _get_gpu_info()
            if gpu_info.available and row < height - 2:
                stdscr.addstr(row, 0, "GPU Memory", section_attr)
                row += 1
                gpu_pct = (gpu_info.used_bytes / gpu_info.total_bytes * 100) if gpu_info.total_bytes else 0
                bar_width = min(60, width - 30)
                bar, color = _draw_bar(bar_width, gpu_pct, use_colors)
                stdscr.addstr(row, 0, "[")
                stdscr.addstr(bar, curses.color_pair(color) if use_colors else 0)
                stdscr.addstr("]")
                info = f" {_format_size(gpu_info.used_bytes)} / {_format_size(gpu_info.total_bytes)} ({gpu_pct:.1f}%)"
                stdscr.addstr(info[:width - bar_width - 3])
                row += 2

            # KVCache segments
            ipc_names = self.ipc_names if self.ipc_names else _detect_kvcache_ipc_names()
            if ipc_names and row < height - 2:
                stdscr.addstr(row, 0, "KVCache Segments", section_attr)
                row += 1
                for ipc_name in ipc_names:
                    if row >= height - 4:
                        break
                    info = _get_kvcache_info(ipc_name)
                    if info.reachable:
                        pct = ((info.used_size + info.prealloc_size) / info.total_size * 100) if info.total_size else 0
                        bar_width = min(40, width - 50)
                        bar, color = _draw_bar(bar_width, pct, use_colors)
                        stdscr.addstr(row, 0, f"{ipc_name[:15]:<15} [")
                        stdscr.addstr(bar, curses.color_pair(color) if use_colors else 0)
                        stdscr.addstr("]")
                        detail = f" Used:{_format_size(info.used_size):>8} Pre:{_format_size(info.prealloc_size):>8} / {_format_size(info.total_size):>8}"
                        stdscr.addstr(detail[:width - bar_width - 20])
                        row += 1
                row += 1

            # Get KVCache info for CSV output
            ipc_list = self.ipc_names if self.ipc_names else _detect_kvcache_ipc_names()
            first_kv_info = _get_kvcache_info(ipc_list[0]) if ipc_list else None

            # Model metrics
            if self.endpoints and row < height - 2:
                stdscr.addstr(row, 0, "Model Endpoints", section_attr)
                row += 1

                for endpoint in self.endpoints:
                    if row >= height - 4:
                        break

                    prev = self.model_metrics.get(endpoint)
                    metrics = _fetch_model_metrics(endpoint, prev)
                    self.model_metrics[endpoint] = metrics

                    # Write to CSV
                    if metrics.reachable:
                        self._write_csv_row(metrics, gpu_info, first_kv_info)

                    if metrics.reachable:
                        # Model name and endpoint
                        model_short = metrics.model_name.split("/")[-1] if metrics.model_name else "unknown"
                        stdscr.addstr(row, 0, f"{model_short[:20]:<20} ", curses.A_BOLD if use_colors else 0)
                        stdscr.addstr(f"({endpoint})", curses.A_DIM)
                        row += 1

                        # Requests and tokens (session delta)
                        if row < height - 2:
                            req_info = f"  Requests: run={metrics.requests_running} wait={metrics.requests_waiting} done={metrics.session_request_success}"
                            stdscr.addstr(row, 0, req_info[:width - 1])
                            row += 1

                        if row < height - 2:
                            tok_info = f"  Tokens: prompt={metrics.session_prompt_tokens} gen={metrics.session_generation_tokens}"
                            stdscr.addstr(row, 0, tok_info[:width - 1])
                            row += 1

                        if row < height - 2:
                            rate_info = f"  Rate: {metrics.prompt_tokens_rate:.1f} prompt/s, {metrics.generation_tokens_rate:.1f} gen/s, {metrics.request_rate:.2f} req/s"
                            stdscr.addstr(row, 0, rate_info[:width - 1])
                            row += 1

                        if row < height - 2:
                            lat_info = f"  Latency: TTFT={metrics.recent_ttft_ms:.1f}ms ITL={metrics.recent_itl_ms:.1f}ms E2E={metrics.recent_e2e_ms:.1f}ms KV={metrics.kv_cache_usage_perc:.1f}%"
                            stdscr.addstr(row, 0, lat_info[:width - 1])
                            row += 1

                        row += 1
                    else:
                        stdscr.addstr(row, 0, f"{endpoint}: ", curses.A_BOLD if use_colors else 0)
                        stdscr.addstr("UNREACHABLE", curses.color_pair(3) if use_colors else 0)
                        if metrics.error:
                            stdscr.addstr(f" ({metrics.error[:30]})", curses.A_DIM)
                        row += 2

            # Collect time series data
            total_gen_rate = sum(
                m.generation_tokens_rate for m in self.model_metrics.values() if m.reachable
            )
            total_req_rate = sum(
                m.request_rate for m in self.model_metrics.values() if m.reachable
            )
            # Average recent TTFT across all models (delta-based, not cumulative)
            ttft_values = [m.recent_ttft_ms for m in self.model_metrics.values() if m.reachable and m.recent_ttft_ms > 0]
            avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0

            kv_pct = 0.0
            if first_kv_info and first_kv_info.reachable and first_kv_info.total_size > 0:
                kv_pct = (first_kv_info.used_size + first_kv_info.prealloc_size) / first_kv_info.total_size * 100

            self.time_series.add_sample(
                gpu_pct=gpu_pct if gpu_info.available else 0,
                kv_pct=kv_pct,
                gen_rate=total_gen_rate,
                req_rate=total_req_rate,
                ttft_ms=avg_ttft,
            )

            # Draw time series graphs
            row += 1
            if row < height - 10:
                stdscr.addstr(row, 0, "Time Series Graphs", section_attr)
                row += 1

                # Calculate graph dimensions
                graph_height = 8
                graph_width = min(70, width - 2)
                graphs_per_row = 2 if width >= 140 else 1

                # (data, label, unit, color, fixed_max)
                graphs = [
                    (self.time_series.gpu_usage, "GPU Mem", "%", 2, 100.0),  # yellow, 0-100%
                    (self.time_series.kvcache_usage, "KVCache", "%", 4, 100.0),  # cyan, 0-100%
                    (self.time_series.gen_tokens_rate, "Gen Rate", "tok/s", 1, None),  # green
                    (self.time_series.request_rate, "Req Rate", "req/s", 5, None),  # magenta
                    (self.time_series.ttft_ms, "TTFT", "ms", 3, None),  # red
                ]

                col = 0
                graphs_drawn = 0
                for data, label, unit, color, fixed_max in graphs:
                    if row + graph_height + 2 >= height - 1:
                        break

                    rows_used = _draw_ascii_graph(
                        stdscr,
                        row,
                        col,
                        graph_width,
                        graph_height,
                        data,
                        label,
                        unit,
                        color,
                        use_colors,
                        fixed_max=fixed_max,
                    )

                    graphs_drawn += 1
                    if graphs_per_row == 2 and graphs_drawn % 2 == 1:
                        col = graph_width + 2
                    else:
                        col = 0
                        row += graph_height + 2

            # Footer
            if row < height - 1:
                footer = "Press 'q' to quit"
                if self.output_file:
                    footer += f" | Logging to: {self.output_file}"
                stdscr.addstr(height - 1, 0, footer[:width - 1], curses.A_DIM)

            stdscr.refresh()

            # Handle input
            ch = stdscr.getch()
            if ch == ord("q") or ch == ord("Q"):
                break

            time.sleep(self.refresh_rate)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark monitor TUI for KVcached",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kvbench --endpoints localhost:12346,localhost:12347
  kvbench --endpoints localhost:12346 --ipc CONTROLLER
  kvbench --endpoints localhost:12346 --output bench.csv
        """,
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        default="",
        help="Comma-separated list of vLLM/SGLang endpoints (host:port)",
    )
    parser.add_argument(
        "--ipc",
        type=str,
        default="",
        help="Comma-separated list of IPC names to monitor (auto-detect if empty)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV file for logging metrics",
    )
    parser.add_argument(
        "--refresh",
        "-r",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=60,
        help="Number of data points to keep in graph history (default: 60)",
    )
    args = parser.parse_args()

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    ipc_names = [i.strip() for i in args.ipc.split(",") if i.strip()] if args.ipc else None

    if not endpoints:
        print("Error: --endpoints is required")
        print("Example: kvbench --endpoints localhost:12346,localhost:12347")
        return 1

    monitor = BenchmarkMonitor(
        endpoints=endpoints,
        ipc_names=ipc_names,
        output_file=args.output,
        refresh_rate=args.refresh,
        graph_history=args.history,
    )
    monitor.run()
    return 0


if __name__ == "__main__":
    exit(main())
