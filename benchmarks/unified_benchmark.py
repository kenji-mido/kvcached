#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""
Unified Benchmark Script for KVcached.

This script runs comprehensive benchmarks comparing KVcached enabled vs disabled,
measuring GPU memory efficiency and user experience metrics (TTFT, ITL, E2E, throughput).

Usage:
    python unified_benchmark.py --models meta-llama/Llama-3.2-1B,Qwen/Qwen2.5-0.5B
    python unified_benchmark.py --models meta-llama/Llama-3.2-1B --num-prompts 50
    python unified_benchmark.py --config benchmark_config.yaml
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    models: List[str]
    base_port: int = 12346
    num_prompts: int = 30
    request_rate: float = 5.0
    gpu_memory_utilization: float = 0.35
    max_model_len: int = 4096
    warmup_prompts: int = 5
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    venv_path: Optional[Path] = None
    cuda_home: str = "/usr/local/cuda"
    ipc_name: str = "BENCHMARK"


@dataclass
class MetricsSample:
    """A single metrics sample."""

    timestamp: float
    model: str
    endpoint: str
    requests_running: int = 0
    requests_waiting: int = 0
    requests_done: int = 0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_rate: float = 0.0
    generation_rate: float = 0.0
    request_rate: float = 0.0
    ttft_ms: float = 0.0
    itl_ms: float = 0.0
    e2e_ms: float = 0.0
    kv_cache_usage_perc: float = 0.0
    kvcache_used_bytes: int = 0
    kvcache_total_bytes: int = 0
    gpu_used_bytes: int = 0
    gpu_total_bytes: int = 0


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    kvcached_enabled: bool
    models: List[str]
    samples: List[MetricsSample] = field(default_factory=list)
    # Summary statistics
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_generation_tokens: int = 0
    avg_ttft_ms: float = 0.0
    avg_itl_ms: float = 0.0
    avg_e2e_ms: float = 0.0
    avg_throughput: float = 0.0
    peak_gpu_memory_bytes: int = 0
    peak_kvcache_bytes: int = 0
    duration_seconds: float = 0.0


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus text format into a dict."""
    metrics: Dict[str, float] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+(.+)$", line)
        if match:
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    return metrics


def fetch_vllm_metrics(endpoint: str) -> Optional[Dict[str, float]]:
    """Fetch metrics from vLLM endpoint."""
    try:
        with urlopen(f"http://{endpoint}/metrics", timeout=2) as resp:
            return parse_prometheus_metrics(resp.read().decode("utf-8"))
    except Exception:
        return None


def check_server_health(endpoint: str) -> bool:
    """Check if server is healthy."""
    try:
        with urlopen(f"http://{endpoint}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_gpu_memory() -> Tuple[int, int]:
    """Get GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]  # First GPU
            used_mb, total_mb = map(int, line.split(","))
            return used_mb * 1024 * 1024, total_mb * 1024 * 1024
    except Exception:
        pass
    return 0, 0


def get_kvcache_info(ipc_name: str) -> Tuple[int, int, int]:
    """Get KVCache info from shared memory."""
    try:
        from kvcached.cli.utils import (
            MemInfoStruct,
            RwLockedShm,
            get_ipc_name,
        )

        with RwLockedShm(
            get_ipc_name(ipc_name), MemInfoStruct.SHM_SIZE, RwLockedShm.RLOCK
        ) as mm:
            info = MemInfoStruct.from_buffer(mm)
            return int(info.used_size), int(info.prealloc_size), int(info.total_size)
    except Exception:
        return 0, 0, 0


class UnifiedBenchmark:
    """Unified benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.server_processes: List[subprocess.Popen] = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_python_path(self) -> str:
        """Get Python executable path."""
        if self.config.venv_path:
            return str(self.config.venv_path / "bin" / "python")
        return sys.executable

    def _get_env(self, kvcached_enabled: bool) -> Dict[str, str]:
        """Get environment variables for server."""
        env = os.environ.copy()
        env["CUDA_HOME"] = self.config.cuda_home
        env["PATH"] = f"{self.config.cuda_home}/bin:{env.get('PATH', '')}"

        if kvcached_enabled:
            env["ENABLE_KVCACHED"] = "true"
            env["KVCACHED_AUTOPATCH"] = "1"
            env["KVCACHED_IPC_NAME"] = self.config.ipc_name
        else:
            env["ENABLE_KVCACHED"] = "false"

        return env

    def start_servers(self, kvcached_enabled: bool) -> List[str]:
        """Start vLLM servers for all models."""
        endpoints = []
        python_path = self._get_python_path()
        env = self._get_env(kvcached_enabled)

        for i, model in enumerate(self.config.models):
            port = self.config.base_port + i
            endpoint = f"localhost:{port}"
            endpoints.append(endpoint)

            cmd = [
                python_path,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
                "--port",
                str(port),
                "--host",
                "localhost",
                "--gpu-memory-utilization",
                str(self.config.gpu_memory_utilization),
                "--max-model-len",
                str(self.config.max_model_len),
                "--disable-log-requests",
            ]

            if kvcached_enabled:
                cmd.append("--no-enable-prefix-caching")

            print(f"Starting server for {model} on port {port}...")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.server_processes.append(proc)

            # Wait between server starts
            if i < len(self.config.models) - 1:
                time.sleep(30)

        # Wait for all servers to be healthy
        print("Waiting for servers to be ready...")
        for endpoint in endpoints:
            for _ in range(120):  # 2 minutes timeout
                if check_server_health(endpoint):
                    print(f"  {endpoint} is ready")
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"Server {endpoint} failed to start")

        return endpoints

    def stop_servers(self):
        """Stop all running servers."""
        for proc in self.server_processes:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.server_processes.clear()

    def run_warmup(self, endpoints: List[str]):
        """Run warmup requests."""
        print("Running warmup...")
        python_path = self._get_python_path()

        for i, (endpoint, model) in enumerate(zip(endpoints, self.config.models)):
            port = endpoint.split(":")[1]
            cmd = [
                python_path,
                "-m",
                "vllm.entrypoints.openai.run_batch_inference",
                "--model",
                model,
                "--port",
                port,
                "--num-prompts",
                str(self.config.warmup_prompts),
            ]
            # Simple warmup - just send a few requests
            try:
                subprocess.run(
                    [
                        python_path,
                        "-c",
                        f"""
import requests
for i in range({self.config.warmup_prompts}):
    requests.post(
        "http://{endpoint}/v1/completions",
        json={{"model": "{model}", "prompt": "Hello", "max_tokens": 10}}
    )
""",
                    ],
                    timeout=60,
                    capture_output=True,
                )
            except Exception:
                pass

    def _get_vllm_cli_path(self) -> str:
        """Get vllm CLI executable path."""
        if self.config.venv_path:
            return str(self.config.venv_path / "bin" / "vllm")
        # Fall back to system vllm
        return "vllm"

    def run_benchmark_clients(self, endpoints: List[str]) -> List[subprocess.Popen]:
        """Run benchmark clients for all endpoints."""
        vllm_path = self._get_vllm_cli_path()
        processes = []

        for endpoint, model in zip(endpoints, self.config.models):
            port = endpoint.split(":")[1]
            cmd = [
                vllm_path,
                "bench",
                "serve",
                "--model",
                model,
                "--port",
                port,
                "--num-prompts",
                str(self.config.num_prompts),
                "--request-rate",
                str(self.config.request_rate),
            ]
            print(f"Starting benchmark client for {model}...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            processes.append(proc)

        return processes

    def collect_metrics(
        self,
        endpoints: List[str],
        kvcached_enabled: bool,
        duration: float,
    ) -> BenchmarkResult:
        """Collect metrics during benchmark."""
        result = BenchmarkResult(
            kvcached_enabled=kvcached_enabled,
            models=self.config.models.copy(),
        )

        start_time = time.time()
        prev_metrics: Dict[str, Dict] = {}

        while time.time() - start_time < duration:
            for endpoint, model in zip(endpoints, self.config.models):
                prom = fetch_vllm_metrics(endpoint)
                if not prom:
                    continue

                # Extract metrics
                sample = MetricsSample(
                    timestamp=time.time(),
                    model=model,
                    endpoint=endpoint,
                )

                # Initialize latency accumulators
                ttft_sum = 0.0
                ttft_count = 0
                itl_sum = 0.0
                itl_count = 0
                e2e_sum = 0.0
                e2e_count = 0

                for key, value in prom.items():
                    if "num_requests_running" in key:
                        sample.requests_running = int(value)
                    elif "num_requests_waiting" in key:
                        sample.requests_waiting = int(value)
                    elif "kv_cache_usage_perc" in key:
                        sample.kv_cache_usage_perc = value * 100
                    elif "prompt_tokens_total" in key and "request_" not in key:
                        sample.prompt_tokens = int(value)
                    elif "generation_tokens_total" in key and "request_" not in key:
                        sample.generation_tokens = int(value)
                    elif "request_success_total" in key:
                        sample.requests_done += int(value)
                    elif "time_to_first_token_seconds_sum" in key:
                        ttft_sum = value
                    elif "time_to_first_token_seconds_count" in key:
                        ttft_count = int(value)
                    elif "inter_token_latency_seconds_sum" in key:
                        itl_sum = value
                    elif "inter_token_latency_seconds_count" in key:
                        itl_count = int(value)
                    elif "e2e_request_latency_seconds_sum" in key:
                        e2e_sum = value
                    elif "e2e_request_latency_seconds_count" in key:
                        e2e_count = int(value)

                # Calculate delta-based latencies
                prev = prev_metrics.get(endpoint, {})
                if prev:
                    delta_ttft_count = ttft_count - prev.get("ttft_count", 0)
                    if delta_ttft_count > 0:
                        sample.ttft_ms = (
                            (ttft_sum - prev.get("ttft_sum", 0)) / delta_ttft_count
                        ) * 1000

                    delta_itl_count = itl_count - prev.get("itl_count", 0)
                    if delta_itl_count > 0:
                        sample.itl_ms = (
                            (itl_sum - prev.get("itl_sum", 0)) / delta_itl_count
                        ) * 1000

                    delta_e2e_count = e2e_count - prev.get("e2e_count", 0)
                    if delta_e2e_count > 0:
                        sample.e2e_ms = (
                            (e2e_sum - prev.get("e2e_sum", 0)) / delta_e2e_count
                        ) * 1000

                    dt = sample.timestamp - prev.get("timestamp", sample.timestamp)
                    if dt > 0:
                        sample.prompt_rate = (
                            sample.prompt_tokens - prev.get("prompt_tokens", 0)
                        ) / dt
                        sample.generation_rate = (
                            sample.generation_tokens - prev.get("generation_tokens", 0)
                        ) / dt

                prev_metrics[endpoint] = {
                    "timestamp": sample.timestamp,
                    "ttft_sum": ttft_sum,
                    "ttft_count": ttft_count,
                    "itl_sum": itl_sum,
                    "itl_count": itl_count,
                    "e2e_sum": e2e_sum,
                    "e2e_count": e2e_count,
                    "prompt_tokens": sample.prompt_tokens,
                    "generation_tokens": sample.generation_tokens,
                }

                # GPU and KVCache info
                sample.gpu_used_bytes, sample.gpu_total_bytes = get_gpu_memory()
                if kvcached_enabled:
                    used, prealloc, total = get_kvcache_info(self.config.ipc_name)
                    sample.kvcache_used_bytes = used + prealloc
                    sample.kvcache_total_bytes = total

                result.samples.append(sample)

                # Track peaks
                if sample.gpu_used_bytes > result.peak_gpu_memory_bytes:
                    result.peak_gpu_memory_bytes = sample.gpu_used_bytes
                if sample.kvcache_used_bytes > result.peak_kvcache_bytes:
                    result.peak_kvcache_bytes = sample.kvcache_used_bytes

            time.sleep(1)

        result.duration_seconds = time.time() - start_time
        self._calculate_summary(result)
        return result

    def _calculate_summary(self, result: BenchmarkResult):
        """Calculate summary statistics from samples."""
        if not result.samples:
            return

        ttft_values = [s.ttft_ms for s in result.samples if s.ttft_ms > 0]
        itl_values = [s.itl_ms for s in result.samples if s.itl_ms > 0]
        e2e_values = [s.e2e_ms for s in result.samples if s.e2e_ms > 0]
        gen_rates = [s.generation_rate for s in result.samples if s.generation_rate > 0]

        result.avg_ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else 0
        result.avg_itl_ms = sum(itl_values) / len(itl_values) if itl_values else 0
        result.avg_e2e_ms = sum(e2e_values) / len(e2e_values) if e2e_values else 0
        result.avg_throughput = sum(gen_rates) / len(gen_rates) if gen_rates else 0

        # Total tokens from last sample of each model
        last_samples = {}
        for s in result.samples:
            last_samples[s.model] = s
        result.total_prompt_tokens = sum(s.prompt_tokens for s in last_samples.values())
        result.total_generation_tokens = sum(
            s.generation_tokens for s in last_samples.values()
        )
        result.total_requests = sum(s.requests_done for s in last_samples.values())

    def save_results(
        self,
        result: BenchmarkResult,
        filename: str,
    ):
        """Save results to CSV."""
        filepath = self.config.output_dir / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "endpoint",
                    "requests_running",
                    "requests_waiting",
                    "requests_done",
                    "prompt_tokens",
                    "generation_tokens",
                    "prompt_rate",
                    "generation_rate",
                    "request_rate",
                    "ttft_ms",
                    "itl_ms",
                    "e2e_ms",
                    "kv_cache_usage_perc",
                    "kvcache_used_bytes",
                    "kvcache_total_bytes",
                    "gpu_used_bytes",
                    "gpu_total_bytes",
                ]
            )
            for s in result.samples:
                writer.writerow(
                    [
                        datetime.fromtimestamp(s.timestamp).isoformat(),
                        s.model,
                        s.endpoint,
                        s.requests_running,
                        s.requests_waiting,
                        s.requests_done,
                        s.prompt_tokens,
                        s.generation_tokens,
                        f"{s.prompt_rate:.2f}",
                        f"{s.generation_rate:.2f}",
                        f"{s.request_rate:.2f}",
                        f"{s.ttft_ms:.2f}",
                        f"{s.itl_ms:.2f}",
                        f"{s.e2e_ms:.2f}",
                        f"{s.kv_cache_usage_perc:.2f}",
                        s.kvcache_used_bytes,
                        s.kvcache_total_bytes,
                        s.gpu_used_bytes,
                        s.gpu_total_bytes,
                    ]
                )
        print(f"Saved results to {filepath}")

    def save_summary(
        self,
        results: List[BenchmarkResult],
        filename: str = "summary.json",
    ):
        """Save summary comparison to JSON."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "models": self.config.models,
                "num_prompts": self.config.num_prompts,
                "request_rate": self.config.request_rate,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
            },
            "results": [],
        }

        for r in results:
            summary["results"].append(
                {
                    "kvcached_enabled": r.kvcached_enabled,
                    "total_requests": r.total_requests,
                    "total_prompt_tokens": r.total_prompt_tokens,
                    "total_generation_tokens": r.total_generation_tokens,
                    "avg_ttft_ms": round(r.avg_ttft_ms, 2),
                    "avg_itl_ms": round(r.avg_itl_ms, 2),
                    "avg_e2e_ms": round(r.avg_e2e_ms, 2),
                    "avg_throughput_tok_s": round(r.avg_throughput, 2),
                    "peak_gpu_memory_gb": round(r.peak_gpu_memory_bytes / 1e9, 2),
                    "peak_kvcache_gb": round(r.peak_kvcache_bytes / 1e9, 2),
                    "duration_seconds": round(r.duration_seconds, 2),
                }
            )

        # Calculate improvement
        if len(results) == 2:
            enabled = next((r for r in results if r.kvcached_enabled), None)
            disabled = next((r for r in results if not r.kvcached_enabled), None)
            if enabled and disabled:
                summary["comparison"] = {
                    "ttft_improvement_pct": round(
                        (disabled.avg_ttft_ms - enabled.avg_ttft_ms)
                        / disabled.avg_ttft_ms
                        * 100,
                        2,
                    )
                    if disabled.avg_ttft_ms > 0
                    else 0,
                    "throughput_improvement_pct": round(
                        (enabled.avg_throughput - disabled.avg_throughput)
                        / disabled.avg_throughput
                        * 100,
                        2,
                    )
                    if disabled.avg_throughput > 0
                    else 0,
                    "gpu_memory_reduction_pct": round(
                        (disabled.peak_gpu_memory_bytes - enabled.peak_gpu_memory_bytes)
                        / disabled.peak_gpu_memory_bytes
                        * 100,
                        2,
                    )
                    if disabled.peak_gpu_memory_bytes > 0
                    else 0,
                }

        filepath = self.config.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {filepath}")

        # Print summary to console
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        for r in results:
            status = "KVcached ENABLED" if r.kvcached_enabled else "KVcached DISABLED"
            print(f"\n{status}:")
            print(f"  TTFT (avg):        {r.avg_ttft_ms:.2f} ms")
            print(f"  ITL (avg):         {r.avg_itl_ms:.2f} ms")
            print(f"  E2E (avg):         {r.avg_e2e_ms:.2f} ms")
            print(f"  Throughput (avg):  {r.avg_throughput:.2f} tok/s")
            print(f"  Peak GPU Memory:   {r.peak_gpu_memory_bytes / 1e9:.2f} GB")
            if r.kvcached_enabled:
                print(f"  Peak KVCache:      {r.peak_kvcache_bytes / 1e9:.2f} GB")

        if "comparison" in summary:
            print("\nCOMPARISON (KVcached enabled vs disabled):")
            print(f"  TTFT improvement:      {summary['comparison']['ttft_improvement_pct']:+.1f}%")
            print(
                f"  Throughput improvement: {summary['comparison']['throughput_improvement_pct']:+.1f}%"
            )
            print(
                f"  GPU memory reduction:   {summary['comparison']['gpu_memory_reduction_pct']:+.1f}%"
            )
        print("=" * 60)

    def run(self, skip_disabled: bool = False):
        """Run the full benchmark."""
        results = []

        try:
            # Run with KVcached enabled
            print("\n" + "=" * 60)
            print("RUNNING BENCHMARK: KVcached ENABLED")
            print("=" * 60)

            endpoints = self.start_servers(kvcached_enabled=True)
            self.run_warmup(endpoints)

            clients = self.run_benchmark_clients(endpoints)
            result_enabled = self.collect_metrics(
                endpoints,
                kvcached_enabled=True,
                duration=self.config.num_prompts / self.config.request_rate + 30,
            )

            for client in clients:
                client.wait()

            self.save_results(result_enabled, "kvcached_enabled.csv")
            results.append(result_enabled)

            self.stop_servers()
            time.sleep(5)

            # Run with KVcached disabled
            if not skip_disabled:
                print("\n" + "=" * 60)
                print("RUNNING BENCHMARK: KVcached DISABLED")
                print("=" * 60)

                endpoints = self.start_servers(kvcached_enabled=False)
                self.run_warmup(endpoints)

                clients = self.run_benchmark_clients(endpoints)
                result_disabled = self.collect_metrics(
                    endpoints,
                    kvcached_enabled=False,
                    duration=self.config.num_prompts / self.config.request_rate + 30,
                )

                for client in clients:
                    client.wait()

                self.save_results(result_disabled, "kvcached_disabled.csv")
                results.append(result_disabled)

            # Save summary
            self.save_summary(results)

        finally:
            self.stop_servers()


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark for KVcached",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=30,
        help="Number of prompts per model (default: 30)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=5.0,
        help="Requests per second (default: 5.0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="GPU memory utilization per model (default: 0.35)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model length (default: 4096)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=None,
        help="Path to virtual environment",
    )
    parser.add_argument(
        "--cuda-home",
        type=str,
        default="/usr/local/cuda",
        help="CUDA home directory",
    )
    parser.add_argument(
        "--skip-disabled",
        action="store_true",
        help="Skip benchmark with KVcached disabled",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        models=[m.strip() for m in args.models.split(",")],
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        output_dir=args.output_dir,
        venv_path=args.venv_path,
        cuda_home=args.cuda_home,
    )

    benchmark = UnifiedBenchmark(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nInterrupted. Stopping servers...")
        benchmark.stop_servers()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    benchmark.run(skip_disabled=args.skip_disabled)


if __name__ == "__main__":
    main()
