#!/usr/bin/env python3
"""GHZ-state benchmark comparing CPU (``qpp-cpu``) and GPU (``nvidia``) simulation.

Extracts, runs, and plots the GHZ scaling benchmark originally developed in
``notebooks/benchmark.ipynb``. Pure helper functions (memory estimation,
fit checking, timing) have no ``cudaq`` import at module load time so they can
be unit-tested independently of a quantum runtime or GPU.

Examples
--------
Full CPU + GPU benchmark::

    python -m src.benchmark

Quick CPU-only run (< 2 minutes)::

    python -m src.benchmark --quick
"""

import argparse
import csv
import os
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any

#: Bytes consumed per statevector amplitude for each floating-point precision.
BYTES_PER_AMPLITUDE: dict[str, int] = {"fp32": 8, "fp64": 16}

_CSV_HEADER: list[str] = ["target", "n", "median_s", "std_s", "est_mem_gb"]
_GHZ_KERNEL: Any = None


def estimate_memory_bytes(n_qubits: int, precision: str) -> int:
    """Estimate statevector memory footprint in bytes.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the quantum register (must be non-negative).
    precision:
        Floating-point precision: ``"fp32"`` (complex64, 8 B/amplitude) or
        ``"fp64"`` (complex128, 16 B/amplitude).

    Returns
    -------
    int
        Estimated statevector size in bytes: ``2**n_qubits * bytes_per_amp``.

    Raises
    ------
    ValueError
        If ``n_qubits`` is negative or ``precision`` is unsupported.
    """
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be non-negative, got {n_qubits}")
    if precision not in BYTES_PER_AMPLITUDE:
        supported = ", ".join(sorted(BYTES_PER_AMPLITUDE))
        raise ValueError(
            f"Unsupported precision {precision!r}; expected one of: {supported}"
        )
    return (2**n_qubits) * BYTES_PER_AMPLITUDE[precision]


def fits(
    n_qubits: int,
    precision: str,
    available_bytes: int,
    safety: float = 0.8,
) -> bool:
    """Return whether an ``n_qubits`` statevector fits within available memory.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the circuit.
    precision:
        ``"fp32"`` or ``"fp64"``.
    available_bytes:
        Available RAM or VRAM in bytes (must be non-negative).
    safety:
        Maximum fraction of ``available_bytes`` allowed (must be in ``(0, 1]``).

    Returns
    -------
    bool
        ``True`` if ``estimate_memory_bytes(n_qubits, precision) <= safety * available_bytes``.

    Raises
    ------
    ValueError
        If ``available_bytes < 0`` or ``safety`` is outside ``(0, 1]``.
    """
    if available_bytes < 0:
        raise ValueError(f"available_bytes must be non-negative, got {available_bytes}")
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"safety must be in (0, 1], got {safety}")
    return estimate_memory_bytes(n_qubits, precision) <= safety * available_bytes


def time_runs(
    fn: Callable[[], object],
    warmup: int = 1,
    repeats: int = 5,
) -> tuple[float, float]:
    """Benchmark a zero-argument callable with warmup and repeated timings.

    Parameters
    ----------
    fn:
        Zero-argument callable to time.
    warmup:
        Number of untimed warmup executions.
    repeats:
        Number of timed executions (must be >= 1).

    Returns
    -------
    tuple[float, float]
        ``(median_seconds, stdev_seconds)``. When ``repeats == 1``,
        ``stdev_seconds`` is ``0.0``.

    Raises
    ------
    ValueError
        If ``warmup < 0`` or ``repeats < 1``.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    median_s = statistics.median(times)
    std_s = statistics.stdev(times) if len(times) >= 2 else 0.0
    return median_s, std_s


def get_ghz_kernel() -> Any:
    """Build (once) and return the compiled CUDA-Q GHZ circuit kernel.

    Returns
    -------
    Any
        CUDA-Q ``@cudaq.kernel`` preparing an ``n``-qubit GHZ state and
        measuring all qubits in the computational basis.
    """
    global _GHZ_KERNEL
    if _GHZ_KERNEL is None:
        import cudaq

        @cudaq.kernel
        def ghz(qubit_count: int):
            q = cudaq.qvector(qubit_count)
            h(q[0])
            for i in range(qubit_count - 1):
                x.ctrl(q[i], q[i + 1])
            mz(q)

        _GHZ_KERNEL = ghz
    return _GHZ_KERNEL


def sample_ghz(
    qubit_count: int,
    shots: int = 100,
    target: str | None = None,
) -> Any:
    """Sample an ``n``-qubit GHZ state using CUDA-Q.

    Parameters
    ----------
    qubit_count:
        Number of qubits (must be >= 1).
    shots:
        Number of measurement shots (must be >= 1).
    target:
        Optional CUDA-Q backend name (e.g. ``"qpp-cpu"`` or ``"nvidia"``).
        If omitted, uses the currently active CUDA-Q target.

    Returns
    -------
    Any
        ``cudaq.SampleResult`` containing bitstring counts.

    Raises
    ------
    ValueError
        If ``qubit_count < 1`` or ``shots < 1``.
    """
    if qubit_count < 1:
        raise ValueError(f"qubit_count must be >= 1, got {qubit_count}")
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")

    import cudaq

    if target is not None:
        cudaq.set_target(target)
    ghz = get_ghz_kernel()
    return cudaq.sample(ghz, qubit_count, shots_count=shots)


def verify_ghz(
    qubit_count: int = 8,
    shots: int = 1000,
    target: str = "qpp-cpu",
    tolerance: float = 0.1,
) -> tuple[int, int]:
    """Run a statistical correctness sanity check on the GHZ circuit.

    Parameters
    ----------
    qubit_count:
        Number of qubits to entangle.
    shots:
        Number of measurement shots.
    target:
        CUDA-Q backend to test.
    tolerance:
        Maximum allowed deviation of the all-zeros fraction from ``0.5``.

    Returns
    -------
    tuple[int, int]
        ``(zeros_count, ones_count)`` observed in the sample.

    Raises
    ------
    AssertionError
        If unexpected bitstrings appear or counts deviate beyond ``tolerance``.
    """
    counts = sample_ghz(qubit_count, shots=shots, target=target)
    expected_zero = "0" * qubit_count
    expected_one = "1" * qubit_count
    observed = set(counts)
    assert observed <= {expected_zero, expected_one}, (
        f"Unexpected GHZ bitstrings on {target}: {observed}"
    )
    zeros = counts.count(expected_zero)
    ones = counts.count(expected_one)
    total = zeros + ones
    assert total == shots, f"Total shots mismatch: {total} != {shots}"
    assert abs(zeros / total - 0.5) < tolerance, (
        f"GHZ distribution skewed on {target}: zeros={zeros}, ones={ones}"
    )
    return zeros, ones


def free_vram_bytes() -> int:
    """Query free GPU memory in bytes via ``nvidia-smi``.

    Returns
    -------
    int
        Free VRAM of GPU 0 in bytes.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0]) * (2**20)


def benchmark(
    target: str,
    precision: str,
    qubit_range: Sequence[int],
    available_bytes: int,
    shots: int = 100,
    warmup: int = 1,
    repeats: int = 5,
) -> list[tuple[str, int, float, float, float]]:
    """Run GHZ runtime benchmarks across a sequence of qubit counts.

    Parameters
    ----------
    target:
        CUDA-Q backend name (e.g. ``"qpp-cpu"`` or ``"nvidia"``).
    precision:
        ``"fp32"`` or ``"fp64"``.
    qubit_range:
        Qubit counts to benchmark.
    available_bytes:
        Available memory (RAM or VRAM) in bytes used by ``fits``.
    shots:
        Number of shots per circuit execution.
    warmup:
        Number of untimed warmup iterations.
    repeats:
        Number of timed repetitions.

    Returns
    -------
    list[tuple[str, int, float, float, float]]
        Rows of ``(target, n, median_s, std_s, est_mem_gb)``.
    """
    import cudaq

    cudaq.set_target(target)
    ghz = get_ghz_kernel()
    rows: list[tuple[str, int, float, float, float]] = []

    for n in qubit_range:
        if not fits(n, precision, available_bytes):
            needed_gib = estimate_memory_bytes(n, precision) / (2**30)
            print(f"skip n={n} on {target}: needs {needed_gib:.2f} GiB")
            continue

        median_s, std_s = time_runs(
            lambda _n=n: cudaq.sample(ghz, _n, shots_count=shots),
            warmup=warmup,
            repeats=repeats,
        )
        est_mem_gb = estimate_memory_bytes(n, precision) / (2**30)
        rows.append((target, n, median_s, std_s, est_mem_gb))
        print(
            f"  {target:<8s} n={n:>2d}  median={median_s:.4f}s  "
            f"std={std_s:.4f}s  mem={est_mem_gb:.4f} GiB"
        )
    return rows


def write_csv(
    path: str,
    rows: Sequence[tuple[str, int, float, float, float]],
) -> None:
    """Write benchmark results to a CSV file.

    Parameters
    ----------
    path:
        Destination CSV file path.
    rows:
        Sequence of ``(target, n, median_s, std_s, est_mem_gb)`` tuples.
    """
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        writer.writerows(rows)
    print(f"CSV saved -> {path}")


def plot_benchmark(csv_path: str, png_path: str) -> None:
    """Plot GHZ median runtime with error bars vs qubit count on a log scale.

    Parameters
    ----------
    csv_path:
        Path to the benchmark results CSV file.
    png_path:
        Destination path for the saved PNG figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target_name, group in df.groupby("target"):
        ax.errorbar(
            group["n"],
            group["median_s"],
            yerr=group["std_s"],
            marker="o",
            capsize=3,
            label=str(target_name),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits (n)")
    ax.set_ylabel("Median runtime (s)")
    ax.set_title("GHZ statevector simulation: runtime vs qubits")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved -> {png_path}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for the GHZ benchmark.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark GHZ statevector simulation on CPU and GPU.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced CPU-only benchmark (qubits 8-14, < 2 min).",
    )
    parser.add_argument(
        "--target",
        choices=["qpp-cpu", "nvidia", "both"],
        default="both",
        help="Simulation target(s) to benchmark (default: both).",
    )
    parser.add_argument(
        "--min-qubits",
        type=int,
        default=None,
        help="Minimum qubit count to benchmark.",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Maximum qubit count to benchmark.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Number of measurement shots per sample call.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup runs before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of timed repetitions per qubit count.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output CSV and PNG files (default: results).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the GHZ benchmark CLI workflow.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    import cudaq
    import psutil

    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "ghz_benchmark.csv")
    png_path = os.path.join(args.output_dir, "ghz_benchmark.png")

    if args.quick:
        min_q = args.min_qubits if args.min_qubits is not None else 8
        max_q = args.max_qubits if args.max_qubits is not None else 14
        shots = args.shots if args.shots is not None else 100
        warmup = args.warmup if args.warmup is not None else 1
        repeats = args.repeats if args.repeats is not None else 3
        targets = ["qpp-cpu"]
    else:
        min_q = args.min_qubits if args.min_qubits is not None else 20
        max_q = args.max_qubits if args.max_qubits is not None else 27
        shots = args.shots if args.shots is not None else 100
        warmup = args.warmup if args.warmup is not None else 4
        repeats = args.repeats if args.repeats is not None else 5
        if args.target == "both":
            targets = ["qpp-cpu", "nvidia"]
        else:
            targets = [args.target]

    qubit_range = list(range(min_q, max_q + 1))
    all_rows: list[tuple[str, int, float, float, float]] = []

    for tgt in targets:
        if tgt == "nvidia":
            if cudaq.num_available_gpus() < 1:
                if args.target == "nvidia":
                    raise RuntimeError("Target 'nvidia' requested, but no GPU is visible to CUDA-Q.")
                print("Warning: no GPU visible to CUDA-Q; skipping 'nvidia' target.")
                continue
            avail = free_vram_bytes()
            prec = "fp32"
        else:
            avail = psutil.virtual_memory().available
            prec = "fp64"

        zeros, ones = verify_ghz(qubit_count=8, shots=1000, target=tgt)
        print(f"GHZ correctness check on {tgt}: |0^8>={zeros}, |1^8>={ones}")
        print(f"\n--- {tgt} (precision={prec}, avail={avail / 2**30:.1f} GiB) ---")
        rows = benchmark(tgt, prec, qubit_range, avail, shots, warmup, repeats)
        all_rows.extend(rows)

    write_csv(csv_path, all_rows)
    plot_benchmark(csv_path, png_path)


if __name__ == "__main__":
    main()
