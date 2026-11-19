"""Pytest suite for GHZ circuit correctness and memory estimation utilities."""

import pytest

from src.benchmark import (
    BYTES_PER_AMPLITUDE,
    build_parser,
    estimate_memory_bytes,
    fits,
    sample_ghz,
    time_runs,
    verify_ghz,
)


class TestMemoryEstimator:
    """Tests for ``estimate_memory_bytes`` and ``fits``."""

    @pytest.mark.parametrize(
        ("n_qubits", "precision", "expected_bytes"),
        [
            (0, "fp32", 8),
            (0, "fp64", 16),
            (1, "fp32", 16),
            (1, "fp64", 32),
            (10, "fp32", 1024 * 8),
            (10, "fp64", 1024 * 16),
            (20, "fp32", (2**20) * 8),
            (20, "fp64", (2**20) * 16),
            (26, "fp64", 2**30),  # exactly 1 GiB
            (27, "fp32", 2**30),  # exactly 1 GiB
        ],
    )
    def test_estimate_memory_bytes_exact(
        self, n_qubits: int, precision: str, expected_bytes: int
    ) -> None:
        """Memory estimate matches 2**n * bytes_per_amplitude."""
        assert estimate_memory_bytes(n_qubits, precision) == expected_bytes

    def test_fp64_is_double_fp32(self) -> None:
        """Double precision requires exactly twice the memory of single precision."""
        for n in (4, 12, 24):
            assert estimate_memory_bytes(n, "fp64") == 2 * estimate_memory_bytes(
                n, "fp32"
            )

    def test_estimate_memory_invalid_inputs(self) -> None:
        """Negative qubit count or unknown precision raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            estimate_memory_bytes(-1, "fp32")
        with pytest.raises(ValueError, match="Unsupported precision"):
            estimate_memory_bytes(10, "fp16")

    def test_fits_boundary_conditions(self) -> None:
        """``fits`` enforces the safety margin accurately on boundary sizes."""
        # 26 qubits fp64 = 1 GiB = 1024 MiB. With safety=0.8, needs >= 1.25 GiB.
        one_gib = 2**30
        assert not fits(26, "fp64", one_gib, safety=0.8)
        assert fits(26, "fp64", int(one_gib / 0.8), safety=0.8)
        assert fits(26, "fp64", one_gib, safety=1.0)

    def test_fits_invalid_inputs(self) -> None:
        """Invalid available_bytes or safety factor raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            fits(10, "fp32", -100)
        with pytest.raises(ValueError, match="safety"):
            fits(10, "fp32", 1024, safety=0.0)
        with pytest.raises(ValueError, match="safety"):
            fits(10, "fp32", 1024, safety=1.5)

    def test_bytes_per_amplitude_constants(self) -> None:
        """Complex64 is 8 bytes and complex128 is 16 bytes."""
        assert BYTES_PER_AMPLITUDE == {"fp32": 8, "fp64": 16}


class TestTimingHelper:
    """Tests for ``time_runs``."""

    def test_time_runs_counts_warmup_and_repeats(self) -> None:
        """Executes exactly warmup + repeats calls and returns non-negative stats."""
        counter = {"calls": 0}

        def work() -> None:
            counter["calls"] += 1

        median_s, std_s = time_runs(work, warmup=2, repeats=4)
        assert counter["calls"] == 6
        assert median_s >= 0.0
        assert std_s >= 0.0

    def test_time_runs_single_repeat_zero_stdev(self) -> None:
        """Single timed repetition returns 0.0 standard deviation."""
        median_s, std_s = time_runs(lambda: None, warmup=0, repeats=1)
        assert median_s >= 0.0
        assert std_s == 0.0

    def test_time_runs_invalid_arguments(self) -> None:
        """Negative warmup or repeats < 1 raises ValueError."""
        with pytest.raises(ValueError, match="warmup"):
            time_runs(lambda: None, warmup=-1, repeats=2)
        with pytest.raises(ValueError, match="repeats"):
            time_runs(lambda: None, warmup=0, repeats=0)


class TestGHZCorrectness:
    """Tests for GHZ quantum state preparation on ``qpp-cpu``."""

    @pytest.mark.parametrize("n_qubits", [2, 4, 8])
    def test_ghz_only_all_zero_and_all_one_bitstrings(self, n_qubits: int) -> None:
        """GHZ circuit produces only |0...0> and |1...1> with ~50/50 probabilities."""
        shots = 2000
        counts = sample_ghz(n_qubits, shots=shots, target="qpp-cpu")
        zero_str = "0" * n_qubits
        one_str = "1" * n_qubits

        observed = set(counts)
        assert observed <= {zero_str, one_str}, f"Unexpected states: {observed}"

        zeros = counts.count(zero_str)
        ones = counts.count(one_str)
        assert zeros + ones == shots
        assert abs(zeros / shots - 0.5) < 0.08, (
            f"Expected ~50/50 split for n={n_qubits}, got zeros={zeros}, ones={ones}"
        )

    def test_verify_ghz_helper(self) -> None:
        """``verify_ghz`` succeeds on 8-qubit CPU simulation."""
        zeros, ones = verify_ghz(qubit_count=8, shots=1000, target="qpp-cpu")
        assert zeros + ones == 1000
        assert zeros > 0 and ones > 0


class TestBenchmarkCLI:
    """Tests for the benchmark CLI argument parser."""

    def test_quick_flag_parsed(self) -> None:
        """Parser accepts ``--quick`` and ``--output-dir``."""
        args = build_parser().parse_args(["--quick", "--output-dir", "tmp_out"])
        assert args.quick is True
        assert args.output_dir == "tmp_out"
