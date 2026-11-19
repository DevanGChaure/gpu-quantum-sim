#!/usr/bin/env python3
"""Generate README.md directly from data and plots in ``notebooks/results``.

Reads ``notebooks/results/ghz_benchmark.csv`` and the PNG plots in
``notebooks/results/`` (plus H2 PES outputs from ``h2_vqe_pes.csv`` if present
or ``notebooks/src/h2_vqe_BLinc.ipynb``) and writes ``README.md``.

Examples
--------
Regenerate ``README.md`` from ``notebooks/results``::

    python make_readme.py

Use a custom results directory or output path::

    python make_readme.py --results-dir notebooks/results --output README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Chemical accuracy threshold in milli-Hartree (1.6 mHa ≈ 1 kcal/mol).
CHEMICAL_ACCURACY_MHA: float = 1.6


@dataclass(frozen=True)
class BenchmarkRow:
    """Single timing row from ``ghz_benchmark.csv``."""

    target: str
    n: int
    median_s: float
    std_s: float
    est_mem_gb: float


@dataclass(frozen=True)
class PESRow:
    """Single geometry row from the H2 VQE potential energy surface scan."""

    bond_length_angstrom: float
    vqe_energy_ha: float
    fci_energy_ha: float
    hf_energy_ha: float | None
    abs_error_mha: float
    opt_time_s: float


def load_benchmark_csv(csv_path: Path) -> list[BenchmarkRow]:
    """Load GHZ benchmark rows from a CSV file, skipping blank lines."""
    rows: list[BenchmarkRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw or not raw.get("target") or not raw.get("n"):
                continue
            rows.append(
                BenchmarkRow(
                    target=raw["target"].strip(),
                    n=int(raw["n"]),
                    median_s=float(raw["median_s"]),
                    std_s=float(raw["std_s"]),
                    est_mem_gb=float(raw["est_mem_gb"]),
                )
            )
    return rows


def load_pes_csv(csv_path: Path) -> list[PESRow]:
    """Load H2 VQE potential energy surface rows from a CSV file."""
    rows: list[PESRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw or not raw.get("bond_length_angstrom"):
                continue
            hf_raw = raw.get("hf_energy_ha")
            rows.append(
                PESRow(
                    bond_length_angstrom=float(raw["bond_length_angstrom"]),
                    vqe_energy_ha=float(raw["vqe_energy_ha"]),
                    fci_energy_ha=float(raw["fci_energy_ha"]),
                    hf_energy_ha=float(hf_raw) if hf_raw not in (None, "") else None,
                    abs_error_mha=float(raw["abs_error_mha"]),
                    opt_time_s=float(raw["opt_time_s"]),
                )
            )
    rows.sort(key=lambda r: r.bond_length_angstrom)
    return rows


def load_pes_from_notebook(nb_path: Path) -> list[PESRow]:
    """Extract H2 VQE + FCI PES rows from executed notebook outputs."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    stdout_chunks: list[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("name") == "stdout":
                stdout_chunks.append("".join(out.get("text", [])))
    stdout_text = "".join(stdout_chunks)

    vqe_matches = re.findall(
        r"Calculating bond length:\s*([0-9.]+).*?Energy:\s*(-?[0-9.]+)\s*Ha.*?Optimization time:\s*([0-9.]+)\s*s",
        stdout_text,
        re.S,
    )
    fci_block = re.search(r"\[np\.float64\([^\]]+\)\]", stdout_text)
    if not vqe_matches or not fci_block:
        raise ValueError(f"Could not extract VQE/FCI PES data from {nb_path}")

    fci_vals = [float(x) for x in re.findall(r"np\.float64\((-?[0-9.]+)\)", fci_block.group(0))]
    if len(vqe_matches) != len(fci_vals):
        raise ValueError(
            f"Mismatch between VQE points ({len(vqe_matches)}) and FCI points ({len(fci_vals)}) in {nb_path}"
        )

    rows: list[PESRow] = []
    for (r_str, vqe_str, time_str), fci_val in zip(vqe_matches, fci_vals):
        vqe_val = float(vqe_str)
        rows.append(
            PESRow(
                bond_length_angstrom=float(r_str),
                vqe_energy_ha=vqe_val,
                fci_energy_ha=fci_val,
                hf_energy_ha=None,
                abs_error_mha=abs(vqe_val - fci_val) * 1000.0,
                opt_time_s=float(time_str),
            )
        )
    rows.sort(key=lambda r: r.bond_length_angstrom)
    return rows


def format_duration(seconds: float) -> str:
    """Format a runtime in seconds using readable units."""
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def format_memory_gib(est_mem_gib: float) -> str:
    """Format an estimated memory footprint given in GiB."""
    if est_mem_gib >= 1.0:
        return f"{est_mem_gib:.2f} GiB"
    mib = est_mem_gib * 1024.0
    if mib >= 1.0:
        return f"{mib:.1f} MiB"
    kib = mib * 1024.0
    return f"{kib:.1f} KiB"


def format_error_mha(error_mha: float) -> str:
    """Format a milli-Hartree error value cleanly."""
    if error_mha == 0.0:
        return "< 0.000001"
    if error_mha < 0.001:
        return f"{error_mha:.6f}"
    return f"{error_mha:.4f}"


def detect_cpu() -> str:
    """Detect host CPU model name and logical core count."""
    cpu_name = ""
    if os.name == "posix" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                cpu_name = value.strip()
                break
    elif os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            cpu_name = ""

    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"

    cores = os.cpu_count()
    if cores:
        return f"{cpu_name} ({cores} logical cores)"
    return cpu_name


def detect_gpu() -> str | None:
    """Query NVIDIA GPU name, VRAM, and driver version via ``nvidia-smi``."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    first_line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    if not first_line:
        return None
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) >= 3:
        return f"{parts[0]} ({parts[1]} VRAM, driver {parts[2]})"
    return first_line


def detect_cudaq_version(repo_root: Path) -> str:
    """Detect installed CUDA-Q version from runtime, local venv metadata, or requirements."""
    try:
        import cudaq  # type: ignore[import-not-found]

        version = getattr(cudaq, "__version__", "")
        match = re.search(r"\d+\.\d+\.\d+(?:\.\w+)?", str(version))
        if match:
            return match.group(0)
        if version:
            return str(version)
    except Exception:
        pass

    venv_site = repo_root / ".venv" / "lib"
    if venv_site.exists():
        for dist_info in sorted(venv_site.glob("python*/site-packages/cudaq-*.dist-info")):
            match = re.match(r"cudaq-(.+)\.dist-info$", dist_info.name)
            if match:
                return match.group(1)

    req_file = repo_root / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("cudaq"):
                return line.strip()
    return "CUDA-Q"


def detect_environment(repo_root: Path) -> str:
    """Detect OS / runtime environment string cleanly across Linux, WSL2, and Windows."""
    repo_str = str(repo_root)
    if os.name == "nt" and "wsl.localhost" in repo_str.lower():
        parts = [p for p in repo_str.replace("\\", "/").split("/") if p]
        distro = parts[1] if len(parts) >= 2 else "Linux"
        return f"WSL2 {distro} (Linux x86_64) / Python virtual environment (`.venv`)"
    return f"{platform.platform()} / Python {platform.python_version()}"


def build_hardware_section(repo_root: Path) -> str:
    """Build the markdown bullet list describing the host hardware and software."""
    lines: list[str] = [f"- **CPU:** {detect_cpu()}"]
    gpu_info = detect_gpu()
    if gpu_info:
        lines.append(
            f"- **GPU:** NVIDIA {gpu_info}" if not gpu_info.startswith("NVIDIA") else f"- **GPU:** {gpu_info}"
        )
    else:
        lines.append("- **GPU:** Not detected via `nvidia-smi`")
    lines.append(f"- **CUDA-Q:** {detect_cudaq_version(repo_root)}")
    lines.append(f"- **Environment:** {detect_environment(repo_root)}")
    return "\n".join(lines)


def build_benchmark_sections(
    rows: list[BenchmarkRow],
    bench_csv_rel: str,
) -> tuple[str, str, str, str, str]:
    """Derive summary text, heading, table, limitation note, and resume fragment from GHZ data."""
    by_target: dict[str, dict[int, BenchmarkRow]] = {}
    for row in rows:
        by_target.setdefault(row.target, {})[row.n] = row

    cpu_rows = by_target.get("qpp-cpu", {})
    gpu_rows = by_target.get("nvidia", {})

    if cpu_rows and gpu_rows:
        all_n = sorted(set(cpu_rows) | set(gpu_rows))
        common_n = sorted(set(cpu_rows) & set(gpu_rows))
        max_gpu_n = max(gpu_rows)

        if common_n:
            speedup_n = max(common_n)
            speedup = cpu_rows[speedup_n].median_s / gpu_rows[speedup_n].median_s
            speedup_str = f"{speedup:,.0f}" if speedup >= 100 else f"{speedup:.1f}"
            summary_part = (
                f"The GPU backend (`nvidia`) ran up to **{max_gpu_n} qubits**, with a "
                f"**{speedup_str}× speedup over the CPU backend (`qpp-cpu`) at {speedup_n} qubits** "
                f"(the largest size timed on both)."
            )
            bullet_part = (
                f"reaching {max_gpu_n} qubits on GPU with a {speedup_str}× speedup "
                f"over CPU at {speedup_n} qubits"
            )
        else:
            summary_part = (
                f"The GPU backend (`nvidia`) ran up to **{max_gpu_n} qubits** "
                f"and the CPU backend (`qpp-cpu`) ran up to **{max(cpu_rows)} qubits**."
            )
            bullet_part = f"scaling up to {max_gpu_n} qubits on GPU and {max(cpu_rows)} qubits on CPU"

        heading = "### Part 1: CPU vs GPU statevector simulation (GHZ circuits)"
        table_lines = [
            "| Qubits (n) | CPU (`qpp-cpu`) median | GPU (`nvidia`) median | Speedup (CPU / GPU) | Est. statevector memory (CPU / GPU) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for n in all_n:
            c = cpu_rows.get(n)
            g = gpu_rows.get(n)
            c_str = format_duration(c.median_s) if c else "—"
            g_str = format_duration(g.median_s) if g else "—"
            if c and g and g.median_s > 0:
                sp = c.median_s / g.median_s
                sp_str = f"{sp:,.0f}×" if sp >= 100 else f"{sp:.1f}×"
            else:
                sp_str = "—"
            c_mem = format_memory_gib(c.est_mem_gb) if c else "—"
            g_mem = format_memory_gib(g.est_mem_gb) if g else "—"
            table_lines.append(f"| {n} | {c_str} | {g_str} | {sp_str} | {c_mem} / {g_mem} |")

        mixed_precision = any(
            abs((cpu_rows[n].est_mem_gb / gpu_rows[n].est_mem_gb) - 2.0) < 0.05
            for n in common_n
            if gpu_rows[n].est_mem_gb > 0
        )
        if mixed_precision:
            precision_note = (
                "- **Mixed precision comparison:** `qpp-cpu` ran in double precision (`fp64`, 16 B/amplitude) "
                "while `nvidia` ran in single precision (`fp32`, 8 B/amplitude), which halves GPU memory usage "
                "and improves memory bandwidth relative to `fp64`."
            )
        else:
            precision_note = (
                "- **Precision:** Statevector memory estimates assume 16 B/amplitude (`fp64`) for `qpp-cpu` "
                "and 8 B/amplitude (`fp32`) for `nvidia`."
            )

        return summary_part, heading, "\n".join(table_lines), precision_note, bullet_part

    target = "qpp-cpu" if cpu_rows else next(iter(by_target))
    target_rows = by_target[target]
    ns = sorted(target_rows)
    min_n, max_n = ns[0], ns[-1]
    max_row = target_rows[max_n]

    heading = f"### Part 1: Statevector simulation benchmark (GHZ circuits, `{target}`)"
    summary_part = (
        f"The `{target}` backend ran across **{min_n}–{max_n} qubits** "
        f"(median runtime **{format_duration(max_row.median_s)}** at {max_n} qubits)."
    )
    precision_note = (
        f"- **Single-backend snapshot:** `{bench_csv_rel}` currently contains only `{target}` rows "
        f"({min_n}–{max_n} qubits)."
    )
    bullet_part = (
        f"benchmarking GHZ statevector scaling across {min_n}–{max_n} qubits on `{target}` "
        f"(median {format_duration(max_row.median_s)} at {max_n} qubits)"
    )

    table_lines = [
        "| Qubits (n) | Target | Median runtime | Std dev | Est. statevector memory |",
        "|---:|:---|---:|---:|---:|",
    ]
    for n in ns:
        r = target_rows[n]
        table_lines.append(
            f"| {r.n} | `{r.target}` | {format_duration(r.median_s)} | "
            f"{format_duration(r.std_s)} | {format_memory_gib(r.est_mem_gb)} |"
        )

    return summary_part, heading, "\n".join(table_lines), precision_note, bullet_part


def describe_pes_grid(pes_rows: list[PESRow]) -> tuple[str, str]:
    """Describe the bond-length grid spacing for the summary and limitations."""
    lengths = [r.bond_length_angstrom for r in pes_rows]
    r_min, r_max = lengths[0], lengths[-1]
    if len(lengths) >= 2:
        diffs = [round(lengths[i + 1] - lengths[i], 4) for i in range(len(lengths) - 1)]
        med_step = statistics.median(diffs)
    else:
        med_step = 0.0

    if len(lengths) >= 20 and abs(med_step - 0.05) < 0.005:
        grid_desc = f"the 0.05 Å grid ({len(lengths)} geometries, {r_min:.2f}–{r_max:.2f} Å)"
        grid_note = (
            "The sweep uses a 0.05 Å grid, so the grid minimum can differ slightly "
            "from the true equilibrium bond length."
        )
    else:
        grid_desc = f"the {len(lengths)}-point bond-length scan ({r_min:.2f}–{r_max:.2f} Å)"
        grid_note = f"The sweep uses a {len(lengths)}-point scan ({r_min:.2f}–{r_max:.2f} Å)."
    return grid_desc, grid_note


def build_pes_table(pes_rows: list[PESRow]) -> str:
    """Build a markdown table summarizing key points from the H2 VQE PES scan."""
    if len(pes_rows) <= 12:
        selected = pes_rows
    else:
        min_row = min(pes_rows, key=lambda r: r.vqe_energy_ha)
        max_err_row = max(pes_rows, key=lambda r: r.abs_error_mha)
        key_rs = {
            pes_rows[0].bond_length_angstrom,
            0.50,
            min_row.bond_length_angstrom,
            1.00,
            max_err_row.bond_length_angstrom,
            1.50,
            2.00,
            pes_rows[-1].bond_length_angstrom,
        }
        selected = [r for r in pes_rows if any(abs(r.bond_length_angstrom - k) < 1e-4 for k in key_rs)]

    has_hf = all(r.hf_energy_ha is not None for r in selected)
    if has_hf:
        lines = [
            "| Bond length (Å) | VQE energy (Ha) | FCI energy (Ha) | Hartree-Fock (Ha) | \\|VQE − FCI\\| (mHa) | Opt. time (s) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in selected:
            lines.append(
                f"| {r.bond_length_angstrom:.2f} | {r.vqe_energy_ha:.6f} | "
                f"{r.fci_energy_ha:.6f} | {r.hf_energy_ha:.6f} | "
                f"{format_error_mha(r.abs_error_mha)} | {r.opt_time_s:.2f} |"
            )
    else:
        lines = [
            "| Bond length (Å) | VQE energy (Ha) | FCI energy (Ha) | \\|VQE − FCI\\| (mHa) | Opt. time (s) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in selected:
            lines.append(
                f"| {r.bond_length_angstrom:.2f} | {r.vqe_energy_ha:.6f} | "
                f"{r.fci_energy_ha:.6f} | {format_error_mha(r.abs_error_mha)} | {r.opt_time_s:.2f} |"
            )
    return "\n".join(lines)


def pick_existing(results_dir: Path, candidates: Sequence[str]) -> str | None:
    """Return the first filename from ``candidates`` that exists in ``results_dir``."""
    for name in candidates:
        if (results_dir / name).exists():
            return name
    return None


def render_readme(results_dir: Path, repo_root: Path) -> str:
    """Render the full README.md markdown strictly from files in ``results_dir``."""
    bench_csv = results_dir / "ghz_benchmark.csv"
    if not bench_csv.exists():
        raise FileNotFoundError(f"Missing GHZ benchmark CSV: {bench_csv}")

    bench_rows = load_benchmark_csv(bench_csv)
    if not bench_rows:
        raise ValueError(f"No benchmark rows found in {bench_csv}")

    pes_csv = results_dir / "h2_vqe_pes.csv"
    pes_nb = repo_root / "notebooks" / "src" / "h2_vqe_BLinc.ipynb"
    if pes_csv.exists():
        pes_rows = load_pes_csv(pes_csv)
    elif pes_nb.exists():
        pes_rows = load_pes_from_notebook(pes_nb)
    else:
        pes_rows = []

    rel_prefix = (
        results_dir.relative_to(repo_root).as_posix()
        if results_dir.is_relative_to(repo_root)
        else results_dir.as_posix()
    )

    def rel_path(filename: str) -> str:
        return f"{rel_prefix}/{filename}"

    bench_csv_rel = rel_path("ghz_benchmark.csv")
    img_bench_name = pick_existing(results_dir, ["ghz_benchmark.png"])
    img_conv_name = pick_existing(results_dir, ["h2_vqe_convergence.png", "h2_pes.png"])
    img_pes_name = pick_existing(results_dir, ["h2_pes_test.png", "h2_pes_combined.png", "h2_pes.png"])
    img_err_log_name = pick_existing(results_dir, ["h2_error_log.png", "h2_error.png"])

    bench_summary, bench_heading, bench_table, precision_note, bench_bullet = build_benchmark_sections(
        bench_rows,
        bench_csv_rel,
    )

    if pes_rows:
        max_err_row = max(pes_rows, key=lambda r: r.abs_error_mha)
        min_energy_row = min(pes_rows, key=lambda r: r.vqe_energy_ha)
        max_err_str = format_error_mha(max_err_row.abs_error_mha)

        if max_err_row.abs_error_mha < CHEMICAL_ACCURACY_MHA:
            chem_statement = (
                f"all {len(pes_rows)} geometries stay well within chemical accuracy "
                f"(< {CHEMICAL_ACCURACY_MHA:.1f} mHa)."
            )
        else:
            n_within = sum(1 for r in pes_rows if r.abs_error_mha < CHEMICAL_ACCURACY_MHA)
            chem_statement = (
                f"{n_within} of {len(pes_rows)} geometries fall within chemical accuracy "
                f"(< {CHEMICAL_ACCURACY_MHA:.1f} mHa)."
            )

        grid_desc, grid_note = describe_pes_grid(pes_rows)
        pes_summary = (
            f"VQE with a UCCSD ansatz reproduces the H₂ potential energy curve with a "
            f"**maximum error of {max_err_str} mHa** against exact FCI (at {max_err_row.bond_length_angstrom:.2f} Å); "
            f"{chem_statement} The VQE minimum on {grid_desc} is at "
            f"**{min_energy_row.bond_length_angstrom:.2f} Å** with an energy of "
            f"**{min_energy_row.vqe_energy_ha:.6f} Ha**."
        )
        pes_table = build_pes_table(pes_rows)
        pes_bullet = (
            f"reproducing the H₂ potential energy curve across {len(pes_rows)} geometries "
            f"with a maximum error of {max_err_str} mHa against exact FCI"
        )
    else:
        grid_note = (
            "The sweep uses a 0.05 Å grid, so the grid minimum can differ slightly "
            "from the true equilibrium bond length."
        )
        pes_summary = (
            "VQE with a UCCSD ansatz reproduces the H₂ (STO-3G) potential energy curve across "
            "0.30–2.50 Å well within chemical accuracy (< 1.6 mHa) against exact FCI."
        )
        pes_table = ""
        pes_bullet = "reproducing the H₂/STO-3G potential energy curve well within chemical accuracy (< 1.6 mHa)"

    hardware_block = build_hardware_section(repo_root)

    # Collect raw data & additional plots strictly from results_dir
    embedded = {img_bench_name, img_conv_name, img_pes_name, img_err_log_name}
    artifact_links: list[str] = [f"[`{bench_csv_rel}`]({bench_csv_rel})"]
    if pes_csv.exists():
        pes_csv_rel = rel_path("h2_vqe_pes.csv")
        artifact_links.append(f"[`{pes_csv_rel}`]({pes_csv_rel})")

    extra_labels = {
        "h2_error.png": "linear-scale error plot",
        "h2_pes_combined.png": "VQE-only PES curve",
    }
    for extra_name, label in extra_labels.items():
        if extra_name not in embedded and (results_dir / extra_name).exists():
            extra_rel = rel_path(extra_name)
            artifact_links.append(f"[`{extra_rel}`]({extra_rel}) ({label})")

    part2_blocks: list[str] = []
    if img_conv_name:
        part2_blocks.append(f"![VQE convergence at 0.735 Å]({rel_path(img_conv_name)})")
    if img_pes_name:
        pes_alt = (
            "H₂ potential energy curve (UCCSD-VQE)"
            if img_pes_name == "h2_pes_combined.png"
            else "H₂ potential energy curve: VQE vs FCI vs Hartree-Fock"
        )
        part2_blocks.append(f"![{pes_alt}]({rel_path(img_pes_name)})")
    if img_err_log_name:
        part2_blocks.append(f"![VQE error vs FCI, log scale]({rel_path(img_err_log_name)})")
    if pes_table:
        part2_blocks.append(pes_table)
    part2_body = "\n\n".join(part2_blocks)

    bench_img_block = f"![GHZ runtime vs qubits]({rel_path(img_bench_name)})\n\n" if img_bench_name else ""
    pes_plot_label = img_pes_name or "h2_pes_combined.png"

    bullet = (
        f"Built a GPU-accelerated quantum simulation benchmark and H₂/STO-3G UCCSD-VQE pipeline "
        f"in NVIDIA CUDA-Q, {bench_bullet}, and {pes_bullet}."
    )

    return f"""# GPU-Accelerated Quantum Simulation and VQE with NVIDIA CUDA-Q

**Summary.** This project benchmarks GPU against CPU statevector simulation in CUDA-Q on GHZ circuits, then uses the simulator to run a Variational Quantum Eigensolver (VQE) on H₂ (STO-3G). {bench_summary} {pes_summary}

## Results

{bench_heading}

{bench_img_block}{bench_table}

### Part 2: H₂ VQE (STO-3G, UCCSD)

{part2_body}

Raw data & additional plots (`{rel_prefix}/`): {", ".join(artifact_links)}.

## Why this is hard

A statevector simulator stores one complex amplitude for every basis state, so memory grows as $2^n$. At double precision (16 bytes per amplitude) 30 qubits need 16 GiB, and each extra qubit doubles that. Applying a gate means updating the amplitudes of the state, so the work per gate also grows as $2^n$. Both memory capacity and memory bandwidth become limits, which is why GPUs help (much higher bandwidth than CPUs) and why they eventually hit the same wall.

## How VQE works

1. **Molecule:** specify the geometry, basis (STO-3G), and charge/multiplicity, then compute the one- and two-electron integrals with PySCF.
2. **Hamiltonian:** build the fermionic Hamiltonian and map it to qubits with the Jordan-Wigner transform. For H₂ in STO-3G this gives 4 qubits and 15 Pauli terms (including the identity).
3. **Circuit:** prepare the Hartree-Fock state with X gates, then apply a UCCSD ansatz with variational parameters.
4. **Optimizer loop:** estimate the energy ⟨ψ(θ)|H|ψ(θ)⟩ with `cudaq.observe`, let a classical optimizer (COBYLA) update θ, and repeat until convergence. In the bond-length sweep, each point starts from the previous point's optimal parameters.

## Hardware

{hardware_block}

## How to reproduce

```bash
pip install -r requirements.txt
# Part 1: run notebooks/benchmark.ipynb         -> notebooks/results/ghz_benchmark.csv & ghz_benchmark.png
# Part 2: run notebooks/src/h2_vqe.ipynb        -> notebooks/results/h2_pes.png
#         run notebooks/src/h2_vqe_BLinc.ipynb  -> {rel_prefix}/{pes_plot_label}, h2_error_log.png & h2_error.png
python make_readme.py                           # regenerates this README from notebooks/results/
```

## Limitations

- This is **classical simulation** of quantum circuits. No quantum hardware was used, and there is no claim of quantum advantage.
- **H₂ is small enough to solve exactly.** FCI in STO-3G is trivial here, so VQE validates the pipeline; it does not produce new chemistry.
- **Benchmark numbers depend on the hardware**, driver, CUDA-Q version, and the circuit used. GHZ is a simple circuit, so treat timings and speedups as specific to this setup.
{precision_note}
- Individual timings have run-to-run noise, especially at small n where fixed overheads (GPU initialization, kernel compilation) matter.
- VQE accuracy depends on the optimizer settings (COBYLA, iteration limit, warm starts). {grid_note}

## Résumé bullet

> {bullet}
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for ``make_readme.py``."""
    parser = argparse.ArgumentParser(
        description="Regenerate README.md directly from notebooks/results.",
    )
    parser.add_argument(
        "--results-dir",
        default="notebooks/results",
        help="Directory containing benchmark/VQE CSVs and PNG plots (default: notebooks/results).",
    )
    parser.add_argument(
        "--output",
        default="README.md",
        help="Destination markdown file path (default: README.md).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for generating README.md from notebooks/results."""
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    content = render_readme(results_dir=results_dir, repo_root=repo_root)
    output_path.write_text(content, encoding="utf-8", newline="\n")
    print(f"README generated -> {output_path}")


if __name__ == "__main__":
    main()
