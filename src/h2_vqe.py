#!/usr/bin/env python3
"""H2 Variational Quantum Eigensolver (VQE) with UCCSD ansatz using CUDA-Q.

Extracts, runs, and plots the H2/STO-3G potential energy surface (PES) and
single-point UCCSD-VQE experiments originally developed in
``notebooks/src/h2_vqe.ipynb`` and ``notebooks/src/h2_vqe_BLinc.ipynb``.

Examples
--------
Full PES scan (0.30-2.50 A in 0.05 A steps, GPU if available)::

    python -m src.h2_vqe

Quick CPU-only run (< 2 minutes)::

    python -m src.h2_vqe --quick
"""

import argparse
import csv
import os
import timeit
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

#: Chemical accuracy threshold in milli-Hartree (1.6 mHa ≈ 1 kcal/mol).
CHEMICAL_ACCURACY_MHA: float = 1.6

#: Equilibrium H-H bond length in Angstroms.
EQUILIBRIUM_BOND_LENGTH: float = 0.735

_UCCSD_KERNEL_CACHE: dict[tuple[int, int], Any] = {}


@dataclass(frozen=True)
class HamiltonianData:
    """Quantum chemistry Hamiltonian and reference energies for a geometry.

    Attributes
    ----------
    bond_length:
        H-H internuclear separation in Angstroms.
    spin_hamiltonian:
        CUDA-Q ``SpinOperator`` representation of the Jordan-Wigner Hamiltonian.
    num_qubits:
        Number of spin-orbitals (qubits) in the active basis.
    num_electrons:
        Total number of electrons.
    num_pauli_terms:
        Number of Pauli terms in the Jordan-Wigner qubit Hamiltonian.
    fci_energy:
        Exact Full Configuration Interaction (FCI) ground-state energy (Ha).
    hf_energy:
        Restricted Hartree-Fock (HF) energy (Ha).
    """

    bond_length: float
    spin_hamiltonian: Any
    num_qubits: int
    num_electrons: int
    num_pauli_terms: int
    fci_energy: float
    hf_energy: float


@dataclass(frozen=True)
class VQEResult:
    """Result of a single-geometry VQE optimization.

    Attributes
    ----------
    bond_length:
        H-H internuclear separation in Angstroms.
    vqe_energy:
        Optimized VQE ground-state energy in Hartree.
    fci_energy:
        Exact FCI ground-state energy in Hartree.
    hf_energy:
        Hartree-Fock energy in Hartree.
    optimal_parameters:
        Optimized variational angles for the UCCSD ansatz.
    history:
        Expectation values recorded at each optimizer callback iteration.
    opt_time_s:
        Wall-clock optimization time in seconds.
    num_qubits:
        Number of qubits used.
    num_pauli_terms:
        Number of Pauli terms in the spin Hamiltonian.
    """

    bond_length: float
    vqe_energy: float
    fci_energy: float
    hf_energy: float
    optimal_parameters: np.ndarray
    history: list[float]
    opt_time_s: float
    num_qubits: int
    num_pauli_terms: int

    @property
    def abs_error_mha(self) -> float:
        """Absolute VQE error relative to FCI in milli-Hartree."""
        return abs(self.vqe_energy - self.fci_energy) * 1000.0


def build_h2_hamiltonian(
    bond_length: float,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    charge: int = 0,
) -> HamiltonianData:
    """Build the Jordan-Wigner spin Hamiltonian and reference energies for H2.

    Parameters
    ----------
    bond_length:
        H-H internuclear distance in Angstroms (must be > 0).
    basis:
        Gaussian atomic orbital basis set (default ``"sto-3g"``).
    multiplicity:
        Spin multiplicity ``2S + 1`` (default ``1`` for singlet H2).
    charge:
        Net molecular charge (default ``0``).

    Returns
    -------
    HamiltonianData
        Spin operator, qubit/electron counts, Pauli term count, and FCI/HF
        reference energies.

    Raises
    ------
    ValueError
        If ``bond_length <= 0``.
    """
    if bond_length <= 0.0:
        raise ValueError(f"bond_length must be positive, got {bond_length}")

    import cudaq
    import openfermion
    import openfermionpyscf
    from openfermion.transforms import get_fermion_operator, jordan_wigner

    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, float(bond_length))),
    ]
    molecule = openfermionpyscf.run_pyscf(
        openfermion.MolecularData(geometry, basis, multiplicity, charge),
        run_fci=True,
    )
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

    spin_ham = cudaq.SpinOperator(qubit_hamiltonian)
    num_qubits = int(molecule.n_orbitals * 2)
    num_electrons = int(molecule.n_electrons)
    num_pauli_terms = len(qubit_hamiltonian.terms)

    return HamiltonianData(
        bond_length=float(bond_length),
        spin_hamiltonian=spin_ham,
        num_qubits=num_qubits,
        num_electrons=num_electrons,
        num_pauli_terms=num_pauli_terms,
        fci_energy=float(molecule.fci_energy),
        hf_energy=float(molecule.hf_energy),
    )


def compute_h_atom_asymptote(basis: str = "sto-3g") -> float:
    """Compute the dissociation limit ``2 * E_HF(H atom)`` in Hartree.

    Parameters
    ----------
    basis:
        Atomic orbital basis set (default ``"sto-3g"``).

    Returns
    -------
    float
        Twice the unrestricted/ROHF Hartree-Fock energy of an isolated H atom.
    """
    import openfermion
    import openfermionpyscf

    atom_geometry = [("H", (0.0, 0.0, 0.0))]
    h_atom = openfermionpyscf.run_pyscf(
        openfermion.MolecularData(atom_geometry, basis, multiplicity=2, charge=0)
    )
    return float(2.0 * h_atom.hf_energy)


def get_uccsd_kernel(num_qubits: int, num_electrons: int) -> Any:
    """Build (and cache) the Hartree-Fock + UCCSD CUDA-Q ansatz kernel.

    Parameters
    ----------
    num_qubits:
        Number of spin-orbital qubits.
    num_electrons:
        Number of occupied spin-orbitals in the Hartree-Fock reference state.

    Returns
    -------
    Any
        Compiled ``@cudaq.kernel`` accepting ``thetas: list[float]``.
    """
    key = (num_qubits, num_electrons)
    if key not in _UCCSD_KERNEL_CACHE:
        import cudaq

        nq = int(num_qubits)
        ne = int(num_electrons)

        @cudaq.kernel
        def uccsd_ansatz(thetas: list[float]):
            qubits = cudaq.qvector(nq)
            for i in range(ne):
                x(qubits[i])
            cudaq.kernels.uccsd(qubits, thetas, ne, nq)

        _UCCSD_KERNEL_CACHE[key] = uccsd_ansatz
    return _UCCSD_KERNEL_CACHE[key]


def run_vqe(
    bond_length: float,
    target: str = "qpp-cpu",
    basis: str = "sto-3g",
    maxiter: int = 100,
    initial_parameters: np.ndarray | None = None,
    seed: int = 42,
    record_history: bool = False,
) -> VQEResult:
    """Run UCCSD-VQE for H2 at a single bond length.

    Parameters
    ----------
    bond_length:
        H-H bond length in Angstroms.
    target:
        CUDA-Q simulation backend (e.g. ``"qpp-cpu"`` or ``"nvidia"``).
    basis:
        Basis set for PySCF integral generation (default ``"sto-3g"``).
    maxiter:
        Maximum COBYLA optimizer iterations.
    initial_parameters:
        Optional starting parameter vector for warm-starting. If omitted,
        initializes at zero (the exact Hartree-Fock reference state).
    seed:
        Random seed (reserved for reproducibility).
    record_history:
        If ``True``, records energy at each COBYLA callback step.

    Returns
    -------
    VQEResult
        Optimized energy, reference energies, parameters, history, and timing.
    """
    import cudaq

    cudaq.set_target(target)
    ham = build_h2_hamiltonian(bond_length, basis=basis)
    kernel = get_uccsd_kernel(ham.num_qubits, ham.num_electrons)
    parameter_count = int(
        cudaq.kernels.uccsd_num_parameters(ham.num_electrons, ham.num_qubits)
    )

    if initial_parameters is None:
        x0 = np.zeros(parameter_count, dtype=float)
    else:
        x0 = np.asarray(initial_parameters, dtype=float).copy()

    history: list[float] = []

    def cost(theta: np.ndarray) -> float:
        return float(
            cudaq.observe(kernel, ham.spin_hamiltonian, theta.tolist()).expectation()
        )

    def callback(xk: np.ndarray) -> None:
        if record_history:
            history.append(cost(xk))

    if record_history:
        history.append(cost(x0))

    start_time = timeit.default_timer()
    opt_result = minimize(
        cost,
        x0,
        method="COBYLA",
        callback=callback if record_history else None,
        options={"maxiter": maxiter},
    )
    elapsed = timeit.default_timer() - start_time

    return VQEResult(
        bond_length=ham.bond_length,
        vqe_energy=float(opt_result.fun),
        fci_energy=ham.fci_energy,
        hf_energy=ham.hf_energy,
        optimal_parameters=np.asarray(opt_result.x, dtype=float),
        history=history,
        opt_time_s=float(elapsed),
        num_qubits=ham.num_qubits,
        num_pauli_terms=ham.num_pauli_terms,
    )


def scan_pes(
    bond_lengths: Sequence[float],
    target: str = "qpp-cpu",
    basis: str = "sto-3g",
    maxiter: int = 100,
) -> list[VQEResult]:
    """Scan the H2 potential energy surface with warm-started UCCSD-VQE.

    Parameters
    ----------
    bond_lengths:
        Ordered sequence of H-H bond lengths in Angstroms.
    target:
        CUDA-Q backend name (e.g. ``"qpp-cpu"`` or ``"nvidia"``).
    basis:
        Atomic orbital basis set (default ``"sto-3g"``).
    maxiter:
        Maximum COBYLA iterations per geometry.

    Returns
    -------
    list[VQEResult]
        Optimization results for each bond length in ``bond_lengths``.
    """
    results: list[VQEResult] = []
    warm_params: np.ndarray | None = None

    for r in bond_lengths:
        res = run_vqe(
            bond_length=float(r),
            target=target,
            basis=basis,
            maxiter=maxiter,
            initial_parameters=warm_params,
            record_history=False,
        )
        warm_params = res.optimal_parameters.copy()
        results.append(res)
        print(
            f"  R={res.bond_length:5.3f} A  VQE={res.vqe_energy:+.8f} Ha  "
            f"FCI={res.fci_energy:+.8f} Ha  err={res.abs_error_mha:.4f} mHa  "
            f"({res.opt_time_s:.2f}s)"
        )
    return results


def write_pes_csv(path: str, results: Sequence[VQEResult]) -> None:
    """Save H2 PES scan results to a CSV file.

    Parameters
    ----------
    path:
        Destination CSV file path.
    results:
        Sequence of ``VQEResult`` objects.
    """
    header = [
        "bond_length_angstrom",
        "vqe_energy_ha",
        "fci_energy_ha",
        "hf_energy_ha",
        "abs_error_mha",
        "opt_time_s",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in results:
            writer.writerow(
                [
                    f"{r.bond_length:.4f}",
                    f"{r.vqe_energy:.10f}",
                    f"{r.fci_energy:.10f}",
                    f"{r.hf_energy:.10f}",
                    f"{r.abs_error_mha:.6f}",
                    f"{r.opt_time_s:.4f}",
                ]
            )
    print(f"CSV saved -> {path}")


def plot_pes(
    results: Sequence[VQEResult],
    asymptote: float,
    png_path: str,
) -> None:
    """Plot H2 potential energy curve (VQE, FCI, Hartree-Fock, and 2*E(H)).

    Parameters
    ----------
    results:
        PES scan results.
    asymptote:
        Dissociation limit ``2 * E_HF(H atom)`` in Hartree.
    png_path:
        Destination PNG file path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bond_lengths = [r.bond_length for r in results]
    vqe_energies = [r.vqe_energy for r in results]
    fci_energies = [r.fci_energy for r in results]
    hf_energies = [r.hf_energy for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        bond_lengths,
        vqe_energies,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="VQE (UCCSD)",
    )
    ax.plot(bond_lengths, fci_energies, linestyle="-", linewidth=1.5, label="FCI")
    ax.plot(
        bond_lengths,
        hf_energies,
        linestyle="--",
        linewidth=1.5,
        label="Hartree-Fock",
    )
    ax.axhline(
        y=asymptote,
        linestyle=":",
        linewidth=1.5,
        color="green",
        label="2 x E(H atom)",
    )
    ax.set_xlabel("H-H Bond Length (Å)")
    ax.set_ylabel("Ground State Energy (Hartree)")
    ax.set_title("H₂ Potential Energy Curve (STO-3G)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved -> {png_path}")


def plot_error(
    results: Sequence[VQEResult],
    png_path: str,
    log_scale: bool = False,
) -> None:
    """Plot absolute VQE energy error relative to FCI in milli-Hartree.

    Parameters
    ----------
    results:
        PES scan results.
    png_path:
        Destination PNG file path.
    log_scale:
        Whether to use a logarithmic y-axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bond_lengths = [r.bond_length for r in results]
    errors_mha = [max(r.abs_error_mha, 1e-9) if log_scale else r.abs_error_mha for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        bond_lengths,
        errors_mha,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="blue",
        label="|E_VQE - E_FCI|",
    )
    ax.axhline(
        y=CHEMICAL_ACCURACY_MHA,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label=f"Chemical accuracy ({CHEMICAL_ACCURACY_MHA} mHa)",
    )
    floor = 1e-6 if log_scale else 0.0
    ax.axhspan(floor, CHEMICAL_ACCURACY_MHA, color="green", alpha=0.12)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("H-H Bond Length (Å)")
    ax.set_ylabel("Absolute Energy Error (mHa)")
    title = "VQE Error Relative to FCI (Log Scale)" if log_scale else "VQE Error Relative to FCI"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved -> {png_path}")


def plot_convergence(result: VQEResult, png_path: str) -> None:
    """Plot VQE energy convergence trajectory for a single geometry.

    Parameters
    ----------
    result:
        Single-point ``VQEResult`` with populated ``history``.
    png_path:
        Destination PNG file path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.history, marker="o", markersize=3, linewidth=1.5, label="UCCSD-VQE")
    ax.axhline(
        y=result.fci_energy,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"FCI ({result.fci_energy:.6f} Ha)",
    )
    ax.set_xlabel("Optimizer Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(f"H₂ UCCSD-VQE Convergence (R = {result.bond_length:.3f} Å)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved -> {png_path}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for the H2 VQE script.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run H2/STO-3G UCCSD-VQE PES scan and regenerate all VQE plots.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced CPU-only scan (5 bond lengths including 0.735 A, < 2 min).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="CUDA-Q target (default: 'qpp-cpu' in --quick mode, else 'nvidia' if GPU available).",
    )
    parser.add_argument(
        "--basis",
        default="sto-3g",
        help="Gaussian basis set for PySCF (default: sto-3g).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Maximum COBYLA iterations per geometry.",
    )
    parser.add_argument(
        "--bond-lengths",
        type=str,
        default=None,
        help="Optional comma-separated list of bond lengths in Angstroms (e.g. '0.5,0.735,1.5').",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output CSV and PNG plots (default: results).",
    )
    return parser


def resolve_target(requested: str | None, quick: bool) -> str:
    """Resolve the CUDA-Q backend target based on CLI flags and GPU availability.

    Parameters
    ----------
    requested:
        Explicitly requested target name, or ``None`` for automatic selection.
    quick:
        Whether ``--quick`` mode was enabled.

    Returns
    -------
    str
        Resolved CUDA-Q target name.
    """
    if requested is not None:
        return requested
    if quick:
        return "qpp-cpu"

    import cudaq

    return "nvidia" if cudaq.num_available_gpus() >= 1 else "qpp-cpu"


def parse_bond_lengths(raw: str | None, quick: bool) -> list[float]:
    """Determine the list of H-H bond lengths to scan.

    Parameters
    ----------
    raw:
        Optional comma-separated bond lengths string from CLI.
    quick:
        Whether ``--quick`` mode was enabled.

    Returns
    -------
    list[float]
        Sorted list of bond lengths in Angstroms.
    """
    if raw is not None:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    if quick:
        return [0.50, 0.735, 1.00, 1.50, 2.00]
    grid = np.round(np.arange(0.30, 2.50 + 0.025, 0.05), 3).tolist()
    if EQUILIBRIUM_BOND_LENGTH not in grid:
        grid.append(EQUILIBRIUM_BOND_LENGTH)
        grid.sort()
    return grid


def main(argv: Sequence[str] | None = None) -> None:
    """Run the H2 VQE CLI workflow and regenerate all plots.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    target = resolve_target(args.target, args.quick)
    maxiter = args.maxiter if args.maxiter is not None else (60 if args.quick else 100)
    bond_lengths = parse_bond_lengths(args.bond_lengths, args.quick)

    print(
        f"Running H2 ({args.basis}) UCCSD-VQE on target={target!r} "
        f"({len(bond_lengths)} geometries, maxiter={maxiter})"
    )

    eq_result = run_vqe(
        bond_length=EQUILIBRIUM_BOND_LENGTH,
        target=target,
        basis=args.basis,
        maxiter=maxiter,
        record_history=True,
    )
    print(
        f"Equilibrium R={EQUILIBRIUM_BOND_LENGTH} A: "
        f"VQE={eq_result.vqe_energy:.8f} Ha, FCI={eq_result.fci_energy:.8f} Ha, "
        f"err={eq_result.abs_error_mha:.4f} mHa, Pauli terms={eq_result.num_pauli_terms}"
    )

    pes_results = scan_pes(
        bond_lengths=bond_lengths,
        target=target,
        basis=args.basis,
        maxiter=maxiter,
    )
    asymptote = compute_h_atom_asymptote(basis=args.basis)

    write_pes_csv(os.path.join(args.output_dir, "h2_vqe_pes.csv"), pes_results)
    plot_pes(
        pes_results,
        asymptote=asymptote,
        png_path=os.path.join(args.output_dir, "h2_pes.png"),
    )
    plot_error(
        pes_results,
        png_path=os.path.join(args.output_dir, "h2_error.png"),
        log_scale=False,
    )
    plot_error(
        pes_results,
        png_path=os.path.join(args.output_dir, "h2_error_log.png"),
        log_scale=True,
    )
    plot_convergence(
        eq_result,
        png_path=os.path.join(args.output_dir, "h2_vqe_convergence.png"),
    )


if __name__ == "__main__":
    main()
