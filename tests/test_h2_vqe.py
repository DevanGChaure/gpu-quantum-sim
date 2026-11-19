"""Pytest suite for H2/STO-3G Hamiltonian construction and UCCSD-VQE accuracy."""

import pytest

from src.h2_vqe import (
    CHEMICAL_ACCURACY_MHA,
    EQUILIBRIUM_BOND_LENGTH,
    build_h2_hamiltonian,
    build_parser,
    parse_bond_lengths,
    run_vqe,
)


class TestH2Hamiltonian:
    """Tests for Jordan-Wigner H2/STO-3G Hamiltonian properties."""

    def test_pauli_term_count_is_15(self) -> None:
        """Jordan-Wigner H2/STO-3G qubit Hamiltonian has exactly 15 Pauli terms."""
        ham = build_h2_hamiltonian(EQUILIBRIUM_BOND_LENGTH, basis="sto-3g")
        assert ham.num_qubits == 4
        assert ham.num_electrons == 2
        assert ham.num_pauli_terms == 15
        assert ham.spin_hamiltonian.term_count == 15
        assert ham.fci_energy < ham.hf_energy

    def test_invalid_bond_length_raises(self) -> None:
        """Non-positive bond length raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            build_h2_hamiltonian(0.0)
        with pytest.raises(ValueError, match="positive"):
            build_h2_hamiltonian(-0.5)


class TestH2VQEAccuracy:
    """Tests for UCCSD-VQE chemical accuracy at equilibrium bond length."""

    def test_vqe_energy_at_0735_within_chemical_accuracy(self) -> None:
        """VQE energy at 0.735 A is within 1.6 mHa (0.0016 Ha) of exact FCI."""
        result = run_vqe(
            bond_length=EQUILIBRIUM_BOND_LENGTH,
            target="qpp-cpu",
            basis="sto-3g",
            maxiter=100,
            record_history=True,
        )
        assert result.num_pauli_terms == 15
        assert result.num_qubits == 4
        assert len(result.history) > 1
        # Variational principle: VQE energy cannot fall significantly below exact FCI
        assert result.vqe_energy >= result.fci_energy - 1e-6
        assert result.abs_error_mha < CHEMICAL_ACCURACY_MHA, (
            f"VQE error {result.abs_error_mha:.4f} mHa exceeds "
            f"{CHEMICAL_ACCURACY_MHA} mHa "
            f"(VQE={result.vqe_energy:.8f}, FCI={result.fci_energy:.8f})"
        )


class TestH2VQECLI:
    """Tests for the H2 VQE CLI argument parser."""

    def test_quick_flag_and_bond_lengths(self) -> None:
        """``--quick`` mode uses a small bond-length grid including 0.735 A."""
        args = build_parser().parse_args(["--quick", "--output-dir", "tmp_vqe"])
        assert args.quick is True
        assert args.output_dir == "tmp_vqe"
        lengths = parse_bond_lengths(args.bond_lengths, quick=True)
        assert len(lengths) <= 6
        assert EQUILIBRIUM_BOND_LENGTH in lengths

    def test_custom_bond_lengths_override(self) -> None:
        """Explicit ``--bond-lengths`` string overrides default grids."""
        lengths = parse_bond_lengths("0.5, 0.735, 1.2", quick=False)
        assert lengths == [0.5, 0.735, 1.2]
