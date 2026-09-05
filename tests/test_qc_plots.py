"""Tests for raw calculation plots and exports."""

import numpy as np

from iqc_dashboard.qc_models import CalculationRecord
from iqc_dashboard.qc_plots import (
    atomic_charge_table,
    convergence_figure,
    coordinates_to_xyz,
    electronic_transition_table,
    ir_spectrum_figure,
    molecular_formula,
    optimization_energy_figure,
    orbital_table,
    vibration_table,
    vibration_to_xyz,
)


def record():
    return CalculationRecord(
        "id", "water.out", 100, "sha", "output",
        atom_numbers=np.array([8, 1, 1]),
        coordinates=np.array(
            [
                [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0.98], [0, 0.98, 0]],
            ]
        ),
        scf_energies_ev=np.array([-10.0, -10.2]),
        geometry_values=np.array([[2.0, 3.0], [0.5, 0.8]]),
        geometry_targets=np.array([1.0, 1.0]),
        frequencies_cm1=np.array([-20.0, 1000.0, 3500.0]),
        ir_intensities=np.array([1.0, 20.0, 50.0]),
        vibration_displacements=np.ones((3, 3, 3)) * 0.01,
        vibration_symmetries=["A", "A", "B"],
        atomic_charges={"mulliken": np.array([-0.8, 0.4, 0.4])},
        mo_energies_ev=[np.array([-10.0, -5.0, 1.0])],
        homos=np.array([1]),
        electronic_transitions_cm1=np.array([20_000.0, 25_000.0]),
        oscillator_strengths=np.array([0.1, 0.2]),
    )


def test_formula_xyz_and_vibration_animation_export():
    item = record()
    assert molecular_formula(item.atom_numbers) == "H2O"
    xyz = coordinates_to_xyz(item.atom_numbers, item.coordinates[-1], "final")
    assert xyz.splitlines()[0] == "3"
    animation = vibration_to_xyz(item, 1, frame_count=10)
    assert animation.count("Mode 2") == 10


def test_optimization_and_convergence_figures_have_scientific_axes():
    item = record()
    energy = optimization_energy_figure(item, "kcal/mol")
    assert energy.layout.yaxis.title.text.endswith("(kcal/mol)")
    assert np.isclose(energy.data[0].y[-1], -0.2 * 23.0605)
    convergence = convergence_figure(item)
    assert convergence.layout.yaxis.type == "log"
    assert len(convergence.data) == 2


def test_ir_plot_excludes_imaginary_mode_and_preserves_source_mode_number():
    figure = ir_spectrum_figure(record(), scale=0.98, width_cm1=15)
    modes = next(trace for trace in figure.data if trace.name == "Modes")
    assert list(modes.customdata) == [2, 3]
    assert min(modes.x) > 0
    assert "Scaled" in figure.layout.xaxis.title.text


def test_vibration_orbital_and_charge_tables_preserve_indices():
    item = record()
    vibrations = vibration_table(item)
    assert vibrations.iloc[0]["Mode"] == 1
    assert vibrations.iloc[0]["Type"] == "Imaginary"
    orbitals = orbital_table(item)
    assert orbitals["Frontier"].tolist() == ["", "HOMO", "LUMO"]
    charges = atomic_charge_table(item)
    assert charges["Atom"].tolist() == [1, 2, 3]
    assert charges["Element"].tolist() == ["O", "H", "H"]
    transitions = electronic_transition_table(item)
    assert transitions["State"].tolist() == [1, 2]
    assert np.isclose(transitions.iloc[0]["Wavelength (nm)"], 500.0)
