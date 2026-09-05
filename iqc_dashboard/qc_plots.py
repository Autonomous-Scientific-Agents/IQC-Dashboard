"""Scientific tables, plots, and geometry serialization for raw calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem

from .qc_models import CalculationRecord


EV_TO_KCAL_MOL = 23.0605


def energy_values(values, unit: str) -> tuple[np.ndarray, str]:
    result = np.asarray(values, dtype=float)
    if unit == "kcal/mol":
        result = result * EV_TO_KCAL_MOL
    return result, unit


def atomic_symbols(atom_numbers) -> list[str]:
    periodic = Chem.GetPeriodicTable()
    return [periodic.GetElementSymbol(int(number)) for number in atom_numbers]


def molecular_formula(atom_numbers) -> str:
    if atom_numbers is None:
        return "Unknown"
    counts: dict[str, int] = {}
    for symbol in atomic_symbols(atom_numbers):
        counts[symbol] = counts.get(symbol, 0) + 1
    order = []
    for symbol in ("C", "H"):
        if symbol in counts:
            order.append(symbol)
    order.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] != 1 else "") for symbol in order)


def coordinates_to_xyz(atom_numbers, coordinates, comment: str = "") -> str:
    symbols = atomic_symbols(atom_numbers)
    coords = np.asarray(coordinates, dtype=float)
    lines = [str(len(symbols)), comment]
    lines.extend(
        f"{symbol:<3} {x: .10f} {y: .10f} {z: .10f}"
        for symbol, (x, y, z) in zip(symbols, coords)
    )
    return "\n".join(lines)


def trajectory_to_xyz(record: CalculationRecord) -> str:
    if record.atom_numbers is None or record.coordinates is None:
        return ""
    return "\n".join(
        coordinates_to_xyz(record.atom_numbers, coords, f"{record.name}: step {index + 1}")
        for index, coords in enumerate(record.coordinates)
    )


def vibration_to_xyz(
    record: CalculationRecord,
    mode_index: int,
    amplitude: float = 0.7,
    frame_count: int = 20,
) -> str:
    if (
        record.atom_numbers is None
        or record.coordinates is None
        or record.vibration_displacements is None
    ):
        return ""
    base = np.asarray(record.coordinates[-1], dtype=float)
    displacement = np.asarray(record.vibration_displacements[mode_index], dtype=float)
    frames = []
    for frame in range(frame_count):
        phase = math.sin(2 * math.pi * frame / frame_count)
        frames.append(
            coordinates_to_xyz(
                record.atom_numbers,
                base + amplitude * phase * displacement,
                f"Mode {mode_index + 1}; amplitude is illustrative",
            )
        )
    return "\n".join(frames)


def optimization_energy_figure(record: CalculationRecord, unit: str) -> go.Figure | None:
    if record.scf_energies_ev is None:
        return None
    energies, label = energy_values(record.scf_energies_ev, unit)
    relative = energies - energies[0]
    step = np.arange(1, len(energies) + 1)
    aligned = record.n_steps == len(energies)
    geometry_steps = (
        np.arange(1, len(energies) + 1) if aligned else np.full(len(energies), -1)
    )
    custom = np.column_stack([energies, geometry_steps])
    fig = go.Figure(
        go.Scatter(
            x=step,
            y=relative,
            mode="lines+markers",
            customdata=custom,
            name="Electronic energy",
            hovertemplate=(
                "Energy point %{x}<br>Energy: %{customdata[0]:.8g} " + label
                + "<br>ΔE: %{y:.6g} " + label
                + ("<br>Geometry step %{customdata[1]}" if aligned else "")
                + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Electronic energy progression",
        xaxis_title="Energy point" if not aligned else "Optimization step",
        yaxis_title=f"ΔE from first point ({label})",
        height=420,
    )
    return fig


def convergence_figure(record: CalculationRecord) -> go.Figure | None:
    values = record.geometry_values
    targets = record.geometry_targets
    if values is None or targets is None or np.asarray(values).ndim != 2:
        return None
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if values.shape[1] != len(targets):
        return None
    names = ["Max force", "RMS force", "Max displacement", "RMS displacement"]
    fig = go.Figure()
    for index in range(values.shape[1]):
        valid_target = targets[index] if targets[index] > 0 else np.nan
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(values) + 1),
                y=values[:, index] / valid_target,
                mode="lines+markers",
                name=names[index] if index < len(names) else f"Criterion {index + 1}",
            )
        )
    fig.add_hline(y=1, line_dash="dash", annotation_text="Reported target")
    fig.update_layout(
        title="Geometry convergence",
        xaxis_title="Optimization step",
        yaxis_title="Value / target",
        yaxis_type="log",
        height=420,
    )
    return fig


def vibration_table(record: CalculationRecord) -> pd.DataFrame:
    frequencies = record.frequencies_cm1
    if frequencies is None:
        return pd.DataFrame(columns=["Mode", "Frequency (cm⁻¹)", "Type"])
    frequencies = np.asarray(frequencies, dtype=float)
    table = pd.DataFrame(
        {
            "Mode": np.arange(1, len(frequencies) + 1),
            "Frequency (cm⁻¹)": frequencies,
            "Type": np.where(frequencies < 0, "Imaginary", "Real"),
        }
    )
    optional = [
        ("IR intensity (km/mol)", record.ir_intensities),
        ("Reduced mass (Da)", record.vibration_masses),
        ("Force constant (mDyne/Å)", record.vibration_force_constants),
        ("Symmetry", record.vibration_symmetries),
    ]
    for label, values in optional:
        if values is not None and len(values) == len(table):
            table[label] = values
    return table


def ir_spectrum_figure(
    record: CalculationRecord,
    scale: float = 1.0,
    width_cm1: float = 20.0,
    line_shape: str = "Gaussian",
) -> go.Figure | None:
    if record.frequencies_cm1 is None:
        return None
    frequencies = np.asarray(record.frequencies_cm1, dtype=float) * scale
    positive = frequencies >= 0
    frequencies = frequencies[positive]
    if not len(frequencies):
        return None
    has_intensities = record.ir_intensities is not None
    intensities = (
        np.asarray(record.ir_intensities, dtype=float)[positive]
        if has_intensities else np.ones(len(frequencies))
    )
    fig = go.Figure()
    stick_x = []
    stick_y = []
    for frequency, intensity in zip(frequencies, intensities):
        stick_x.extend([frequency, frequency, None])
        stick_y.extend([0, intensity, None])
    fig.add_trace(
        go.Scatter(
            x=stick_x, y=stick_y, mode="lines", name="Sticks",
            line={"color": "#7f7f7f", "width": 1}, hoverinfo="skip",
        )
    )
    original_mode_numbers = np.flatnonzero(positive) + 1
    fig.add_trace(
        go.Scatter(
            x=frequencies, y=intensities, mode="markers", name="Modes",
            customdata=original_mode_numbers,
            hovertemplate=(
                "Mode %{customdata}<br>Frequency: %{x:.2f} cm⁻¹"
                "<br>Intensity: %{y:.4g}<extra></extra>"
            ),
        )
    )
    if has_intensities and width_cm1 > 0:
        low = max(0.0, float(frequencies.min() - 5 * width_cm1))
        high = float(frequencies.max() + 5 * width_cm1)
        grid = np.linspace(low, high, min(12_000, max(1_500, int(high - low) * 2)))
        broadened = np.zeros_like(grid)
        if line_shape == "Lorentzian":
            gamma = width_cm1 / 2
            for frequency, intensity in zip(frequencies, intensities):
                broadened += intensity * gamma**2 / ((grid - frequency) ** 2 + gamma**2)
        else:
            sigma = width_cm1 / (2 * math.sqrt(2 * math.log(2)))
            for frequency, intensity in zip(frequencies, intensities):
                broadened += intensity * np.exp(-0.5 * ((grid - frequency) / sigma) ** 2)
        fig.add_trace(go.Scatter(x=grid, y=broadened, mode="lines", name=line_shape))
    fig.update_layout(
        title="Calculated IR spectrum" if has_intensities else "Vibrational frequency positions",
        xaxis_title="Scaled frequency (cm⁻¹)" if scale != 1 else "Frequency (cm⁻¹)",
        yaxis_title="IR intensity (km/mol)" if has_intensities else "Mode position",
        yaxis={"showticklabels": has_intensities},
        height=430,
    )
    return fig


def orbital_table(record: CalculationRecord) -> pd.DataFrame:
    rows = []
    for spin, energies in enumerate(record.mo_energies_ev):
        homo = int(record.homos[spin]) if record.homos is not None and spin < len(record.homos) else None
        for index, energy in enumerate(np.asarray(energies, dtype=float)):
            occupied = homo is not None and index <= homo
            frontier = ""
            if homo is not None:
                if index == homo:
                    frontier = "HOMO"
                elif index == homo + 1:
                    frontier = "LUMO"
            rows.append({
                "Spin": "Alpha" if spin == 0 else "Beta",
                "Orbital": index + 1,
                "Energy (eV)": energy,
                "Occupation": "Occupied" if occupied else "Virtual",
                "Frontier": frontier,
            })
    return pd.DataFrame(rows)


def atomic_charge_table(record: CalculationRecord) -> pd.DataFrame:
    if record.atom_numbers is None or not record.atomic_charges:
        return pd.DataFrame()
    result = pd.DataFrame({
        "Atom": np.arange(1, len(record.atom_numbers) + 1),
        "Element": atomic_symbols(record.atom_numbers),
    })
    for scheme, values in record.atomic_charges.items():
        if len(values) == len(result):
            result[scheme] = values
    return result


def electronic_transition_table(record: CalculationRecord) -> pd.DataFrame:
    values = record.electronic_transitions_cm1
    if values is None:
        return pd.DataFrame()
    wavenumbers = np.asarray(values, dtype=float)
    table = pd.DataFrame({
        "State": np.arange(1, len(wavenumbers) + 1),
        "Energy (cm⁻¹)": wavenumbers,
        "Energy (eV)": wavenumbers / 8065.544005,
        "Wavelength (nm)": np.divide(
            1.0e7, wavenumbers, out=np.full_like(wavenumbers, np.nan), where=wavenumbers != 0
        ),
    })
    if record.oscillator_strengths is not None and len(record.oscillator_strengths) == len(table):
        table["Oscillator strength"] = record.oscillator_strengths
    if len(record.electronic_transition_symmetries) == len(table):
        table["Symmetry"] = record.electronic_transition_symmetries
    return table
