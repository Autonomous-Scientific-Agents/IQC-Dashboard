"""Normalized records for uploaded quantum-chemistry files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


Severity = Literal["error", "warning", "info"]
Status = Literal["success", "warning", "error", "unknown", "input"]


@dataclass
class QCDiagnostic:
    severity: Severity
    title: str
    message: str
    code: str
    line: int | None = None
    excerpt: str | None = None


@dataclass
class QCFile:
    path: str
    content: bytes
    size: int
    sha256: str


@dataclass
class CalculationRecord:
    record_id: str
    path: str
    size: int
    sha256: str
    kind: Literal["output", "input", "unsupported"]
    parser: str | None = None
    program: str | None = None
    program_version: str | None = None
    methods: list[str] = field(default_factory=list)
    basis: str | None = None
    charge: int | None = None
    multiplicity: int | None = None
    atom_numbers: np.ndarray | None = None
    coordinates: np.ndarray | None = None
    scf_energies_ev: np.ndarray | None = None
    energy_series: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)
    geometry_values: np.ndarray | None = None
    geometry_targets: np.ndarray | None = None
    optimization_status: np.ndarray | None = None
    frequencies_cm1: np.ndarray | None = None
    ir_intensities: np.ndarray | None = None
    vibration_displacements: np.ndarray | None = None
    vibration_symmetries: list[str] = field(default_factory=list)
    vibration_masses: np.ndarray | None = None
    vibration_force_constants: np.ndarray | None = None
    thermochemistry: dict[str, tuple[float, str]] = field(default_factory=dict)
    moments: Any = None
    atomic_charges: dict[str, np.ndarray] = field(default_factory=dict)
    mo_energies_ev: list[np.ndarray] = field(default_factory=list)
    homos: np.ndarray | None = None
    electronic_transitions_cm1: np.ndarray | None = None
    oscillator_strengths: np.ndarray | None = None
    electronic_transition_symmetries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    requested_settings: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[QCDiagnostic] = field(default_factory=list)
    parse_status: Status = "unknown"
    termination_status: Status = "unknown"
    optimization_status_label: Status = "unknown"
    scf_status_label: Status = "unknown"
    source_text: str = ""

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def n_atoms(self) -> int | None:
        if self.atom_numbers is not None:
            return int(len(self.atom_numbers))
        if self.coordinates is not None and self.coordinates.ndim >= 2:
            return int(self.coordinates.shape[-2])
        return None

    @property
    def n_steps(self) -> int:
        return int(len(self.coordinates)) if self.coordinates is not None else 0

    @property
    def warning_count(self) -> int:
        return sum(d.severity == "warning" for d in self.diagnostics)

    @property
    def error_count(self) -> int:
        return sum(d.severity == "error" for d in self.diagnostics)

    @property
    def overall_status(self) -> Status:
        if self.kind == "input":
            return "input"
        if self.parse_status == "error" or self.termination_status == "error":
            return "error"
        if self.warning_count or self.parse_status == "warning":
            return "warning"
        if self.termination_status == "success":
            return "success"
        return "unknown"

    def capability_names(self) -> list[str]:
        capabilities = []
        if self.coordinates is not None:
            capabilities.append("Structure")
        if self.n_steps > 1:
            capabilities.append("Trajectory")
        if self.scf_energies_ev is not None:
            capabilities.append("Energies")
        if self.frequencies_cm1 is not None:
            capabilities.append("Frequencies")
        if self.ir_intensities is not None:
            capabilities.append("IR")
        if self.vibration_displacements is not None:
            capabilities.append("Mode animation")
        if self.thermochemistry:
            capabilities.append("Thermochemistry")
        if self.electronic_transitions_cm1 is not None:
            capabilities.append("Electronic transitions")
        if self.mo_energies_ev:
            capabilities.append("Orbitals")
        if self.atomic_charges:
            capabilities.append("Atomic charges")
        return capabilities
