"""Parse quantum-chemistry outputs and common text input formats."""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from .qc_models import CalculationRecord, QCDiagnostic, QCFile


INPUT_SUFFIXES = {".gjf", ".com", ".inp", ".in"}
OUTPUT_HINTS = (
    "gaussian, inc.",
    "o   r   c   a",
    "q-chem",
    "gamess",
    "nwchem",
    "psi4",
    "molpro",
    "turbomole",
)
ERROR_PATTERNS = (
    (re.compile(r"error termination", re.I), "Program reported error termination."),
    (re.compile(r"\bfatal error\b", re.I), "Program reported a fatal error."),
    (re.compile(r"scf (?:has )?not converged|scf convergence failure", re.I),
     "The output reports that an SCF cycle did not converge."),
    (re.compile(r"maximum number of optimization cycles", re.I),
     "The optimization reached its cycle limit."),
)
WARNING_PATTERNS = (
    (re.compile(r"\bwarning\b", re.I), "The program emitted a warning."),
    (re.compile(r"optimization stopped|optimization did not converge", re.I),
     "The output indicates an incomplete optimization."),
)
WARNING_IGNORE_PATTERNS = (
    re.compile(r"this program may not be used in any manner that", re.I),
)
SUCCESS_PATTERNS = (
    re.compile(r"normal termination of gaussian", re.I),
    re.compile(r"orca terminated normally", re.I),
    re.compile(r"thank you very much for using q-chem", re.I),
    re.compile(r"execution of gamess terminated normally", re.I),
    re.compile(r"psi4 exiting successfully", re.I),
)
MAX_SOURCE_CHARS = 2_000_000


def _record_id(file: QCFile) -> str:
    return hashlib.sha256(f"{file.path}\0{file.sha256}".encode()).hexdigest()[:16]


def _array(data: Any, name: str, ndim: int | None = None) -> np.ndarray | None:
    value = getattr(data, name, None)
    if value is None:
        return None
    try:
        result = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if result.size == 0 or (ndim is not None and result.ndim != ndim):
        return None
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _array_list(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        try:
            return [np.asarray(item) for item in value if np.asarray(item).size]
        except (TypeError, ValueError):
            return []
    if array.size == 0:
        return []
    if array.ndim == 1:
        return [array]
    return [np.asarray(item) for item in value]


def _source_for_display(text: str, diagnostics: list[QCDiagnostic]) -> str:
    if len(text) <= MAX_SOURCE_CHARS:
        return text
    diagnostics.append(
        QCDiagnostic(
            "info",
            "Source preview truncated",
            "Parsing used the complete file; the source tab shows the first 2 million characters.",
            "source-preview-truncated",
        )
    )
    return text[:MAX_SOURCE_CHARS]


def _scan_text(text: str) -> tuple[list[QCDiagnostic], bool]:
    diagnostics: list[QCDiagnostic] = []
    normal_termination = False
    capped = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        normal_termination = normal_termination or any(p.search(line) for p in SUCCESS_PATTERNS)
        if capped:
            continue
        for pattern, message in ERROR_PATTERNS:
            if pattern.search(line):
                diagnostics.append(
                    QCDiagnostic(
                        "error", "Calculation error reported", message,
                        "reported-calculation-error", line_number, line.strip()[:500]
                    )
                )
                break
        else:
            for pattern, message in WARNING_PATTERNS:
                if pattern.search(line) and not any(
                    ignored.search(line) for ignored in WARNING_IGNORE_PATTERNS
                ):
                    diagnostics.append(
                        QCDiagnostic(
                            "warning", "Program warning", message,
                            "reported-program-warning", line_number, line.strip()[:500]
                        )
                    )
                    break
        if len(diagnostics) >= 100:
            diagnostics.append(
                QCDiagnostic(
                    "info", "Diagnostic limit reached",
                    "Only the first 100 source warnings and errors are listed.",
                    "diagnostic-limit",
                )
            )
            capped = True
    return diagnostics, normal_termination


def _multi_job_diagnostic(text: str) -> QCDiagnostic | None:
    markers = (
        (r"(?im)^\s*--link1--\s*$", "Gaussian Link1"),
        (r"(?im)^\s*running job\s+\d+\s+of\s+\d+", "Q-Chem job"),
        (r"(?im)^\s*ORCA JOB NUMBER\s+\d+", "ORCA job"),
    )
    for pattern, label in markers:
        matches = list(re.finditer(pattern, text))
        if matches:
            line = text.count("\n", 0, matches[0].start()) + 1
            return QCDiagnostic(
                "warning",
                "Multiple calculation sections detected",
                f"{label} separators were found. cclib exposes this file as one record, so "
                "properties may span stages whose boundaries cannot be aligned reliably.",
                "multiple-job-sections",
                line,
                matches[0].group(0).strip(),
            )
    return None


def _deduplicate_diagnostics(items: list[QCDiagnostic]) -> list[QCDiagnostic]:
    seen = set()
    result = []
    for item in items:
        key = (item.severity, item.code, item.line, item.excerpt)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _input_program(text: str, suffix: str) -> str | None:
    lower = text.lower()
    if suffix in {".gjf", ".com"} or re.search(r"(?m)^\s*#(?:p|n|t)?\s", lower):
        return "Gaussian"
    if re.search(r"(?m)^\s*!\s*\S+", text) or re.search(r"(?mi)^\s*\*\s*xyz(?:file)?\s", text):
        return "ORCA"
    if "$molecule" in lower or "$rem" in lower:
        return "Q-Chem"
    if suffix == ".xyz" or re.match(r"^\s*\d+\s*(?:\r?\n)", text):
        return "XYZ"
    return None


def _parse_xyz_input(text: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    lines = text.splitlines()
    if not lines:
        return None
    try:
        count = int(lines[0].strip())
    except ValueError:
        return None
    atoms = _parse_cartesian_lines(lines[2 : 2 + count])
    if atoms is None or len(atoms[0]) != count:
        return None
    return atoms[0], atoms[1][None, :, :], {"title": lines[1].strip() if len(lines) > 1 else ""}


def _parse_cartesian_lines(lines) -> tuple[np.ndarray, np.ndarray] | None:
    numbers = []
    coordinates = []
    try:
        from rdkit import Chem

        periodic = Chem.GetPeriodicTable()
        for line in lines:
            fields = line.split()
            if len(fields) < 4:
                continue
            symbol = fields[0].capitalize()
            atomic_number = int(periodic.GetAtomicNumber(symbol))
            if atomic_number <= 0:
                continue
            xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
            numbers.append(atomic_number)
            coordinates.append(xyz)
    except (TypeError, ValueError, RuntimeError):
        return None
    if not numbers:
        return None
    return np.asarray(numbers, dtype=int), np.asarray(coordinates, dtype=float)


def _parse_gaussian_input(text: str):
    lines = text.splitlines()
    route = []
    route_start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if route_start is None:
        return None
    index = route_start
    while index < len(lines) and (index == route_start or lines[index].strip()):
        route.append(lines[index].strip())
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    title = lines[index].strip() if index < len(lines) else ""
    index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    charge = multiplicity = None
    if index < len(lines):
        match = re.match(r"\s*(-?\d+)\s+(\d+)\s*$", lines[index])
        if match:
            charge, multiplicity = map(int, match.groups())
            index += 1
    atoms = _parse_cartesian_lines(lines[index:])
    if atoms is None:
        return None
    return atoms[0], atoms[1][None, :, :], {
        "route": " ".join(route), "title": title,
        "charge": charge, "multiplicity": multiplicity,
    }


def _parse_orca_input(text: str):
    lines = text.splitlines()
    settings = [line.strip() for line in lines if line.lstrip().startswith("!")]
    start = None
    charge = multiplicity = None
    for i, line in enumerate(lines):
        match = re.match(r"\s*\*\s*xyz\s+(-?\d+)\s+(\d+)", line, re.I)
        if match:
            start = i + 1
            charge, multiplicity = map(int, match.groups())
            break
    if start is None:
        return None
    end = next((i for i in range(start, len(lines)) if lines[i].strip() == "*"), len(lines))
    atoms = _parse_cartesian_lines(lines[start:end])
    if atoms is None:
        return None
    return atoms[0], atoms[1][None, :, :], {
        "keywords": " ".join(settings), "charge": charge, "multiplicity": multiplicity,
    }


def _parse_qchem_input(text: str):
    molecule = re.search(r"(?is)\$molecule\s*(.*?)\$end", text)
    if not molecule:
        return None
    lines = [line for line in molecule.group(1).splitlines() if line.strip()]
    charge = multiplicity = None
    if lines:
        match = re.match(r"\s*(-?\d+)\s+(\d+)\s*$", lines[0])
        if match:
            charge, multiplicity = map(int, match.groups())
            lines = lines[1:]
    atoms = _parse_cartesian_lines(lines)
    if atoms is None:
        return None
    rem = re.search(r"(?is)\$rem\s*(.*?)\$end", text)
    settings = {}
    if rem:
        for line in rem.group(1).splitlines():
            fields = line.split(None, 1)
            if len(fields) == 2:
                settings[fields[0].lower()] = fields[1].strip()
    settings.update({"charge": charge, "multiplicity": multiplicity})
    return atoms[0], atoms[1][None, :, :], settings


def _parse_input(file: QCFile, text: str, program: str) -> CalculationRecord:
    parser = {
        "Gaussian": _parse_gaussian_input,
        "ORCA": _parse_orca_input,
        "Q-Chem": _parse_qchem_input,
        "XYZ": _parse_xyz_input,
    }[program]
    parsed = parser(text)
    diagnostics = []
    if parsed is None:
        diagnostics.append(
            QCDiagnostic(
                "warning", "Input structure not recognized",
                "The source is available, but its Cartesian structure could not be extracted.",
                "input-structure-unavailable",
            )
        )
        atom_numbers = coordinates = None
        settings = {}
        parse_status = "warning"
    else:
        atom_numbers, coordinates, settings = parsed
        parse_status = "success"
    return CalculationRecord(
        record_id=_record_id(file), path=file.path, size=file.size, sha256=file.sha256,
        kind="input", parser=f"{program} input adapter", program=program,
        charge=settings.get("charge"), multiplicity=settings.get("multiplicity"),
        atom_numbers=atom_numbers, coordinates=coordinates,
        requested_settings=settings, diagnostics=diagnostics,
        parse_status=parse_status, termination_status="input",
        optimization_status_label="input", source_text=_source_for_display(text, diagnostics),
    )


def _thermochemistry(data) -> dict[str, tuple[float, str]]:
    fields = {
        "Electronic energy": ("scfenergies", "eV", True),
        "Zero-point correction": ("zpve", "hartree/particle", False),
        "Enthalpy": ("enthalpy", "hartree/particle", False),
        "Entropy": ("entropy", "hartree/(particle·K)", False),
        "Gibbs free energy": ("freeenergy", "hartree/particle", False),
        "Temperature": ("temperature", "K", False),
        "Pressure": ("pressure", "Pa", False),
    }
    result = {}
    for label, (name, unit, last) in fields.items():
        value = getattr(data, name, None)
        if value is None:
            continue
        try:
            if last:
                value = np.asarray(value).reshape(-1)[-1]
            result[label] = (float(value), unit)
        except (TypeError, ValueError, IndexError):
            continue
    return result


def _energy_series(data) -> dict[str, tuple[np.ndarray, str]]:
    result = {}
    for label, attribute in (
        ("SCF/DFT electronic energy", "scfenergies"),
        ("MP energies", "mpenergies"),
        ("Coupled-cluster energies", "ccenergies"),
        ("Dispersion corrections", "dispersionenergies"),
    ):
        values = _array(data, attribute)
        if values is not None:
            result[label] = (np.asarray(values, dtype=float), "eV")
    return result


def _parse_output(file: QCFile, text: str) -> CalculationRecord | None:
    try:
        import cclib
    except ImportError as exc:
        diagnostics = [QCDiagnostic(
            "error", "cclib is unavailable", str(exc), "cclib-unavailable"
        )]
        source_text = _source_for_display(text, diagnostics)
        return CalculationRecord(
            _record_id(file), file.path, file.size, file.sha256, "output",
            diagnostics=diagnostics, parse_status="error",
            source_text=source_text,
        )

    suffix = Path(file.path).suffix
    suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,12}", suffix) else ".out"
    temp_path = None
    parser = None
    parser_messages = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(file.content)
            temp_path = handle.name
        parser = cclib.io.ccopen(temp_path, quiet=True)
        if parser is None:
            return None
        message_stream = io.StringIO()
        logger = getattr(parser, "logger", None)
        if logger is None:
            with contextlib.redirect_stderr(message_stream):
                data = parser.parse()
        else:
            original_handlers = list(logger.handlers)
            original_propagate = logger.propagate
            handler = logging.StreamHandler(message_stream)
            logger.handlers = [handler]
            logger.propagate = False
            try:
                data = parser.parse()
            finally:
                logger.handlers = original_handlers
                logger.propagate = original_propagate
        parser_messages = message_stream.getvalue().strip()
    except Exception as exc:  # cclib parsers expose several exception types
        if "message_stream" in locals():
            parser_messages = message_stream.getvalue().strip()
        diagnostics, _ = _scan_text(text)
        parser_name = type(parser).__name__.removesuffix("Parser") if parser is not None else None
        detail = f"{type(exc).__name__}: {exc}"
        if parser_messages:
            detail += f"\n\nParser log:\n{parser_messages[-1_500:]}"
        diagnostics.insert(0, QCDiagnostic(
            "error", "Parser failed",
            detail, "parser-exception"
        ))
        source_text = _source_for_display(text, diagnostics)
        diagnostics = _deduplicate_diagnostics(diagnostics)
        return CalculationRecord(
            _record_id(file), file.path, file.size, file.sha256, "output",
            parser=type(parser).__name__ if parser is not None else None,
            program=parser_name,
            diagnostics=diagnostics, parse_status="error",
            source_text=source_text,
        )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    diagnostics, normal_marker = _scan_text(text)
    multi_job = _multi_job_diagnostic(text)
    if multi_job is not None:
        diagnostics.append(multi_job)
    if parser_messages:
        diagnostics.append(QCDiagnostic(
            "warning", "Parser message",
            parser_messages[-2_000:], "parser-message"
        ))

    raw_metadata = getattr(data, "metadata", {}) or {}
    try:
        metadata = dict(raw_metadata)
    except (TypeError, ValueError):
        metadata = {"raw_metadata": str(raw_metadata)}
    metadata["cclib_version"] = cclib.__version__
    program = metadata.get("package") or type(parser).__name__.removesuffix("Parser")
    methods = _string_list(metadata.get("methods"))
    basis = getattr(data, "basis_descript", None) or metadata.get("basis_set")
    coordinates = _array(data, "atomcoords", 3)
    energies = _array(data, "scfenergies")
    atom_numbers = _array(data, "atomnos", 1)
    frequencies = _array(data, "vibfreqs", 1)
    intensities = _array(data, "vibirs", 1)
    displacements = _array(data, "vibdisps", 3)
    transitions = _array(data, "etenergies", 1)
    oscillator_strengths = _array(data, "etoscs", 1)
    if frequencies is not None and intensities is not None and len(frequencies) != len(intensities):
        diagnostics.append(QCDiagnostic(
            "warning", "Vibrational arrays do not align",
            "Frequencies and IR intensities have different lengths; intensities are hidden.",
            "vibration-length-mismatch"
        ))
        intensities = None
    if frequencies is not None and displacements is not None and len(frequencies) != len(displacements):
        diagnostics.append(QCDiagnostic(
            "warning", "Mode displacements do not align",
            "Frequencies and displacement vectors have different lengths; animation is disabled.",
            "mode-displacement-mismatch"
        ))
        displacements = None
    if (
        transitions is not None
        and oscillator_strengths is not None
        and len(transitions) != len(oscillator_strengths)
    ):
        diagnostics.append(QCDiagnostic(
            "warning", "Electronic transition arrays do not align",
            "Transition energies and oscillator strengths have different lengths; strengths are hidden.",
            "transition-length-mismatch"
        ))
        oscillator_strengths = None

    metadata_success = metadata.get("success")
    if any(d.severity == "error" for d in diagnostics):
        termination = "error"
    elif metadata_success is True or normal_marker:
        termination = "success"
    elif metadata_success is False:
        termination = "error"
    else:
        termination = "unknown"
        diagnostics.append(QCDiagnostic(
            "warning", "Completion not confirmed",
            "No recognized normal-termination marker was found. Available results are still shown.",
            "termination-unknown"
        ))

    try:
        optdone = bool(getattr(data, "optdone", False))
    except ValueError:
        optdone = bool(np.asarray(getattr(data, "optdone")).any())
    optstatus = _array(data, "optstatus")
    if optdone:
        optimization = "success"
    elif coordinates is not None and len(coordinates) > 1:
        optimization = "warning"
        diagnostics.append(QCDiagnostic(
            "warning", "Optimization not confirmed",
            "Multiple geometries were parsed, but completed geometry convergence was not reported.",
            "optimization-unconfirmed"
        ))
    else:
        optimization = "unknown"

    scf_failed = any(
        diagnostic.code == "reported-calculation-error"
        and "SCF" in (diagnostic.excerpt or "").upper()
        for diagnostic in diagnostics
    )
    if scf_failed:
        scf_status = "error"
    elif energies is not None and termination == "success":
        scf_status = "success"
    elif energies is not None:
        scf_status = "unknown"
    else:
        scf_status = "unknown"

    imaginary = int(np.count_nonzero(frequencies < 0)) if frequencies is not None else 0
    if imaginary:
        diagnostics.append(QCDiagnostic(
            "warning", "Imaginary vibrational modes",
            f"Found {imaginary} negative frequenc{'y' if imaginary == 1 else 'ies'}. "
            "Inspect their motion before assigning a minimum or transition state.",
            "imaginary-frequencies"
        ))

    raw_charges = getattr(data, "atomcharges", {}) or {}
    charges = raw_charges if isinstance(raw_charges, dict) else {}
    moenergies = _array_list(getattr(data, "moenergies", None))
    source_text = _source_for_display(text, diagnostics)
    diagnostics = _deduplicate_diagnostics(diagnostics)
    return CalculationRecord(
        record_id=_record_id(file), path=file.path, size=file.size, sha256=file.sha256,
        kind="output", parser=type(parser).__name__, program=str(program),
        program_version=(
            str(metadata["package_version"]) if metadata.get("package_version") is not None else None
        ),
        methods=list(methods), basis=basis,
        charge=getattr(data, "charge", None), multiplicity=getattr(data, "mult", None),
        atom_numbers=atom_numbers, coordinates=coordinates,
        scf_energies_ev=energies, energy_series=_energy_series(data),
        geometry_values=_array(data, "geovalues"),
        geometry_targets=_array(data, "geotargets"), optimization_status=optstatus,
        frequencies_cm1=frequencies, ir_intensities=intensities,
        vibration_displacements=displacements,
        vibration_symmetries=_string_list(getattr(data, "vibsyms", None)),
        vibration_masses=_array(data, "vibrmasses", 1),
        vibration_force_constants=_array(data, "vibfconsts", 1),
        thermochemistry=_thermochemistry(data), moments=getattr(data, "moments", None),
        atomic_charges={str(k): np.asarray(v) for k, v in charges.items()},
        mo_energies_ev=moenergies,
        homos=_array(data, "homos", 1), metadata=metadata,
        electronic_transitions_cm1=transitions,
        oscillator_strengths=oscillator_strengths,
        electronic_transition_symmetries=_string_list(getattr(data, "etsyms", None)),
        diagnostics=diagnostics,
        parse_status="warning" if parser_messages else "success",
        termination_status=termination, optimization_status_label=optimization,
        scf_status_label=scf_status,
        source_text=source_text,
    )


def parse_calculation_file(file: QCFile) -> CalculationRecord:
    """Parse one file without allowing its failure to poison a batch."""
    text = file.content.decode("utf-8", errors="replace")
    suffix = Path(file.path).suffix.lower()
    looks_like_output = any(hint in text[:100_000].lower() for hint in OUTPUT_HINTS)
    if suffix not in INPUT_SUFFIXES or looks_like_output:
        parsed = _parse_output(file, text)
        if parsed is not None:
            return parsed

    program = _input_program(text, suffix)
    if program:
        return _parse_input(file, text, program)

    diagnostics = [QCDiagnostic(
        "info", "Unsupported file",
        "No supported quantum-chemistry output or structured input format was detected.",
        "unsupported-file"
    )]
    return CalculationRecord(
        _record_id(file), file.path, file.size, file.sha256, "unsupported",
        diagnostics=diagnostics, parse_status="unknown",
        source_text=_source_for_display(text, diagnostics),
    )


def parse_calculation_files(files: list[QCFile]) -> list[CalculationRecord]:
    """Parse a batch with deterministic ordering and per-file containment."""
    return [parse_calculation_file(file) for file in sorted(files, key=lambda item: item.path)]
