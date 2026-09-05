"""Streamlit workspace for uploaded quantum-chemistry calculation files."""

from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import streamlit as st

from .qc_ingest import expand_uploaded_files
from .qc_models import CalculationRecord, QCDiagnostic
from .qc_parser import parse_calculation_file
from .qc_plots import (
    atomic_charge_table,
    coordinates_to_xyz,
    electronic_transition_table,
    convergence_figure,
    ir_spectrum_figure,
    molecular_formula,
    optimization_energy_figure,
    orbital_table,
    trajectory_to_xyz,
    vibration_table,
    vibration_to_xyz,
)


STATUS_ICON = {
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "unknown": "❔",
    "input": "📝",
}


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _candidate_type(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".gjf", ".com", ".inp", ".in", ".xyz"}:
        return "Input / geometry candidate"
    if suffix in {".out", ".log", ".output", ".dat"}:
        return "Output candidate"
    return "Will attempt detection"


def _diagnostic_box(diagnostic: QCDiagnostic) -> None:
    body = diagnostic.message
    if diagnostic.line is not None:
        body += f" (source line {diagnostic.line})"
    renderer = {
        "error": st.error,
        "warning": st.warning,
        "info": st.info,
    }[diagnostic.severity]
    renderer(f"**{diagnostic.title}** — {body}")
    if diagnostic.excerpt:
        st.code(diagnostic.excerpt, language=None)


def _render_multiframe_xyz(xyz: str, *, animate: bool, key: str) -> None:
    if not xyz:
        st.info("No coordinates are available for this view.")
        return
    try:
        import py3Dmol
        import stmol

        view = py3Dmol.view(width=700, height=480)
        if animate:
            view.addModelsAsFrames(xyz, "xyz")
        else:
            view.addModel(xyz, "xyz")
        view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.3}})
        if animate:
            view.animate({"loop": "forward", "reps": 0, "interval": 100})
        view.zoomTo()
        view.render()
        stmol.showmol(view, height=480, width=700)
    except Exception as exc:  # render failures should not hide parsed results
        st.warning(f"The 3D view could not be rendered: {type(exc).__name__}: {exc}")


def _summary_table(records: list[CalculationRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Status": f"{STATUS_ICON[record.overall_status]} {record.overall_status.title()}",
                "File": record.path,
                "Kind": record.kind.title(),
                "Program": record.program or "Unknown",
                "Method": ", ".join(record.methods) or "Unknown",
                "Formula": molecular_formula(record.atom_numbers),
                "Atoms": record.n_atoms,
                "Geometry steps": record.n_steps,
                "Properties": ", ".join(record.capability_names()) or "Source only",
                "Warnings": record.warning_count,
                "Errors": record.error_count,
            }
            for record in records
        ]
    )


def _diagnostic_table(records: list[CalculationRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "File": record.path,
                "Severity": diagnostic.severity,
                "Code": diagnostic.code,
                "Title": diagnostic.title,
                "Message": diagnostic.message,
                "Source line": diagnostic.line,
                "Evidence": diagnostic.excerpt,
            }
            for record in records
            for diagnostic in record.diagnostics
        ]
    )


def _related_records(selected: CalculationRecord, records: list[CalculationRecord]) -> list[str]:
    stem = PurePosixPath(selected.path).stem.lower()
    parent = PurePosixPath(selected.path).parent
    return [
        record.path
        for record in records
        if record.path != selected.path
        and PurePosixPath(record.path).parent == parent
        and PurePosixPath(record.path).stem.lower() == stem
    ]


def _render_status(record: CalculationRecord) -> None:
    cols = st.columns(5)
    cols[0].metric("File status", f"{STATUS_ICON[record.overall_status]} {record.overall_status.title()}")
    cols[1].metric("Parsing", f"{STATUS_ICON[record.parse_status]} {record.parse_status.title()}")
    termination = "Input file" if record.kind == "input" else record.termination_status.title()
    cols[2].metric("Termination", termination)
    optimization = "Not applicable" if record.kind == "input" else record.optimization_status_label.title()
    cols[3].metric("Geometry convergence", optimization)
    cols[4].metric("SCF", record.scf_status_label.title())


def _render_overview(record: CalculationRecord, energy_unit: str) -> None:
    _render_status(record)
    if record.diagnostics:
        important = [d for d in record.diagnostics if d.severity in {"error", "warning"}]
        for diagnostic in important[:3]:
            _diagnostic_box(diagnostic)
        if len(important) > 3:
            st.caption(f"{len(important) - 3} more warning(s) or error(s) are in Diagnostics.")

    details = [
        ("Program", record.program),
        ("Version", record.program_version),
        ("Method", ", ".join(record.methods) if record.methods else None),
        ("Basis", record.basis),
        ("Formula", molecular_formula(record.atom_numbers)),
        ("Atoms", record.n_atoms),
        ("Charge", record.charge),
        ("Multiplicity", record.multiplicity),
        ("Parser", record.parser),
    ]
    st.dataframe(
        pd.DataFrame([(label, str(value)) for label, value in details if value is not None],
                     columns=["Property", "Value"]),
        width="stretch", hide_index=True,
    )

    if record.requested_settings:
        with st.expander("Requested calculation settings", expanded=True):
            st.dataframe(
                pd.DataFrame(
                    [(key, str(value)) for key, value in record.requested_settings.items()],
                    columns=["Setting", "Value"],
                ),
                width="stretch", hide_index=True,
            )

    if record.coordinates is not None and record.atom_numbers is not None:
        title = "Input structure" if record.kind == "input" else (
            "Final optimized structure"
            if record.optimization_status_label == "success"
            else "Last available structure"
        )
        st.subheader(title)
        _render_multiframe_xyz(
            coordinates_to_xyz(record.atom_numbers, record.coordinates[-1], title),
            animate=False, key=f"overview-{record.record_id}",
        )
        st.download_button(
            "Download displayed XYZ",
            coordinates_to_xyz(record.atom_numbers, record.coordinates[-1], title),
            file_name=f"{PurePosixPath(record.name).stem}_displayed.xyz",
            mime="chemical/x-xyz", key=f"overview-download-{record.record_id}",
        )


def _render_optimization(record: CalculationRecord, energy_unit: str) -> None:
    if record.n_steps <= 1 and record.scf_energies_ev is None:
        st.info("No optimization trajectory or energy progression was parsed from this file.")
        return

    energy_fig = optimization_energy_figure(record, energy_unit)
    if energy_fig is not None:
        if record.n_steps and len(record.scf_energies_ev) != record.n_steps:
            st.info(
                "The number of energy points differs from the number of geometries. "
                "They are plotted independently and are not assumed to correspond step by step."
            )
        st.plotly_chart(energy_fig, width="stretch", key=f"opt-energy-{record.record_id}")

    convergence = convergence_figure(record)
    if convergence is not None:
        st.plotly_chart(convergence, width="stretch", key=f"opt-convergence-{record.record_id}")
    elif record.n_steps > 1:
        st.info("The program's geometry convergence values or targets were not available.")

    if record.n_steps > 1 and record.atom_numbers is not None:
        st.subheader("Geometry trajectory")
        animate = st.toggle("Play trajectory", key=f"play-trajectory-{record.record_id}")
        if animate:
            _render_multiframe_xyz(
                trajectory_to_xyz(record), animate=True, key=f"trajectory-{record.record_id}"
            )
        else:
            step = st.slider(
                "Geometry step", 1, record.n_steps, record.n_steps,
                key=f"geometry-step-{record.record_id}",
            )
            xyz = coordinates_to_xyz(
                record.atom_numbers, record.coordinates[step - 1],
                f"{record.name}: geometry step {step}",
            )
            _render_multiframe_xyz(xyz, animate=False, key=f"trajectory-step-{record.record_id}")
        st.download_button(
            "Download trajectory XYZ", trajectory_to_xyz(record),
            file_name=f"{PurePosixPath(record.name).stem}_trajectory.xyz",
            mime="chemical/x-xyz", key=f"trajectory-download-{record.record_id}",
        )


def _render_vibrations(record: CalculationRecord) -> None:
    table = vibration_table(record)
    if table.empty:
        st.info("No vibrational frequencies were parsed from this calculation.")
        return
    imaginary = int((table["Frequency (cm⁻¹)"] < 0).sum())
    st.metric("Imaginary modes", imaginary)
    if imaginary:
        st.warning(
            "Negative frequencies are shown separately. Inspect their displacement patterns before "
            "assigning this structure as a minimum or transition state."
        )

    controls = st.columns(3)
    scale = controls[0].number_input(
        "Frequency scale", min_value=0.5, max_value=1.5, value=1.0, step=0.001,
        format="%.3f", key=f"frequency-scale-{record.record_id}",
        help="The table retains original frequencies; scaling is applied only to the plot.",
    )
    width = controls[1].slider(
        "FWHM (cm⁻¹)", 1.0, 100.0, 20.0, 1.0,
        key=f"spectrum-width-{record.record_id}",
    )
    shape = controls[2].selectbox(
        "Line shape", ["Gaussian", "Lorentzian"],
        key=f"line-shape-{record.record_id}",
    )
    if record.ir_intensities is None:
        st.info(
            "IR intensities were not available, so the plot shows frequency positions rather than "
            "an intensity-weighted IR spectrum."
        )
    spectrum = ir_spectrum_figure(record, scale=scale, width_cm1=width, line_shape=shape)
    if spectrum is not None:
        st.plotly_chart(spectrum, width="stretch", key=f"ir-spectrum-{record.record_id}")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download vibration table CSV", table.to_csv(index=False),
        file_name=f"{PurePosixPath(record.name).stem}_vibrations.csv",
        mime="text/csv", key=f"vibration-download-{record.record_id}",
    )

    if record.vibration_displacements is not None and record.coordinates is not None:
        st.subheader("Normal-mode animation")
        mode = st.selectbox(
            "Mode", range(1, len(table) + 1),
            format_func=lambda number: (
                f"Mode {number}: {table.iloc[number - 1]['Frequency (cm⁻¹)']:.2f} cm⁻¹ "
                f"({table.iloc[number - 1]['Type']})"
            ),
            key=f"mode-select-{record.record_id}",
        )
        amplitude = st.slider(
            "Visual amplitude", 0.1, 2.0, 0.7, 0.1,
            key=f"mode-amplitude-{record.record_id}",
            help="This display scale is illustrative and is not a thermal displacement.",
        )
        _render_multiframe_xyz(
            vibration_to_xyz(record, mode - 1, amplitude=amplitude),
            animate=True, key=f"mode-animation-{record.record_id}",
        )
        st.caption("Animation amplitude is illustrative; the parsed displacement directions are preserved.")
    else:
        st.info("Mode displacement vectors were not available, so animation is disabled.")


def _render_results(record: CalculationRecord) -> None:
    if record.energy_series:
        st.subheader("Electronic energy series")
        energy_rows = []
        for name, (values, unit) in record.energy_series.items():
            flat = np.asarray(values).reshape(-1)
            energy_rows.extend(
                {"Series": name, "Point": index + 1, "Value": value, "Unit": unit}
                for index, value in enumerate(flat)
            )
        energy_table = pd.DataFrame(energy_rows)
        st.dataframe(energy_table, width="stretch", hide_index=True)
        st.download_button(
            "Download energy series CSV", energy_table.to_csv(index=False),
            file_name=f"{PurePosixPath(record.name).stem}_energies.csv",
            mime="text/csv", key=f"energy-download-{record.record_id}",
        )
    if record.thermochemistry:
        st.subheader("Energies and thermochemistry")
        st.dataframe(
            pd.DataFrame(
                [(name, value, unit) for name, (value, unit) in record.thermochemistry.items()],
                columns=["Property", "Value", "Unit"],
            ), width="stretch", hide_index=True,
        )
    charges = atomic_charge_table(record)
    if not charges.empty:
        st.subheader("Atomic charges")
        st.dataframe(charges, width="stretch", hide_index=True)
    orbitals = orbital_table(record)
    if not orbitals.empty:
        st.subheader("Molecular orbital energies")
        frontier = orbitals[orbitals["Frontier"].isin(["HOMO", "LUMO"])]
        if not frontier.empty:
            st.dataframe(frontier, width="stretch", hide_index=True)
        with st.expander("All molecular orbitals"):
            st.dataframe(orbitals, width="stretch", hide_index=True)
    transitions = electronic_transition_table(record)
    if not transitions.empty:
        st.subheader("Electronic transitions")
        st.dataframe(transitions, width="stretch", hide_index=True)
    if record.moments is not None:
        with st.expander("Moments"):
            st.write(record.moments)
    if (
        not record.energy_series
        and not record.thermochemistry
        and charges.empty
        and orbitals.empty
        and transitions.empty
        and record.moments is None
    ):
        st.info("No additional result tables were available for this file.")
    if record.metadata:
        with st.expander("Parser metadata and provenance"):
            metadata = pd.DataFrame(
                [(key, str(value)) for key, value in sorted(record.metadata.items())],
                columns=["Field", "Value"],
            )
            st.dataframe(metadata, width="stretch", hide_index=True)


def _render_diagnostics(record: CalculationRecord) -> None:
    if not record.diagnostics:
        st.success("No parser or calculation diagnostics were detected.")
        return
    severity = st.multiselect(
        "Show severity", ["error", "warning", "info"],
        default=["error", "warning", "info"], key=f"diagnostic-filter-{record.record_id}",
    )
    visible = [diagnostic for diagnostic in record.diagnostics if diagnostic.severity in severity]
    for diagnostic in visible:
        _diagnostic_box(diagnostic)


def _render_source(record: CalculationRecord) -> None:
    query = st.text_input("Find in source", key=f"source-search-{record.record_id}")
    lines = record.source_text.splitlines()
    if query:
        matches = [i for i, line in enumerate(lines) if query.lower() in line.lower()]
        if matches:
            selected = st.selectbox(
                f"{len(matches)} match(es)", matches,
                format_func=lambda index: f"Line {index + 1}: {lines[index][:100]}",
                key=f"source-match-{record.record_id}",
            )
            start, end = max(0, selected - 8), min(len(lines), selected + 9)
            excerpt = "\n".join(f"{i + 1:>7}  {lines[i]}" for i in range(start, end))
            st.code(excerpt, language=None)
        else:
            st.info("No matches found in the available source preview.")
    else:
        preview = lines[:500]
        st.caption(
            f"Showing the first {len(preview):,} of {len(lines):,} available source line(s). "
            "Search above to focus on any matching excerpt."
        )
        st.code("\n".join(preview), language=None)


def _render_record(record: CalculationRecord, records: list[CalculationRecord], energy_unit: str) -> None:
    st.header(record.name)
    st.caption(record.path)
    related = _related_records(record, records)
    if related:
        st.info("Related file(s) with the same directory and stem: " + ", ".join(related))
    overview, optimization, vibrations, results, diagnostics, source = st.tabs(
        ["Overview", "Optimization", "Vibrations & IR", "Results", "Diagnostics", "Source"]
    )
    with overview:
        _render_overview(record, energy_unit)
    with optimization:
        _render_optimization(record, energy_unit)
    with vibrations:
        _render_vibrations(record)
    with results:
        _render_results(record)
    with diagnostics:
        _render_diagnostics(record)
    with source:
        _render_source(record)


def render_calculation_workspace(energy_unit: str) -> None:
    """Render upload controls, batch inventory, and a capability-driven inspector."""
    st.subheader("Calculation Files")
    st.write(
        "Inspect quantum-chemistry outputs through cclib, common text inputs, XYZ files, "
        "folders, or ZIP archives. Files stay within this application session."
    )

    with st.sidebar:
        st.header("Calculation files")
        upload_mode = st.radio(
            "Upload", ["Files", "Folder"], horizontal=True, key="qc-upload-mode"
        )
        generation = st.session_state.get("qc-uploader-generation", 0)
        uploaded = st.file_uploader(
            "Choose calculation files" if upload_mode == "Files" else "Choose a calculation folder",
            accept_multiple_files=True if upload_mode == "Files" else "directory",
            key=f"qc-file-uploader-{generation}",
            help="Outputs are detected by content. Files mode also accepts ZIP archives.",
        )
        clear = st.button("Clear calculation session", width="stretch")
        if clear:
            for key in ["qc-records", "qc-analyzed-signature", "qc-parse-cache"]:
                st.session_state.pop(key, None)
            st.session_state["qc-uploader-generation"] = generation + 1
            st.rerun()

    files, ingestion_warnings = expand_uploaded_files(uploaded)
    for warning in ingestion_warnings:
        st.warning(warning)

    if not files and not st.session_state.get("qc-records"):
        st.info("Upload one or more calculation files, select a folder, or add a ZIP archive.")
        st.markdown(
            "**Broad output support:** cclib-compatible formats are detected from their content. "
            "Structured input previews currently support Gaussian, ORCA, Q-Chem, and XYZ."
        )
        return

    signature = tuple((file.path, file.sha256) for file in files)
    if files:
        st.subheader("Upload inventory")
        st.dataframe(
            pd.DataFrame(
                [{"File": file.path, "Size": _human_size(file.size), "Detection": _candidate_type(file.path)}
                 for file in files]
            ), width="stretch", hide_index=True,
        )
        needs_analysis = signature != st.session_state.get("qc-analyzed-signature")
        if needs_analysis:
            st.info("Review the inventory, then analyze it. Existing results remain until analysis starts.")
        if st.button(
            "Analyze calculation files", type="primary", disabled=not needs_analysis,
            width="stretch",
        ):
            cache = st.session_state.setdefault("qc-parse-cache", {})
            records = []
            progress = st.progress(0, text="Preparing calculation files…")
            for index, file in enumerate(files):
                cache_key = (file.path, file.sha256)
                progress.progress(
                    index / len(files), text=f"Parsing {file.path} ({index + 1}/{len(files)})"
                )
                if cache_key not in cache:
                    cache[cache_key] = parse_calculation_file(file)
                records.append(cache[cache_key])
            progress.progress(1.0, text="Analysis complete")
            st.session_state["qc-parse-cache"] = {
                (file.path, file.sha256): cache[(file.path, file.sha256)] for file in files
            }
            st.session_state["qc-records"] = records
            st.session_state["qc-analyzed-signature"] = signature
            st.rerun()

    records = st.session_state.get("qc-records", [])
    if not records:
        return
    st.subheader("Calculation inventory")
    summary = _summary_table(records)
    st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        "Download inventory CSV", summary.to_csv(index=False),
        file_name="calculation_inventory.csv", mime="text/csv",
    )
    diagnostic_table = _diagnostic_table(records)
    if not diagnostic_table.empty:
        st.download_button(
            "Download diagnostic report CSV", diagnostic_table.to_csv(index=False),
            file_name="calculation_diagnostics.csv", mime="text/csv",
        )
    record_by_path = {record.path: record for record in records}
    selected_path = st.selectbox(
        "Inspect calculation", list(record_by_path),
        format_func=lambda path: (
            f"{STATUS_ICON[record_by_path[path].overall_status]} {path} · "
            f"{record_by_path[path].program or record_by_path[path].kind.title()}"
        ), key="qc-selected-record",
    )
    _render_record(record_by_path[selected_path], records, energy_unit)
