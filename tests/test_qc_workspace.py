"""Smoke coverage for the calculation-files Streamlit workspace."""

from streamlit.testing.v1 import AppTest


def test_calculation_files_workspace_opens_without_dataset():
    app = AppTest.from_file("streamlit_app.py", default_timeout=20).run()
    workspace = next(radio for radio in app.radio if radio.label == "Workspace")
    workspace.set_value("Calculation files").run()

    assert not app.exception
    assert any("Upload one or more calculation files" in message.value for message in app.info)


def test_parsed_record_renders_all_capability_tabs():
    script = """
import numpy as np
import streamlit as st
from iqc_dashboard.qc_models import CalculationRecord, QCDiagnostic
from iqc_dashboard.qc_ui import render_calculation_workspace

st.session_state['qc-records'] = [CalculationRecord(
    'record', 'water.out', 42, 'sha', 'output', program='Gaussian',
    atom_numbers=np.array([8, 1, 1]), scf_energies_ev=np.array([-10.0, -10.1]),
    energy_series={'SCF/DFT electronic energy': (np.array([-10.0, -10.1]), 'eV')},
    frequencies_cm1=np.array([-20.0, 1000.0]), ir_intensities=np.array([1.0, 5.0]),
    thermochemistry={'Enthalpy': (-75.0, 'hartree/particle')},
    diagnostics=[QCDiagnostic('warning', 'Imaginary mode', 'Inspect mode 1.', 'imaginary')],
    parse_status='success', termination_status='success', optimization_status_label='unknown',
    source_text='line one\\nline two',
)]
render_calculation_workspace('eV')
"""
    app = AppTest.from_string(script, default_timeout=20).run()

    assert not app.exception
    assert any(tab.label == "Vibrations & IR" for tab in app.tabs)
    assert any(metric.label == "Imaginary modes" and metric.value == "1" for metric in app.metric)
