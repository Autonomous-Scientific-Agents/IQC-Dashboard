"""Tests for common input adapters and normalized cclib output records."""

import hashlib

import numpy as np
import pytest

from iqc_dashboard.qc_models import QCFile
from iqc_dashboard.qc_parser import parse_calculation_file, parse_calculation_files


def qc_file(path, text):
    content = text.encode()
    return QCFile(path, content, len(content), hashlib.sha256(content).hexdigest())


@pytest.mark.parametrize(
    "path,text,program,charge,multiplicity,atoms",
    [
        (
            "water.gjf",
            "%mem=1GB\n#p b3lyp/6-31g(d) opt freq\n\nWater\n\n0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n",
            "Gaussian", 0, 1, 3,
        ),
        (
            "water.inp",
            "! B3LYP def2-SVP Opt Freq\n* xyz 0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n*\n",
            "ORCA", 0, 1, 3,
        ),
        (
            "water.in",
            "$molecule\n0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n$end\n$rem\njobtype opt\nmethod b3lyp\n$end\n",
            "Q-Chem", 0, 1, 3,
        ),
        ("water.xyz", "3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", "XYZ", None, None, 3),
    ],
)
def test_structured_input_formats(path, text, program, charge, multiplicity, atoms):
    record = parse_calculation_file(qc_file(path, text))
    assert record.kind == "input"
    assert record.program == program
    assert record.charge == charge
    assert record.multiplicity == multiplicity
    assert record.n_atoms == atoms
    assert record.coordinates.shape == (1, atoms, 3)
    assert record.overall_status == "input"


class FakeData:
    metadata = {
        "package": "Gaussian",
        "package_version": "16-C.01",
        "methods": ["B3LYP"],
        "success": True,
    }
    atomnos = np.array([8, 1, 1])
    atomcoords = np.array(
        [
            [[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]],
            [[0, 0, 0], [0, 0, 0.98], [0, 0.98, 0]],
        ]
    )
    scfenergies = np.array([-2040.0, -2041.0])
    geovalues = np.array([[0.01, 0.008], [0.0001, 0.00008]])
    geotargets = np.array([0.001, 0.001])
    optstatus = np.array([1, 4])
    optdone = True
    vibfreqs = np.array([-20.0, 1200.0, 3600.0])
    vibirs = np.array([1.0, 25.0, 50.0])
    vibdisps = np.ones((3, 3, 3)) * 0.01
    vibsyms = ["A", "A", "B"]
    vibrmasses = np.array([1.0, 2.0, 3.0])
    vibfconsts = np.array([0.1, 0.2, 0.3])
    charge = 0
    mult = 1
    enthalpy = -75.9
    entropy = 0.0001
    freeenergy = -75.95
    zpve = 0.02
    temperature = 298.15
    atomcharges = {"mulliken": np.array([-0.8, 0.4, 0.4])}
    moenergies = [np.array([-20.0, -10.0, 1.0])]
    homos = np.array([1])


class FakeParser:
    def parse(self):
        return FakeData()


def test_cclib_output_is_normalized_and_scientific_warning_is_visible(monkeypatch):
    import cclib

    monkeypatch.setattr(cclib.io, "ccopen", lambda *args, **kwargs: FakeParser())
    record = parse_calculation_file(qc_file("water.log", "Gaussian, Inc.\nNormal termination of Gaussian"))

    assert record.kind == "output"
    assert record.program == "Gaussian"
    assert record.parse_status == "success"
    assert record.termination_status == "success"
    assert record.optimization_status_label == "success"
    assert set(record.capability_names()) >= {"Structure", "Trajectory", "Energies"}
    assert record.scf_status_label == "success"
    assert any(d.code == "imaginary-frequencies" for d in record.diagnostics)
    assert record.overall_status == "warning"
    assert record.thermochemistry["Electronic energy"] == (-2041.0, "eV")


def test_misaligned_vibrational_arrays_disable_unsafe_association(monkeypatch):
    import cclib

    class Misaligned(FakeData):
        vibirs = np.array([1.0])
        vibdisps = np.ones((2, 3, 3))

    class Parser:
        def parse(self):
            return Misaligned()

    monkeypatch.setattr(cclib.io, "ccopen", lambda *args, **kwargs: Parser())
    record = parse_calculation_file(qc_file("water.log", "Gaussian, Inc."))

    assert record.ir_intensities is None
    assert record.vibration_displacements is None
    assert {d.code for d in record.diagnostics} >= {
        "vibration-length-mismatch", "mode-displacement-mismatch"
    }


def test_multi_job_output_is_flagged_as_ambiguous(monkeypatch):
    import cclib

    monkeypatch.setattr(cclib.io, "ccopen", lambda *args, **kwargs: FakeParser())
    text = "Gaussian, Inc.\n--Link1--\nNormal termination of Gaussian"
    record = parse_calculation_file(qc_file("linked.log", text))
    assert any(d.code == "multiple-job-sections" for d in record.diagnostics)


def test_parser_failure_is_contained_with_source_evidence(monkeypatch):
    import cclib

    class BrokenParser:
        def parse(self):
            raise ValueError("unexpected section")

    monkeypatch.setattr(cclib.io, "ccopen", lambda *args, **kwargs: BrokenParser())
    record = parse_calculation_file(qc_file("broken.out", "Gaussian, Inc.\nError termination"))

    assert record.parse_status == "error"
    assert record.overall_status == "error"
    assert {d.code for d in record.diagnostics} >= {
        "parser-exception", "reported-calculation-error"
    }


def test_batch_continues_after_one_parser_fails(monkeypatch):
    import cclib

    class BrokenParser:
        def parse(self):
            raise ValueError("unexpected section")

    parsers = iter([BrokenParser(), FakeParser()])
    monkeypatch.setattr(cclib.io, "ccopen", lambda *args, **kwargs: next(parsers))

    records = parse_calculation_files([
        qc_file("a-broken.out", "Gaussian, Inc.\nError termination"),
        qc_file("b-good.out", "Gaussian, Inc.\nNormal termination of Gaussian"),
    ])

    assert [record.path for record in records] == ["a-broken.out", "b-good.out"]
    assert [record.parse_status for record in records] == ["error", "success"]


def test_unknown_text_file_remains_inspectable():
    record = parse_calculation_file(qc_file("notes.txt", "not a calculation"))
    assert record.kind == "unsupported"
    assert record.source_text == "not a calculation"
    assert record.diagnostics[0].code == "unsupported-file"


def test_gaussian_license_notice_is_not_a_scientific_warning(monkeypatch):
    import cclib

    monkeypatch.setattr(cclib.io, "ccopen", lambda *args, **kwargs: FakeParser())
    text = (
        "Gaussian, Inc.\nWarning -- This program may not be used in any manner that\n"
        "Normal termination of Gaussian"
    )
    record = parse_calculation_file(qc_file("water.log", text))

    assert not any(d.code == "reported-program-warning" for d in record.diagnostics)
