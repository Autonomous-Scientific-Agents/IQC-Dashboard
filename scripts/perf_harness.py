"""Interaction-latency harness for the dashboard, driven by streamlit AppTest.

Generates a synthetic IQC-style thermo dataset (Ni_CO2 naming scheme so the
Reactions tab computes) and times the interactions users feel most:
initial load, idle rerun, molecule navigation, energy-unit toggle, descriptor
change, and filter changes. Use it to check for latency regressions before
merging dashboard changes:

    python scripts/perf_harness.py                  # reuse cached dataset
    python scripts/perf_harness.py --rebuild        # regenerate dataset
    python scripts/perf_harness.py --molecules 800  # smaller/faster run

The harness fails loudly (non-zero exit) if any interaction raises.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def make_xyz(n_atoms: int, rng: np.random.Generator) -> str:
    elements = rng.choice(["C", "H", "N", "O"], size=n_atoms, p=[0.35, 0.45, 0.1, 0.1])
    coords = rng.normal(scale=4.0, size=(n_atoms, 3))
    lines = [str(n_atoms), "synthetic"]
    lines += [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in zip(elements, coords)]
    return "\n".join(lines)


def build_dataset(path: Path, n_bipy: int, n_alkyne: int) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for b in range(n_bipy):
        for a in range(n_alkyne):
            for role in ("reactant", "product"):
                n_atoms = int(rng.integers(100, 180))
                g = float(rng.normal(-450.0, 5.0))
                rows.append(
                    {
                        "unique_name": f"bipy-B{b}_A{a}C2H2_{role}_conf1",
                        "formula": f"C{n_atoms // 2}H{n_atoms // 2}NiN2O2",
                        "number_of_atoms": n_atoms,
                        "number_of_electrons": n_atoms * 3,
                        "calculator": rng.choice(["dft", "xtb"]),
                        "task": "thermo",
                        "initial_energy_eV": g + 0.5,
                        "opt_energy_eV": g + 0.1,
                        "opt_converged": bool(rng.random() > 0.05),
                        "opt_steps": int(rng.integers(5, 60)),
                        "opt_time": float(rng.uniform(10, 500)),
                        "initial_xyz": make_xyz(n_atoms, rng),
                        "opt_xyz": make_xyz(n_atoms, rng),
                        "initial_smiles": f"C{'C' * (b % 5)}N{a}",
                        "opt_smiles": f"C{'C' * (b % 5)}N{a}",
                        "smiles_changed": False,
                        "number_of_imaginary": int(rng.integers(0, 3)),
                        "vibrational_frequencies_cm^-1": rng.uniform(
                            -50, 3500, size=3 * n_atoms - 6
                        ).tolist(),
                        "G_eV": g,
                        "H_eV": g + 0.3,
                        "S_eV/K": 0.001,
                    }
                )
    rows.append(
        {
            "unique_name": "co2",
            "formula": "CO2",
            "number_of_atoms": 3,
            "number_of_electrons": 22,
            "calculator": "dft",
            "task": "thermo",
            "initial_energy_eV": -19.0,
            "opt_energy_eV": -19.1,
            "opt_converged": True,
            "opt_steps": 3,
            "opt_time": 5.0,
            "initial_xyz": "3\nCO2\nC 0 0 0\nO 1.16 0 0\nO -1.16 0 0",
            "opt_xyz": "3\nCO2\nC 0 0 0\nO 1.17 0 0\nO -1.17 0 0",
            "initial_smiles": "O=C=O",
            "opt_smiles": "O=C=O",
            "smiles_changed": False,
            "number_of_imaginary": 0,
            "vibrational_frequencies_cm^-1": [600.0, 600.0, 1300.0, 2400.0],
            "G_eV": -19.5,
            "H_eV": -19.2,
            "S_eV/K": 0.001,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def timed(label: str, fn) -> None:
    start = time.time()
    fn()
    print(f"{label}: {time.time() - start:.2f}s")


def run(data_path: Path) -> int:
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from iqc_dashboard.app import main\n"
        f"main(data_paths=[{str(data_path)!r}])\n"
    )
    at = AppTest.from_string(script, default_timeout=900)

    timed("initial load", at.run)
    timed("idle rerun (no state change)", at.run)

    boxes = {s.key: s for s in at.selectbox}
    molecule_box = boxes.get("selected_molecule_select")
    if molecule_box is not None and len(molecule_box.options) > 1:
        timed(
            "molecule change rerun",
            lambda: molecule_box.select(molecule_box.options[1]).run(),
        )

    radios = {r.key: r for r in at.radio}
    if "energy_unit_select" in radios:
        timed(
            "energy unit toggle rerun",
            lambda: radios["energy_unit_select"].set_value("eV").run(),
        )

    boxes = {s.key: s for s in at.selectbox}
    descriptor_box = boxes.get("descriptor_id_select")
    if descriptor_box is not None and len(descriptor_box.options) > 1:
        timed(
            "descriptor change rerun",
            lambda: descriptor_box.select(descriptor_box.options[1]).run(),
        )

    formula_box = boxes.get("filter_formula_select")
    if formula_box is not None and len(formula_box.options) > 1:
        timed(
            "formula filter change rerun",
            lambda: formula_box.select(formula_box.options[1]).run(),
        )

    nav = {b.key: b for b in at.button if b.key}
    if "single_calc_next_button" in nav:
        timed(
            "next-molecule button rerun",
            lambda: nav["single_calc_next_button"].click().run(),
        )

    if at.exception:
        print(f"FAILED: {len(at.exception)} exception(s)")
        for exc in at.exception:
            print("  EXC:", str(exc.value)[:300])
        return 1

    print("no exceptions")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(tempfile.gettempdir()) / "iqc_perf_harness" / "thermo_synth.parquet",
        help="Where to keep the synthetic dataset (default: system temp dir).",
    )
    parser.add_argument(
        "--molecules",
        type=int,
        default=3200,
        help="Approximate molecule count for the synthetic dataset.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate the synthetic dataset even if it exists.",
    )
    args = parser.parse_args()

    if args.rebuild or not args.data_path.exists():
        side = max(2, int(round((args.molecules / 2) ** 0.5)))
        timed(
            f"build synthetic dataset ({2 * side * side} molecules)",
            lambda: build_dataset(args.data_path, side, side),
        )
    size_mb = args.data_path.stat().st_size / 1e6
    print(f"dataset: {args.data_path} ({size_mb:.1f} MB)")

    return run(args.data_path)


if __name__ == "__main__":
    raise SystemExit(main())
