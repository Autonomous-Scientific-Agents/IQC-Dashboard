"""Self-contained demo — runs offline from the bundled xyz files, no data files
or pandas required.

    .venv/bin/python3 descriptor_kit/example/run_example.py
      (or, from the directory containing the package:)
    python -m descriptor_kit.example.run_example

It computes the descriptor dict for one reactant+product pair, then the 23
tdelta_* deltas for the bundled Type_I / Type_II regioisomer pair.
"""
from __future__ import annotations
import os
import sys

# Make ``import descriptor_kit`` work no matter the CWD: add the directory that
# contains the descriptor_kit package to sys.path.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from descriptor_kit import compute_descriptors, compute_tdelta  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _xyz(name):
    with open(os.path.join(_HERE, name)) as fh:
        return fh.read()


def main():
    # --- single-row descriptors for the Type_I pair ---
    desc = compute_descriptors(_xyz("type_I_reactant.xyz"),
                               _xyz("type_I_product.xyz"),
                               stereo_type="S")
    print(f"{len(desc)} single-row descriptors (bundled Type_I reaction):")
    for k, v in desc.items():
        print(f"  {k:32s} {v:.6g}")

    # --- tdelta_* for the Type_I / Type_II regioisomer pair ---
    dI = desc
    dII = compute_descriptors(_xyz("type_II_reactant.xyz"),
                              _xyz("type_II_product.xyz"),
                              stereo_type="S")
    deltas = compute_tdelta(dI, dII)
    print(f"\n{len(deltas)} tdelta_* descriptors (Type_I - Type_II):")
    for k, v in deltas.items():
        print(f"  {k:32s} {v:.6g}")


if __name__ == "__main__":
    main()
