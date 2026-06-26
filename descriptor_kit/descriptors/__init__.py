"""Descriptor function modules: one public function per output key.

* ``reactant`` — 54 ``reac_*`` functions, each taking a ``Reactant``.
* ``product``  — 25 ``prod_*`` functions, each taking a ``Product``.
* ``pair``     — 23 ``tdelta_*`` functions, each taking two single-row results.
"""
from . import reactant, product, pair  # noqa: F401
