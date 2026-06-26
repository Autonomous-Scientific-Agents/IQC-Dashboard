"""Public entry points for the descriptor kit.

``compute_descriptors(reactant_xyz, product_xyz, *, stereo_type="", insertion_type="")``
    The single-row API.  Takes the two xyz blocks of one reaction (a
    ``reactions.parquet`` row: a Ni·bpy·alkyne reactant and its Ni·bpy·carboxylate
    metallacycle product) and returns a flat ``dict`` of all ``reac_*`` /
    ``prod_*`` descriptors plus the 16 regiochemistry-resolved ``reac_*`` α/β
    descriptors (NaN unless ``insertion_type`` is ``Type_I``/``Type_II``) — 95 keys.

``compute_tdelta(result_type_I, result_type_II)``
    The pair API.  Takes two ``compute_descriptors`` results — the Type_I and
    Type_II regioisomers of the same ligand pair / stereochemistry — and returns
    the 23 ``tdelta_*`` regioisomer-Δ descriptors.

Failure policy mirrors the production pipeline: with ``strict=False`` (default) a
descriptor whose preconditions fail becomes ``NaN`` (other descriptors are
unaffected); with ``strict=True`` the underlying exception propagates.
"""
from __future__ import annotations

from .core import geometry as geom_mod
from .core import topology as topo
from .descriptors import reactant as reac_mod
from .descriptors import product as prod_mod
from .descriptors import pair as pair_mod
from .regio import REGIO_KEYS, add_regio_descriptors, compute_regio


# Output key order: reactant block, then product block, then regio block
# (matches the pipeline's DESCRIPTOR_COLUMNS ordering for the single-row columns).
REACTANT_KEYS = [fn.__name__ for fn in reac_mod.ALL]
PRODUCT_KEYS = [fn.__name__ for fn in prod_mod.ALL]
DESCRIPTOR_KEYS = REACTANT_KEYS + PRODUCT_KEYS + REGIO_KEYS
TDELTA_KEYS = [fn.__name__ for fn in pair_mod.ALL]


def _run_descriptor(fn, obj, out, strict):
    """Call one descriptor fn on ``obj`` and merge its dict into ``out``.

    On failure: re-raise if strict, else leave the fn's key(s) NaN (already
    pre-seeded in ``out``).  Returns the exception (or None) for diagnostics.
    """
    try:
        out.update(fn(obj))
        return None
    except Exception as exc:  # noqa: BLE001 - per-descriptor NaN containment
        if strict:
            raise
        return exc


def _run_block(fns, obj, out, strict, diagnostics):
    """Run a block of descriptor fns on ``obj``, merging results into ``out``."""
    for fn in fns:
        exc = _run_descriptor(fn, obj, out, strict)
        if exc is not None and diagnostics is not None:
            diagnostics.append((fn.__name__, f"{type(exc).__name__}: {exc}"))


def compute_descriptors(reactant_xyz: str, product_xyz: str, *,
                        stereo_type: str = "",
                        insertion_type: str = "",
                        strict: bool = False,
                        diagnostics: list | None = None) -> dict:
    """Compute every single-row descriptor for one reactant+product pair.

    Parameters
    ----------
    reactant_xyz, product_xyz : str
        xyz blocks (standard xyz: count line, comment line, then ``El x y z``).
    stereo_type : str
        Optional "S" / "O" label used to assign PyA/PyB for per-ring reactant
        descriptors.
    insertion_type : str
        Optional "Type_I" / "Type_II" label used to compute regio α/β descriptors.
    strict : bool
        If True, the first failing descriptor (or identification step) raises.
        If False (default), failures become NaN and computation continues.
    diagnostics : list | None
        If provided, ``(key, "ExcType: msg")`` tuples are appended for every
        descriptor that failed (and ``("_identification", ...)`` if the whole row
        could not be built).

    Returns
    -------
    dict
        ``{descriptor_key: float}`` for all ``reac_*`` / ``prod_*`` / ``reac_*_ab`` keys.
    """
    out = {k: float("nan") for k in DESCRIPTOR_KEYS}

    # --- build geoms + identify (one barrier: if this fails, all NaN) ---
    try:
        reactant = topo.identify_reactant(
            geom_mod.build_geom(*geom_mod.parse_xyz(reactant_xyz)),
            stereo_type=stereo_type,
        )
        product = topo.identify_product(
            geom_mod.build_geom(*geom_mod.parse_xyz(product_xyz)))
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        if diagnostics is not None:
            diagnostics.append(("_identification", f"{type(exc).__name__}: {exc}"))
        return out

    # --- per-descriptor calls, each independently contained ---
    _run_block(reac_mod.ALL, reactant, out, strict, diagnostics)
    _run_block(prod_mod.ALL, product, out, strict, diagnostics)
    out.update(compute_regio(out, insertion_type))
    return out


def compute_regio_descriptors(reactant_xyz: str, *,
                              stereo_type: str = "",
                              insertion_type: str = "",
                              strict: bool = False,
                              diagnostics: list | None = None) -> dict:
    """Compute the 16 regio (α/β) descriptors from a reactant geometry alone.

    All regio sources are ``reac_*``, so only the reactant is needed.  Returns
    all ``REGIO_KEYS`` (NaN where ``insertion_type`` is empty/unknown, the
    reactant cannot be identified, or a source descriptor fails).
    """
    out = {k: float("nan") for k in REGIO_KEYS}
    try:
        reactant = topo.identify_reactant(
            geom_mod.build_geom(*geom_mod.parse_xyz(reactant_xyz)),
            stereo_type=stereo_type,
        )
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        if diagnostics is not None:
            diagnostics.append(("_identification", f"{type(exc).__name__}: {exc}"))
        return out

    reac_values = {k: float("nan") for k in REACTANT_KEYS}
    _run_block(reac_mod.ALL, reactant, reac_values, strict, diagnostics)
    out.update(compute_regio(reac_values, insertion_type))
    return out


def compute_tdelta(result_type_I: dict, result_type_II: dict) -> dict:
    """Compute the 23 ``tdelta_*`` regioisomer-Δ descriptors from two single-row
    results.

    Parameters
    ----------
    result_type_I, result_type_II : dict
        ``compute_descriptors`` outputs for the Type_I and Type_II regioisomers
        (only their ``prod_*`` keys are read).

    Returns
    -------
    dict
        ``{tdelta_key: float}`` for all 23 keys (NaN where a source value is
        missing or NaN).
    """
    out = {}
    for fn in pair_mod.ALL:
        out.update(fn(result_type_I, result_type_II))
    return out
