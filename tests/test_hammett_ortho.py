"""Regression tests for Hammett position handling and XYZ input validation."""

import math
from pathlib import Path

import pytest

from descriptor_kit import compute_descriptors
from descriptor_kit.core import geometry, hammett, sigma_data


EXAMPLE_DIR = Path(__file__).parent.parent / "descriptor_kit" / "example"
METHYL_SMILES = "[H]C([H])([H])[H]"


class _StubGeom:
    """Minimal geom stand-in: sigma_for_fragment only reads ``elements``."""

    elements = ["C"]


@pytest.fixture
def methyl_fragment(monkeypatch):
    """Make sigma_for_fragment perceive every fragment as methyl."""
    monkeypatch.setattr(
        hammett,
        "fragment_smiles",
        lambda geom, root_idx, frag_atoms: METHYL_SMILES,
    )
    return sigma_data.SIGMA_TABLE[METHYL_SMILES]


class TestSigmaForFragmentPositions:
    """Position 6 is ortho and must NOT receive the para constant.

    See ortho_bug.md: the old code returned sigma_p for ``position in (4, 6)``,
    silently assigning every ortho substituent the para electronic parameter.
    """

    def test_position_4_returns_sigma_p(self, methyl_fragment):
        value = hammett.sigma_for_fragment(_StubGeom(), 0, {0}, 4)
        assert value == pytest.approx(methyl_fragment["sigma_p"])

    @pytest.mark.parametrize("position", [3, 5])
    def test_meta_positions_return_sigma_m(self, methyl_fragment, position):
        value = hammett.sigma_for_fragment(_StubGeom(), 0, {0}, position)
        assert value == pytest.approx(methyl_fragment["sigma_m"])

    def test_position_6_returns_nan_not_sigma_p(self, methyl_fragment):
        value = hammett.sigma_for_fragment(_StubGeom(), 0, {0}, 6)
        assert math.isnan(value)
        # Guard the guard: methyl has a tabulated sigma_p, so a regression to
        # the old 4/6 branch would return a finite value here.
        assert not math.isnan(methyl_fragment["sigma_p"])

    @pytest.mark.parametrize("position", [1, 2])
    def test_skipped_positions_return_nan(self, methyl_fragment, position):
        assert math.isnan(hammett.sigma_for_fragment(_StubGeom(), 0, {0}, position))


class TestParseXyzValidation:
    def test_valid_block_round_trips(self):
        block = "2\ncomment\nC 0.0 0.0 0.0\nO 1.2 0.0 0.0"
        elements, coords = geometry.parse_xyz(block)
        assert elements == ["C", "O"]
        assert coords.shape == (2, 3)

    def test_truncated_block_raises(self):
        block = "5\ncomment\nC 0.0 0.0 0.0\nO 1.2 0.0 0.0"
        with pytest.raises(ValueError, match="declares 5 atoms"):
            geometry.parse_xyz(block)

    def test_truncated_reactant_is_contained_with_diagnostics(self):
        product_xyz = (EXAMPLE_DIR / "type_I_product.xyz").read_text(
            encoding="utf-8"
        )
        truncated_reactant = "9999\ncomment\nC 0.0 0.0 0.0"

        diagnostics = []
        values = compute_descriptors(
            truncated_reactant,
            product_xyz,
            diagnostics=diagnostics,
        )

        assert all(math.isnan(value) for value in values.values())
        assert diagnostics, "identification failure must be recorded"
        assert diagnostics[0][0] == "_identification"
        assert "9999" in diagnostics[0][1]
