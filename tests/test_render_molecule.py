"""Tests for render_molecule function."""

from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import pytest

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from iqc_dashboard.app import (
    ENERGY_UNIT_EV,
    build_geometry_optimization_summary,
    build_vibrational_frequency_table,
    create_ir_spectrum_plot,
    create_molecule_spectrum_plot,
    create_vibrational_stick_plot,
    normalize_spectrum_intensities,
    normalize_vibrational_frequencies,
    parse_xyz_coordinates,
    render_molecule,
    set_molecule_style,
)


class TestRenderMolecule:
    """Test suite for render_molecule function."""

    def test_render_molecule_with_none_xyz(self):
        """Test render_molecule with None XYZ string."""
        with patch("iqc_dashboard.app.st") as mock_st:
            render_molecule(None, label="Test")
            mock_st.warning.assert_called()

    def test_render_molecule_with_empty_xyz(self):
        """Test render_molecule with empty XYZ string."""
        with patch("iqc_dashboard.app.st") as mock_st:
            render_molecule("", label="Test")
            mock_st.warning.assert_called()

    def test_render_molecule_without_stmol(self):
        """Test render_molecule when stmol is not available."""
        with patch("iqc_dashboard.app.STMOL_AVAILABLE", False):
            with patch("iqc_dashboard.app.st") as mock_st:
                xyz = "3\nH2O\nH 0.0 0.0 0.0\nO 0.0 0.0 0.96\nH 0.87 0.0 0.39"
                render_molecule(xyz, label="Test")
                mock_st.error.assert_called()

    def test_render_molecule_without_py3dmol(self):
        """Test render_molecule when py3Dmol is not available."""
        with patch("iqc_dashboard.app.STMOL_AVAILABLE", True):
            with patch("iqc_dashboard.app.PY3DMOL_AVAILABLE", False):
                with patch("iqc_dashboard.app.st") as mock_st:
                    xyz = "3\nH2O\nH 0.0 0.0 0.0\nO 0.0 0.0 0.96\nH 0.87 0.0 0.39"
                    render_molecule(xyz, label="Test")
                    mock_st.error.assert_called()

    def test_render_molecule_success(self):
        """Test successful molecule rendering."""
        # Mock the libraries
        with patch("iqc_dashboard.app.STMOL_AVAILABLE", True):
            with patch("iqc_dashboard.app.PY3DMOL_AVAILABLE", True):
                with patch("iqc_dashboard.app.st"):
                    # Setup mocks for imports inside the function
                    mock_view = MagicMock()
                    mock_py3dmol = MagicMock()
                    mock_py3dmol.view.return_value = mock_view
                    mock_stmol = MagicMock()

                    # Patch sys.modules to intercept imports
                    import sys

                    with patch.dict(sys.modules, {"py3Dmol": mock_py3dmol, "stmol": mock_stmol}):
                        xyz = "3\nH2O\nH 0.0 0.0 0.0\nO 0.0 0.0 0.96\nH 0.87 0.0 0.39"
                        render_molecule(xyz, style="stick", label="Test", show_labels=True)

                    # Verify py3Dmol was called
                    mock_py3dmol.view.assert_called_once_with(width=400, height=400)
                    mock_view.addModel.assert_called_once_with(xyz, "xyz")
                    mock_view.setStyle.assert_called()
                    assert mock_view.addLabel.call_count == 3
                    mock_view.zoomTo.assert_called_once()
                    mock_view.render.assert_called_once()
                    mock_stmol.showmol.assert_called_once()

    def test_render_molecule_different_styles(self):
        """Test render_molecule with different styles."""
        with patch("iqc_dashboard.app.STMOL_AVAILABLE", True):
            with patch("iqc_dashboard.app.PY3DMOL_AVAILABLE", True):
                with patch("iqc_dashboard.app.st"):
                    # Setup mocks for imports inside the function
                    mock_view = MagicMock()
                    mock_py3dmol = MagicMock()
                    mock_py3dmol.view.return_value = mock_view
                    mock_stmol = MagicMock()

                    # Patch sys.modules to intercept imports
                    import sys

                    with patch.dict(sys.modules, {"py3Dmol": mock_py3dmol, "stmol": mock_stmol}):
                        xyz = "3\nH2O\nH 0.0 0.0 0.0\nO 0.0 0.0 0.96\nH 0.87 0.0 0.39"

                        # Test sphere style
                        render_molecule(xyz, style="sphere")
                        mock_view.setStyle.assert_called_with({"sphere": {"radius": 0.5}})

                        # Test wireframe style
                        mock_view.reset_mock()
                        render_molecule(xyz, style="wireframe")
                        mock_view.setStyle.assert_called_with({"line": {"linewidth": 2}})

                        # Test ball and stick style
                        mock_view.reset_mock()
                        render_molecule(xyz, style="ball_and_stick")
                        mock_view.setStyle.assert_called_with(
                            {"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}}
                        )

                        # Test cartoon style
                        mock_view.reset_mock()
                        render_molecule(xyz, style="cartoon")
                        mock_view.setStyle.assert_called_with({"cartoon": {}})

                        # Test default style
                        mock_view.reset_mock()
                        render_molecule(xyz, style="unknown")
                        mock_view.setStyle.assert_called_with(
                            {"stick": {}, "sphere": {"radius": 0.3}}
                        )

    def test_set_molecule_style_wireframe(self):
        """Test wireframe style maps to py3Dmol line rendering."""
        mock_view = MagicMock()

        set_molecule_style(mock_view, "wireframe")

        mock_view.setStyle.assert_called_once_with({"line": {"linewidth": 2}})

    def test_build_geometry_optimization_summary(self):
        """Test geometry optimization summary from initial and optimized XYZ structures."""
        molecule_data = {
            "initial_xyz": (
                "4\ninitial\n"
                "C 0.0 0.0 0.0\n"
                "C 1.5 0.0 0.0\n"
                "C 2.5 1.0 0.0\n"
                "C 3.5 1.0 1.0"
            ),
            "opt_xyz": (
                "4\noptimized\n"
                "C 0.0 0.0 0.0\n"
                "C 1.4 0.0 0.0\n"
                "C 2.5 1.0 0.0\n"
                "C 3.4 1.2 -1.0"
            ),
            "initial_energy_eV": -10.0,
            "opt_energy_eV": -10.2,
        }

        summary = build_geometry_optimization_summary(molecule_data, ENERGY_UNIT_EV)

        assert summary is not None
        assert summary["energy_change"] == pytest.approx(-0.2)
        assert summary["heavy_atom_rmsd"] > 0
        assert summary["max_atom_displacement"] > 0
        assert not summary["bond_changes"].empty
        assert not summary["angle_changes"].empty
        assert not summary["dihedral_changes"].empty

    def test_parse_xyz_coordinates_preserves_blank_comment_line(self):
        """Test standard XYZ with an empty comment line keeps the first atom."""
        xyz = "3\n\nNi 0.0 0.0 0.0\nN 1.0 0.0 0.0\nN -1.0 0.0 0.0\n"

        result = parse_xyz_coordinates(xyz)

        assert result is not None
        elements, coords = result
        assert elements == ["Ni", "N", "N"]
        assert coords.shape == (3, 3)

    def test_build_geometry_optimization_summary_requires_matching_atoms(self):
        """Test geometry summary returns None when atom lists do not match."""
        molecule_data = {
            "initial_xyz": "2\ninitial\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7",
            "opt_xyz": "2\noptimized\nH 0.0 0.0 0.0\nO 0.0 0.0 0.7",
        }

        summary = build_geometry_optimization_summary(molecule_data, ENERGY_UNIT_EV)

        assert summary is None

    def test_normalize_vibrational_frequencies(self):
        """Test vibrational frequency normalization handles iterable numeric input."""
        result = normalize_vibrational_frequencies([100, -25.5, 300])

        assert result is not None
        assert result.tolist() == [100.0, -25.5, 300.0]

    def test_build_vibrational_frequency_table(self):
        """Test vibrational frequency table labels real and imaginary modes."""
        result = build_vibrational_frequency_table([-100, 250, 350])

        assert result["Mode"].tolist() == [1, 2, 3]
        assert result["Frequency (cm^-1)"].tolist() == [-100.0, 250.0, 350.0]
        assert result["Type"].tolist() == ["Imaginary", "Real", "Real"]

    def test_build_vibrational_frequency_table_with_intensities(self):
        """Test vibrational frequency table includes matched spectrum intensities."""
        result = build_vibrational_frequency_table([-100, 250, 350], [0.1, 12.5, 8.0])

        assert result["Intensity"].tolist() == [0.1, 12.5, 8.0]

    def test_normalize_spectrum_intensities_rejects_mismatched_lengths(self):
        """Test spectrum intensity normalization rejects arrays that do not align."""
        result = normalize_spectrum_intensities([1.0, 2.0], expected_length=3)

        assert result is None

    def test_create_vibrational_stick_plot(self):
        """Test vibrational stick plot generation for valid frequencies."""
        fig = create_vibrational_stick_plot([-50, 100, 200], "Test Spectrum")

        assert fig is not None
        assert fig.layout.title.text == "Test Spectrum"
        assert len(fig.data) == 3

    def test_create_vibrational_stick_plot_uses_intensities(self):
        """Test vibrational stick plot uses provided spectrum intensities."""
        fig = create_vibrational_stick_plot(
            [-50, 100, 200],
            "Test Spectrum",
            [2.0, 10.0, 5.0],
        )

        assert fig is not None
        assert list(fig.data[-1].y) == [2.0, 10.0, 5.0]
        assert fig.layout.yaxis.title.text == "IR Intensity"
        assert list(fig.layout.yaxis.range) == [0, 11.0]

    def test_create_vibrational_stick_plot_falls_back_for_mismatched_intensities(self):
        """Test vibrational stick plot ignores spectrum intensities that do not align."""
        fig = create_vibrational_stick_plot(
            [-50, 100, 200],
            "Test Spectrum",
            [2.0, 10.0],
        )

        assert fig is not None
        assert list(fig.data[-1].y) == [1.0, 1.0, 1.0]
        assert fig.layout.yaxis.title.text == "Relative Intensity"

    def test_create_ir_spectrum_plot_uses_schema_arrays_and_units(self):
        """Test IR spectrum plot uses paired spectrum frequency and intensity arrays."""
        fig = create_ir_spectrum_plot(
            [100, 200, 300],
            [0.0, 4.5, 1.2],
            "IR Spectrum",
            "cm^-1",
            "km/mol",
        )

        assert fig is not None
        assert list(fig.data[0].x) == [100.0, 200.0, 300.0]
        assert list(fig.data[0].y) == [0.0, 4.5, 1.2]
        assert fig.data[0].mode == "lines"
        assert fig.layout.xaxis.title.text == "Frequency (cm^-1)"
        assert fig.layout.yaxis.title.text == "IR Intensity (km/mol)"

    def test_create_ir_spectrum_plot_rejects_mismatched_schema_arrays(self):
        """Test IR spectrum plot returns None when spectrum arrays do not align."""
        fig = create_ir_spectrum_plot([100, 200, 300], [0.0, 4.5], "IR Spectrum")

        assert fig is None

    def test_create_molecule_spectrum_plot_prefers_spectrum_schema(self):
        """Test molecule spectrum plot prefers spectrum arrays over vibrational modes."""
        molecule_data = {
            "spectrum_frequencies": [100, 200, 300],
            "spectrum_frequencies_units": "cm^-1",
            "spectrum_intensities": [0.0, 4.5, 1.2],
            "spectrum_intensities_units": "km/mol",
            "vibrational_frequencies_cm^-1": [10, 20],
        }

        fig = create_molecule_spectrum_plot(molecule_data, "IR Spectrum")

        assert fig is not None
        assert list(fig.data[0].x) == [100.0, 200.0, 300.0]
        assert list(fig.data[0].y) == [0.0, 4.5, 1.2]
        assert fig.layout.yaxis.title.text == "IR Intensity (km/mol)"

    def test_create_vibrational_stick_plot_with_invalid_input(self):
        """Test vibrational stick plot returns None for invalid input."""
        fig = create_vibrational_stick_plot("not-a-frequency-list", "Test Spectrum")

        assert fig is None
