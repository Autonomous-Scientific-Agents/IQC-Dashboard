"""Tests for DataManager class."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from iqc_dashboard.app import (
    DataManager,
    ENERGY_UNIT_EV,
    ENERGY_UNIT_KCAL,
    build_all_data_table,
    build_comparison_metric_table,
    build_optimized_geometry_comparison,
    build_row_comparison_table,
    build_ligand_selector_df,
    calculate_reaction_gibbs,
    calculate_reaction_table,
    build_descriptor_dataframe,
    convert_energy_value,
    create_comparison_spectrum_plot,
    energy_metadata_label,
    parquet_files_have_same_dimensions,
    parquet_files_have_same_schema,
)


class TestDataManager:
    """Test suite for DataManager class."""

    def test_init(self, temp_dir):
        """Test DataManager initialization."""
        dm = DataManager(temp_dir)
        assert dm.temp_dir == Path(temp_dir)
        assert dm.parquet_files == []
        assert dm.temp_dir.exists()

    def test_save_uploaded_files(self, temp_dir):
        """Test saving uploaded files."""
        dm = DataManager(temp_dir)

        # Create mock uploaded files
        mock_file1 = Mock()
        mock_file1.name = "test1.parquet"
        mock_file1.getbuffer.return_value = b"test data 1"

        mock_file2 = Mock()
        mock_file2.name = "test2.parquet"
        mock_file2.getbuffer.return_value = b"test data 2"

        uploaded_files = [mock_file1, mock_file2]
        saved_paths = dm.save_uploaded_files(uploaded_files)

        assert len(saved_paths) == 2
        assert all(Path(p).exists() for p in saved_paths)
        assert dm.parquet_files == saved_paths

    def test_save_uploaded_files_with_none(self, temp_dir):
        """Test saving uploaded files with None values."""
        dm = DataManager(temp_dir)

        mock_file = Mock()
        mock_file.name = "test.parquet"
        mock_file.getbuffer.return_value = b"test data"

        uploaded_files = [mock_file, None]
        saved_paths = dm.save_uploaded_files(uploaded_files)

        assert len(saved_paths) == 1
        assert Path(saved_paths[0]).exists()

    def test_load_data_paths_converts_generic_json(self, temp_dir):
        """Test loading a JSON records file through the Parquet-backed query path."""
        json_path = Path(temp_dir) / "molecules.json"
        pd.DataFrame(
            {
                "unique_name": ["mol_json_001", "mol_json_002"],
                "formula": ["H2O", "CO2"],
                "opt_converged": [True, False],
            }
        ).to_json(json_path, orient="records")

        dm = DataManager(temp_dir)
        loaded_paths = dm.load_data_paths([str(json_path)])

        assert len(loaded_paths) == 1
        assert loaded_paths[0].endswith(".parquet")
        assert Path(loaded_paths[0]).exists()
        loaded_df = pd.read_parquet(loaded_paths[0])
        assert loaded_df["unique_name"].tolist() == [
            "mol_json_001",
            "mol_json_002",
        ]

    def test_load_data_paths_expands_reaction_json(self, temp_dir):
        """Test reaction-level JSON expands into descriptor-ready molecule rows."""
        reactant_xyz = """5

Ni 0.0 0.0 0.0
N -2.0 0.0 0.0
N 2.0 0.0 0.0
C 0.0 2.0 0.0
O 0.0 -2.0 0.0
"""
        product_xyz = """6

Ni 0.0 0.0 0.2
N -2.0 0.0 0.0
N 2.4 0.0 0.0
O 0.0 2.0 0.2
C 0.0 -2.0 0.2
C -2.0 1.3 0.0
"""
        json_path = Path(temp_dir) / "reaction_data.json"
        pd.DataFrame(
            {
                "ligand_pair": ["bipy-aaeaaeaa_f-C2H2-e"],
                "stereo_type": ["Type_I"],
                "insertion_type": ["intermediate"],
                "reaction_gibbs_kcal": [-2.45],
                "reactant_gibbs": [-307.8],
                "product_gibbs": [-330.1],
                "reactant_geometry": [reactant_xyz],
                "product_geometry": [product_xyz],
                "reactant_configuration": ["reactant_conf"],
                "product_configuration": ["product_conf"],
            }
        ).to_json(json_path)

        dm = DataManager(temp_dir)
        loaded_paths = dm.load_data_paths([str(json_path)])
        loaded_df = pd.read_parquet(loaded_paths[0])

        assert loaded_df["reaction_role"].tolist() == ["reactant", "product"]
        assert loaded_df["unique_name"].tolist() == [
            "bipy-aaeaaeaa_f-C2H2-e_reactant_reactant_conf_Type_I_intermediate_0",
            "bipy-aaeaaeaa_f-C2H2-e_product_product_conf_Type_I_intermediate_0",
        ]
        assert loaded_df["opt_xyz"].tolist() == [reactant_xyz, product_xyz]
        assert loaded_df["number_of_atoms"].tolist() == [5, 6]

        descriptor_df = build_descriptor_dataframe(loaded_df)
        assert not descriptor_df.empty
        assert set(descriptor_df["role"]) == {"reactant", "product"}

    @patch("iqc_dashboard.app.duckdb")
    def test_get_connection(self, mock_duckdb, temp_dir):
        """Test getting DuckDB connection."""
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn

        # Mock streamlit cache_resource
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            conn = DataManager.get_connection()
            assert conn == mock_conn
            mock_duckdb.connect.assert_called_once()

    def test_get_summary_stats_empty(self, temp_dir):
        """Test get_summary_stats with no files."""
        dm = DataManager(temp_dir)
        parquet_hash = dm._get_parquet_files_hash()
        result = dm.get_summary_stats(parquet_hash)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_summary_stats_with_data(self, temp_dir, sample_parquet_file):
        """Test get_summary_stats with actual data."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Setup mock connection
        mock_conn = Mock()
        mock_execute = Mock()
        mock_conn.execute.return_value = mock_execute

        # Return actual DataFrame with summary stats
        summary_df = pd.DataFrame(
            {
                "total_rows": [3],
                "unique_formulas": [3],
                "converged_count": [2],
                "not_converged_count": [1],
            }
        )
        mock_execute.df.return_value = summary_df

        # Mock streamlit and patch get_connection
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            with patch.object(DataManager, "get_connection", return_value=mock_conn):
                parquet_hash = dm._get_parquet_files_hash()
                result = dm.get_summary_stats(parquet_hash)

                assert isinstance(result, pd.DataFrame)
                assert not result.empty
                assert "total_rows" in result.columns
                assert result["total_rows"].iloc[0] == 3

    def test_get_filtered_data_empty(self, temp_dir):
        """Test get_filtered_data with no files."""
        dm = DataManager(temp_dir)
        result = dm.get_filtered_data()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_filtered_data_with_filters(self, temp_dir, sample_parquet_file):
        """Test get_filtered_data with filters."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Setup mock connection
        mock_conn = Mock()
        mock_execute = Mock()
        mock_conn.execute.return_value = mock_execute

        # Return filtered DataFrame
        filtered_df = pd.DataFrame(
            {
                "unique_name": ["mol_001", "mol_002"],
                "formula": ["H2O", "CO2"],
                "calculator": ["dft", "dft"],
                "opt_converged": [True, True],
            }
        )
        mock_execute.df.return_value = filtered_df

        # Mock streamlit and patch get_connection
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            with patch.object(DataManager, "get_connection", return_value=mock_conn):
                result = dm.get_filtered_data(calculator="dft", opt_converged=True)

                assert isinstance(result, pd.DataFrame)
                # Should filter to converged molecules
                if not result.empty:
                    assert all(result["opt_converged"])

    def test_get_unique_values(self, temp_dir, sample_parquet_file):
        """Test get_unique_values."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Setup mock connection
        mock_conn = Mock()
        mock_execute = Mock()
        mock_conn.execute.return_value = mock_execute

        # Return DataFrame with unique values
        values_df = pd.DataFrame({"formula": ["H2O", "CO2", "NH3"]})
        mock_execute.df.return_value = values_df

        # Mock streamlit and patch get_connection
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            with patch.object(DataManager, "get_connection", return_value=mock_conn):
                parquet_hash = dm._get_parquet_files_hash()
                values = dm.get_unique_values("formula", parquet_hash)

                assert isinstance(values, list)
                assert "H2O" in values
                assert "CO2" in values
                assert "NH3" in values

    def test_get_unique_values_empty(self, temp_dir):
        """Test get_unique_values with no files."""
        dm = DataManager(temp_dir)
        parquet_hash = dm._get_parquet_files_hash()
        values = dm.get_unique_values("formula", parquet_hash)
        assert values == []

    def test_get_molecule_by_name(self, temp_dir, sample_parquet_file):
        """Test get_molecule_by_name."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Setup mock connection
        mock_conn = Mock()
        mock_execute = Mock()
        mock_conn.execute.return_value = mock_execute

        # Return DataFrame with molecule data
        molecule_df = pd.DataFrame(
            {"unique_name": ["mol_001"], "formula": ["H2O"], "number_of_atoms": [3]}
        )
        mock_execute.df.return_value = molecule_df

        # Mock streamlit and patch get_connection
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            with patch.object(DataManager, "get_connection", return_value=mock_conn):
                molecule = dm.get_molecule_by_name("mol_001")

                assert molecule is not None
                assert molecule["formula"] == "H2O"
                assert molecule["unique_name"] == "mol_001"

    def test_get_molecule_by_name_not_found(self, temp_dir, sample_parquet_file):
        """Test get_molecule_by_name with non-existent molecule."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Mock streamlit
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            molecule = dm.get_molecule_by_name("nonexistent")
            assert molecule is None

    def test_get_molecule_by_index(self, temp_dir, sample_parquet_file):
        """Test get_molecule_by_index."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Setup mock connection
        mock_conn = Mock()
        mock_execute = Mock()
        mock_conn.execute.return_value = mock_execute

        # Return DataFrame with molecule data
        molecule_df = pd.DataFrame(
            {"unique_name": ["mol_001"], "formula": ["H2O"], "number_of_atoms": [3]}
        )
        mock_execute.df.return_value = molecule_df

        # Mock streamlit and patch get_connection
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            with patch.object(DataManager, "get_connection", return_value=mock_conn):
                molecule = dm.get_molecule_by_index(0)

                assert molecule is not None
                assert molecule["unique_name"] == "mol_001"

    def test_get_all_molecule_names(self, temp_dir, sample_parquet_file):
        """Test get_all_molecule_names."""
        dm = DataManager(temp_dir)
        dm.parquet_files = [sample_parquet_file]

        # Setup mock connection
        mock_conn = Mock()
        mock_execute = Mock()
        mock_conn.execute.return_value = mock_execute

        # Return DataFrame with molecule names
        names_df = pd.DataFrame({"unique_name": ["mol_001", "mol_002", "mol_003"]})
        mock_execute.df.return_value = names_df

        # Mock streamlit and patch get_connection
        with patch("iqc_dashboard.app.st") as mock_st:
            mock_st.cache_resource = lambda x: x
            with patch.object(DataManager, "get_connection", return_value=mock_conn):
                parquet_hash = dm._get_parquet_files_hash()
                names = dm.get_all_molecule_names(parquet_hash)

                assert isinstance(names, list)
                assert len(names) == 3
                assert "mol_001" in names
                assert "mol_002" in names
                assert "mol_003" in names

    def test_get_comparison_data_matches_rows_by_initial_smiles(self, temp_dir):
        """Test comparison data matches rows with identical initial molecules."""
        df_one = pd.DataFrame(
            {
                "unique_name": ["mol_water_a", "mol_co2_a", "mol_nh3_a"],
                "initial_smiles": ["O", "O=C=O", "N"],
                "formula": ["H2O", "CO2", "NH3"],
                "opt_energy_eV": [-10.0, -20.0, -15.0],
                "opt_converged": [True, True, False],
            }
        )
        df_two = pd.DataFrame(
            {
                "unique_name": ["mol_nh3_b", "mol_water_b", "mol_co2_b"],
                "initial_smiles": ["N", "O", "O=C=O"],
                "formula": ["NH3", "H2O", "CO2"],
                "opt_energy_eV": [-14.8, -10.2, -20.3],
                "opt_converged": [True, True, True],
            }
        )

        file_one = Path(temp_dir) / "method_a.parquet"
        file_two = Path(temp_dir) / "method_b.parquet"
        df_one.to_parquet(file_one)
        df_two.to_parquet(file_two)

        dm = DataManager(temp_dir)
        dm.parquet_files = [str(file_one), str(file_two)]
        parquet_hash = dm._get_parquet_files_hash()

        summaries = dm.get_parquet_file_summaries(parquet_hash)
        assert parquet_files_have_same_dimensions(summaries)
        assert parquet_files_have_same_schema(summaries)

        comparison = dm.get_comparison_data(parquet_hash, "initial_smiles")
        matched_rows = comparison["matched_rows"]

        assert comparison["error"] is None
        assert matched_rows["_comparison_match_id"].nunique() == 3
        assert len(matched_rows) == 6

        water_rows = matched_rows[matched_rows["_comparison_key"] == "O"].sort_values(
            "_comparison_file_order"
        )
        assert water_rows["unique_name"].tolist() == ["mol_water_a", "mol_water_b"]
        assert water_rows["_comparison_row_number"].tolist() == [1, 2]

        metric_table = build_comparison_metric_table(
            matched_rows,
            "opt_energy_eV",
            ENERGY_UNIT_EV,
        )
        water_metric = metric_table[metric_table["Initial Molecule"] == "O"].iloc[0]
        assert water_metric["method_a.parquet"] == pytest.approx(-10.0)
        assert water_metric["method_b.parquet"] == pytest.approx(-10.2)
        assert water_metric["Range"] == pytest.approx(0.2)

        row_comparison = build_row_comparison_table(water_rows, ENERGY_UNIT_EV)
        assert "unique_name" in row_comparison["Field"].tolist()
        assert "opt_energy (eV)" in row_comparison["Field"].tolist()

    def test_get_comparison_data_rejects_dimension_mismatch(self, temp_dir):
        """Test comparison data rejects parquet files with different dimensions."""
        df_one = pd.DataFrame(
            {
                "unique_name": ["mol_water_a", "mol_co2_a", "mol_nh3_a"],
                "initial_smiles": ["O", "O=C=O", "N"],
                "opt_energy_eV": [-10.0, -20.0, -15.0],
            }
        )
        df_two = df_one.head(2).copy()

        file_one = Path(temp_dir) / "method_a.parquet"
        file_two = Path(temp_dir) / "method_b.parquet"
        df_one.to_parquet(file_one)
        df_two.to_parquet(file_two)

        dm = DataManager(temp_dir)
        dm.parquet_files = [str(file_one), str(file_two)]
        parquet_hash = dm._get_parquet_files_hash()

        summaries = dm.get_parquet_file_summaries(parquet_hash)
        assert not parquet_files_have_same_dimensions(summaries)

        comparison = dm.get_comparison_data(parquet_hash, "initial_smiles")

        assert comparison["matched_rows"].empty
        assert comparison["error"] == "Parquet files must have matching row and column counts."

    def test_build_optimized_geometry_comparison(self):
        """Test optimized geometry comparison reports structure deltas across files."""
        matched_rows = pd.DataFrame(
            {
                "_comparison_file_label": ["method_a.parquet", "method_b.parquet"],
                "_comparison_file_order": [0, 1],
                "unique_name": ["mol_a", "mol_b"],
                "opt_xyz": [
                    (
                        "4\nA\n"
                        "C 0.0 0.0 0.0\n"
                        "C 1.5 0.0 0.0\n"
                        "C 2.5 1.0 0.0\n"
                        "C 3.5 1.0 1.0"
                    ),
                    (
                        "4\nB\n"
                        "C 0.0 0.0 0.0\n"
                        "C 1.4 0.0 0.0\n"
                        "C 2.5 1.1 0.0\n"
                        "C 3.6 1.2 1.0"
                    ),
                ],
            }
        )

        result = build_optimized_geometry_comparison(
            matched_rows,
            "method_a.parquet",
        )

        assert result["errors"] == []
        assert result["metrics"]["Comparison File"].tolist() == ["method_b.parquet"]
        assert result["metrics"]["Heavy-atom RMSD (Å)"].iloc[0] > 0
        assert not result["bond_changes"].empty
        assert not result["angle_changes"].empty
        assert {"Reference (Å)", "Comparison (Å)", "Δ (Å)"}.issubset(
            result["bond_changes"].columns
        )
        assert {"Reference (°)", "Comparison (°)", "Δ (°)"}.issubset(
            result["angle_changes"].columns
        )

    def test_create_comparison_spectrum_plot_prefers_ir_spectrum(self):
        """Test comparison spectrum plot overlays paired IR spectra when available."""
        matched_rows = pd.DataFrame(
            {
                "_comparison_file_label": ["method_a.parquet", "method_b.parquet"],
                "_comparison_file_order": [0, 1],
                "spectrum_frequencies": [[100, 200, 300], [100, 200, 300]],
                "spectrum_intensities": [[0.1, 0.5, 0.2], [0.2, 0.4, 0.3]],
                "spectrum_frequencies_units": ["cm^-1", "cm^-1"],
                "spectrum_intensities_units": ["km/mol", "km/mol"],
                "vibrational_frequencies_cm^-1": [[90, 190, 290], [95, 195, 295]],
            }
        )

        fig, summary, mode = create_comparison_spectrum_plot(
            matched_rows,
            "Spectrum Comparison",
        )

        assert fig is not None
        assert mode == "IR spectrum"
        assert len(fig.data) == 2
        assert summary["Data"].tolist() == ["IR spectrum", "IR spectrum"]

    def test_create_comparison_spectrum_plot_falls_back_to_frequencies(self):
        """Test comparison spectrum plot falls back to vibrational frequencies."""
        matched_rows = pd.DataFrame(
            {
                "_comparison_file_label": ["method_a.parquet", "method_b.parquet"],
                "_comparison_file_order": [0, 1],
                "vibrational_frequencies_cm^-1": [[-25, 100, 200], [10, 110, 205]],
            }
        )

        fig, summary, mode = create_comparison_spectrum_plot(
            matched_rows,
            "Frequency Comparison",
        )

        assert fig is not None
        assert mode == "Vibrational frequencies"
        assert len(fig.data) >= 2
        assert summary["Data"].tolist() == [
            "Vibrational frequencies",
            "Vibrational frequencies",
        ]

    def test_calculate_reaction_gibbs_kcal_conversion(self):
        """Test reaction Gibbs calculation converts eV to kcal/mol and computes ΔG."""
        test_df = pd.DataFrame(
            {
                # parse_unique_name expects underscore-separated name elements
                "unique_name": ["bipy-A_C2H2_reactant", "bipy-A_C2H2_product", "CO2"],
                "G_eV": [0.0, 1.0, -0.5],
            }
        )

        result = calculate_reaction_gibbs(test_df)

        assert "deltaG" in result.columns
        assert "G_reactant" in result.columns
        assert "G_product" in result.columns
        assert "G_CO2" in result.columns

        # Check conversion from eV to kcal/mol: 1 eV ~ 23.0605 kcal/mol
        expected_reactant = 0.0 * 23.0605
        expected_product = 1.0 * 23.0605
        expected_co2 = -0.5 * 23.0605
        expected_delta = expected_product - (expected_reactant + expected_co2)

        assert result["G_reactant"].iloc[0] == pytest.approx(expected_reactant)
        assert result["G_product"].iloc[0] == pytest.approx(expected_product)
        assert result["G_CO2"].iloc[0] == pytest.approx(expected_co2)
        assert result["deltaG"].iloc[0] == pytest.approx(expected_delta)

    def test_calculate_reaction_gibbs_ev_output(self):
        """Test reaction Gibbs calculation can return eV values."""
        test_df = pd.DataFrame(
            {
                "unique_name": ["bipy-A_C2H2_reactant", "bipy-A_C2H2_product", "CO2"],
                "G_eV": [0.0, 1.0, -0.5],
            }
        )

        result = calculate_reaction_gibbs(test_df, energy_unit=ENERGY_UNIT_EV)

        assert result["G_reactant"].iloc[0] == pytest.approx(0.0)
        assert result["G_product"].iloc[0] == pytest.approx(1.0)
        assert result["G_CO2"].iloc[0] == pytest.approx(-0.5)
        assert result["deltaG"].iloc[0] == pytest.approx(1.5)

    def test_calculate_reaction_table_uses_precomputed_json_delta(self):
        """Test reaction table uses reaction_gibbs_kcal when G_eV is unavailable."""
        test_df = pd.DataFrame(
            {
                "source_json_row": [0, 0],
                "reaction_role": ["reactant", "product"],
                "unique_name": [
                    "bipy-A_C2H2_reactant_conf",
                    "bipy-A_C2H2_product_conf",
                ],
                "reaction_gibbs_kcal": [-2.5, -2.5],
                "source_gibbs": [-307.8, -330.1],
            }
        )

        result = calculate_reaction_table(test_df, energy_unit=ENERGY_UNIT_KCAL)

        assert len(result) == 1
        assert result["reaction_data_source"].iloc[0] == "precomputed_json"
        assert result["deltaG"].iloc[0] == pytest.approx(-2.5)
        assert result["reaction_gibbs_kcal"].iloc[0] == pytest.approx(-2.5)
        assert result["G_reactant"].iloc[0] == pytest.approx(-307.8)
        assert result["G_product"].iloc[0] == pytest.approx(-330.1)
        assert pd.isna(result["G_CO2"].iloc[0])

    def test_calculate_reaction_table_converts_precomputed_json_delta_to_ev(self):
        """Test precomputed reaction_gibbs_kcal values convert to eV display units."""
        test_df = pd.DataFrame(
            {
                "source_json_row": [0, 0],
                "reaction_role": ["reactant", "product"],
                "unique_name": [
                    "bipy-A_C2H2_reactant_conf",
                    "bipy-A_C2H2_product_conf",
                ],
                "reaction_gibbs_kcal": [23.0605, 23.0605],
                "source_gibbs": [-307.8, -330.1],
            }
        )

        result = calculate_reaction_table(test_df, energy_unit=ENERGY_UNIT_EV)

        assert result["deltaG"].iloc[0] == pytest.approx(1.0)

    def test_energy_unit_conversion_helpers(self):
        """Test scalar energy conversion and metadata labels."""
        assert convert_energy_value(1.0, ENERGY_UNIT_KCAL) == pytest.approx(23.0605)
        assert convert_energy_value(1.0, ENERGY_UNIT_EV) == pytest.approx(1.0)
        assert energy_metadata_label("G_eV", ENERGY_UNIT_KCAL) == "G (kcal/mol)"
        assert energy_metadata_label("S_eV/K", ENERGY_UNIT_KCAL) == "S (kcal/mol/K)"

    def test_invalid_energy_unit_raises(self):
        """Test invalid energy units fail clearly."""
        with pytest.raises(ValueError):
            convert_energy_value(1.0, "hartree")

    def test_build_all_data_table_includes_every_field(self):
        """Test selected molecule All Data table includes all row fields."""
        molecule_data = pd.Series(
            {
                "unique_name": "mol_001",
                "G_eV": 1.0,
                "spectrum_intensities": [0.1, 0.2],
                "custom_field": "custom value",
                "missing_value": float("nan"),
            }
        )

        result = build_all_data_table(molecule_data, ENERGY_UNIT_KCAL)

        assert result["Field"].tolist() == [
            "unique_name",
            "G (kcal/mol)",
            "spectrum_intensities",
            "custom_field",
            "missing_value",
        ]
        assert result["Value"].tolist() == [
            "mol_001",
            "23.060500 kcal/mol",
            "[0.1, 0.2]",
            "custom value",
            "N/A",
        ]

    def test_build_ligand_selector_df_filters_non_reaction_rows(self):
        """Test ligand selector data only keeps rows with parsed bipyridine and ligand values."""
        test_df = pd.DataFrame(
            {
                "unique_name": [
                    "bipy-A_C2H2_reactant",
                    "bipy-A_C2H2_product",
                    "CO2",
                    "plain_molecule_name",
                ]
            }
        )

        result = build_ligand_selector_df(test_df)

        assert list(result.columns) == ["unique_name", "bipyridine", "alkyne", "role"]
        assert result["unique_name"].tolist() == [
            "bipy-A_C2H2_reactant",
            "bipy-A_C2H2_product",
        ]
        assert result["bipyridine"].tolist() == ["A", "A"]
        assert result["alkyne"].tolist() == ["C2H2", "C2H2"]
        assert result["role"].tolist() == ["reactant", "product"]
