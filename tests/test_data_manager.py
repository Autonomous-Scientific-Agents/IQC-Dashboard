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
    build_ligand_selector_df,
    calculate_reaction_gibbs,
    convert_energy_value,
    energy_metadata_label,
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
