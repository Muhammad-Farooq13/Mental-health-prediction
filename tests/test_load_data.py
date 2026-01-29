"""
Unit Tests for Data Loading Module
"""

import pytest
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_raw_data, save_data, get_data_info


class TestLoadData:
    """Test cases for data loading functionality"""
    
    def test_load_raw_data_success(self, tmp_path):
        """Test successful data loading"""
        # Create a temporary CSV file
        test_file = tmp_path / "test_data.csv"
        df_test = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        df_test.to_csv(test_file, index=False)
        
        # Load the data
        df_loaded = load_raw_data(str(test_file))
        
        # Assertions
        assert df_loaded.shape == (3, 2)
        assert list(df_loaded.columns) == ['col1', 'col2']
        pd.testing.assert_frame_equal(df_loaded, df_test)
    
    def test_load_raw_data_file_not_found(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_raw_data('non_existent_file.csv')
    
    def test_save_data(self, tmp_path):
        """Test data saving functionality"""
        # Create test dataframe
        df_test = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Save data
        output_file = tmp_path / "output" / "saved_data.csv"
        save_data(df_test, str(output_file))
        
        # Load and verify
        df_loaded = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(df_loaded, df_test)
    
    def test_get_data_info(self):
        """Test data info extraction"""
        # Create test dataframe
        df_test = pd.DataFrame({
            'col1': [1, 2, 3, None],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        # Get info
        info = get_data_info(df_test)
        
        # Assertions
        assert info['shape'] == (4, 2)
        assert 'col1' in info['columns']
        assert 'col2' in info['columns']
        assert info['missing_values']['col1'] == 1
        assert info['missing_values']['col2'] == 0
