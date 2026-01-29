"""
Unit Tests for Data Preprocessing Module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for data preprocessing functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'Z']
        })
    
    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values"""
        return pd.DataFrame({
            'num1': [1, 2, None, 4, 5],
            'num2': [10, None, 30, 40, 50],
            'cat1': ['A', 'B', None, 'C', 'B']
        })
    
    def test_identify_column_types(self, sample_data):
        """Test column type identification"""
        preprocessor = DataPreprocessor()
        preprocessor.identify_column_types(sample_data)
        
        assert len(preprocessor.numerical_cols) == 2
        assert len(preprocessor.categorical_cols) == 2
        assert 'num1' in preprocessor.numerical_cols
        assert 'cat1' in preprocessor.categorical_cols
    
    def test_handle_missing_values_mean(self, data_with_missing):
        """Test missing value handling with mean strategy"""
        preprocessor = DataPreprocessor()
        preprocessor.identify_column_types(data_with_missing)
        
        df_cleaned = preprocessor.handle_missing_values(data_with_missing, strategy='mean')
        
        # Check no missing values in numerical columns
        assert df_cleaned['num1'].isnull().sum() == 0
        assert df_cleaned['num2'].isnull().sum() == 0
    
    def test_handle_missing_values_drop(self, data_with_missing):
        """Test missing value handling with drop strategy"""
        preprocessor = DataPreprocessor()
        df_cleaned = preprocessor.handle_missing_values(data_with_missing, strategy='drop')
        
        # Should have fewer rows
        assert len(df_cleaned) < len(data_with_missing)
        assert df_cleaned.isnull().sum().sum() == 0
    
    def test_encode_categorical_variables(self, sample_data):
        """Test categorical variable encoding"""
        preprocessor = DataPreprocessor()
        preprocessor.identify_column_types(sample_data)
        
        df_encoded = preprocessor.encode_categorical_variables(sample_data, fit=True)
        
        # Check that categorical columns are now numeric
        assert df_encoded['cat1'].dtype in [np.int32, np.int64]
        assert df_encoded['cat2'].dtype in [np.int32, np.int64]
    
    def test_scale_numerical_features(self, sample_data):
        """Test numerical feature scaling"""
        preprocessor = DataPreprocessor()
        preprocessor.identify_column_types(sample_data)
        
        df_scaled = preprocessor.scale_numerical_features(sample_data, fit=True)
        
        # Check that scaled values have mean ~0 and std ~1
        assert abs(df_scaled['num1'].mean()) < 0.1
        assert abs(df_scaled['num1'].std() - 1.0) < 0.1
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        df_with_dupes = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        preprocessor = DataPreprocessor()
        df_cleaned = preprocessor.remove_duplicates(df_with_dupes)
        
        assert len(df_cleaned) == 3
    
    def test_preprocess_pipeline(self, sample_data):
        """Test complete preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_pipeline(sample_data, fit=True)
        
        # Check that data is processed
        assert df_processed.isnull().sum().sum() == 0
        assert df_processed.shape[0] <= sample_data.shape[0]
