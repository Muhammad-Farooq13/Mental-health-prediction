"""
Data Loading Module
This module provides functions for loading raw data from various sources.
"""

import pandas as pd
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Args:
        filepath (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return load_raw_data(filepath)


def save_data(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        filepath (str): Output filepath
        index (bool): Whether to save index column
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=index)
        logger.info(f"Data saved successfully to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return info
