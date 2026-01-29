"""
Helper Utilities Module
This module provides utility functions used across the project.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Any, Dict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
        log_file (str): Path to log file
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    file_extension = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r') as f:
        if file_extension in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif file_extension == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_extension}")
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    file_extension = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'w') as f:
        if file_extension in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif file_extension == '.json':
            json.dump(config, f, indent=4)
        else:
            raise ValueError(f"Unsupported config file format: {file_extension}")
    
    logger.info(f"Configuration saved to {config_path}")


def create_timestamp() -> str:
    """
    Create timestamp string.
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def get_project_root() -> str:
    """
    Get project root directory.
    
    Returns:
        str: Project root path
    """
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results (Dict[str, Any]): Results dictionary
        output_path (str): Output file path
    """
    ensure_dir(os.path.dirname(output_path))
    
    # Convert numpy types to native Python types
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.integer):
            results_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            results_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    logger.info(f"Results saved to {output_path}")


def memory_usage(df: pd.DataFrame) -> float:
    """
    Calculate memory usage of dataframe in MB.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        float: Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage of dataframe by optimizing dtypes.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Memory-optimized dataframe
    """
    start_mem = memory_usage(df)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = memory_usage(df)
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"{self.name} started at {self.start_time.strftime('%H:%M:%S')}")
        return self
    
    def __exit__(self, *args):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f} seconds")
