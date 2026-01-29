"""
Data Preprocessing Module
This module provides functions for cleaning and preprocessing data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing mental health dataset"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        
    def identify_column_types(self, df: pd.DataFrame) -> None:
        """
        Identify numerical and categorical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_cols)} numerical columns")
        logger.info(f"Identified {len(self.categorical_cols)} categorical columns")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_copy = df.copy()
        
        missing_count = df_copy.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return df_copy
        
        logger.info(f"Handling {missing_count} missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            df_copy = df_copy.dropna()
        elif strategy == 'mean':
            for col in self.numerical_cols:
                if col in df_copy.columns:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numerical_cols:
                if col in df_copy.columns:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in df_copy.columns:
                if df_copy[col].isnull().sum() > 0:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        
        logger.info(f"Missing values handled. Remaining: {df_copy.isnull().sum().sum()}")
        return df_copy
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoder or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_copy = df.copy()
        
        for col in self.categorical_cols:
            if col in df_copy.columns:
                if fit:
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Encoded column: {col} with {len(le.classes_)} unique values")
                else:
                    if col in self.label_encoders:
                        df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        
        return df_copy
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features
        """
        df_copy = df.copy()
        
        numerical_cols_present = [col for col in self.numerical_cols if col in df_copy.columns]
        
        if len(numerical_cols_present) > 0:
            if fit:
                df_copy[numerical_cols_present] = self.scaler.fit_transform(df_copy[numerical_cols_present])
                logger.info(f"Scaled {len(numerical_cols_present)} numerical columns")
            else:
                df_copy[numerical_cols_present] = self.scaler.transform(df_copy[numerical_cols_present])
        
        return df_copy
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        original_shape = df.shape
        df_copy = df.drop_duplicates()
        removed = original_shape[0] - df_copy.shape[0]
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return df_copy
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit transformers or use existing ones
            
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        logger.info("Starting preprocessing pipeline")
        
        # Identify column types
        if fit:
            self.identify_column_types(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='mean')
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)
        
        logger.info("Preprocessing pipeline completed")
        return df
