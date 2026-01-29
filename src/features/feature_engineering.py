"""
Feature Engineering Module
This module provides functions for creating new features and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for feature engineering operations"""
    
    def __init__(self):
        self.selected_features = []
        self.pca = None
        self.feature_selector = None
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_pairs (List[Tuple[str, str]]): List of feature pairs to create interactions
            
        Returns:
            pd.DataFrame: Dataframe with new interaction features
        """
        df_copy = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_copy.columns and feat2 in df_copy.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df_copy[interaction_name] = df_copy[feat1] * df_copy[feat2]
                logger.info(f"Created interaction feature: {interaction_name}")
        
        return df_copy
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to create polynomial features for
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: Dataframe with polynomial features
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                for d in range(2, degree + 1):
                    new_col_name = f"{col}_poly_{d}"
                    df_copy[new_col_name] = df_copy[col] ** d
                    logger.info(f"Created polynomial feature: {new_col_name}")
        
        return df_copy
    
    def select_features_statistical(self, X: pd.DataFrame, y: pd.Series, k: int = 10, 
                                   score_func=f_classif) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using statistical tests.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target variable
            k (int): Number of features to select
            score_func: Scoring function (f_classif, chi2, mutual_info_classif)
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features and their names
        """
        self.feature_selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        feature_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[feature_mask].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features using {score_func.__name__}")
        logger.info(f"Selected features: {self.selected_features}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features), self.selected_features
    
    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            n_components (float or int): Number of components or variance ratio to preserve
            
        Returns:
            Tuple[pd.DataFrame, PCA]: Transformed features and PCA object
        """
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        n_components_actual = self.pca.n_components_
        variance_explained = self.pca.explained_variance_ratio_.sum()
        
        logger.info(f"PCA reduced dimensions from {X.shape[1]} to {n_components_actual}")
        logger.info(f"Explained variance: {variance_explained:.4f}")
        
        column_names = [f"PC{i+1}" for i in range(n_components_actual)]
        return pd.DataFrame(X_pca, columns=column_names), self.pca
    
    def create_binned_features(self, df: pd.DataFrame, column: str, bins: int = 5, 
                              labels: List[str] = None) -> pd.DataFrame:
        """
        Create binned (discretized) features from continuous variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to bin
            bins (int): Number of bins
            labels (List[str]): Labels for bins
            
        Returns:
            pd.DataFrame: Dataframe with binned feature
        """
        df_copy = df.copy()
        
        if column in df_copy.columns:
            new_col_name = f"{column}_binned"
            df_copy[new_col_name] = pd.cut(df_copy[column], bins=bins, labels=labels)
            logger.info(f"Created binned feature: {new_col_name}")
        
        return df_copy
    
    def create_aggregated_features(self, df: pd.DataFrame, group_by: str, 
                                  agg_columns: List[str], agg_funcs: List[str]) -> pd.DataFrame:
        """
        Create aggregated features based on grouping.
        
        Args:
            df (pd.DataFrame): Input dataframe
            group_by (str): Column to group by
            agg_columns (List[str]): Columns to aggregate
            agg_funcs (List[str]): Aggregation functions (mean, sum, std, etc.)
            
        Returns:
            pd.DataFrame: Dataframe with aggregated features
        """
        df_copy = df.copy()
        
        for col in agg_columns:
            if col in df_copy.columns:
                for func in agg_funcs:
                    new_col_name = f"{col}_{func}_by_{group_by}"
                    agg_values = df_copy.groupby(group_by)[col].transform(func)
                    df_copy[new_col_name] = agg_values
                    logger.info(f"Created aggregated feature: {new_col_name}")
        
        return df_copy
    
    def get_feature_importance_scores(self) -> pd.DataFrame:
        """
        Get feature importance scores from the selector.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.feature_selector is None:
            logger.warning("No feature selector fitted yet")
            return None
        
        scores = pd.DataFrame({
            'feature': self.selected_features,
            'score': self.feature_selector.scores_[self.feature_selector.get_support()]
        })
        
        return scores.sort_values('score', ascending=False)
