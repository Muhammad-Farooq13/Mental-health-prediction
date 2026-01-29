"""
Visualization Module
This module provides functions for data visualization and plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class DataVisualizer:
    """Class for creating various data visualizations"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_distribution(self, df: pd.DataFrame, column: str, save: bool = True) -> None:
        """
        Plot distribution of a column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to plot
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        if df[column].dtype in ['int64', 'float64']:
            sns.histplot(data=df, x=column, kde=True)
            plt.title(f'Distribution of {column}')
        else:
            df[column].value_counts().plot(kind='bar')
            plt.title(f'Count Plot of {column}')
            plt.xticks(rotation=45)
        
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'distribution_{column}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df (pd.DataFrame): Input dataframe
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        
        if numerical_df.empty:
            logger.warning("No numerical columns found for correlation matrix")
            return
        
        correlation = numerical_df.corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: list = None, save: bool = True) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (list): Class labels
            save (bool): Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      save: bool = True) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            save (bool): Whether to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'roc_curve.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame, 
                               top_n: int = 20, save: bool = True) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to plot
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        top_features = feature_importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot model comparison results.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_missing_values(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot missing values heatmap.
        
        Args:
            df (pd.DataFrame): Input dataframe
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        missing_data = df.isnull()
        
        if missing_data.sum().sum() == 0:
            logger.info("No missing values found")
            return
        
        sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'missing_values.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
    
    def plot_pairplot(self, df: pd.DataFrame, target_col: str = None, 
                     sample_size: int = 1000, save: bool = True) -> None:
        """
        Create pairplot for numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column for coloring
            sample_size (int): Sample size for large datasets
            save (bool): Whether to save the plot
        """
        # Sample data if too large
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        # Select only numerical columns
        numerical_cols = df_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) < 2:
            logger.warning("Not enough numerical columns for pairplot")
            return
        
        # Limit to first 5 numerical columns to avoid overcrowding
        if len(numerical_cols) > 5:
            numerical_cols = numerical_cols[:5]
        
        plot_data = df_sample[numerical_cols]
        
        if target_col and target_col in df_sample.columns:
            plot_data[target_col] = df_sample[target_col]
            pairplot = sns.pairplot(plot_data, hue=target_col, diag_kind='kde')
        else:
            pairplot = sns.pairplot(plot_data, diag_kind='kde')
        
        pairplot.fig.suptitle('Pairplot of Numerical Features', y=1.01)
        
        if save:
            filepath = os.path.join(self.output_dir, 'pairplot.png')
            pairplot.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        plt.close()
