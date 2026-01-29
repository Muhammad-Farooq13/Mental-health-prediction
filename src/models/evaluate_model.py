"""
Model Evaluation Module
This module provides functions for evaluating machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import logging
from typing import Dict, Any, Tuple
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating machine learning models"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model (Any): Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
        
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"{model_name} Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            np.ndarray: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return cm
    
    def get_classification_report(self, y_true: pd.Series, y_pred: np.ndarray, 
                                 target_names: list = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            target_names (list): Names of target classes
            
        Returns:
            str: Classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        logger.info(f"Classification Report:\n{report}")
        
        return report
    
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models and return results as dataframe.
        
        Args:
            models (Dict[str, Any]): Dictionary of models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            metrics['model'] = model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('model').sort_values('accuracy', ascending=False)
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df}")
        
        return comparison_df
    
    def get_best_model(self, comparison_df: pd.DataFrame, metric: str = 'accuracy') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            metric (str): Metric to use for selection
            
        Returns:
            Tuple[str, float]: Best model name and its score
        """
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        
        logger.info(f"Best model: {best_model} with {metric}: {best_score:.4f}")
        
        return best_model, best_score
    
    def save_evaluation_results(self, filepath: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            filepath (str): Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def calculate_prediction_confidence(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate prediction confidence scores.
        
        Args:
            model (Any): Trained model
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Confidence scores
        """
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            confidence = np.max(probabilities, axis=1)
            return confidence
        else:
            logger.warning("Model does not support probability predictions")
            return None
    
    def get_misclassified_samples(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                 y_pred: np.ndarray) -> pd.DataFrame:
        """
        Get samples that were misclassified.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            pd.DataFrame: Misclassified samples
        """
        misclassified_mask = y_test != y_pred
        misclassified_samples = X_test[misclassified_mask].copy()
        misclassified_samples['true_label'] = y_test[misclassified_mask]
        misclassified_samples['predicted_label'] = y_pred[misclassified_mask]
        
        logger.info(f"Found {len(misclassified_samples)} misclassified samples")
        
        return misclassified_samples
