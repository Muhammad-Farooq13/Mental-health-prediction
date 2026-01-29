"""
Model Training Module
This module provides functions for training machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import logging
from typing import Tuple, Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training and managing machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    random_state: int = 42) -> None:
        """
        Split data into train and test sets.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize multiple classification models.
        
        Returns:
            Dict[str, Any]: Dictionary of model instances
        """
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_single_model(self, model_name: str, model: Any) -> Any:
        """
        Train a single model.
        
        Args:
            model_name (str): Name of the model
            model (Any): Model instance
            
        Returns:
            Any: Trained model
        """
        logger.info(f"Training {model_name}...")
        model.fit(self.X_train, self.y_train)
        logger.info(f"{model_name} training completed")
        
        return model
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            trained_model = self.train_single_model(model_name, model)
            trained_models[model_name] = trained_model
        
        logger.info("All models trained successfully")
        return trained_models
    
    def cross_validate_model(self, model: Any, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model (Any): Model instance
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation scores
        """
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        
        results = {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'min_cv_score': scores.min(),
            'max_cv_score': scores.max()
        }
        
        logger.info(f"Cross-validation scores - Mean: {results['mean_cv_score']:.4f} (+/- {results['std_cv_score']:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, model: Any, param_grid: Dict[str, list], 
                            cv: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model (Any): Model instance
            param_grid (Dict[str, list]): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Best model and best parameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model (Any): Trained model
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def get_feature_importance(self, model: Any, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model (Any): Trained model
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
