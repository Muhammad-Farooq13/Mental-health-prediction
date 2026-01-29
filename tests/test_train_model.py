"""
Unit Tests for Model Training Module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_model import ModelTrainer


class TestModelTrainer:
    """Test cases for model training functionality"""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series
    
    def test_prepare_data(self, sample_classification_data):
        """Test data preparation"""
        X, y = sample_classification_data
        
        trainer = ModelTrainer()
        trainer.prepare_data(X, y, test_size=0.3, random_state=42)
        
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        assert len(trainer.X_train) + len(trainer.X_test) == len(X)
    
    def test_initialize_models(self):
        """Test model initialization"""
        trainer = ModelTrainer()
        models = trainer.initialize_models()
        
        assert len(models) > 0
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models
    
    def test_train_single_model(self, sample_classification_data):
        """Test training a single model"""
        X, y = sample_classification_data
        
        trainer = ModelTrainer()
        trainer.prepare_data(X, y)
        trainer.initialize_models()
        
        model = trainer.models['Logistic Regression']
        trained_model = trainer.train_single_model('Logistic Regression', model)
        
        assert hasattr(trained_model, 'predict')
        
        # Test prediction
        predictions = trained_model.predict(trainer.X_test)
        assert len(predictions) == len(trainer.X_test)
    
    def test_train_all_models(self, sample_classification_data):
        """Test training all models"""
        X, y = sample_classification_data
        
        trainer = ModelTrainer()
        trainer.prepare_data(X, y)
        trained_models = trainer.train_all_models()
        
        assert len(trained_models) > 0
        
        # Test that all models can make predictions
        for model_name, model in trained_models.items():
            predictions = model.predict(trainer.X_test)
            assert len(predictions) == len(trainer.X_test)
    
    def test_cross_validate_model(self, sample_classification_data):
        """Test cross-validation"""
        X, y = sample_classification_data
        
        trainer = ModelTrainer()
        trainer.prepare_data(X, y)
        trainer.initialize_models()
        
        model = trainer.models['Logistic Regression']
        model.fit(trainer.X_train, trainer.y_train)
        
        cv_results = trainer.cross_validate_model(model, cv=3)
        
        assert 'mean_cv_score' in cv_results
        assert 'std_cv_score' in cv_results
        assert 0 <= cv_results['mean_cv_score'] <= 1
    
    def test_save_and_load_model(self, sample_classification_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_classification_data
        
        trainer = ModelTrainer()
        trainer.prepare_data(X, y)
        trainer.initialize_models()
        
        model = trainer.models['Logistic Regression']
        model.fit(trainer.X_train, trainer.y_train)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        trainer.save_model(model, str(model_path))
        
        # Load model
        loaded_model = trainer.load_model(str(model_path))
        
        # Test that loaded model works
        predictions = loaded_model.predict(trainer.X_test)
        assert len(predictions) == len(trainer.X_test)
