"""
MLOps Pipeline for Mental Health Prediction Project
This module provides automated pipeline for model training, evaluation, and deployment.
"""

import os
import logging
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import project modules
from src.data.load_data import load_raw_data, save_data, get_data_info
from src.data.preprocess import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.visualization.visualize import DataVisualizer
from src.utils.helpers import Timer, ensure_dir, save_results, create_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """Complete MLOps pipeline for model development and deployment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MLOps pipeline.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config or self._default_config()
        self.timestamp = create_timestamp()
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.visualizer = DataVisualizer(output_dir='visualizations')
        
        # Data placeholders
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_model_name = None
        
        # Create necessary directories
        self._setup_directories()
        
        logger.info("MLOps Pipeline initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'data': {
                'raw_path': 'data/raw/mental_health.csv',
                'processed_path': 'data/processed/mental_health_processed.csv',
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'handle_missing': True,
                'missing_strategy': 'mean',
                'encode_categorical': True,
                'scale_numerical': True,
                'remove_duplicates': True
            },
            'feature_engineering': {
                'apply_pca': False,
                'pca_variance': 0.95,
                'select_features': False,
                'n_features': 10
            },
            'training': {
                'cross_validation_folds': 5,
                'hyperparameter_tuning': False
            },
            'models': {
                'output_dir': 'models',
                'save_best_model': True
            },
            'monitoring': {
                'log_metrics': True,
                'save_visualizations': True
            }
        }
    
    def _setup_directories(self) -> None:
        """Create necessary directories for the pipeline"""
        directories = [
            'data/raw',
            'data/processed',
            'models',
            'logs',
            'visualizations',
            'reports'
        ]
        
        for directory in directories:
            ensure_dir(directory)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data.
        
        Returns:
            pd.DataFrame: Loaded raw data
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 80)
        
        with Timer("Data Loading"):
            raw_path = self.config['data']['raw_path']
            self.raw_data = load_raw_data(raw_path)
            
            # Log data information
            data_info = get_data_info(self.raw_data)
            logger.info(f"Data shape: {data_info['shape']}")
            logger.info(f"Columns: {len(data_info['columns'])}")
            logger.info(f"Memory usage: {data_info['memory_usage']:.2f} MB")
        
        return self.raw_data
    
    def preprocess_data(self, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data.
        
        Args:
            target_column (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Preprocessing Data")
        logger.info("=" * 80)
        
        with Timer("Data Preprocessing"):
            # Separate features and target
            if target_column not in self.raw_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            y = self.raw_data[target_column]
            X = self.raw_data.drop(columns=[target_column])
            
            # Apply preprocessing pipeline
            X_processed = self.preprocessor.preprocess_pipeline(X, fit=True)
            
            # Save processed data
            processed_data = X_processed.copy()
            processed_data[target_column] = y
            processed_path = self.config['data']['processed_path']
            save_data(processed_data, processed_path)
            
            self.processed_data = processed_data
        
        return X_processed, y
    
    def engineer_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Engineer features.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Engineered features
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 80)
        
        with Timer("Feature Engineering"):
            X_engineered = X.copy()
            
            # Apply feature selection if configured
            if self.config['feature_engineering']['select_features']:
                n_features = self.config['feature_engineering']['n_features']
                X_engineered, selected_features = self.feature_engineer.select_features_statistical(
                    X_engineered, y, k=n_features
                )
                logger.info(f"Selected {len(selected_features)} features")
            
            # Apply PCA if configured
            if self.config['feature_engineering']['apply_pca']:
                variance = self.config['feature_engineering']['pca_variance']
                X_engineered, pca = self.feature_engineer.apply_pca(X_engineered, n_components=variance)
                logger.info(f"PCA reduced features to {X_engineered.shape[1]} components")
        
        return X_engineered
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into train and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Splitting Data")
        logger.info("=" * 80)
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        self.trainer.prepare_data(X, y, test_size=test_size, random_state=random_state)
        self.X_train = self.trainer.X_train
        self.X_test = self.trainer.X_test
        self.y_train = self.trainer.y_train
        self.y_test = self.trainer.y_test
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Returns:
            Dict[str, Any]: Trained models
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Training Models")
        logger.info("=" * 80)
        
        with Timer("Model Training"):
            # Initialize and train all models
            self.trainer.initialize_models()
            trained_models = self.trainer.train_all_models()
            
            # Perform cross-validation
            cv_folds = self.config['training']['cross_validation_folds']
            logger.info(f"\nPerforming {cv_folds}-fold cross-validation...")
            
            for model_name, model in trained_models.items():
                cv_results = self.trainer.cross_validate_model(model, cv=cv_folds)
                logger.info(f"{model_name} - CV Score: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")
        
        return trained_models
    
    def evaluate_models(self, models: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate trained models.
        
        Args:
            models (Dict[str, Any]): Trained models
            
        Returns:
            pd.DataFrame: Comparison results
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Evaluating Models")
        logger.info("=" * 80)
        
        with Timer("Model Evaluation"):
            # Compare all models
            comparison_df = self.evaluator.compare_models(models, self.X_test, self.y_test)
            
            # Get best model
            self.best_model_name, best_score = self.evaluator.get_best_model(comparison_df, metric='accuracy')
            self.best_model = models[self.best_model_name]
            
            logger.info(f"\nBest Model: {self.best_model_name}")
            logger.info(f"Best Accuracy: {best_score:.4f}")
            
            # Generate visualizations
            if self.config['monitoring']['save_visualizations']:
                self._generate_visualizations(comparison_df)
        
        return comparison_df
    
    def _generate_visualizations(self, comparison_df: pd.DataFrame) -> None:
        """
        Generate and save visualizations.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison results
        """
        logger.info("\nGenerating visualizations...")
        
        # Model comparison plot
        self.visualizer.plot_model_comparison(comparison_df)
        
        # Confusion matrix for best model
        y_pred = self.best_model.predict(self.X_test)
        self.visualizer.plot_confusion_matrix(self.y_test, y_pred)
        
        # ROC curve for binary classification
        if len(np.unique(self.y_test)) == 2 and hasattr(self.best_model, 'predict_proba'):
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
            self.visualizer.plot_roc_curve(self.y_test, y_pred_proba)
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = self.trainer.get_feature_importance(self.best_model, self.X_train.columns.tolist())
            if importance_df is not None:
                self.visualizer.plot_feature_importance(importance_df)
        
        # Data visualizations
        self.visualizer.plot_correlation_matrix(self.processed_data)
        
        logger.info("Visualizations saved")
    
    def save_model(self) -> None:
        """Save the best model and artifacts"""
        logger.info("=" * 80)
        logger.info("STEP 7: Saving Model and Artifacts")
        logger.info("=" * 80)
        
        if self.best_model is None:
            logger.warning("No model to save")
            return
        
        with Timer("Model Saving"):
            models_dir = self.config['models']['output_dir']
            
            # Save best model
            model_path = os.path.join(models_dir, 'best_model.pkl')
            self.trainer.save_model(self.best_model, model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
            joblib.dump(self.preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
            
            # Save feature names
            feature_names_path = os.path.join(models_dir, 'feature_names.json')
            with open(feature_names_path, 'w') as f:
                json.dump(self.X_train.columns.tolist(), f)
            logger.info(f"Feature names saved to {feature_names_path}")
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'model_type': type(self.best_model).__name__,
                'timestamp': self.timestamp,
                'n_features': self.X_train.shape[1],
                'n_samples_train': self.X_train.shape[0],
                'n_samples_test': self.X_test.shape[0],
                'feature_names': self.X_train.columns.tolist()
            }
            
            metadata_path = os.path.join(models_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Model metadata saved to {metadata_path}")
    
    def generate_report(self, comparison_df: pd.DataFrame) -> None:
        """
        Generate a comprehensive report.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison results
        """
        logger.info("=" * 80)
        logger.info("STEP 8: Generating Report")
        logger.info("=" * 80)
        
        report = {
            'pipeline_execution': {
                'timestamp': self.timestamp,
                'data_shape': self.raw_data.shape,
                'processed_shape': self.processed_data.shape,
                'train_size': self.X_train.shape[0],
                'test_size': self.X_test.shape[0]
            },
            'best_model': {
                'name': self.best_model_name,
                'type': type(self.best_model).__name__,
                'metrics': comparison_df.loc[self.best_model_name].to_dict()
            },
            'all_models': comparison_df.to_dict('index')
        }
        
        report_path = f'reports/pipeline_report_{self.timestamp}.json'
        save_results(report, report_path)
        logger.info(f"Report saved to {report_path}")
    
    def run_pipeline(self, target_column: str) -> Dict[str, Any]:
        """
        Run the complete MLOps pipeline.
        
        Args:
            target_column (str): Name of the target column
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING MLOPS PIPELINE")
        logger.info("=" * 80 + "\n")
        
        try:
            with Timer("Complete Pipeline"):
                # Step 1: Load data
                self.load_data()
                
                # Step 2: Preprocess data
                X, y = self.preprocess_data(target_column)
                
                # Step 3: Engineer features
                X = self.engineer_features(X, y)
                
                # Step 4: Split data
                self.split_data(X, y)
                
                # Step 5: Train models
                trained_models = self.train_models()
                
                # Step 6: Evaluate models
                comparison_df = self.evaluate_models(trained_models)
                
                # Step 7: Save best model
                if self.config['models']['save_best_model']:
                    self.save_model()
                
                # Step 8: Generate report
                self.generate_report(comparison_df)
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80 + "\n")
            
            return {
                'status': 'success',
                'best_model': self.best_model_name,
                'comparison': comparison_df,
                'timestamp': self.timestamp
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': self.timestamp
            }


def main():
    """Main function to run the pipeline"""
    # Create pipeline
    pipeline = MLOpsPipeline()
    
    # Note: Adjust target_column based on your actual dataset
    # You may need to inspect the dataset first to determine the target column
    target_column = 'target'  # Change this to your actual target column name
    
    # Run pipeline
    results = pipeline.run_pipeline(target_column)
    
    if results['status'] == 'success':
        logger.info(f"\nBest Model: {results['best_model']}")
        logger.info("\nModel Comparison:")
        logger.info(f"\n{results['comparison']}")
    else:
        logger.error(f"\nPipeline failed: {results['error']}")


if __name__ == '__main__':
    ensure_dir('logs')
    main()
