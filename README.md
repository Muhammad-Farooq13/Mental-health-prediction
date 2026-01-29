# 🧠 Mental Health Prediction - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive, production-ready machine learning project for mental health prediction. This project follows MLOps best practices and includes complete data preprocessing, model training, evaluation, deployment via Flask API, and Docker containerization.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [MLOps Pipeline](#mlops-pipeline)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete machine learning pipeline for mental health prediction. It includes:

- **Data Processing**: Automated data cleaning, preprocessing, and feature engineering
- **Model Training**: Multiple ML algorithms with hyperparameter tuning and cross-validation
- **Model Evaluation**: Comprehensive metrics, visualizations, and performance comparisons
- **API Deployment**: RESTful API using Flask for model serving
- **Containerization**: Docker support for easy deployment
- **Testing**: Complete unit test coverage with pytest
- **MLOps**: Automated pipeline for continuous integration and deployment

### Dataset

The project uses a mental health dataset (`mental_health.csv`) containing various features related to mental health indicators. The dataset includes:

- **Source**: [Add your dataset source here]
- **Features**: [Describe the main features in your dataset]
- **Target Variable**: [Describe your target variable]
- **Size**: [Number of samples and features]

### Methodology

The project follows a systematic approach:

1. **Data Loading**: Load and inspect raw data
2. **Preprocessing**: Handle missing values, encode categorical variables, scale numerical features
3. **Feature Engineering**: Create new features, select important features, dimensionality reduction
4. **Model Training**: Train multiple ML models (Logistic Regression, Random Forest, SVM, etc.)
5. **Model Evaluation**: Compare models using accuracy, precision, recall, F1-score, and ROC-AUC
6. **Model Selection**: Select the best performing model
7. **Deployment**: Deploy via Flask API and Docker container

## ✨ Features

- **Modular Architecture**: Clean, maintainable code with separation of concerns
- **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes
- **Automated Pipeline**: End-to-end MLOps pipeline for training and deployment
- **RESTful API**: Flask-based API for model inference
- **Docker Support**: Containerized application for easy deployment
- **Comprehensive Testing**: Unit tests with pytest
- **Visualization**: Automatic generation of plots and charts
- **Logging**: Structured logging for monitoring and debugging
- **Configuration Management**: YAML-based configuration
- **CI/CD Ready**: GitHub Actions compatible structure

## 📁 Project Structure

```
mental-health-prediction/
│
├── data/
│   ├── raw/                    # Raw datasets
│   │   └── mental_health.csv
│   └── processed/              # Processed datasets
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/                 # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── visualization/          # Data visualization
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_load_data.py
│   ├── test_preprocess.py
│   ├── test_train_model.py
│   └── test_flask_app.py
│
├── models/                     # Saved models
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
│
├── visualizations/             # Generated plots
│
├── logs/                       # Application logs
│
├── reports/                    # Pipeline reports
│
├── flask_app.py               # Flask API application
├── mlops_pipeline.py          # MLOps pipeline script
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── config.yaml                # Configuration file
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore              # Docker ignore file
├── .gitignore                 # Git ignore file
├── .env.example               # Environment variables example
├── LICENSE                    # License file
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Muhammad-Farooq-13/mental-health-prediction.git
   cd mental-health-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 💻 Usage

### 1. Running the MLOps Pipeline

Train models using the automated pipeline:

```bash
python mlops_pipeline.py
```

This will:
- Load and preprocess the data
- Engineer features
- Train multiple models
- Evaluate and compare models
- Save the best model
- Generate visualizations and reports

### 2. Starting the Flask API

Serve the trained model via REST API:

```bash
python flask_app.py
```

The API will be available at `http://localhost:5000`

### 3. Making Predictions

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 1.5,
      "feature2": 2.3,
      "feature3": 0.8
    }
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '[
    {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8},
    {"feature1": 2.0, "feature2": 1.8, "feature3": 1.2}
  ]'
```

### 4. Using Python

```python
from src.data.load_data import load_raw_data
from src.data.preprocess import DataPreprocessor
from src.models.train_model import ModelTrainer

# Load data
df = load_raw_data('data/raw/mental_health.csv')

# Preprocess
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_pipeline(df)

# Train model
trainer = ModelTrainer()
trainer.prepare_data(X, y)
trained_models = trainer.train_all_models()
```

## 🔬 Model Development

### Models Implemented

1. **Logistic Regression**: Linear model for binary/multiclass classification
2. **Random Forest**: Ensemble of decision trees
3. **Gradient Boosting**: Boosting algorithm for improved performance
4. **Decision Tree**: Single tree-based model
5. **Naive Bayes**: Probabilistic classifier
6. **Support Vector Machine (SVM)**: Kernel-based classification

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (for binary classification)
- **Confusion Matrix**: Visual representation of predictions

### Hyperparameter Tuning

The project supports hyperparameter tuning via GridSearchCV:

```python
from src.models.train_model import ModelTrainer

trainer = ModelTrainer()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

best_model, best_params = trainer.hyperparameter_tuning(
    model, param_grid, cv=5
)
```

## 📡 API Documentation

### Endpoints

#### `GET /`
Home page with API documentation

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

#### `POST /predict`
Make a single prediction

**Request:**
```json
{
  "features": {
    "feature1": value1,
    "feature2": value2,
    ...
  }
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "status": "success"
}
```

#### `POST /predict_batch`
Make batch predictions

**Request:**
```json
[
  {"feature1": value1, "feature2": value2},
  {"feature1": value3, "feature2": value4}
]
```

**Response:**
```json
{
  "predictions": [1, 0],
  "probabilities": [0.85, 0.72],
  "count": 2,
  "status": "success"
}
```

#### `GET /model_info`
Get model information

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "model_loaded": true,
  "n_features": 10,
  "classes": [0, 1],
  "status": "success"
}
```

## 🐳 Docker Deployment

### Building the Docker Image

```bash
docker build -t mental-health-prediction .
```

### Running the Container

```bash
docker run -p 5000:5000 mental-health-prediction
```

### Using Docker Compose

```bash
docker-compose up -d
```

This will start the application with persistent volumes for models and data.

### Stopping the Container

```bash
docker-compose down
```

## 🧪 Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_train_model.py
```

Run with verbose output:
```bash
pytest -v
```

### Test Structure

- `test_load_data.py`: Tests for data loading functionality
- `test_preprocess.py`: Tests for preprocessing pipeline
- `test_train_model.py`: Tests for model training
- `test_flask_app.py`: Tests for Flask API endpoints

## 🔄 MLOps Pipeline

### Pipeline Stages

1. **Data Validation**: Check data quality and schema
2. **Preprocessing**: Clean and transform data
3. **Feature Engineering**: Create and select features
4. **Model Training**: Train multiple models
5. **Model Evaluation**: Compare and select best model
6. **Model Registry**: Save model artifacts
7. **Deployment**: Deploy to production

### Continuous Integration

The project is structured for CI/CD pipelines (e.g., GitHub Actions):

```yaml
# Example: .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

### Model Monitoring

- **Performance Tracking**: Log model metrics over time
- **Data Drift Detection**: Monitor input distribution changes
- **Model Versioning**: Track model versions and metadata
- **Alerting**: Set up alerts for performance degradation

## 📊 Visualizations

The project automatically generates:

- **Distribution plots**: Feature distributions
- **Correlation matrix**: Feature correlations
- **Confusion matrix**: Model predictions vs actual
- **ROC curves**: Model performance visualization
- **Feature importance**: Most influential features
- **Model comparison**: Performance across models

All visualizations are saved in the `visualizations/` directory.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Maintain test coverage above 80%

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
pylint src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Muhammad Farooq** - *Initial work* - [Muhammad-Farooq-13](https://github.com/Muhammad-Farooq-13)

## 🙏 Acknowledgments

- Thanks to all contributors
- Dataset source: [Add source]
- Inspired by MLOps best practices
- Built with scikit-learn, Flask, and Docker

## 📧 Contact

For questions or support, please contact:
- Email: mfarooqshafee333@gmail.com
- GitHub Issues: [Project Issues](https://github.com/Muhammad-Farooq-13/mental-health-prediction/issues)

## 🗺️ Roadmap

- [ ] Add more ML models (XGBoost, LightGBM)
- [ ] Implement model explainability (SHAP, LIME)
- [ ] Add real-time monitoring dashboard
- [ ] Integrate with MLflow for experiment tracking
- [ ] Add automated retraining pipeline
- [ ] Implement A/B testing framework
- [ ] Add GraphQL API support
- [ ] Create web UI for predictions

---

**Note**: Remember to replace placeholder values (like repository URL, dataset source, email) with your actual information before deploying.
