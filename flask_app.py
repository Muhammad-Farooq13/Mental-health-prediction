"""
Flask Application for Mental Health Prediction
This application serves the trained machine learning model via REST API.
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import os
import logging
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None

# Configuration
MODEL_PATH = os.path.join('models', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join('models', 'preprocessor.pkl')
FEATURE_NAMES_PATH = os.path.join('models', 'feature_names.json')


def load_model_artifacts():
    """Load model, preprocessor, and feature names"""
    global model, preprocessor, feature_names
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
        
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            logger.warning(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
        
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Feature names loaded from {FEATURE_NAMES_PATH}")
        else:
            logger.warning(f"Feature names file not found at {FEATURE_NAMES_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")


# HTML template for home page
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mental Health Prediction API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .endpoint {
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        .method {
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
        }
        .status.ready {
            background-color: #d4edda;
            color: #155724;
        }
        .status.not-ready {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Mental Health Prediction API</h1>
        
        <div class="status {{ status_class }}">
            <strong>Status:</strong> {{ status_message }}
        </div>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/</strong>
            <p>This page - API documentation</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/health</strong>
            <p>Health check endpoint - returns API status</p>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <strong>/predict</strong>
            <p>Make prediction on input data</p>
            <p><strong>Request body:</strong></p>
            <pre><code>{
    "features": {
        "feature1": value1,
        "feature2": value2,
        ...
    }
}</code></pre>
            <p><strong>Response:</strong></p>
            <pre><code>{
    "prediction": prediction_value,
    "probability": confidence_score,
    "status": "success"
}</code></pre>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <strong>/predict_batch</strong>
            <p>Make predictions on multiple records</p>
            <p><strong>Request body:</strong> Array of feature dictionaries</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/model_info</strong>
            <p>Get information about the loaded model</p>
        </div>
        
        <h2>Example Usage</h2>
        <pre><code>curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1, "feature2": 2}}'</code></pre>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    """Home page with API documentation"""
    status_class = "ready" if model is not None else "not-ready"
    status_message = "Model loaded and ready for predictions" if model is not None else "Model not loaded - please train and save a model first"
    
    return render_template_string(HOME_TEMPLATE, status_class=status_class, status_message=status_message)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'feature_names_loaded': feature_names is not None
    }
    
    return jsonify(health_status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({
                'error': 'Missing "features" in request body',
                'status': 'error'
            }), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data['features']])
        
        # Preprocess if preprocessor is available
        if preprocessor is not None:
            input_df = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            probability = float(max(proba))
        
        response = {
            'prediction': int(prediction) if isinstance(prediction, (int, float)) else str(prediction),
            'probability': probability,
            'status': 'success'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({
                'error': 'Request body must be an array of feature dictionaries',
                'status': 'error'
            }), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data)
        
        # Preprocess if preprocessor is available
        if preprocessor is not None:
            input_df = preprocessor.transform(input_df)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)
            probabilities = [float(max(p)) for p in proba]
        
        response = {
            'predictions': [int(p) if isinstance(p, (int, float)) else str(p) for p in predictions],
            'probabilities': probabilities,
            'count': len(predictions),
            'status': 'success'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        info = {
            'model_type': type(model).__name__,
            'model_loaded': True,
            'preprocessor_loaded': preprocessor is not None,
            'feature_names': feature_names,
            'n_features': len(feature_names) if feature_names else None,
            'status': 'success'
        }
        
        # Add model-specific info
        if hasattr(model, 'n_features_in_'):
            info['n_features_in'] = int(model.n_features_in_)
        
        if hasattr(model, 'classes_'):
            info['classes'] = [int(c) if isinstance(c, (int, float)) else str(c) for c in model.classes_]
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


if __name__ == '__main__':
    # Load model artifacts on startup
    load_model_artifacts()
    
    # Get configuration from environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Run the application
    logger.info(f"Starting Flask application on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
