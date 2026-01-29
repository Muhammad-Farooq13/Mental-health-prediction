"""
Unit Tests for Flask Application
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask_app import app


class TestFlaskApp:
    """Test cases for Flask application"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_home_page(self, client):
        """Test home page loads"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Mental Health Prediction API' in response.data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get('/model_info')
        assert response.status_code in [200, 500]  # May fail if model not loaded
        
        data = json.loads(response.data)
        assert 'status' in data
    
    def test_predict_missing_features(self, client):
        """Test prediction with missing features"""
        response = client.post('/predict',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_predict_batch_invalid_format(self, client):
        """Test batch prediction with invalid format"""
        response = client.post('/predict_batch',
                             data=json.dumps({'invalid': 'format'}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
