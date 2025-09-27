"""
Unit Tests för EmotionClassifierApp.
Kör: pytest tests/test_emotion.py -v
För Data Analyst/AI-roller: Visar test-driven development.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.emotion_classifier import EmotionClassifierApp, load_config

@pytest.fixture
def app_instance():
    config = {'model_path': 'dummy.keras', 'confidence_threshold': 0.5, 'img_size': (48, 48)}
    app = EmotionClassifierApp(config)
    app.model = Mock()  # Mock model för tests
    return app

def test_load_config():
    """Testa att config laddas med defaults."""
    config = load_config()
    assert isinstance(config, dict)
    assert 'confidence_threshold' in config
    assert config['confidence_threshold'] == 0.5

def test_predict_emotion_high_conf(app_instance):
    """Testa high-confidence prediction."""
    app_instance.model.predict.return_value = np.array([[[0.1, 0.1, 0.1, 0.6, 0.1, 0.0, 0.0]]])  # Happy=0.6
    mock_roi = np.zeros((100, 100), dtype=np.uint8)
    emotion, conf = app_instance.predict_emotion(mock_roi)
    assert emotion == 'Happy'
    assert conf == 0.6

def test_predict_emotion_low_conf(app_instance):
    """Testa low-confidence -> 'Uncertain'."""
    app_instance.model.predict.return_value = np.array([[[0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1]]])  # Max=0.3 < 0.5
    mock_roi = np.zeros((100, 100), dtype=np.uint8)
    emotion, conf = app_instance.predict_emotion(mock_roi)
    assert emotion == 'Uncertain'
    assert conf == 0.3

def test_preprocess_face(app_instance):
    """Testa preprocess returnerar shape (1,48,48,1)."""
    mock_roi = np.zeros((100, 100), dtype=np.uint8)
    processed = app_instance.preprocess_face(mock_roi)
    assert processed.shape == (1, 48, 48, 1)
    assert processed.max() <= 1.0  # Normalized