import sys
import os

# Добавляем src/ в sys.path
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# flake8: noqa: E402
from model import dummy_model


def test_dummy_model():
    assert dummy_model() == "Model is OK"