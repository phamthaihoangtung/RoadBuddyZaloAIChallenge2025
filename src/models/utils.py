"""Backwards compatibility utilities for model loading.

This module re-exports load_model from src/utils/utils.py to avoid duplication.
"""

from utils.utils import load_model  # noqa: F401
