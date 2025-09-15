"""
UI package for the Lecture Summarizer & Note Enhancer.
Provides web interface for interacting with the system.
"""

from .web_app import create_app

__all__ = ["create_app"]