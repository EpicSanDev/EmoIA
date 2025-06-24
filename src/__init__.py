"""
EmoIA - Emotional Intelligent Assistant
Une IA émotionnelle avancée avec modèles locaux et architecture modulaire.
"""

__version__ = "2.0.0"
__author__ = "EmoIA Team"
__description__ = "Advanced Emotional AI Assistant with Local Models"

from .core import EmoIA
from .config import Config

__all__ = ["EmoIA", "Config"]