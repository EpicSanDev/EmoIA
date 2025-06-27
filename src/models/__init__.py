"""
Module de Modèles pour EmoIA
Contient tous les modèles d'IA locaux et leurs gestionnaires.
"""

from .local_llm import LocalLanguageModel, GenerationConfig

__all__ = [
    "LocalLanguageModel",
    "GenerationConfig"
]