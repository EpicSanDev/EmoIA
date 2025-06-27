"""
Configuration système pour EmoIA
Gestion centralisée de tous les paramètres et modèles.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class ModelConfig(BaseSettings):
    """Configuration des modèles IA locaux"""
    
    # Modèle de langage principal (local)
    language_model: str = "microsoft/DialoGPT-medium"
    language_model_device: str = "auto"
    language_model_max_length: int = 512
    
    # Modèle d'embedding pour la similarité sémantique
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Modèle d'analyse émotionnelle
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Modèle de personnalité
    personality_model: str = "martin-ha/toxic-comment-model"
    
    # Modèles audio et vision
    speech_model: str = "openai/whisper-base"
    tts_model: str = "local"  # Utiliser gTTS local
    vision_model: str = "google/vit-base-patch16-224"
    
    class Config:
        env_prefix = "EMOIA_MODEL_"


class EmotionalConfig(BaseSettings):
    """Configuration du système émotionnel"""
    
    # Paramètres d'empathie
    empathy_threshold: float = 0.7
    emotional_memory_decay: float = 0.95
    mood_adaptation_rate: float = 0.3
    
    # Personnalité de base
    base_personality: Dict[str, float] = {
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.9,
        "neuroticism": 0.2
    }
    
    # Paramètres émotionnels
    emotional_intensity: float = 0.8
    emotional_consistency: float = 0.7
    emotional_learning_rate: float = 0.1
    
    # Types d'émotions supportées
    supported_emotions: List[str] = [
        "joy", "sadness", "anger", "fear", "surprise", "disgust",
        "love", "excitement", "anxiety", "contentment", "curiosity"
    ]
    
    class Config:
        env_prefix = "EMOIA_EMOTION_"


class MemoryConfig(BaseSettings):
    """Configuration du système de mémoire"""
    
    # Mémoire à court terme
    short_term_capacity: int = 100
    short_term_relevance_threshold: float = 0.6
    
    # Mémoire à long terme
    long_term_capacity: int = 10000
    long_term_importance_threshold: float = 0.8
    memory_consolidation_interval: int = 3600  # secondes
    
    # Mémoire sémantique
    semantic_similarity_threshold: float = 0.75
    knowledge_retention_days: int = 365
    
    # Base de données
    database_url: str = "sqlite:///emoia_memory.db"
    redis_url: Optional[str] = None
    
    class Config:
        env_prefix = "EMOIA_MEMORY_"


class CommunicationConfig(BaseSettings):
    """Configuration des interfaces de communication"""
    
    # Telegram Bot
    telegram_token: Optional[str] = None
    telegram_webhook_url: Optional[str] = None
    
    # API REST
    api_host: str = "localhost"
    api_port: int = 8000
    api_debug: bool = False
    
    # WebSocket
    websocket_enabled: bool = True
    websocket_port: int = 8000
    
    # Sécurité
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    
    class Config:
        env_prefix = "EMOIA_COMM_"


class LearningConfig(BaseSettings):
    """Configuration du système d'apprentissage"""
    
    # Apprentissage continu
    continuous_learning: bool = True
    learning_rate: float = 0.001
    adaptation_speed: float = 0.1
    
    # Apprentissage par renforcement
    rl_enabled: bool = True
    rl_exploration_rate: float = 0.1
    rl_discount_factor: float = 0.95
    
    # Mise à jour des modèles
    model_update_interval: int = 86400  # 24h en secondes
    auto_fine_tuning: bool = True
    fine_tuning_batch_size: int = 32
    
    # Feedback utilisateur
    feedback_weight: float = 2.0
    negative_feedback_penalty: float = 0.5
    
    class Config:
        env_prefix = "EMOIA_LEARNING_"


class Config(BaseSettings):
    """Configuration principale d'EmoIA"""
    
    # Métadonnées
    app_name: str = "EmoIA"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Répertoires
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    cache_dir: Path = Path("cache")
    
    # Sous-configurations
    models: ModelConfig = Field(default_factory=ModelConfig)
    emotional: EmotionalConfig = Field(default_factory=EmotionalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Charge la configuration depuis un fichier YAML"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, yaml_path: str):
        """Sauvegarde la configuration dans un fichier YAML"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, allow_unicode=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Retourne le chemin vers un modèle spécifique"""
        return self.models_dir / model_name
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Met à jour la configuration depuis un dictionnaire"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)