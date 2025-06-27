"""
Module de gestion des préférences utilisateur
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserPreferencesDB(Base):
    """Modèle SQLAlchemy pour les préférences utilisateur"""
    __tablename__ = "user_preferences"
    
    user_id = Column(String, primary_key=True, index=True)
    language = Column(String, default="fr")
    theme = Column(String, default="light")
    notification_settings = Column(JSON, default={
        "email": True,
        "push": False,
        "sound": True
    })
    ai_settings = Column(JSON, default={
        "personality_adaptation": True,
        "emotion_intensity": 0.8,
        "response_style": "balanced"
    })


class UserPreferences:
    """Schémas Pydantic pour les préférences utilisateur"""
    
    class PreferencesUpdate(BaseModel):
        """Schéma pour la mise à jour des préférences"""
        language: Optional[str] = None
        theme: Optional[str] = None
        notification_settings: Optional[Dict[str, bool]] = None
        ai_settings: Optional[Dict[str, Any]] = None
    
    class PreferencesResponse(BaseModel):
        """Schéma pour la réponse des préférences"""
        user_id: str
        language: str
        theme: str
        notification_settings: Dict[str, bool]
        ai_settings: Dict[str, Any]
        
        class Config:
            from_attributes = True