from pydantic import BaseModel
from typing import Optional, Dict
from sqlalchemy import Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserPreferencesDB(Base):
    """Modèle SQLAlchemy pour les préférences utilisateur"""
    __tablename__ = 'user_preferences'
    
    user_id = Column(String, primary_key=True)
    language = Column(String, default="fr")
    theme = Column(String, default="light")
    notification_settings = Column(JSON, default={
        "email": True,
        "push": False,
        "sound": True
    })

class UserPreferences(BaseModel):
    """Modèle Pydantic pour les préférences utilisateur"""
    user_id: str
    language: str = "fr"
    theme: str = "light"
    notification_settings: Dict[str, bool] = {
        "email": True,
        "push": False,
        "sound": True
    }

    class PreferencesUpdate(BaseModel):
        """Modèle pour les mises à jour de préférences"""
        language: Optional[str] = None
        theme: Optional[str] = None
        notification_settings: Optional[Dict[str, bool]] = None