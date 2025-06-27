"""
Gestionnaire des Profils Utilisateur pour EmoIA
Gestion complète des profils avec persistance et fonctionnalités avancées
"""

import asyncio
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import uuid

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """Profil utilisateur complet"""
    user_id: str
    display_name: str
    email: Optional[str] = None
    telegram_id: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    preferences: Dict[str, Any] = None
    personality_traits: Dict[str, float] = None
    created_at: datetime = None
    last_active: datetime = None
    settings: Dict[str, Any] = None
    privacy_settings: Dict[str, bool] = None
    notification_preferences: Dict[str, bool] = None
    language: str = "fr"
    timezone: str = "Europe/Paris"
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.personality_traits is None:
            self.personality_traits = {}
        if self.settings is None:
            self.settings = {}
        if self.privacy_settings is None:
            self.privacy_settings = {
                "share_emotions": True,
                "share_memories": False,
                "public_profile": False
            }
        if self.notification_preferences is None:
            self.notification_preferences = {
                "email_notifications": True,
                "telegram_notifications": True,
                "proactive_messages": True,
                "reminders": True
            }
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le profil en dictionnaire pour la sérialisation"""
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "email": self.email,
            "telegram_id": self.telegram_id,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "preferences": self.preferences,
            "personality_traits": self.personality_traits,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "settings": self.settings,
            "privacy_settings": self.privacy_settings,
            "notification_preferences": self.notification_preferences,
            "language": self.language,
            "timezone": self.timezone
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Crée un profil depuis un dictionnaire"""
        profile = cls(
            user_id=data["user_id"],
            display_name=data["display_name"],
            email=data.get("email"),
            telegram_id=data.get("telegram_id"),
            avatar_url=data.get("avatar_url"),
            bio=data.get("bio"),
            preferences=data.get("preferences", {}),
            personality_traits=data.get("personality_traits", {}),
            settings=data.get("settings", {}),
            privacy_settings=data.get("privacy_settings", {}),
            notification_preferences=data.get("notification_preferences", {}),
            language=data.get("language", "fr"),
            timezone=data.get("timezone", "Europe/Paris")
        )
        
        if data.get("created_at"):
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_active"):
            profile.last_active = datetime.fromisoformat(data["last_active"])
        
        return profile

class UserProfileManager:
    """Gestionnaire des profils utilisateur"""
    
    def __init__(self, database_path: str = "data/user_profiles.db"):
        self.database_path = Path(database_path)
        self.profiles = {}  # Cache en mémoire
        self.session_data = {}  # Données de session temporaires
        
    async def initialize(self):
        """Initialise le gestionnaire de profils"""
        try:
            logger.info("Initialisation du gestionnaire de profils...")
            
            # Créer le répertoire si nécessaire
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialiser la base de données
            await self._init_database()
            
            # Charger les profils existants
            await self._load_existing_profiles()
            
            logger.info(f"Gestionnaire de profils initialisé - {len(self.profiles)} profils chargés")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du gestionnaire de profils: {e}")
            raise
    
    async def _init_database(self):
        """Initialise la base de données SQLite"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    email TEXT,
                    telegram_id TEXT,
                    avatar_url TEXT,
                    bio TEXT,
                    preferences TEXT,
                    personality_traits TEXT,
                    created_at TEXT,
                    last_active TEXT,
                    settings TEXT,
                    privacy_settings TEXT,
                    notification_preferences TEXT,
                    language TEXT DEFAULT 'fr',
                    timezone TEXT DEFAULT 'Europe/Paris'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telegram_id ON user_profiles(telegram_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_email ON user_profiles(email)
            """)
            
            conn.commit()
    
    async def _load_existing_profiles(self):
        """Charge les profils existants depuis la base de données"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("SELECT * FROM user_profiles")
                
                for row in cursor.fetchall():
                    profile_data = self._row_to_dict(row)
                    profile = UserProfile.from_dict(profile_data)
                    self.profiles[profile.user_id] = profile
                
                logger.info(f"Chargé {len(self.profiles)} profils utilisateur")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des profils: {e}")
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convertit une ligne de base de données en dictionnaire"""
        (user_id, display_name, email, telegram_id, avatar_url, bio,
         preferences_json, personality_traits_json, created_at, last_active,
         settings_json, privacy_settings_json, notification_preferences_json,
         language, timezone) = row
        
        # Désérialiser les champs JSON
        preferences = json.loads(preferences_json) if preferences_json else {}
        personality_traits = json.loads(personality_traits_json) if personality_traits_json else {}
        settings = json.loads(settings_json) if settings_json else {}
        privacy_settings = json.loads(privacy_settings_json) if privacy_settings_json else {}
        notification_preferences = json.loads(notification_preferences_json) if notification_preferences_json else {}
        
        return {
            "user_id": user_id,
            "display_name": display_name,
            "email": email,
            "telegram_id": telegram_id,
            "avatar_url": avatar_url,
            "bio": bio,
            "preferences": preferences,
            "personality_traits": personality_traits,
            "created_at": created_at,
            "last_active": last_active,
            "settings": settings,
            "privacy_settings": privacy_settings,
            "notification_preferences": notification_preferences,
            "language": language or "fr",
            "timezone": timezone or "Europe/Paris"
        }
    
    async def create_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """Crée un nouveau profil utilisateur"""
        try:
            # Générer un ID unique si non fourni
            if "user_id" not in user_data:
                user_data["user_id"] = f"user_{uuid.uuid4().hex[:12]}"
            
            # Créer le profil
            profile = UserProfile.from_dict(user_data)
            
            # Sauvegarder en base
            await self._save_profile(profile)
            
            # Ajouter au cache
            self.profiles[profile.user_id] = profile
            
            logger.info(f"Nouveau profil créé: {profile.user_id} - {profile.display_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du profil: {e}")
            raise
    
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Récupère un profil par ID utilisateur"""
        if user_id in self.profiles:
            return self.profiles[user_id]
        
        # Essayer de charger depuis la base
        profile = await self._load_profile_from_db(user_id)
        if profile:
            self.profiles[user_id] = profile
            return profile
        
        return None
    
    async def get_profile_by_telegram_id(self, telegram_id: str) -> Optional[UserProfile]:
        """Récupère un profil par ID Telegram"""
        # Chercher dans le cache
        for profile in self.profiles.values():
            if profile.telegram_id == telegram_id:
                return profile
        
        # Chercher en base
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM user_profiles WHERE telegram_id = ?",
                    (telegram_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    profile_data = self._row_to_dict(row)
                    profile = UserProfile.from_dict(profile_data)
                    self.profiles[profile.user_id] = profile
                    return profile
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche par Telegram ID: {e}")
        
        return None
    
    async def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[UserProfile]:
        """Met à jour un profil utilisateur"""
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return None
            
            # Appliquer les mises à jour
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            # Mettre à jour la dernière activité
            profile.last_active = datetime.now()
            
            # Sauvegarder
            await self._save_profile(profile)
            
            logger.info(f"Profil mis à jour: {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du profil: {e}")
            raise
    
    async def delete_profile(self, user_id: str) -> bool:
        """Supprime un profil utilisateur"""
        try:
            # Supprimer du cache
            if user_id in self.profiles:
                del self.profiles[user_id]
            
            # Supprimer de la base
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Profil supprimé: {user_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du profil: {e}")
            return False
    
    async def list_profiles(self, limit: int = 100, offset: int = 0) -> List[UserProfile]:
        """Liste les profils utilisateur"""
        try:
            profiles = list(self.profiles.values())
            
            # Trier par dernière activité
            profiles.sort(key=lambda p: p.last_active, reverse=True)
            
            # Appliquer pagination
            return profiles[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Erreur lors de la liste des profils: {e}")
            return []
    
    async def search_profiles(self, query: str) -> List[UserProfile]:
        """Recherche des profils par nom ou email"""
        try:
            results = []
            query_lower = query.lower()
            
            for profile in self.profiles.values():
                if (query_lower in profile.display_name.lower() or
                    (profile.email and query_lower in profile.email.lower()) or
                    (profile.bio and query_lower in profile.bio.lower())):
                    results.append(profile)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de profils: {e}")
            return []
    
    async def update_last_activity(self, user_id: str):
        """Met à jour la dernière activité d'un utilisateur"""
        try:
            profile = await self.get_profile(user_id)
            if profile:
                profile.last_active = datetime.now()
                await self._save_profile(profile)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'activité: {e}")
    
    async def get_profile_stats(self, user_id: str) -> Dict[str, Any]:
        """Récupère les statistiques d'un profil"""
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return {}
            
            # Calculer les statistiques
            days_since_creation = (datetime.now() - profile.created_at).days
            days_since_last_activity = (datetime.now() - profile.last_active).days
            
            return {
                "profile_completion": self._calculate_profile_completion(profile),
                "days_since_creation": days_since_creation,
                "days_since_last_activity": days_since_last_activity,
                "has_telegram": profile.telegram_id is not None,
                "has_email": profile.email is not None,
                "personality_traits_count": len(profile.personality_traits),
                "preferences_count": len(profile.preferences)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des stats: {e}")
            return {}
    
    def _calculate_profile_completion(self, profile: UserProfile) -> float:
        """Calcule le pourcentage de complétion du profil"""
        total_fields = 8  # Nombre de champs importants
        completed_fields = 0
        
        if profile.display_name:
            completed_fields += 1
        if profile.email:
            completed_fields += 1
        if profile.bio:
            completed_fields += 1
        if profile.avatar_url:
            completed_fields += 1
        if profile.telegram_id:
            completed_fields += 1
        if profile.personality_traits:
            completed_fields += 1
        if len(profile.preferences) > 0:
            completed_fields += 1
        if profile.timezone and profile.language:
            completed_fields += 1
        
        return completed_fields / total_fields
    
    async def _save_profile(self, profile: UserProfile):
        """Sauvegarde un profil en base de données"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_profiles (
                        user_id, display_name, email, telegram_id, avatar_url, bio,
                        preferences, personality_traits, created_at, last_active,
                        settings, privacy_settings, notification_preferences,
                        language, timezone
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    profile.display_name,
                    profile.email,
                    profile.telegram_id,
                    profile.avatar_url,
                    profile.bio,
                    json.dumps(profile.preferences),
                    json.dumps(profile.personality_traits),
                    profile.created_at.isoformat(),
                    profile.last_active.isoformat(),
                    json.dumps(profile.settings),
                    json.dumps(profile.privacy_settings),
                    json.dumps(profile.notification_preferences),
                    profile.language,
                    profile.timezone
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du profil: {e}")
            raise
    
    async def _load_profile_from_db(self, user_id: str) -> Optional[UserProfile]:
        """Charge un profil depuis la base de données"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    profile_data = self._row_to_dict(row)
                    return UserProfile.from_dict(profile_data)
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du profil: {e}")
        
        return None
    
    async def backup_profiles(self, backup_path: str):
        """Sauvegarde tous les profils dans un fichier JSON"""
        try:
            profiles_data = []
            for profile in self.profiles.values():
                profiles_data.append(profile.to_dict())
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Sauvegarde des profils créée: {backup_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    async def restore_profiles(self, backup_path: str):
        """Restaure les profils depuis un fichier JSON"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
            
            for profile_data in profiles_data:
                profile = UserProfile.from_dict(profile_data)
                await self._save_profile(profile)
                self.profiles[profile.user_id] = profile
            
            logger.info(f"Profils restaurés depuis: {backup_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration: {e}")