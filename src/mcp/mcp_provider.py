"""
Classe de base pour les providers MCP
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MCPProvider(ABC):
    """
    Classe abstraite pour les providers MCP
    Tous les providers doivent hériter de cette classe
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.capabilities = []
        self.default_model = None
        self._initialized = False
        
    async def initialize(self):
        """Initialise le provider"""
        if self._initialized:
            return
            
        logger.info(f"Initialisation du provider {self.name}")
        await self._setup()
        self._initialized = True
        
    @abstractmethod
    async def _setup(self):
        """Configuration spécifique du provider"""
        pass
        
    @abstractmethod
    async def send_completion(self, 
                            model: str,
                            messages: List[Dict[str, str]],
                            max_tokens: int = 2048,
                            temperature: float = 0.7,
                            **kwargs) -> Dict[str, Any]:
        """
        Envoie une requête de complétion au modèle
        
        Args:
            model: Nom du modèle à utiliser
            messages: Liste des messages de la conversation
            max_tokens: Nombre maximum de tokens dans la réponse
            temperature: Température pour la génération
            **kwargs: Paramètres supplémentaires spécifiques au provider
            
        Returns:
            Dict contenant la réponse et les métadonnées
        """
        pass
        
    @abstractmethod
    async def list_models(self) -> List[str]:
        """Liste tous les modèles disponibles"""
        pass
        
    async def stream_completion(self,
                              model: str,
                              messages: List[Dict[str, str]],
                              max_tokens: int = 2048,
                              temperature: float = 0.7,
                              **kwargs):
        """
        Stream une complétion (optionnel)
        Retourne un générateur asynchrone
        """
        raise NotImplementedError(f"Le provider {self.name} ne supporte pas le streaming")
        
    def is_active(self) -> bool:
        """Vérifie si le provider est actif et fonctionnel"""
        return self._initialized
        
    async def test_connection(self) -> bool:
        """Teste la connexion au provider"""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception as e:
            logger.error(f"Erreur lors du test de connexion pour {self.name}: {e}")
            return False
            
    async def cleanup(self):
        """Nettoie les ressources du provider"""
        logger.info(f"Nettoyage du provider {self.name}")
        self._initialized = False
        
    def get_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le provider"""
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "default_model": self.default_model,
            "initialized": self._initialized
        }