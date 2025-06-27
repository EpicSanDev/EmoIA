"""
Client MCP pour simplifier les interactions avec les modèles
"""

from typing import List, Dict, Any, Optional, Union
import logging
from .mcp_manager import MCPManager, MCPMessage, MCPContext

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client simplifié pour interagir avec les modèles via MCP
    """
    
    def __init__(self, manager: Optional[MCPManager] = None):
        self.manager = manager or MCPManager()
        self._contexts: Dict[str, MCPContext] = {}
        
    async def initialize(self):
        """Initialise le client"""
        await self.manager.initialize()
        
    async def chat(self,
                   user_id: str,
                   message: str,
                   provider: Optional[str] = None,
                   model: Optional[str] = None,
                   context_id: Optional[str] = None,
                   **kwargs) -> str:
        """
        Interface simplifiée pour envoyer un message
        
        Args:
            user_id: ID de l'utilisateur
            message: Message à envoyer
            provider: Provider à utiliser (optionnel)
            model: Modèle à utiliser (optionnel)
            context_id: ID du contexte existant (optionnel)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Réponse du modèle
        """
        # Obtenir ou créer le contexte
        if context_id and context_id in self._contexts:
            context = self._contexts[context_id]
        else:
            context = await self.manager.create_context(
                user_id=user_id,
                provider=provider,
                model=model,
                **kwargs
            )
            context_id = f"{user_id}_{provider or 'default'}"
            self._contexts[context_id] = context
            
        # Créer le message
        user_message = MCPMessage(
            role="user",
            content=message,
            model_context=context
        )
        
        # Récupérer l'historique si disponible
        history = kwargs.get('history', [])
        messages = history + [user_message]
        
        # Envoyer le message
        response = await self.manager.send_message(
            context=context,
            messages=messages,
            **kwargs
        )
        
        return response.content
        
    async def chat_with_history(self,
                               user_id: str,
                               message: str,
                               history: List[Dict[str, str]],
                               **kwargs) -> str:
        """
        Chat avec historique de conversation
        
        Args:
            user_id: ID de l'utilisateur
            message: Message à envoyer
            history: Historique de la conversation
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Réponse du modèle
        """
        # Convertir l'historique en MCPMessages
        mcp_history = []
        for msg in history:
            mcp_history.append(MCPMessage(
                role=msg.get('role', 'user'),
                content=msg.get('content', '')
            ))
            
        return await self.chat(
            user_id=user_id,
            message=message,
            history=mcp_history,
            **kwargs
        )
        
    async def complete(self,
                      prompt: str,
                      provider: Optional[str] = None,
                      model: Optional[str] = None,
                      **kwargs) -> str:
        """
        Complétion simple sans contexte utilisateur
        
        Args:
            prompt: Prompt à compléter
            provider: Provider à utiliser
            model: Modèle à utiliser
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Texte complété
        """
        return await self.chat(
            user_id="anonymous",
            message=prompt,
            provider=provider,
            model=model,
            **kwargs
        )
        
    async def list_available_models(self) -> Dict[str, List[str]]:
        """Liste tous les modèles disponibles"""
        return await self.manager.list_models()
        
    async def get_providers(self) -> List[str]:
        """Liste tous les providers disponibles"""
        return list(self.manager.providers.keys())
        
    async def switch_model(self,
                          user_id: str,
                          provider: str,
                          model: str) -> bool:
        """
        Change le modèle pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            provider: Nouveau provider
            model: Nouveau modèle
            
        Returns:
            True si le changement a réussi
        """
        try:
            # Créer un nouveau contexte
            context = await self.manager.create_context(
                user_id=user_id,
                provider=provider,
                model=model
            )
            
            # Stocker le nouveau contexte
            context_id = f"{user_id}_{provider}"
            self._contexts[context_id] = context
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors du changement de modèle: {e}")
            return False
            
    async def get_model_info(self, provider: str, model: str) -> Dict[str, Any]:
        """Obtient des informations sur un modèle spécifique"""
        provider_info = await self.manager.get_provider_info(provider)
        
        return {
            "provider": provider,
            "model": model,
            "available": model in provider_info.get('models', []),
            "capabilities": provider_info.get('capabilities', []),
            "info": provider_info
        }
        
    def clear_context(self, user_id: str, provider: Optional[str] = None):
        """Efface le contexte d'un utilisateur"""
        if provider:
            context_id = f"{user_id}_{provider}"
            if context_id in self._contexts:
                del self._contexts[context_id]
        else:
            # Effacer tous les contextes de l'utilisateur
            to_remove = [k for k in self._contexts.keys() if k.startswith(f"{user_id}_")]
            for k in to_remove:
                del self._contexts[k]
                
    async def cleanup(self):
        """Nettoie les ressources"""
        self._contexts.clear()
        await self.manager.cleanup()