"""
Gestionnaire principal MCP pour EmoIA
Gère l'orchestration des différents modèles et protocoles
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MCPContext:
    """Contexte pour un MCP"""
    model_id: str
    provider: str
    capabilities: List[str]
    max_tokens: int
    temperature: float
    metadata: Dict[str, Any]

@dataclass
class MCPMessage:
    """Message MCP standardisé"""
    role: str
    content: str
    model_context: Optional[MCPContext] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class MCPManager:
    """
    Gestionnaire central pour les Model Context Protocols
    Permet d'ajouter, gérer et orchestrer différents modèles IA
    """
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.active_contexts: Dict[str, MCPContext] = {}
        self.default_provider = "ollama"
        self._initialized = False
        
    async def initialize(self):
        """Initialise le gestionnaire MCP"""
        if self._initialized:
            return
            
        logger.info("Initialisation du gestionnaire MCP")
        
        # Charger les providers par défaut
        await self._load_default_providers()
        
        self._initialized = True
        logger.info(f"MCP Manager initialisé avec {len(self.providers)} providers")
        
    async def _load_default_providers(self):
        """Charge les providers par défaut"""
        # Charger la configuration
        from ..config import Config
        config = Config()
        
        # Azure provider (priorité en premier)
        if hasattr(config, 'azure') and config.azure.get('openai', {}).get('enabled', False):
            try:
                from .providers.azure_provider import AzureProvider
                azure_config = {
                    'openai_endpoint': config.azure['openai']['endpoint'],
                    'openai_api_key': config.azure['openai']['api_key'],
                    'openai_api_version': config.azure['openai']['api_version'],
                    'default_model': config.azure['openai']['default_model'],
                    # Services cognitifs
                    'speech_api_key': config.azure.get('speech', {}).get('api_key'),
                    'speech_region': config.azure.get('speech', {}).get('region', 'westeurope'),
                    'vision_endpoint': config.azure.get('vision', {}).get('endpoint'),
                    'vision_api_key': config.azure.get('vision', {}).get('api_key'),
                    'translator_endpoint': config.azure.get('translator', {}).get('endpoint'),
                    'translator_api_key': config.azure.get('translator', {}).get('api_key'),
                    'translator_region': config.azure.get('translator', {}).get('region', 'westeurope'),
                    'language_endpoint': config.azure.get('language', {}).get('endpoint'),
                    'language_api_key': config.azure.get('language', {}).get('api_key'),
                    'search_endpoint': config.azure.get('search', {}).get('endpoint'),
                    'search_api_key': config.azure.get('search', {}).get('api_key'),
                }
                azure = AzureProvider(azure_config)
                await self.register_provider("azure", azure)
                self.default_provider = "azure"  # Azure comme provider par défaut
                logger.info("✅ Provider Azure configuré comme défaut")
            except Exception as e:
                logger.warning(f"Impossible de charger Azure provider: {e}")
        
        # Ollama provider (fallback)
        try:
            from .providers.ollama_provider import OllamaProvider
            ollama = OllamaProvider()
            await self.register_provider("ollama", ollama)
            # Si Azure n'est pas disponible, utiliser Ollama comme défaut
            if self.default_provider == "ollama" and "azure" not in self.providers:
                logger.info("🦙 Ollama configuré comme provider par défaut")
        except Exception as e:
            logger.warning(f"Impossible de charger Ollama provider: {e}")
            
        # OpenAI-compatible provider (commenté jusqu'à implémentation)
        # try:
        #     from .providers.openai_provider import OpenAIProvider
        #     openai = OpenAIProvider()
        #     await self.register_provider("openai", openai)
        # except Exception as e:
        #     logger.warning(f"Impossible de charger OpenAI provider: {e}")
            
    async def register_provider(self, name: str, provider: Any):
        """Enregistre un nouveau provider MCP"""
        logger.info(f"Enregistrement du provider: {name}")
        self.providers[name] = provider
        
        # Initialiser le provider
        if hasattr(provider, 'initialize'):
            await provider.initialize()
            
    async def create_context(self, 
                           user_id: str,
                           provider: Optional[str] = None,
                           model: Optional[str] = None,
                           **kwargs) -> MCPContext:
        """Crée un nouveau contexte MCP pour un utilisateur"""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider inconnu: {provider_name}")
            
        provider_instance = self.providers[provider_name]
        
        # Créer le contexte
        context = MCPContext(
            model_id=model or provider_instance.default_model,
            provider=provider_name,
            capabilities=provider_instance.capabilities,
            max_tokens=kwargs.get('max_tokens', 2048),
            temperature=kwargs.get('temperature', 0.7),
            metadata={
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                **kwargs
            }
        )
        
        # Stocker le contexte
        context_id = f"{user_id}_{provider_name}_{datetime.now().timestamp()}"
        self.active_contexts[context_id] = context
        
        logger.info(f"Contexte MCP créé: {context_id}")
        return context
        
    async def send_message(self,
                          context: MCPContext,
                          messages: List[MCPMessage],
                          **kwargs) -> MCPMessage:
        """Envoie un message à travers le MCP approprié"""
        provider = self.providers.get(context.provider)
        
        if not provider:
            raise ValueError(f"Provider non trouvé: {context.provider}")
            
        # Préparer les messages
        formatted_messages = self._format_messages(messages, context)
        
        # Envoyer au provider
        response = await provider.send_completion(
            model=context.model_id,
            messages=formatted_messages,
            max_tokens=context.max_tokens,
            temperature=context.temperature,
            **kwargs
        )
        
        # Créer le message de réponse
        return MCPMessage(
            role="assistant",
            content=response['content'],
            model_context=context,
            timestamp=datetime.now(),
            metadata=response.get('metadata', {})
        )
        
    def _format_messages(self, messages: List[MCPMessage], context: MCPContext) -> List[Dict]:
        """Formate les messages pour le provider"""
        formatted = []
        
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Ajouter les métadonnées si nécessaire
            if msg.metadata:
                formatted_msg["metadata"] = msg.metadata
                
            formatted.append(formatted_msg)
            
        return formatted
        
    async def list_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """Liste tous les modèles disponibles"""
        result = {}
        
        if provider:
            if provider in self.providers:
                models = await self.providers[provider].list_models()
                result[provider] = models
        else:
            # Lister tous les modèles de tous les providers
            for name, provider_instance in self.providers.items():
                try:
                    models = await provider_instance.list_models()
                    result[name] = models
                except Exception as e:
                    logger.error(f"Erreur lors de la liste des modèles pour {name}: {e}")
                    result[name] = []
                    
        return result
        
    async def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Obtient les informations sur un provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider inconnu: {provider}")
            
        provider_instance = self.providers[provider]
        
        return {
            "name": provider,
            "capabilities": getattr(provider_instance, 'capabilities', []),
            "default_model": getattr(provider_instance, 'default_model', None),
            "models": await provider_instance.list_models() if hasattr(provider_instance, 'list_models') else [],
            "status": "active" if hasattr(provider_instance, 'is_active') and provider_instance.is_active() else "unknown"
        }
        
    async def cleanup(self):
        """Nettoie les ressources"""
        logger.info("Nettoyage du gestionnaire MCP")
        
        # Nettoyer chaque provider
        for name, provider in self.providers.items():
            if hasattr(provider, 'cleanup'):
                try:
                    await provider.cleanup()
                except Exception as e:
                    logger.error(f"Erreur lors du nettoyage du provider {name}: {e}")
                    
        self.providers.clear()
        self.active_contexts.clear()
        self._initialized = False