"""
Provider Ollama pour les modèles locaux
"""

import aiohttp
import json
from typing import List, Dict, Any, AsyncGenerator
import logging
from ..mcp_provider import MCPProvider

logger = logging.getLogger(__name__)

class OllamaProvider(MCPProvider):
    """
    Provider pour Ollama - modèles IA locaux
    """
    
    def __init__(self, base_url: str = "http://ollama:11434"):
        super().__init__()
        self.base_url = base_url
        self.capabilities = [
            "text-generation",
            "code-generation",
            "chat",
            "embeddings",
            "multimodal",
            "streaming"
        ]
        self.default_model = "llama2"
        self._session = None
        
    async def _setup(self):
        """Configure la connexion à Ollama"""
        self._session = aiohttp.ClientSession()
        
        # Vérifier la disponibilité d'Ollama
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    logger.info("Connexion à Ollama établie")
                    # Obtenir le premier modèle disponible comme défaut
                    data = await resp.json()
                    if data.get('models'):
                        self.default_model = data['models'][0]['name']
                        logger.info(f"Modèle par défaut: {self.default_model}")
                else:
                    logger.warning(f"Ollama non disponible: status {resp.status}")
        except Exception as e:
            logger.error(f"Impossible de se connecter à Ollama: {e}")
            
    async def send_completion(self,
                            model: str,
                            messages: List[Dict[str, str]],
                            max_tokens: int = 2048,
                            temperature: float = 0.7,
                            **kwargs) -> Dict[str, Any]:
        """Envoie une requête de complétion à Ollama"""
        
        # Formater le prompt
        prompt = self._format_prompt(messages)
        
        # Préparer la requête
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 40)
            }
        }
        
        # Support du format chat si disponible
        if self._supports_chat_format(model):
            payload = {
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            endpoint = f"{self.base_url}/api/chat"
        else:
            endpoint = f"{self.base_url}/api/generate"
            
        # Envoyer la requête
        try:
            async with self._session.post(endpoint, json=payload) as resp:
                if resp.status == 200:
                    # Ollama retourne du NDJSON pour le streaming
                    full_response = ""
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    full_response += data['response']
                                elif 'message' in data and 'content' in data['message']:
                                    full_response = data['message']['content']
                            except json.JSONDecodeError:
                                continue
                                
                    return {
                        "content": full_response,
                        "metadata": {
                            "model": model,
                            "provider": "ollama",
                            "finish_reason": "complete"
                        }
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"Erreur Ollama {resp.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Erreur lors de la complétion Ollama: {e}")
            raise
            
    async def stream_completion(self,
                              model: str,
                              messages: List[Dict[str, str]],
                              max_tokens: int = 2048,
                              temperature: float = 0.7,
                              **kwargs) -> AsyncGenerator[str, None]:
        """Stream une complétion depuis Ollama"""
        
        # Préparer la requête
        if self._supports_chat_format(model):
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            endpoint = f"{self.base_url}/api/chat"
        else:
            prompt = self._format_prompt(messages)
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            endpoint = f"{self.base_url}/api/generate"
            
        # Stream la réponse
        try:
            async with self._session.post(endpoint, json=payload) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    yield data['response']
                                elif 'message' in data and 'content' in data['message']:
                                    yield data['message']['content']
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await resp.text()
                    raise Exception(f"Erreur Ollama {resp.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Erreur lors du streaming Ollama: {e}")
            raise
            
    async def list_models(self) -> List[str]:
        """Liste tous les modèles Ollama disponibles"""
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [model['name'] for model in data.get('models', [])]
                else:
                    logger.warning(f"Impossible de lister les modèles: status {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Erreur lors de la liste des modèles: {e}")
            return []
            
    async def pull_model(self, model_name: str) -> bool:
        """Télécharge un modèle depuis le registry Ollama"""
        try:
            payload = {"name": model_name}
            async with self._session.post(f"{self.base_url}/api/pull", json=payload) as resp:
                if resp.status == 200:
                    # Suivre le progrès du téléchargement
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if 'status' in data:
                                    logger.info(f"Pull {model_name}: {data['status']}")
                            except json.JSONDecodeError:
                                continue
                    return True
                else:
                    logger.error(f"Impossible de télécharger {model_name}: status {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du modèle: {e}")
            return False
            
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Formate les messages en prompt pour les modèles non-chat"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
                
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
        
    def _supports_chat_format(self, model: str) -> bool:
        """Vérifie si le modèle supporte le format chat"""
        # Les modèles récents supportent généralement le format chat
        chat_models = ['llama2', 'mistral', 'mixtral', 'gemma', 'phi', 'neural-chat']
        return any(cm in model.lower() for cm in chat_models)
        
    async def cleanup(self):
        """Nettoie les ressources"""
        if self._session:
            await self._session.close()
        await super().cleanup()
        
    async def get_embeddings(self, text: str, model: str = "llama2") -> List[float]:
        """Obtient les embeddings d'un texte"""
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            async with self._session.post(f"{self.base_url}/api/embeddings", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('embedding', [])
                else:
                    logger.error(f"Erreur embeddings: status {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention des embeddings: {e}")
            return []