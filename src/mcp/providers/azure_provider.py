"""
Provider Azure pour Azure OpenAI et Services Cognitifs
Intégration complète des services intelligents Azure
"""

import aiohttp
import json
import base64
from typing import List, Dict, Any, AsyncGenerator, Optional
import logging
from datetime import datetime
import asyncio
from ..mcp_provider import MCPProvider

logger = logging.getLogger(__name__)

class AzureProvider(MCPProvider):
    """
    Provider pour Azure OpenAI et Services Cognitifs Azure
    Intègre : OpenAI, Speech, Vision, Language, Cognitive Search, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration Azure OpenAI
        self.openai_endpoint = config.get('openai_endpoint')
        self.openai_api_key = config.get('openai_api_key')
        self.openai_api_version = config.get('openai_api_version', '2024-02-15-preview')
        self.default_model = config.get('default_model', 'gpt-4')
        
        # Configuration Services Cognitifs
        self.cognitive_endpoint = config.get('cognitive_endpoint')
        self.cognitive_api_key = config.get('cognitive_api_key')
        
        # Configuration Speech Services
        self.speech_endpoint = config.get('speech_endpoint')
        self.speech_api_key = config.get('speech_api_key')
        self.speech_region = config.get('speech_region', 'westeurope')
        
        # Configuration Vision
        self.vision_endpoint = config.get('vision_endpoint')
        self.vision_api_key = config.get('vision_api_key')
        
        # Configuration Translator
        self.translator_endpoint = config.get('translator_endpoint')
        self.translator_api_key = config.get('translator_api_key')
        self.translator_region = config.get('translator_region', 'westeurope')
        
        # Configuration Cognitive Search
        self.search_endpoint = config.get('search_endpoint')
        self.search_api_key = config.get('search_api_key')
        
        # Configuration Language Services
        self.language_endpoint = config.get('language_endpoint')
        self.language_api_key = config.get('language_api_key')
        
        self.capabilities = [
            "text-generation",
            "chat-completion", 
            "embeddings",
            "speech-to-text",
            "text-to-speech",
            "image-analysis",
            "face-detection",
            "emotion-detection",
            "translation",
            "sentiment-analysis",
            "key-phrase-extraction",
            "entity-recognition",
            "language-detection",
            "content-moderation",
            "cognitive-search",
            "streaming",
            "function-calling"
        ]
        
        self._session = None
        self._initialized = False
        
    async def initialize(self):
        """Initialise le provider Azure"""
        if self._initialized:
            return
            
        self._session = aiohttp.ClientSession()
        
        # Vérifier la connectivité Azure OpenAI
        if self.openai_endpoint and self.openai_api_key:
            try:
                headers = {
                    "api-key": self.openai_api_key,
                    "Content-Type": "application/json"
                }
                
                url = f"{self.openai_endpoint}/openai/deployments?api-version={self.openai_api_version}"
                async with self._session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        deployments = data.get('data', [])
                        if deployments:
                            self.default_model = deployments[0]['id']
                            logger.info(f"✅ Azure OpenAI connecté - Modèle par défaut: {self.default_model}")
                        else:
                            logger.warning("⚠️ Aucun déploiement Azure OpenAI trouvé")
                    else:
                        logger.error(f"❌ Erreur connexion Azure OpenAI: {resp.status}")
            except Exception as e:
                logger.error(f"❌ Impossible de se connecter à Azure OpenAI: {e}")
        
        # Vérifier les services cognitifs
        await self._verify_cognitive_services()
        
        self._initialized = True
        logger.info("🚀 Provider Azure initialisé avec succès")
        
    async def _verify_cognitive_services(self):
        """Vérifie la connectivité des services cognitifs"""
        
        # Vérifier Speech Services
        if self.speech_api_key:
            try:
                headers = {"Ocp-Apim-Subscription-Key": self.speech_api_key}
                url = f"https://{self.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
                async with self._session.post(url, headers=headers) as resp:
                    if resp.status == 200:
                        logger.info("✅ Azure Speech Services connecté")
                    else:
                        logger.warning(f"⚠️ Azure Speech Services: {resp.status}")
            except Exception as e:
                logger.error(f"❌ Erreur Speech Services: {e}")
        
        # Vérifier Language Services
        if self.language_endpoint and self.language_api_key:
            try:
                headers = {
                    "Ocp-Apim-Subscription-Key": self.language_api_key,
                    "Content-Type": "application/json"
                }
                url = f"{self.language_endpoint}/language/:analyze-text?api-version=2022-05-01"
                
                # Test simple avec un texte court
                payload = {
                    "kind": "SentimentAnalysis",
                    "parameters": {"opinionMining": False},
                    "analysisInput": {
                        "documents": [{"id": "1", "text": "Hello world", "language": "en"}]
                    }
                }
                
                async with self._session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        logger.info("✅ Azure Language Services connecté")
                    else:
                        logger.warning(f"⚠️ Azure Language Services: {resp.status}")
            except Exception as e:
                logger.error(f"❌ Erreur Language Services: {e}")
                
    async def _setup(self):
        """Configuration spécifique du provider"""
        pass
        
    async def _ensure_session(self):
        """S'assure que la session est initialisée"""
        if not self._session or self._session.closed:
            await self.initialize()
            
    async def send_completion(self,
                            model: str,
                            messages: List[Dict[str, str]],
                            max_tokens: int = 2048,
                            temperature: float = 0.7,
                            **kwargs) -> Dict[str, Any]:
        """Envoie une requête de complétion à Azure OpenAI"""
        
        await self._ensure_session()
        
        # Enrichir les messages avec des insights Azure
        enriched_messages = await self._enrich_messages_with_azure_insights(messages)
        
        # Préparer la requête Azure OpenAI
        headers = {
            "api-key": self.openai_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": enriched_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get('top_p', 0.95),
            "frequency_penalty": kwargs.get('frequency_penalty', 0),
            "presence_penalty": kwargs.get('presence_penalty', 0),
            "stop": kwargs.get('stop'),
            "stream": False
        }
        
        # Ajouter les fonctions si disponibles
        if kwargs.get('functions'):
            payload["functions"] = kwargs['functions']
            payload["function_call"] = kwargs.get('function_call', 'auto')
        
        url = f"{self.openai_endpoint}/openai/deployments/{model}/chat/completions?api-version={self.openai_api_version}"
        
        try:
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choice = data['choices'][0]
                    
                    # Analyser la réponse avec les services cognitifs
                    response_content = choice['message']['content']
                    cognitive_analysis = await self._analyze_response_with_cognitive_services(response_content)
                    
                    return {
                        "content": response_content,
                        "function_call": choice['message'].get('function_call'),
                        "metadata": {
                            "model": model,
                            "provider": "azure",
                            "finish_reason": choice['finish_reason'],
                            "usage": data.get('usage', {}),
                            "cognitive_analysis": cognitive_analysis
                        }
                    }
                else:
                    error_data = await resp.json()
                    raise Exception(f"Erreur Azure OpenAI {resp.status}: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur lors de la complétion Azure: {e}")
            raise
            
    async def stream_completion(self,
                              model: str,
                              messages: List[Dict[str, str]],
                              max_tokens: int = 2048,
                              temperature: float = 0.7,
                              **kwargs) -> AsyncGenerator[str, None]:
        """Stream une complétion depuis Azure OpenAI"""
        
        await self._ensure_session()
        
        # Enrichir les messages
        enriched_messages = await self._enrich_messages_with_azure_insights(messages)
        
        headers = {
            "api-key": self.openai_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": enriched_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        url = f"{self.openai_endpoint}/openai/deployments/{model}/chat/completions?api-version={self.openai_api_version}"
        
        try:
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if 'choices' in data and data['choices']:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield delta['content']
                                except json.JSONDecodeError:
                                    continue
                else:
                    error_data = await resp.json()
                    raise Exception(f"Erreur Azure streaming {resp.status}: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur lors du streaming Azure: {e}")
            raise
            
    async def _enrich_messages_with_azure_insights(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enrichit les messages avec des insights des services cognitifs Azure"""
        
        enriched_messages = []
        
        for message in messages:
            enriched_message = message.copy()
            
            if message.get('role') == 'user':
                content = message.get('content', '')
                
                # Analyse de sentiment et émotions
                sentiment_data = await self._analyze_sentiment(content)
                
                # Détection de langue
                language_data = await self._detect_language(content)
                
                # Extraction d'entités clés
                entities_data = await self._extract_entities(content)
                
                # Ajouter les métadonnées cognitives
                if any([sentiment_data, language_data, entities_data]):
                    cognitive_metadata = {
                        "azure_insights": {
                            "sentiment": sentiment_data,
                            "language": language_data,
                            "entities": entities_data,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # Ajouter un message système avec les insights
                    system_insight = self._format_cognitive_insights(cognitive_metadata["azure_insights"])
                    if system_insight:
                        enriched_messages.append({
                            "role": "system",
                            "content": f"Insights cognitifs Azure: {system_insight}"
                        })
            
            enriched_messages.append(enriched_message)
            
        return enriched_messages
        
    async def _analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyse le sentiment avec Azure Language Services"""
        
        if not self.language_endpoint or not self.language_api_key:
            return None
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.language_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "kind": "SentimentAnalysis",
                "parameters": {"opinionMining": True},
                "analysisInput": {
                    "documents": [{"id": "1", "text": text[:5000], "language": "auto"}]  # Limite Azure
                }
            }
            
            url = f"{self.language_endpoint}/language/:analyze-text?api-version=2022-05-01"
            
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('results', {}).get('documents'):
                        doc = data['results']['documents'][0]
                        return {
                            "sentiment": doc.get('sentiment'),
                            "confidence_scores": doc.get('confidenceScores'),
                            "opinions": doc.get('sentences', [])
                        }
        except Exception as e:
            logger.debug(f"Erreur analyse sentiment: {e}")
            
        return None
        
    async def _detect_language(self, text: str) -> Optional[Dict[str, Any]]:
        """Détecte la langue avec Azure Language Services"""
        
        if not self.language_endpoint or not self.language_api_key:
            return None
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.language_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "kind": "LanguageDetection",
                "analysisInput": {
                    "documents": [{"id": "1", "text": text[:1000]}]  # Limite pour détection
                }
            }
            
            url = f"{self.language_endpoint}/language/:analyze-text?api-version=2022-05-01"
            
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('results', {}).get('documents'):
                        doc = data['results']['documents'][0]
                        return {
                            "language": doc.get('detectedLanguage', {}).get('name'),
                            "iso6391_name": doc.get('detectedLanguage', {}).get('iso6391Name'),
                            "confidence_score": doc.get('detectedLanguage', {}).get('confidenceScore')
                        }
        except Exception as e:
            logger.debug(f"Erreur détection langue: {e}")
            
        return None
        
    async def _extract_entities(self, text: str) -> Optional[Dict[str, Any]]:
        """Extrait les entités avec Azure Language Services"""
        
        if not self.language_endpoint or not self.language_api_key:
            return None
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.language_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "kind": "EntityRecognition",
                "analysisInput": {
                    "documents": [{"id": "1", "text": text[:5000], "language": "auto"}]
                }
            }
            
            url = f"{self.language_endpoint}/language/:analyze-text?api-version=2022-05-01"
            
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('results', {}).get('documents'):
                        doc = data['results']['documents'][0]
                        return {
                            "entities": doc.get('entities', []),
                            "entity_count": len(doc.get('entities', []))
                        }
        except Exception as e:
            logger.debug(f"Erreur extraction entités: {e}")
            
        return None
        
    def _format_cognitive_insights(self, insights: Dict[str, Any]) -> str:
        """Formate les insights cognitifs pour le prompt"""
        
        parts = []
        
        if insights.get('sentiment'):
            sentiment_info = insights['sentiment']
            parts.append(f"Sentiment: {sentiment_info['sentiment']}")
            
        if insights.get('language'):
            lang_info = insights['language']
            parts.append(f"Langue détectée: {lang_info['language']}")
            
        if insights.get('entities') and insights['entities']['entity_count'] > 0:
            entity_count = insights['entities']['entity_count']
            parts.append(f"Entités détectées: {entity_count}")
            
        return " | ".join(parts) if parts else ""
        
    async def _analyze_response_with_cognitive_services(self, response: str) -> Dict[str, Any]:
        """Analyse la réponse générée avec les services cognitifs"""
        
        analysis = {}
        
        # Analyse de sentiment de la réponse
        sentiment_data = await self._analyze_sentiment(response)
        if sentiment_data:
            analysis['response_sentiment'] = sentiment_data
            
        # Détection de langue de la réponse
        language_data = await self._detect_language(response)
        if language_data:
            analysis['response_language'] = language_data
            
        return analysis
        
    async def speech_to_text(self, audio_data: bytes, language: str = "fr-FR") -> Dict[str, Any]:
        """Convertit la parole en texte avec Azure Speech Services"""
        
        if not self.speech_api_key:
            raise Exception("Azure Speech Services non configuré")
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.speech_api_key,
                "Content-Type": "audio/wav",
                "Accept": "application/json"
            }
            
            url = f"https://{self.speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            params = {
                "language": language,
                "format": "detailed"
            }
            
            async with self._session.post(url, headers=headers, params=params, data=audio_data) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "text": data.get('DisplayText', ''),
                        "confidence": data.get('NBest', [{}])[0].get('Confidence', 0),
                        "language": language
                    }
                else:
                    error_data = await resp.text()
                    raise Exception(f"Erreur Speech-to-Text: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur speech-to-text: {e}")
            raise
            
    async def text_to_speech(self, text: str, voice: str = "fr-FR-DeniseNeural") -> bytes:
        """Convertit le texte en parole avec Azure Speech Services"""
        
        if not self.speech_api_key:
            raise Exception("Azure Speech Services non configuré")
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.speech_api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
            }
            
            ssml = f"""
            <speak version='1.0' xml:lang='fr-FR'>
                <voice xml:lang='fr-FR' name='{voice}'>
                    {text}
                </voice>
            </speak>
            """
            
            url = f"https://{self.speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
            
            async with self._session.post(url, headers=headers, data=ssml.encode('utf-8')) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    error_data = await resp.text()
                    raise Exception(f"Erreur Text-to-Speech: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur text-to-speech: {e}")
            raise
            
    async def translate_text(self, text: str, to_language: str, from_language: str = None) -> Dict[str, Any]:
        """Traduit le texte avec Azure Translator"""
        
        if not self.translator_api_key:
            raise Exception("Azure Translator non configuré")
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.translator_api_key,
                "Ocp-Apim-Subscription-Region": self.translator_region,
                "Content-Type": "application/json"
            }
            
            params = {"api-version": "3.0", "to": to_language}
            if from_language:
                params["from"] = from_language
                
            payload = [{"text": text}]
            
            url = f"{self.translator_endpoint}/translate"
            
            async with self._session.post(url, headers=headers, params=params, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    translation = data[0]
                    return {
                        "translated_text": translation["translations"][0]["text"],
                        "detected_language": translation.get("detectedLanguage"),
                        "to_language": to_language
                    }
                else:
                    error_data = await resp.text()
                    raise Exception(f"Erreur traduction: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur traduction: {e}")
            raise
            
    async def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """Analyse une image avec Azure Vision Services"""
        
        if not self.vision_api_key:
            raise Exception("Azure Vision Services non configuré")
            
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": self.vision_api_key,
                "Content-Type": "application/octet-stream"
            }
            
            params = {
                "visualFeatures": "Categories,Description,Faces,Objects,Tags,Color",
                "details": "Celebrities,Landmarks"
            }
            
            url = f"{self.vision_endpoint}/vision/v3.2/analyze"
            
            async with self._session.post(url, headers=headers, params=params, data=image_data) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_data = await resp.text()
                    raise Exception(f"Erreur analyse image: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur analyse image: {e}")
            raise
            
    async def get_embeddings(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """Obtient les embeddings avec Azure OpenAI"""
        
        await self._ensure_session()
        
        headers = {
            "api-key": self.openai_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {"input": text}
        
        url = f"{self.openai_endpoint}/openai/deployments/{model}/embeddings?api-version={self.openai_api_version}"
        
        try:
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["data"][0]["embedding"]
                else:
                    error_data = await resp.json()
                    raise Exception(f"Erreur embeddings Azure: {error_data}")
                    
        except Exception as e:
            logger.error(f"Erreur embeddings: {e}")
            return []
            
    async def list_models(self) -> List[str]:
        """Liste tous les modèles Azure OpenAI disponibles"""
        
        await self._ensure_session()
        
        if not self.openai_endpoint or not self.openai_api_key:
            return []
            
        try:
            headers = {"api-key": self.openai_api_key}
            url = f"{self.openai_endpoint}/openai/deployments?api-version={self.openai_api_version}"
            
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [deployment['id'] for deployment in data.get('data', [])]
                else:
                    logger.warning(f"Impossible de lister les modèles Azure: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Erreur liste modèles Azure: {e}")
            return []
            
    async def cleanup(self):
        """Nettoie les ressources"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._initialized = False
        await super().cleanup()
        
    def get_capabilities(self) -> List[str]:
        """Retourne les capacités du provider"""
        return self.capabilities
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Retourne les informations du provider"""
        return {
            "name": "Azure Cognitive Services",
            "version": "1.0.0",
            "capabilities": self.capabilities,
            "models_available": True,
            "streaming_support": True,
            "multimodal_support": True,
            "cognitive_services": {
                "speech": bool(self.speech_api_key),
                "vision": bool(self.vision_api_key),
                "translator": bool(self.translator_api_key),
                "language": bool(self.language_api_key),
                "search": bool(self.search_api_key)
            }
        }