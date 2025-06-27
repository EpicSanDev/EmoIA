"""
Système de Modèles de Langage Locaux pour EmoIA
Remplace les API externes avec des modèles locaux optimisés pour l'intelligence émotionnelle.
"""

import asyncio
import logging
import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BloomForCausalLM, BloomTokenizerFast,
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
from langdetect import detect

from ..config import Config, ModelConfig
from ..emotional import EmotionalState, PersonalityProfile


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration pour la génération de texte"""
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    early_stopping: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None


class LocalLanguageModel:
    """Modèle de langage local avec intelligence émotionnelle"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modèles principaux
        self.main_model = None
        self.main_tokenizer = None
        self.conversation_model = None
        self.conversation_tokenizer = None
        self.mistral_llm = None  # Ajouté pour wrapper Mistral
        
        # Modèles spécialisés
        self.empathy_model = None
        self.summarization_model = None
        self.emotion_conditioned_model = None
        
        # Configuration de génération par défaut
        self.default_gen_config = GenerationConfig()
        
        # Cache des réponses
        self._response_cache = {}
        self._cache_ttl = 3600
        
    async def initialize(self):
        """Initialise tous les modèles locaux"""
        try:
            logger.info("Initialisation des modèles de langage locaux...")
            
            # Modèle principal de conversation
            await self._load_conversation_model()
            
            # Modèles spécialisés
            await self._load_specialized_models()
            
            logger.info("Modèles de langage locaux initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles: {e}")
            raise
    
    async def _load_conversation_model(self):
        """Charge le modèle principal de conversation"""
        try:
            model_name = self.config.language_model
            
            if "mistral" in model_name.lower():
                from .mistral_llm import MistralLLM  # Import local pour éviter la dépendance circulaire
                self.mistral_llm = MistralLLM(model_name=model_name, device=str(self.device))
                logger.info(f"Modèle Mistral chargé: {model_name}")
                return
            
            # Charger selon le type de modèle
            if "dialogpt" in model_name.lower():
                self.conversation_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.conversation_model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Configurer le token de padding
                if self.conversation_tokenizer.pad_token is None:
                    self.conversation_tokenizer.pad_token = self.conversation_tokenizer.eos_token
                    
            elif "bloom" in model_name.lower():
                self.conversation_tokenizer = BloomTokenizerFast.from_pretrained(model_name)
                self.conversation_model = BloomForCausalLM.from_pretrained(model_name)
                
            elif "gpt2" in model_name.lower():
                self.conversation_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                self.conversation_model = GPT2LMHeadModel.from_pretrained(model_name)
                
                if self.conversation_tokenizer.pad_token is None:
                    self.conversation_tokenizer.pad_token = self.conversation_tokenizer.eos_token
            
            # Déplacer vers le device approprié
            self.conversation_model.to(self.device)
            self.conversation_model.eval()
            
            # Configurer les tokens spéciaux
            self.default_gen_config.pad_token_id = self.conversation_tokenizer.pad_token_id
            
            logger.info(f"Modèle de conversation chargé: {model_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle de conversation: {e}")
            raise
    
    async def _load_specialized_models(self):
        """Charge les modèles spécialisés"""
        try:
            # Modèle de résumé
            self.summarization_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modèle d'empathie (utilise un modèle d'émotion fine-tuné)
            self.empathy_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Modèles spécialisés chargés")
            
        except Exception as e:
            logger.warning(f"Certains modèles spécialisés n'ont pas pu être chargés: {e}")
    
    async def generate_response(
        self,
        user_input: str,
        context: str = "",
        emotional_state: Optional[EmotionalState] = None,
        personality: Optional[PersonalityProfile] = None,
        response_type: str = "conversational",
        **kwargs
    ) -> str:
        """Génère une réponse intelligente et émotionnellement adaptée"""
        
        # Vérifier le cache
        cache_key = hash(f"{user_input}_{context}_{response_type}")
        if cache_key in self._response_cache:
            cached_response, cached_time = self._response_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_response
        
        try:
            # Détection de langue si multilingue activé
            lang = "fr"
            if getattr(self.config, 'multilingual', False):
                try:
                    lang = detect(user_input)
                    if lang not in ["fr", "en", "es"]:
                        lang = "fr"
                except Exception:
                    lang = "fr"
            # Préparer le prompt en fonction du type de réponse
            prompt = await self._prepare_emotional_prompt(
                user_input, context, emotional_state, personality, response_type
            )
            # Si Mistral est utilisé, déléguer au wrapper avec la langue
            if self.mistral_llm is not None:
                return await self.mistral_llm.generate(prompt, lang=lang)
            # Sinon, adapter le prompt pour la langue
            prompt = f"[Langue: {lang}]\n" + prompt
            
            # Générer la réponse
            if response_type == "empathetic":
                response = await self._generate_empathetic_response(prompt, **kwargs)
            elif response_type == "creative":
                response = await self._generate_creative_response(prompt, **kwargs)
            elif response_type == "analytical":
                response = await self._generate_analytical_response(prompt, **kwargs)
            else:
                response = await self._generate_conversational_response(prompt, **kwargs)
            
            # Post-traitement émotionnel
            response = await self._apply_emotional_conditioning(
                response, emotional_state, personality
            )
            
            # Mise en cache
            self._response_cache[cache_key] = (response, datetime.now())
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {e}")
            return "Je suis désolé, j'ai des difficultés à formuler une réponse appropriée en ce moment."
    
    async def _prepare_emotional_prompt(
        self,
        user_input: str,
        context: str,
        emotional_state: Optional[EmotionalState],
        personality: Optional[PersonalityProfile],
        response_type: str
    ) -> str:
        """Prépare un prompt conditionné émotionnellement"""
        
        prompt_parts = []
        
        # Contexte de personnalité
        if personality:
            personality_desc = self._personality_to_description(personality)
            prompt_parts.append(f"[Personnalité: {personality_desc}]")
        
        # Contexte émotionnel
        if emotional_state:
            emotion, intensity = emotional_state.dominant_emotion()
            prompt_parts.append(f"[Émotion détectée: {emotion} ({intensity:.2f})]")
        
        # Type de réponse souhaité
        response_instructions = {
            "conversational": "Répondez de manière naturelle et engageante.",
            "empathetic": "Répondez avec empathie et compréhension.",
            "creative": "Répondez de manière créative et imaginative.",
            "analytical": "Répondez de manière analytique et structurée.",
            "supportive": "Répondez de manière encourageante et soutenante."
        }
        
        instruction = response_instructions.get(response_type, response_instructions["conversational"])
        prompt_parts.append(f"[Instruction: {instruction}]")
        
        # Contexte conversationnel
        if context:
            prompt_parts.append(f"[Contexte: {context}]")
        
        # Message utilisateur
        prompt_parts.append(f"Utilisateur: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _personality_to_description(self, personality: PersonalityProfile) -> str:
        """Convertit un profil de personnalité en description textuelle"""
        traits = []
        
        if personality.extraversion > 0.6:
            traits.append("extraverti")
        elif personality.extraversion < 0.4:
            traits.append("introverti")
            
        if personality.agreeableness > 0.7:
            traits.append("bienveillant")
        if personality.openness > 0.7:
            traits.append("ouvert d'esprit")
        if personality.conscientiousness > 0.6:
            traits.append("consciencieux")
        if personality.emotional_intelligence > 0.7:
            traits.append("émotionnellement intelligent")
        if personality.empathy_level > 0.8:
            traits.append("très empathique")
        
        return ", ".join(traits) if traits else "équilibré"
    
    async def _generate_conversational_response(self, prompt: str, **kwargs) -> str:
        """Génère une réponse conversationnelle standard"""
        gen_config = GenerationConfig(**kwargs) if kwargs else self.default_gen_config
        
        # Encoder le prompt
        inputs = self.conversation_tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Générer la réponse
        with torch.no_grad():
            outputs = self.conversation_model.generate(
                inputs,
                max_length=min(inputs.size(1) + gen_config.max_length, 1024),
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repetition_penalty=gen_config.repetition_penalty,
                do_sample=gen_config.do_sample,
                early_stopping=gen_config.early_stopping,
                pad_token_id=gen_config.pad_token_id,
                num_return_sequences=gen_config.num_return_sequences
            )
        
        # Décoder la réponse
        response = self.conversation_tokenizer.decode(
            outputs[0][inputs.size(1):], 
            skip_special_tokens=True
        ).strip()
        
        return response
    
    async def _generate_empathetic_response(self, prompt: str, **kwargs) -> str:
        """Génère une réponse empathique"""
        # Modifier le prompt pour l'empathie
        empathy_prompt = f"{prompt}\n[Répondez avec beaucoup d'empathie et de compréhension]"
        
        # Utiliser une température plus basse pour plus de cohérence
        empathy_config = GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2
        )
        
        return await self._generate_conversational_response(empathy_prompt, **empathy_config.__dict__)
    
    async def _generate_creative_response(self, prompt: str, **kwargs) -> str:
        """Génère une réponse créative"""
        creative_prompt = f"{prompt}\n[Soyez créatif et imaginatif dans votre réponse]"
        
        # Augmenter la température pour plus de créativité
        creative_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            top_k=100
        )
        
        return await self._generate_conversational_response(creative_prompt, **creative_config.__dict__)
    
    async def _generate_analytical_response(self, prompt: str, **kwargs) -> str:
        """Génère une réponse analytique"""
        analytical_prompt = f"{prompt}\n[Répondez de manière structurée et analytique]"
        
        # Réduire la température pour plus de cohérence
        analytical_config = GenerationConfig(
            temperature=0.6,
            top_p=0.7,
            repetition_penalty=1.3
        )
        
        return await self._generate_conversational_response(analytical_prompt, **analytical_config.__dict__)
    
    async def _apply_emotional_conditioning(
        self,
        response: str,
        emotional_state: Optional[EmotionalState],
        personality: Optional[PersonalityProfile]
    ) -> str:
        """Applique un conditionnement émotionnel à la réponse"""
        
        if not emotional_state:
            return response
        
        # Détecter l'émotion dominante
        emotion, intensity = emotional_state.dominant_emotion()
        
        # Ajouter des modifications émotionnelles appropriées
        if emotion == "sadness" and intensity > 0.6:
            # Réponse plus douce et soutenante
            if not any(word in response.lower() for word in ["désolé", "comprends", "soutien"]):
                response = f"Je comprends que ce soit difficile. {response}"
                
        elif emotion == "joy" and intensity > 0.6:
            # Réponse plus enthusiaste
            if "!" not in response:
                response = response.rstrip(".") + " !"
                
        elif emotion == "anger" and intensity > 0.5:
            # Réponse plus apaisante
            if not any(word in response.lower() for word in ["calme", "respire", "comprends"]):
                response = f"Je comprends votre frustration. {response}"
        
        elif emotion == "anxiety" and intensity > 0.5:
            # Réponse rassurante
            if not any(word in response.lower() for word in ["rassure", "normal", "inquiet"]):
                response = f"C'est normal de se sentir ainsi. {response}"
        
        return response
    
    async def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Résume un texte donné"""
        try:
            if self.summarization_model:
                result = self.summarization_model(
                    text,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                return result[0]['summary_text']
            else:
                # Fallback simple
                sentences = text.split('.')
                return '. '.join(sentences[:3]) + '.'
                
        except Exception as e:
            logger.error(f"Erreur lors du résumé: {e}")
            return text[:max_length] + "..."
    
    async def generate_questions(self, context: str, num_questions: int = 3) -> List[str]:
        """Génère des questions pertinentes basées sur le contexte"""
        question_prompt = f"""
        Contexte: {context}
        
        Générez {num_questions} questions pertinentes et engageantes basées sur ce contexte:
        1.
        """
        
        try:
            response = await self._generate_conversational_response(question_prompt)
            # Extraire les questions de la réponse
            questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de questions: {e}")
            return ["Comment vous sentez-vous à propos de cela ?"]
    
    async def translate_emotion_to_text(self, emotional_state: EmotionalState) -> str:
        """Traduit un état émotionnel en description textuelle"""
        emotion, intensity = emotional_state.dominant_emotion()
        
        emotion_descriptions = {
            "joy": ["joyeux", "heureux", "ravi", "enchanté"],
            "sadness": ["triste", "mélancolique", "abattu", "chagriné"],
            "anger": ["en colère", "irrité", "furieux", "agacé"],
            "fear": ["effrayé", "anxieux", "inquiet", "nerveux"],
            "surprise": ["surpris", "étonné", "stupéfait", "ébahi"],
            "love": ["amoureux", "affectueux", "tendre", "aimant"],
            "excitement": ["excité", "enthousiaste", "stimulé", "emballé"],
            "anxiety": ["anxieux", "stressé", "tendu", "préoccupé"],
            "contentment": ["content", "satisfait", "serein", "paisible"],
            "curiosity": ["curieux", "intéressé", "intrigué", "questionneur"]
        }
        
        descriptions = emotion_descriptions.get(emotion, ["neutre"])
        
        # Choisir la description selon l'intensité
        if intensity > 0.8:
            desc_index = min(len(descriptions) - 1, 3)  # Plus intense
        elif intensity > 0.6:
            desc_index = min(len(descriptions) - 1, 2)
        elif intensity > 0.4:
            desc_index = min(len(descriptions) - 1, 1)
        else:
            desc_index = 0
        
        return descriptions[desc_index]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur les modèles chargés"""
        return {
            "conversation_model": self.config.language_model,
            "device": str(self.device),
            "models_loaded": {
                "conversation": self.conversation_model is not None,
                "summarization": self.summarization_model is not None,
                "empathy": self.empathy_model is not None
            },
            "cache_size": len(self._response_cache)
        }

def get_model(model_name: str, config: GenerationConfig) -> LocalLanguageModel:
    """
    Factory function to get the appropriate model instance based on the model name.
    """
    from .mistral_llm import MistralLLM  # Import inside function to avoid circular import
    
    if model_name.startswith("mistral"):
        return MistralLLM(model_name, config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
