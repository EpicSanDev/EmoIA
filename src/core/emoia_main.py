"""
Classe Principale EmoIA - Intelligence Artificielle Émotionnelle
Orchestration de tous les composants pour une IA émotionnelle avancée.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from langdetect import detect

from ..config import Config
from ..emotional import EmotionalState, PersonalityProfile, LocalEmotionAnalyzer, PersonalityAnalyzer
from ..models.mistral_llm import MistralLLM
from ..memory import IntelligentMemorySystem, MemoryItem


logger = logging.getLogger(__name__)


class ConversationContext:
    """Contexte d'une conversation avec un utilisateur"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = []
        self.current_topic = ""
        self.emotional_flow = []
        self.personality_profile = PersonalityProfile()
        self.last_interaction = datetime.now()
        self.conversation_depth = 0
        self.user_preferences = {}
        
    def add_exchange(
        self, 
        user_message: str, 
        ai_response: str, 
        emotional_state: EmotionalState,
        importance: float = 0.5
    ):
        """Ajoute un échange à l'historique de conversation"""
        exchange = {
            "timestamp": datetime.now(),
            "user_message": user_message,
            "ai_response": ai_response,
            "emotional_state": emotional_state,
            "importance": importance
        }
        self.conversation_history.append(exchange)
        self.emotional_flow.append(emotional_state)
        self.last_interaction = datetime.now()
        self.conversation_depth += 1
    
    def get_recent_context(self, n: int = 5) -> str:
        """Récupère le contexte conversationnel récent"""
        recent_exchanges = self.conversation_history[-n:]
        context_parts = []
        
        for exchange in recent_exchanges:
            context_parts.append(f"User: {exchange['user_message']}")
            context_parts.append(f"AI: {exchange['ai_response']}")
        
        return "\n".join(context_parts)
    
    def get_emotional_trend(self) -> Dict[str, float]:
        """Analyse la tendance émotionnelle récente"""
        if not self.emotional_flow:
            return {}
        
        recent_emotions = self.emotional_flow[-10:]  # 10 dernières émotions
        
        # Calculer les moyennes
        emotion_sums = {}
        for state in recent_emotions:
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love', 'excitement', 'anxiety', 'contentment', 'curiosity']:
                emotion_sums[emotion] = emotion_sums.get(emotion, 0) + getattr(state, emotion)
        
        # Normaliser
        num_states = len(recent_emotions)
        return {emotion: total / num_states for emotion, total in emotion_sums.items()}


class ProactivityEngine:
    """Moteur de proactivité pour décisions intelligentes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.proactivity_rules = []
        self.user_patterns = {}
        
    def should_initiate_conversation(self, user_id: str, context: ConversationContext) -> Tuple[bool, str]:
        """Détermine si l'IA doit initier une conversation"""
        
        time_since_last = datetime.now() - context.last_interaction
        hours_passed = time_since_last.total_seconds() / 3600
        
        # Règles de proactivité
        if hours_passed > 24:
            return True, "check_in_daily"
        
        if hours_passed > 6 and context.emotional_flow:
            last_emotion = context.emotional_flow[-1]
            dominant_emotion, intensity = last_emotion.dominant_emotion()
            
            if dominant_emotion in ['sadness', 'anxiety', 'anger'] and intensity > 0.7:
                return True, "emotional_support"
        
        # Détecter des patterns d'activité
        if user_id in self.user_patterns:
            pattern = self.user_patterns[user_id]
            current_hour = datetime.now().hour
            
            if current_hour in pattern.get('active_hours', []) and hours_passed > 2:
                return True, "activity_pattern"
        
        return False, ""
    
    def generate_proactive_message(
        self, 
        reason: str, 
        context: ConversationContext,
        personality: PersonalityProfile
    ) -> str:
        """Génère un message proactif approprié"""
        
        templates = {
            "check_in_daily": [
                "Bonjour ! Comment allez-vous aujourd'hui ?",
                "Hello ! J'espère que vous passez une bonne journée !",
                "Salut ! Comment vous sentez-vous ce matin ?"
            ],
            "emotional_support": [
                "J'ai remarqué que vous sembliez préoccupé lors de notre dernière conversation. Comment vous sentez-vous maintenant ?",
                "Je pensais à vous. Voulez-vous parler de ce qui vous tracasse ?",
                "Je suis là si vous avez besoin de parler. Comment ça va ?"
            ],
            "activity_pattern": [
                "C'est généralement l'heure où vous êtes actif. Comment se passe votre journée ?",
                "J'espère que vous allez bien ! Quelque chose d'intéressant aujourd'hui ?",
                "Bonjour ! Prêt pour une nouvelle conversation ?"
            ]
        }
        
        # Adapter selon la personnalité
        if personality.extraversion > 0.7:
            # Plus énergique pour les extravertis
            return templates[reason][0].replace("Comment", "Comment") + " 😊"
        elif personality.agreeableness > 0.8:
            # Plus bienveillant
            return "Je pensais à vous et j'espère que tout va bien. " + templates[reason][0].lower()
        else:
            # Standard
            import random
            return random.choice(templates[reason])


class EmoIA:
    """
    Classe principale d'EmoIA - Intelligence Artificielle Émotionnelle
    Orchestre tous les composants pour une expérience d'IA émotionnelle complète.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Composants principaux
        self.emotion_analyzer = LocalEmotionAnalyzer(self.config.emotional)
        self.personality_analyzer = PersonalityAnalyzer(self.config.emotional)
        self.language_model = MistralLLM(self.config.models)  # Modifié pour utiliser MistralLLM
        self.memory_system = IntelligentMemorySystem(self.config.memory)
        
        # Gestionnaires
        self.conversation_contexts = {}  # user_id -> ConversationContext
        self.proactivity_engine = ProactivityEngine(self.config)
        
        # État global
        self.is_initialized = False
        self.startup_time = datetime.now()
        self.total_interactions = 0
        self.active_users = set()
        
        # Cache et optimisations
        self._response_cache = {}
        self._personality_cache = {}
        
    async def initialize(self):
        """Initialise tous les composants d'EmoIA"""
        try:
            logger.info("🚀 Initialisation d'EmoIA...")
            
            # Initialiser les composants dans l'ordre
            await self.emotion_analyzer.initialize()
            logger.info("✅ Analyseur d'émotions initialisé")
            
            await self.personality_analyzer.initialize()
            logger.info("✅ Analyseur de personnalité initialisé")
            
            await self.language_model.initialize()
            logger.info("✅ Modèle de langage local initialisé")
            
            await self.memory_system.initialize()
            logger.info("✅ Système de mémoire intelligent initialisé")
            
            self.is_initialized = True
            logger.info("🎉 EmoIA entièrement initialisé et prêt !")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation d'EmoIA: {e}")
            raise
    
    async def process_message(
        self,
        user_input: str,
        user_id: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Traite un message utilisateur avec intelligence émotionnelle complète
        
        Returns:
            Dict contenant la réponse et les métadonnées d'analyse
        """
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Étape 1: Récupérer ou créer le contexte conversationnel
            if user_id not in self.conversation_contexts:
                self.conversation_contexts[user_id] = ConversationContext(user_id)
            
            context = self.conversation_contexts[user_id]
            
            # Étape 1.5: Déterminer la langue
            # Priorité : context_data > détection automatique
            language = "en"  # Valeur par défaut
            if context_data and "language" in context_data:
                language = context_data["language"]
            else:
                try:
                    language = detect(user_input)
                except:
                    pass  # Garde la valeur par défaut
            
            # Étape 2: Analyser l'émotion du message
            emotional_state = await self.emotion_analyzer.analyze_emotion(
                user_input, 
                context.get_recent_context(),
                language=language
            )
            
            # Étape 3: Récupérer les mémoires pertinentes
            relevant_memories = await self.memory_system.retrieve_memories(
                user_input, 
                user_id, 
                k=5
            )
            
            # Étape 4: Analyser/mettre à jour le profil de personnalité
            user_texts = [ex['user_message'] for ex in context.conversation_history[-20:]]
            user_texts.append(user_input)
            
            if user_id not in self._personality_cache:
                personality_profile = await self.personality_analyzer.analyze_personality(
                    user_texts,
                    context.emotional_flow
                )
                self._personality_cache[user_id] = personality_profile
            else:
                personality_profile = self._personality_cache[user_id]
                # Mettre à jour périodiquement
                if len(context.conversation_history) % 10 == 0:
                    personality_profile = await self.personality_analyzer.analyze_personality(
                        user_texts,
                        context.emotional_flow
                    )
                    self._personality_cache[user_id] = personality_profile
            
            # Étape 5: Déterminer le type de réponse approprié
            response_type = self._determine_response_type(emotional_state, personality_profile, context)
            
            # Étape 6: Préparer le contexte pour la génération
            memory_context = self._format_memory_context(relevant_memories)
            conversation_context = context.get_recent_context()
            
            # Étape 7: Générer la réponse émotionnellement intelligente
            ai_response = await self.language_model.generate_response(
                user_input=user_input,
                context=f"{conversation_context}\n{memory_context}",
                emotional_state=emotional_state,
                personality=personality_profile,
                language=language
            )
            
            # Étape 8: Calculer l'importance de cette interaction
            importance = self._calculate_interaction_importance(
                user_input, ai_response, emotional_state, personality_profile
            )
            
            # Étape 9: Stocker en mémoire
            await self.memory_system.store_memory(
                content=f"User: {user_input}\nAI: {ai_response}",
                user_id=user_id,
                importance=importance,
                emotional_state=emotional_state,
                context=conversation_context,
                memory_type="episodic",
                tags=self._extract_tags(user_input, ai_response)
            )
            
            # Étape 10: Mettre à jour le contexte
            context.add_exchange(user_input, ai_response, emotional_state, importance)
            context.personality_profile = personality_profile
            
            # Étape 11: Analyser pour proactivité future
            await self._update_proactivity_patterns(user_id, context, emotional_state)
            
            # Statistiques
            self.total_interactions += 1
            self.active_users.add(user_id)
            
            # Retour structuré
            return {
                "response": ai_response,
                "emotional_analysis": {
                    "detected_emotion": emotional_state.dominant_emotion()[0],
                    "emotion_intensity": emotional_state.dominant_emotion()[1],
                    "emotional_state": emotional_state.__dict__,
                    "confidence": emotional_state.confidence
                },
                "personality_insights": {
                    "profile": personality_profile.__dict__,
                    "dominant_traits": self._get_dominant_traits(personality_profile)
                },
                "interaction_metadata": {
                    "importance": importance,
                    "response_type": response_type,
                    "memories_used": len(relevant_memories),
                    "conversation_depth": context.conversation_depth
                },
                "system_info": {
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_interactions": self.total_interactions
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
            
            # Réponse de fallback
            return {
                "response": "Je suis désolé, j'ai rencontré une difficulté technique. Pouvez-vous reformuler votre message ?",
                "emotional_analysis": {"detected_emotion": "neutral", "emotion_intensity": 0.0},
                "personality_insights": {},
                "interaction_metadata": {"importance": 0.1, "response_type": "fallback"},
                "system_info": {"error": str(e)}
            }
    
    def _determine_response_type(
        self, 
        emotional_state: EmotionalState, 
        personality: PersonalityProfile,
        context: ConversationContext
    ) -> str:
        """Détermine le type de réponse le plus approprié"""
        
        emotion, intensity = emotional_state.dominant_emotion()
        
        # Réponses basées sur l'émotion
        if emotion in ['sadness', 'anxiety', 'fear'] and intensity > 0.6:
            return "empathetic"
        
        if emotion in ['joy', 'excitement'] and intensity > 0.7:
            return "enthusiastic"
        
        if emotion == 'curiosity' and intensity > 0.5:
            return "informative"
        
        if emotion == 'anger' and intensity > 0.5:
            return "calming"
        
        # Adaptations selon la personnalité
        if personality.creativity > 0.7 and personality.openness > 0.6:
            return "creative"
        
        if personality.conscientiousness > 0.7:
            return "analytical"
        
        if personality.extraversion > 0.7:
            return "engaging"
        
        # Adaptation selon le contexte
        if context.conversation_depth > 10:
            return "deep_conversation"
        
        return "conversational"
    
    def _format_memory_context(self, memories: List[Tuple[MemoryItem, float]]) -> str:
        """Formate les mémoires pertinentes en contexte textuel"""
        if not memories:
            return ""
        
        context_parts = ["=== Mémoires pertinentes ==="]
        for memory_item, similarity in memories[:3]:  # Top 3 mémoires
            context_parts.append(f"• {memory_item.content} (pertinence: {similarity:.2f})")
        
        return "\n".join(context_parts)
    
    def _calculate_interaction_importance(
        self,
        user_input: str,
        ai_response: str,
        emotional_state: EmotionalState,
        personality: PersonalityProfile
    ) -> float:
        """Calcule l'importance d'une interaction pour la mémoire"""
        
        importance = 0.5  # Base
        
        # Facteurs émotionnels
        emotion, intensity = emotional_state.dominant_emotion()
        importance += intensity * 0.3
        
        if emotion in ['sadness', 'anxiety', 'joy', 'love']:
            importance += 0.2  # Émotions importantes
        
        # Facteurs de contenu
        if len(user_input.split()) > 20:  # Message long
            importance += 0.1
        
        if any(word in user_input.lower() for word in ['important', 'grave', 'urgent', 'problème']):
            importance += 0.3
        
        if '?' in user_input:  # Question
            importance += 0.1
        
        # Facteurs de personnalité
        if personality.emotional_intelligence > 0.8:
            importance += 0.1  # Utilisateur émotionnellement intelligent
        
        return min(importance, 1.0)
    
    def _extract_tags(self, user_input: str, ai_response: str) -> List[str]:
        """Extrait des tags pertinents d'une interaction"""
        tags = []
        
        # Tags basés sur le contenu
        content = f"{user_input} {ai_response}".lower()
        
        # Catégories thématiques
        if any(word in content for word in ['travail', 'job', 'bureau', 'collègue']):
            tags.append('travail')
        
        if any(word in content for word in ['famille', 'parent', 'enfant', 'frère', 'sœur']):
            tags.append('famille')
        
        if any(word in content for word in ['amour', 'relation', 'couple', 'ami']):
            tags.append('relation')
        
        if any(word in content for word in ['santé', 'médecin', 'malade', 'douleur']):
            tags.append('santé')
        
        if any(word in content for word in ['projet', 'objectif', 'but', 'rêve']):
            tags.append('objectifs')
        
        # Tags émotionnels
        if any(word in content for word in ['triste', 'déprimé', 'mélancolie']):
            tags.append('tristesse')
        
        if any(word in content for word in ['heureux', 'joyeux', 'content']):
            tags.append('joie')
        
        if any(word in content for word in ['stress', 'anxieux', 'inquiet']):
            tags.append('stress')
        
        return tags
    
    def _get_dominant_traits(self, personality: PersonalityProfile) -> List[str]:
        """Identifie les traits de personnalité dominants"""
        traits = []
        
        trait_map = {
            'openness': ('ouvert d\'esprit', 0.6),
            'conscientiousness': ('consciencieux', 0.6),
            'extraversion': ('extraverti', 0.6),
            'agreeableness': ('bienveillant', 0.7),
            'neuroticism': ('sensible émotionnellement', 0.6),
            'emotional_intelligence': ('émotionnellement intelligent', 0.7),
            'empathy_level': ('empathique', 0.8),
            'creativity': ('créatif', 0.7),
            'humor_appreciation': ('apprécie l\'humour', 0.6),
            'optimism': ('optimiste', 0.6)
        }
        
        for trait, (description, threshold) in trait_map.items():
            if getattr(personality, trait) > threshold:
                traits.append(description)
        
        return traits
    
    async def _update_proactivity_patterns(
        self,
        user_id: str,
        context: ConversationContext,
        emotional_state: EmotionalState
    ):
        """Met à jour les patterns de proactivité pour l'utilisateur"""
        
        current_hour = datetime.now().hour
        
        if user_id not in self.proactivity_engine.user_patterns:
            self.proactivity_engine.user_patterns[user_id] = {
                'active_hours': [],
                'emotional_patterns': {},
                'conversation_preferences': {}
            }
        
        pattern = self.proactivity_engine.user_patterns[user_id]
        
        # Enregistrer l'heure d'activité
        if current_hour not in pattern['active_hours']:
            pattern['active_hours'].append(current_hour)
        
        # Pattern émotionnel
        emotion, intensity = emotional_state.dominant_emotion()
        if emotion not in pattern['emotional_patterns']:
            pattern['emotional_patterns'][emotion] = []
        pattern['emotional_patterns'][emotion].append({
            'intensity': intensity,
            'hour': current_hour,
            'timestamp': datetime.now().isoformat()
        })
    
    async def check_proactivity(self, user_id: str) -> Optional[str]:
        """Vérifie si l'IA doit être proactive avec cet utilisateur"""
        
        if user_id not in self.conversation_contexts:
            return None
        
        context = self.conversation_contexts[user_id]
        should_initiate, reason = self.proactivity_engine.should_initiate_conversation(user_id, context)
        
        if should_initiate:
            personality = self._personality_cache.get(user_id, PersonalityProfile())
            message = self.proactivity_engine.generate_proactive_message(reason, context, personality)
            return message
        
        return None
    
    async def get_emotional_insights(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Génère des insights émotionnels pour un utilisateur"""
        
        if user_id not in self.conversation_contexts:
            return {"error": "Utilisateur non trouvé"}
        
        # Récupérer la timeline émotionnelle
        timeline = await self.memory_system.get_emotional_timeline(user_id, days)
        
        if not timeline:
            return {"message": "Pas assez de données émotionnelles"}
        
        # Analyser les tendances
        emotions_by_day = {}
        for timestamp, emotional_state in timeline:
            day = timestamp.date().isoformat()
            if day not in emotions_by_day:
                emotions_by_day[day] = []
            emotions_by_day[day].append(emotional_state)
        
        # Calculer les moyennes quotidiennes
        daily_averages = {}
        for day, emotions in emotions_by_day.items():
            avg_emotions = {}
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'anxiety', 'contentment']:
                values = [getattr(state, emotion) for state in emotions]
                avg_emotions[emotion] = sum(values) / len(values)
            daily_averages[day] = avg_emotions
        
        # Détecter les tendances
        recent_days = list(daily_averages.keys())[-7:]  # 7 derniers jours
        
        # Insights générés
        insights = {
            "period_analyzed": f"{days} derniers jours",
            "total_interactions": len(timeline),
            "emotional_timeline": daily_averages,
            "trends": {
                "most_frequent_emotion": self._get_most_frequent_emotion(timeline),
                "emotional_stability": self._calculate_emotional_stability(timeline),
                "positive_ratio": self._calculate_positive_ratio(timeline)
            },
            "recommendations": self._generate_emotional_recommendations(timeline)
        }
        
        return insights
    
    def _get_most_frequent_emotion(self, timeline: List[Tuple[datetime, EmotionalState]]) -> str:
        """Trouve l'émotion la plus fréquente"""
        emotion_counts = {}
        for _, emotional_state in timeline:
            emotion, _ = emotional_state.dominant_emotion()
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.keys(), key=lambda k: emotion_counts[k]) if emotion_counts else "neutral"
    
    def _calculate_emotional_stability(self, timeline: List[Tuple[datetime, EmotionalState]]) -> float:
        """Calcule la stabilité émotionnelle (faible variance = plus stable)"""
        if len(timeline) < 2:
            return 1.0
        
        dominant_emotions = [state.dominant_emotion()[1] for _, state in timeline]
        
        # Variance des intensités émotionnelles
        mean_intensity = sum(dominant_emotions) / len(dominant_emotions)
        variance = sum((x - mean_intensity) ** 2 for x in dominant_emotions) / len(dominant_emotions)
        
        # Convertir en score de stabilité (0-1, où 1 = très stable)
        return max(0, 1 - variance)
    
    def _calculate_positive_ratio(self, timeline: List[Tuple[datetime, EmotionalState]]) -> float:
        """Calcule le ratio d'émotions positives"""
        if not timeline:
            return 0.5
        
        positive_emotions = ['joy', 'love', 'excitement', 'contentment', 'curiosity']
        positive_count = 0
        
        for _, emotional_state in timeline:
            emotion, intensity = emotional_state.dominant_emotion()
            if emotion in positive_emotions and intensity > 0.5:
                positive_count += 1
        
        return positive_count / len(timeline)
    
    def _generate_emotional_recommendations(self, timeline: List[Tuple[datetime, EmotionalState]]) -> List[str]:
        """Génère des recommandations basées sur l'analyse émotionnelle"""
        recommendations = []
        
        positive_ratio = self._calculate_positive_ratio(timeline)
        stability = self._calculate_emotional_stability(timeline)
        
        if positive_ratio < 0.4:
            recommendations.append("Essayez d'intégrer plus d'activités qui vous apportent de la joie dans votre routine quotidienne.")
        
        if stability < 0.5:
            recommendations.append("Vos émotions semblent fluctuer beaucoup. Des techniques de relaxation pourraient vous aider.")
        
        # Analyser les patterns temporels
        morning_emotions = []
        evening_emotions = []
        
        for timestamp, emotional_state in timeline:
            hour = timestamp.hour
            emotion, intensity = emotional_state.dominant_emotion()
            
            if 6 <= hour <= 12:
                morning_emotions.append((emotion, intensity))
            elif 18 <= hour <= 23:
                evening_emotions.append((emotion, intensity))
        
        if morning_emotions:
            morning_positive = sum(1 for e, i in morning_emotions if e in ['joy', 'contentment', 'excitement'] and i > 0.5)
            if morning_positive / len(morning_emotions) < 0.3:
                recommendations.append("Vos matinées semblent difficiles. Une routine matinale positive pourrait vous aider.")
        
        if not recommendations:
            recommendations.append("Votre bien-être émotionnel semble équilibré. Continuez ainsi !")
        
        return recommendations
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système EmoIA"""
        return {
            "uptime": str(datetime.now() - self.startup_time),
            "total_interactions": self.total_interactions,
            "active_users": len(self.active_users),
            "conversations_active": len(self.conversation_contexts),
            "memory_stats": self.memory_system.get_memory_stats(),
            "model_info": self.language_model.get_model_info(),
            "config_summary": {
                "emotional_intensity": self.config.emotional.emotional_intensity,
                "empathy_threshold": self.config.emotional.empathy_threshold,
                "learning_enabled": self.config.learning.continuous_learning
            }
        }

    async def generate_suggestions(
        self,
        context: str,
        user_input: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Génère des suggestions intelligentes basées sur le contexte
        
        Args:
            context: Contexte de conversation
            user_input: Entrée utilisateur optionnelle
            emotional_state: État émotionnel optionnel
            max_suggestions: Nombre maximum de suggestions
            
        Returns:
            Liste de suggestions avec confiance
        """
        try:
            suggestions = []
            
            # Analyser le contexte émotionnel
            if emotional_state:
                dominant_emotion = emotional_state.get('dominant_emotion', 'neutral')
                
                # Suggestions basées sur l'émotion
                emotion_suggestions = {
                    'sadness': [
                        "Voulez-vous parler de ce qui vous préoccupe ?",
                        "Que diriez-vous d'une activité qui vous remonte le moral ?",
                        "Parfois, partager ses sentiments peut aider..."
                    ],
                    'anxiety': [
                        "Prenons un moment pour respirer ensemble.",
                        "Voulez-vous essayer un exercice de relaxation ?",
                        "Qu'est-ce qui vous aiderait à vous sentir plus calme ?"
                    ],
                    'joy': [
                        "C'est merveilleux de vous voir heureux ! Qu'est-ce qui vous réjouit ?",
                        "Partagez votre bonne nouvelle !",
                        "Comment pouvons-nous célébrer ensemble ?"
                    ],
                    'anger': [
                        "Je comprends votre frustration. Voulez-vous en discuter ?",
                        "Parfois, exprimer sa colère aide à se sentir mieux.",
                        "Qu'est-ce qui pourrait améliorer la situation ?"
                    ],
                    'curiosity': [
                        "Excellente question ! Explorons cela ensemble.",
                        "Je peux vous aider à en apprendre davantage sur ce sujet.",
                        "Quels aspects vous intéressent particulièrement ?"
                    ]
                }
                
                if dominant_emotion in emotion_suggestions:
                    for suggestion in emotion_suggestions[dominant_emotion][:max_suggestions]:
                        suggestions.append({
                            "text": suggestion,
                            "type": "emotional_support",
                            "confidence": 0.8
                        })
            
            # Suggestions contextuelles basées sur les mots-clés
            if context or user_input:
                combined_text = f"{context} {user_input or ''}".lower()
                
                # Détection de sujets
                topic_suggestions = {
                    'travail': [
                        "Comment se passe votre journée de travail ?",
                        "Y a-t-il des défis professionnels dont vous aimeriez discuter ?",
                        "Quels sont vos objectifs professionnels actuels ?"
                    ],
                    'famille': [
                        "Comment va votre famille ?",
                        "Y a-t-il des moments familiaux que vous aimeriez partager ?",
                        "Qu'est-ce qui vous rend fier de votre famille ?"
                    ],
                    'santé': [
                        "Comment vous sentez-vous physiquement ?",
                        "Avez-vous des préoccupations de santé ?",
                        "Que faites-vous pour prendre soin de vous ?"
                    ],
                    'loisirs': [
                        "Quels sont vos loisirs préférés ?",
                        "Avez-vous découvert de nouvelles activités récemment ?",
                        "Comment aimez-vous vous détendre ?"
                    ]
                }
                
                for topic, topic_suggestions_list in topic_suggestions.items():
                    if topic in combined_text and len(suggestions) < max_suggestions:
                        for suggestion in topic_suggestions_list[:2]:
                            if len(suggestions) < max_suggestions:
                                suggestions.append({
                                    "text": suggestion,
                                    "type": "topic_based",
                                    "confidence": 0.7
                                })
            
            # Suggestions générales si pas assez spécifiques
            if len(suggestions) < max_suggestions:
                general_suggestions = [
                    "Comment puis-je vous aider aujourd'hui ?",
                    "Y a-t-il quelque chose dont vous aimeriez parler ?",
                    "Qu'est-ce qui occupe vos pensées en ce moment ?",
                    "Comment s'est passée votre journée ?",
                    "Avez-vous des projets intéressants ?"
                ]
                
                for suggestion in general_suggestions:
                    if len(suggestions) < max_suggestions:
                        suggestions.append({
                            "text": suggestion,
                            "type": "general",
                            "confidence": 0.5
                        })
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de suggestions: {e}")
            return [{
                "text": "Comment puis-je vous aider ?",
                "type": "fallback",
                "confidence": 0.3
            }]
    
    async def get_conversation_insights(
        self,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère les insights détaillés d'une conversation
        
        Args:
            user_id: ID de l'utilisateur
            conversation_id: ID de conversation spécifique (optionnel)
            
        Returns:
            Dict contenant les insights de conversation
        """
        try:
            if user_id not in self.conversation_contexts:
                return {
                    "error": "Aucune conversation trouvée pour cet utilisateur",
                    "user_id": user_id
                }
            
            context = self.conversation_contexts[user_id]
            
            # Analyser l'historique de conversation
            total_exchanges = len(context.conversation_history)
            if total_exchanges == 0:
                return {
                    "message": "Pas encore de conversation avec cet utilisateur",
                    "user_id": user_id
                }
            
            # Calculer les métriques
            avg_message_length = sum(len(ex['user_message'].split()) for ex in context.conversation_history) / total_exchanges
            
            # Analyser les sujets abordés
            all_messages = " ".join([ex['user_message'] for ex in context.conversation_history])
            topics = self._extract_conversation_topics(all_messages)
            
            # Analyser l'évolution émotionnelle
            emotional_journey = []
            for i, exchange in enumerate(context.conversation_history[-10:]):  # 10 derniers échanges
                state = exchange.get('emotional_state')
                if state:
                    emotional_journey.append({
                        "exchange_number": i + 1,
                        "emotion": state.dominant_emotion()[0],
                        "intensity": state.dominant_emotion()[1],
                        "timestamp": exchange['timestamp'].isoformat()
                    })
            
            # Calculer l'engagement
            recent_exchanges = context.conversation_history[-20:]
            engagement_score = self._calculate_engagement_score(recent_exchanges)
            
            # Points clés de la conversation
            key_moments = self._identify_key_moments(context.conversation_history)
            
            insights = {
                "user_id": user_id,
                "conversation_stats": {
                    "total_exchanges": total_exchanges,
                    "conversation_depth": context.conversation_depth,
                    "average_message_length": avg_message_length,
                    "last_interaction": context.last_interaction.isoformat(),
                    "conversation_duration": str(datetime.now() - context.conversation_history[0]['timestamp'])
                },
                "topics_discussed": topics,
                "emotional_journey": emotional_journey,
                "emotional_trends": context.get_emotional_trend(),
                "engagement_metrics": {
                    "engagement_score": engagement_score,
                    "response_consistency": self._calculate_response_consistency(recent_exchanges),
                    "question_ratio": self._calculate_question_ratio(recent_exchanges)
                },
                "personality_insights": {
                    "profile": context.personality_profile.__dict__ if context.personality_profile else {},
                    "dominant_traits": self._get_dominant_traits(context.personality_profile) if context.personality_profile else []
                },
                "key_moments": key_moments,
                "recommendations": self._generate_conversation_recommendations(context)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des insights: {e}")
            return {
                "error": f"Erreur lors de l'analyse: {str(e)}",
                "user_id": user_id
            }
    
    def _extract_conversation_topics(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les principaux sujets de conversation"""
        topics = []
        topic_keywords = {
            "travail": ["travail", "job", "bureau", "collègue", "projet", "réunion"],
            "famille": ["famille", "parent", "enfant", "frère", "sœur", "mère", "père"],
            "santé": ["santé", "médecin", "malade", "fatigue", "douleur", "sommeil"],
            "loisirs": ["loisir", "sport", "musique", "film", "livre", "voyage"],
            "émotions": ["triste", "heureux", "anxieux", "stress", "content", "inquiet"],
            "relations": ["ami", "amour", "relation", "couple", "rencontre"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                topics.append({
                    "topic": topic,
                    "relevance": min(count / 10.0, 1.0),
                    "keyword_matches": count
                })
        
        return sorted(topics, key=lambda x: x['relevance'], reverse=True)
    
    def _calculate_engagement_score(self, exchanges: List[Dict]) -> float:
        """Calcule le score d'engagement basé sur plusieurs facteurs"""
        if not exchanges:
            return 0.0
        
        factors = []
        
        # Longueur moyenne des messages
        avg_length = sum(len(ex['user_message'].split()) for ex in exchanges) / len(exchanges)
        length_score = min(avg_length / 20.0, 1.0)  # Normaliser sur 20 mots
        factors.append(length_score)
        
        # Fréquence des questions
        questions = sum(1 for ex in exchanges if '?' in ex['user_message'])
        question_score = questions / len(exchanges)
        factors.append(question_score)
        
        # Diversité émotionnelle
        emotions = set()
        for ex in exchanges:
            if 'emotional_state' in ex and ex['emotional_state']:
                emotions.add(ex['emotional_state'].dominant_emotion()[0])
        emotion_diversity_score = len(emotions) / 7.0  # 7 émotions principales
        factors.append(emotion_diversity_score)
        
        # Score final
        return sum(factors) / len(factors)
    
    def _calculate_response_consistency(self, exchanges: List[Dict]) -> float:
        """Calcule la cohérence des réponses"""
        if len(exchanges) < 2:
            return 1.0
        
        # Analyser la cohérence temporelle
        timestamps = [ex['timestamp'] for ex in exchanges]
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_diffs.append(diff)
        
        if not time_diffs:
            return 1.0
        
        # Plus la variance est faible, plus c'est cohérent
        avg_diff = sum(time_diffs) / len(time_diffs)
        variance = sum((d - avg_diff) ** 2 for d in time_diffs) / len(time_diffs)
        
        # Normaliser (variance élevée = score bas)
        consistency = 1.0 / (1.0 + variance / 10000)
        return consistency
    
    def _calculate_question_ratio(self, exchanges: List[Dict]) -> float:
        """Calcule le ratio de questions dans les échanges"""
        if not exchanges:
            return 0.0
        
        questions = sum(1 for ex in exchanges if '?' in ex['user_message'])
        return questions / len(exchanges)
    
    def _identify_key_moments(self, history: List[Dict]) -> List[Dict[str, Any]]:
        """Identifie les moments clés de la conversation"""
        key_moments = []
        
        for i, exchange in enumerate(history):
            importance = exchange.get('importance', 0)
            
            # Moments avec haute importance
            if importance > 0.8:
                key_moments.append({
                    "type": "high_importance",
                    "exchange_index": i,
                    "message": exchange['user_message'][:100] + "...",
                    "timestamp": exchange['timestamp'].isoformat(),
                    "importance": importance
                })
            
            # Changements émotionnels significatifs
            if i > 0 and 'emotional_state' in exchange and 'emotional_state' in history[i-1]:
                prev_emotion = history[i-1]['emotional_state'].dominant_emotion()[0]
                curr_emotion = exchange['emotional_state'].dominant_emotion()[0]
                
                if prev_emotion != curr_emotion:
                    key_moments.append({
                        "type": "emotion_shift",
                        "exchange_index": i,
                        "from_emotion": prev_emotion,
                        "to_emotion": curr_emotion,
                        "timestamp": exchange['timestamp'].isoformat()
                    })
        
        return key_moments[-5:]  # Retourner les 5 derniers moments clés
    
    def _generate_conversation_recommendations(self, context: ConversationContext) -> List[str]:
        """Génère des recommandations pour améliorer la conversation"""
        recommendations = []
        
        # Analyser les patterns
        emotional_trend = context.get_emotional_trend()
        
        # Recommandations basées sur les émotions
        if emotional_trend.get('sadness', 0) > 0.6:
            recommendations.append("L'utilisateur semble traverser une période difficile. Continuez à offrir du soutien émotionnel.")
        
        if emotional_trend.get('anxiety', 0) > 0.5:
            recommendations.append("Proposez des techniques de relaxation ou des exercices de respiration.")
        
        if emotional_trend.get('joy', 0) > 0.7:
            recommendations.append("L'utilisateur est de bonne humeur. C'est le moment idéal pour des discussions constructives.")
        
        # Recommandations basées sur l'engagement
        if context.conversation_depth < 5:
            recommendations.append("La conversation est encore superficielle. Posez des questions ouvertes pour approfondir.")
        elif context.conversation_depth > 20:
            recommendations.append("Excellente profondeur de conversation. Continuez à maintenir cet engagement.")
        
        if not recommendations:
            recommendations.append("La conversation se déroule bien. Continuez à être attentif et empathique.")
        
        return recommendations
    
    async def get_mood_history(
        self,
        user_id: str,
        period: str = "week"
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique d'humeur de l'utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            period: Période ('day', 'week', 'month')
            
        Returns:
            Liste des points d'humeur
        """
        try:
            # Déterminer la période
            days = {"day": 1, "week": 7, "month": 30}.get(period, 7)
            
            # Récupérer la timeline émotionnelle
            timeline = await self.memory_system.get_emotional_timeline(user_id, days)
            
            # Formater l'historique
            mood_history = []
            for timestamp, emotional_state in timeline:
                emotion, intensity = emotional_state.dominant_emotion()
                
                # Calculer valence et arousal
                positive_emotions = ['joy', 'love', 'excitement', 'contentment']
                negative_emotions = ['sadness', 'anger', 'fear', 'anxiety', 'disgust']
                
                valence = 0.0
                if emotion in positive_emotions:
                    valence = intensity
                elif emotion in negative_emotions:
                    valence = -intensity
                
                arousal = 0.5  # Par défaut
                high_arousal_emotions = ['excitement', 'anger', 'fear', 'anxiety']
                low_arousal_emotions = ['contentment', 'sadness']
                
                if emotion in high_arousal_emotions:
                    arousal = 0.5 + (intensity * 0.5)
                elif emotion in low_arousal_emotions:
                    arousal = 0.5 - (intensity * 0.3)
                
                mood_history.append({
                    "timestamp": timestamp.isoformat(),
                    "emotion": emotion,
                    "intensity": intensity,
                    "valence": valence,
                    "arousal": arousal,
                    "confidence": emotional_state.confidence
                })
            
            return mood_history
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique d'humeur: {e}")
            return []
    
    async def get_personality_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Récupère le profil de personnalité détaillé de l'utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Profil de personnalité avec insights
        """
        try:
            # Vérifier le cache
            if user_id in self._personality_cache:
                profile = self._personality_cache[user_id]
            else:
                # Créer un profil par défaut si l'utilisateur n'existe pas
                if user_id not in self.conversation_contexts:
                    return {
                        "error": "Utilisateur non trouvé",
                        "user_id": user_id
                    }
                
                context = self.conversation_contexts[user_id]
                user_texts = [ex['user_message'] for ex in context.conversation_history]
                
                if user_texts:
                    profile = await self.personality_analyzer.analyze_personality(
                        user_texts,
                        context.emotional_flow
                    )
                    self._personality_cache[user_id] = profile
                else:
                    profile = PersonalityProfile()
            
            # Générer des insights
            insights = self._generate_personality_insights(profile)
            
            return {
                "user_id": user_id,
                "profile": {
                    "big_five": {
                        "openness": profile.openness,
                        "conscientiousness": profile.conscientiousness,
                        "extraversion": profile.extraversion,
                        "agreeableness": profile.agreeableness,
                        "neuroticism": profile.neuroticism
                    },
                    "emotional_traits": {
                        "emotional_intelligence": profile.emotional_intelligence,
                        "empathy_level": profile.empathy_level,
                        "optimism": profile.optimism
                    },
                    "behavioral_traits": {
                        "creativity": profile.creativity,
                        "humor_appreciation": profile.humor_appreciation
                    }
                },
                "dominant_traits": self._get_dominant_traits(profile),
                "insights": insights,
                "recommendations": self._generate_personality_recommendations(profile),
                "last_updated": profile.last_updated.isoformat(),
                "confidence": profile.confidence
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du profil de personnalité: {e}")
            return {
                "error": f"Erreur lors de l'analyse: {str(e)}",
                "user_id": user_id
            }
    
    def _generate_personality_insights(self, profile: PersonalityProfile) -> List[str]:
        """Génère des insights basés sur le profil de personnalité"""
        insights = []
        
        # Insights Big Five
        if profile.openness > 0.7:
            insights.append("Vous êtes très ouvert aux nouvelles expériences et idées.")
        elif profile.openness < 0.3:
            insights.append("Vous préférez les approches familières et éprouvées.")
        
        if profile.conscientiousness > 0.7:
            insights.append("Vous êtes organisé et orienté vers les objectifs.")
        
        if profile.extraversion > 0.7:
            insights.append("Vous tirez de l'énergie des interactions sociales.")
        elif profile.extraversion < 0.3:
            insights.append("Vous préférez les environnements calmes et la réflexion.")
        
        if profile.agreeableness > 0.8:
            insights.append("Vous êtes naturellement empathique et bienveillant.")
        
        if profile.neuroticism > 0.7:
            insights.append("Vous êtes sensible et ressentez les émotions intensément.")
        
        # Insights émotionnels
        if profile.emotional_intelligence > 0.8:
            insights.append("Vous avez une excellente compréhension des émotions.")
        
        if profile.creativity > 0.7:
            insights.append("Vous avez un esprit créatif et innovant.")
        
        return insights
    
    def _generate_personality_recommendations(self, profile: PersonalityProfile) -> List[str]:
        """Génère des recommandations basées sur la personnalité"""
        recommendations = []
        
        if profile.neuroticism > 0.7:
            recommendations.append("Des techniques de gestion du stress pourraient vous être bénéfiques.")
        
        if profile.extraversion < 0.3:
            recommendations.append("Respectez votre besoin de solitude pour vous ressourcer.")
        
        if profile.openness > 0.7 and profile.creativity > 0.7:
            recommendations.append("Explorez de nouveaux projets créatifs pour nourrir votre curiosité.")
        
        if profile.conscientiousness < 0.4:
            recommendations.append("Des outils d'organisation pourraient vous aider dans vos projets.")
        
        if not recommendations:
            recommendations.append("Votre profil est équilibré. Continuez à cultiver vos forces.")
        
        return recommendations
    
    async def get_current_emotions(self, user_id: str) -> Dict[str, Any]:
        """
        Récupère l'état émotionnel actuel de l'utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            État émotionnel actuel
        """
        try:
            if user_id not in self.conversation_contexts:
                return {
                    "error": "Utilisateur non trouvé",
                    "user_id": user_id
                }
            
            context = self.conversation_contexts[user_id]
            
            # Récupérer le dernier état émotionnel
            if context.emotional_flow:
                current_state = context.emotional_flow[-1]
                emotion, intensity = current_state.dominant_emotion()
                
                # Calculer la tendance
                trend = "stable"
                if len(context.emotional_flow) >= 2:
                    prev_emotion, prev_intensity = context.emotional_flow[-2].dominant_emotion()
                    if intensity > prev_intensity + 0.2:
                        trend = "increasing"
                    elif intensity < prev_intensity - 0.2:
                        trend = "decreasing"
                
                return {
                    "user_id": user_id,
                    "current_emotion": {
                        "name": emotion,
                        "intensity": intensity,
                        "confidence": current_state.confidence
                    },
                    "emotional_state": current_state.to_dict(),
                    "trend": trend,
                    "recent_emotions": [
                        {
                            "emotion": state.dominant_emotion()[0],
                            "intensity": state.dominant_emotion()[1],
                            "timestamp": state.timestamp.isoformat()
                        }
                        for state in context.emotional_flow[-5:]
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "user_id": user_id,
                    "current_emotion": {
                        "name": "neutral",
                        "intensity": 0.0,
                        "confidence": 0.0
                    },
                    "message": "Pas encore d'état émotionnel enregistré",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des émotions actuelles: {e}")
            return {
                "error": f"Erreur lors de l'analyse: {str(e)}",
                "user_id": user_id
            }
