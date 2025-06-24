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

from ..config import Config
from ..emotional import EmotionalState, PersonalityProfile, LocalEmotionAnalyzer, PersonalityAnalyzer
from ..models import LocalLanguageModel
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
        self.language_model = LocalLanguageModel(self.config.models)
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
            
            # Étape 2: Analyser l'émotion du message
            emotional_state = await self.emotion_analyzer.analyze_emotion(
                user_input, 
                context.get_recent_context()
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
                response_type=response_type
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