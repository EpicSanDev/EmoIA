"""
Tests unitaires pour la classe principale EmoIA
"""

import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.core.emoia_main import EmoIA, ConversationContext, ProactivityEngine
from src.emotional import EmotionalState, PersonalityProfile
from src.config import Config


class TestConversationContext(unittest.TestCase):
    """Tests pour la classe ConversationContext"""
    
    def setUp(self):
        self.context = ConversationContext("test_user")
    
    def test_init(self):
        """Test de l'initialisation"""
        self.assertEqual(self.context.user_id, "test_user")
        self.assertEqual(len(self.context.conversation_history), 0)
        self.assertEqual(self.context.conversation_depth, 0)
    
    def test_add_exchange(self):
        """Test de l'ajout d'un échange"""
        emotional_state = EmotionalState(joy=0.8, confidence=0.9)
        self.context.add_exchange(
            "Hello", "Hi there!", emotional_state, importance=0.7
        )
        
        self.assertEqual(len(self.context.conversation_history), 1)
        self.assertEqual(self.context.conversation_depth, 1)
        self.assertEqual(self.context.conversation_history[0]['user_message'], "Hello")
        self.assertEqual(self.context.conversation_history[0]['ai_response'], "Hi there!")
    
    def test_get_recent_context(self):
        """Test de récupération du contexte récent"""
        for i in range(10):
            self.context.add_exchange(
                f"Message {i}", 
                f"Response {i}", 
                EmotionalState()
            )
        
        recent = self.context.get_recent_context(n=3)
        self.assertIn("Message 7", recent)
        self.assertIn("Response 9", recent)
        self.assertNotIn("Message 0", recent)
    
    def test_get_emotional_trend(self):
        """Test de l'analyse de tendance émotionnelle"""
        # Ajouter des états émotionnels variés
        emotions = [
            EmotionalState(joy=0.8, sadness=0.1),
            EmotionalState(joy=0.6, sadness=0.3),
            EmotionalState(joy=0.4, sadness=0.5)
        ]
        
        for i, emotion in enumerate(emotions):
            self.context.add_exchange(f"Msg {i}", f"Resp {i}", emotion)
        
        trend = self.context.get_emotional_trend()
        self.assertIn('joy', trend)
        self.assertIn('sadness', trend)
        self.assertGreater(trend['joy'], 0)
        self.assertGreater(trend['sadness'], 0)


class TestProactivityEngine(unittest.TestCase):
    """Tests pour le moteur de proactivité"""
    
    def setUp(self):
        self.config = Config()
        self.engine = ProactivityEngine(self.config)
        self.context = ConversationContext("test_user")
    
    def test_should_initiate_conversation_daily_check(self):
        """Test de l'initiation de conversation quotidienne"""
        # Simuler une dernière interaction il y a 25 heures
        from datetime import timedelta
        self.context.last_interaction = datetime.now() - timedelta(hours=25)
        
        should_initiate, reason = self.engine.should_initiate_conversation(
            "test_user", self.context
        )
        
        self.assertTrue(should_initiate)
        self.assertEqual(reason, "check_in_daily")
    
    def test_generate_proactive_message(self):
        """Test de génération de message proactif"""
        personality = PersonalityProfile(extraversion=0.8)
        
        message = self.engine.generate_proactive_message(
            "check_in_daily", self.context, personality
        )
        
        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)
        # Pour un extraverti, on s'attend à un emoji
        self.assertIn("😊", message)


class TestEmoIA(unittest.IsolatedAsyncioTestCase):
    """Tests pour la classe principale EmoIA"""
    
    async def asyncSetUp(self):
        """Configuration des tests asynchrones"""
        self.config = Config()
        self.emoia = EmoIA(self.config)
        
        # Mock des composants
        self.emoia.emotion_analyzer = AsyncMock()
        self.emoia.personality_analyzer = AsyncMock()
        self.emoia.language_model = AsyncMock()
        self.emoia.memory_system = AsyncMock()
        
        # Configuration des mocks
        self.emoia.emotion_analyzer.initialize = AsyncMock()
        self.emoia.personality_analyzer.initialize = AsyncMock()
        self.emoia.language_model.initialize = AsyncMock()
        self.emoia.memory_system.initialize = AsyncMock()
        
        self.emoia.is_initialized = True
    
    async def test_initialize(self):
        """Test de l'initialisation"""
        emoia = EmoIA(self.config)
        emoia.emotion_analyzer = AsyncMock()
        emoia.personality_analyzer = AsyncMock()
        emoia.language_model = AsyncMock()
        emoia.memory_system = AsyncMock()
        
        await emoia.initialize()
        
        self.assertTrue(emoia.is_initialized)
        emoia.emotion_analyzer.initialize.assert_called_once()
        emoia.personality_analyzer.initialize.assert_called_once()
        emoia.language_model.initialize.assert_called_once()
        emoia.memory_system.initialize.assert_called_once()
    
    async def test_process_message(self):
        """Test du traitement d'un message"""
        # Configuration des mocks
        emotional_state = EmotionalState(joy=0.8, confidence=0.9)
        self.emoia.emotion_analyzer.analyze_emotion = AsyncMock(
            return_value=emotional_state
        )
        
        personality = PersonalityProfile(openness=0.7)
        self.emoia.personality_analyzer.analyze_personality = AsyncMock(
            return_value=personality
        )
        
        self.emoia.memory_system.retrieve_memories = AsyncMock(
            return_value=[]
        )
        self.emoia.memory_system.store_memory = AsyncMock()
        
        self.emoia.language_model.generate_response = AsyncMock(
            return_value="Je suis heureux de vous aider!"
        )
        
        # Appel de la méthode
        result = await self.emoia.process_message(
            "Bonjour, comment allez-vous?",
            "test_user",
            {"language": "fr"}
        )
        
        # Vérifications
        self.assertIn("response", result)
        self.assertEqual(result["response"], "Je suis heureux de vous aider!")
        self.assertIn("emotional_analysis", result)
        self.assertEqual(result["emotional_analysis"]["detected_emotion"], "joy")
        self.assertIn("personality_insights", result)
        self.assertIn("interaction_metadata", result)
    
    async def test_generate_suggestions(self):
        """Test de génération de suggestions"""
        # Test avec état émotionnel triste
        suggestions = await self.emoia.generate_suggestions(
            context="Je me sens un peu triste aujourd'hui",
            emotional_state={"dominant_emotion": "sadness"},
            max_suggestions=3
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 3)
        self.assertTrue(any("parler" in s["text"].lower() for s in suggestions))
        
        # Test avec contexte de travail
        suggestions = await self.emoia.generate_suggestions(
            context="J'ai beaucoup de travail en ce moment",
            max_suggestions=2
        )
        
        self.assertTrue(any("travail" in s["text"].lower() for s in suggestions))
    
    async def test_get_conversation_insights(self):
        """Test de récupération des insights de conversation"""
        # Créer un contexte avec historique
        context = ConversationContext("test_user")
        for i in range(5):
            context.add_exchange(
                f"Question {i}?",
                f"Réponse {i}",
                EmotionalState(joy=0.5 + i*0.1),
                importance=0.5
            )
        
        self.emoia.conversation_contexts["test_user"] = context
        
        insights = await self.emoia.get_conversation_insights("test_user")
        
        self.assertIn("conversation_stats", insights)
        self.assertIn("topics_discussed", insights)
        self.assertIn("emotional_journey", insights)
        self.assertIn("engagement_metrics", insights)
        self.assertEqual(insights["conversation_stats"]["total_exchanges"], 5)
    
    async def test_get_mood_history(self):
        """Test de l'historique d'humeur"""
        # Mock de la timeline émotionnelle
        timeline = [
            (datetime.now(), EmotionalState(joy=0.8, confidence=0.9)),
            (datetime.now(), EmotionalState(sadness=0.6, confidence=0.8)),
            (datetime.now(), EmotionalState(excitement=0.7, confidence=0.85))
        ]
        
        self.emoia.memory_system.get_emotional_timeline = AsyncMock(
            return_value=timeline
        )
        
        history = await self.emoia.get_mood_history("test_user", "week")
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 3)
        self.assertIn("timestamp", history[0])
        self.assertIn("emotion", history[0])
        self.assertIn("valence", history[0])
        self.assertIn("arousal", history[0])
    
    async def test_get_personality_profile(self):
        """Test du profil de personnalité"""
        # Créer un profil mock
        profile = PersonalityProfile(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.9,
            neuroticism=0.3
        )
        
        self.emoia._personality_cache["test_user"] = profile
        
        result = await self.emoia.get_personality_profile("test_user")
        
        self.assertIn("profile", result)
        self.assertIn("big_five", result["profile"])
        self.assertIn("dominant_traits", result)
        self.assertIn("insights", result)
        self.assertIn("recommendations", result)
        self.assertEqual(result["profile"]["big_five"]["openness"], 0.8)
    
    async def test_get_current_emotions(self):
        """Test des émotions actuelles"""
        # Créer un contexte avec flux émotionnel
        context = ConversationContext("test_user")
        emotions = [
            EmotionalState(joy=0.5, confidence=0.8),
            EmotionalState(joy=0.7, confidence=0.9),
            EmotionalState(joy=0.9, confidence=0.95)
        ]
        
        for emotion in emotions:
            context.emotional_flow.append(emotion)
        
        self.emoia.conversation_contexts["test_user"] = context
        
        current = await self.emoia.get_current_emotions("test_user")
        
        self.assertIn("current_emotion", current)
        self.assertEqual(current["current_emotion"]["name"], "joy")
        self.assertEqual(current["current_emotion"]["intensity"], 0.9)
        self.assertEqual(current["trend"], "increasing")
        self.assertIn("recent_emotions", current)
    
    def test_calculate_interaction_importance(self):
        """Test du calcul d'importance"""
        emotional_state = EmotionalState(sadness=0.8, confidence=0.9)
        personality = PersonalityProfile(emotional_intelligence=0.9)
        
        # Message important avec émotion forte
        importance = self.emoia._calculate_interaction_importance(
            "C'est très important et urgent, j'ai un gros problème!",
            "Je comprends, parlons-en.",
            emotional_state,
            personality
        )
        
        self.assertGreater(importance, 0.8)
        self.assertLessEqual(importance, 1.0)
        
        # Message simple
        importance = self.emoia._calculate_interaction_importance(
            "Ok",
            "D'accord",
            EmotionalState(),
            PersonalityProfile()
        )
        
        self.assertLess(importance, 0.6)
    
    def test_extract_tags(self):
        """Test de l'extraction de tags"""
        tags = self.emoia._extract_tags(
            "Je suis stressé par mon travail et ma famille",
            "Je comprends votre stress concernant le travail"
        )
        
        self.assertIn("travail", tags)
        self.assertIn("famille", tags)
        self.assertIn("stress", tags)
    
    def test_get_dominant_traits(self):
        """Test de l'identification des traits dominants"""
        personality = PersonalityProfile(
            openness=0.9,
            conscientiousness=0.8,
            emotional_intelligence=0.85,
            creativity=0.9
        )
        
        traits = self.emoia._get_dominant_traits(personality)
        
        self.assertIn("ouvert d'esprit", traits)
        self.assertIn("consciencieux", traits)
        self.assertIn("émotionnellement intelligent", traits)
        self.assertIn("créatif", traits)


class TestErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Tests de gestion d'erreur"""
    
    async def test_process_message_error_handling(self):
        """Test de la gestion d'erreur dans process_message"""
        emoia = EmoIA()
        emoia.emotion_analyzer = AsyncMock()
        emoia.emotion_analyzer.analyze_emotion = AsyncMock(
            side_effect=Exception("Erreur de test")
        )
        emoia.is_initialized = True
        
        result = await emoia.process_message("Test", "user1")
        
        self.assertIn("error", result["system_info"])
        self.assertIn("désolé", result["response"])
        self.assertEqual(result["emotional_analysis"]["detected_emotion"], "neutral")


if __name__ == "__main__":
    unittest.main()