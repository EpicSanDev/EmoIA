"""
Tests unitaires pour le module d'intelligence émotionnelle
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.emotional.core import (
    EmotionalState, PersonalityProfile, 
    LocalEmotionAnalyzer, PersonalityAnalyzer
)
from src.config import EmotionalConfig


class TestEmotionalState(unittest.TestCase):
    """Tests pour la classe EmotionalState"""
    
    def test_init(self):
        """Test de l'initialisation"""
        state = EmotionalState(joy=0.8, sadness=0.2, confidence=0.9)
        
        self.assertEqual(state.joy, 0.8)
        self.assertEqual(state.sadness, 0.2)
        self.assertEqual(state.confidence, 0.9)
        self.assertIsInstance(state.timestamp, datetime)
    
    def test_dominant_emotion(self):
        """Test de détection de l'émotion dominante"""
        state = EmotionalState(
            joy=0.8, sadness=0.2, anger=0.1, 
            fear=0.3, love=0.9
        )
        
        emotion, intensity = state.dominant_emotion()
        self.assertEqual(emotion, "love")
        self.assertEqual(intensity, 0.9)
    
    def test_emotional_vector(self):
        """Test de conversion en vecteur"""
        state = EmotionalState(joy=0.5, sadness=0.3)
        vector = state.emotional_vector()
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), 11)  # 11 émotions
        self.assertEqual(vector[0], 0.5)  # joy
        self.assertEqual(vector[1], 0.3)  # sadness
    
    def test_similarity(self):
        """Test de calcul de similarité"""
        state1 = EmotionalState(joy=0.8, sadness=0.2)
        state2 = EmotionalState(joy=0.7, sadness=0.3)
        state3 = EmotionalState(anger=0.8, fear=0.7)
        
        similarity_close = state1.similarity(state2)
        similarity_far = state1.similarity(state3)
        
        self.assertGreater(similarity_close, 0.8)
        self.assertLess(similarity_far, 0.3)
    
    def test_to_dict(self):
        """Test de sérialisation"""
        state = EmotionalState(joy=0.8, confidence=0.9)
        dict_state = state.to_dict()
        
        self.assertIn("joy", dict_state)
        self.assertIn("timestamp", dict_state)
        self.assertIn("confidence", dict_state)
        self.assertEqual(dict_state["joy"], 0.8)


class TestPersonalityProfile(unittest.TestCase):
    """Tests pour la classe PersonalityProfile"""
    
    def test_init(self):
        """Test de l'initialisation"""
        profile = PersonalityProfile(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.6
        )
        
        self.assertEqual(profile.openness, 0.7)
        self.assertEqual(profile.conscientiousness, 0.8)
        self.assertEqual(profile.extraversion, 0.6)
        # Valeurs par défaut
        self.assertEqual(profile.agreeableness, 0.5)
        self.assertEqual(profile.neuroticism, 0.5)
    
    def test_to_vector(self):
        """Test de conversion en vecteur"""
        profile = PersonalityProfile(
            openness=0.7,
            emotional_intelligence=0.8
        )
        vector = profile.to_vector()
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), 10)
        self.assertEqual(vector[0], 0.7)  # openness
    
    def test_distance(self):
        """Test de calcul de distance"""
        profile1 = PersonalityProfile(openness=0.8, extraversion=0.7)
        profile2 = PersonalityProfile(openness=0.7, extraversion=0.8)
        profile3 = PersonalityProfile(openness=0.2, extraversion=0.1)
        
        distance_close = profile1.distance(profile2)
        distance_far = profile1.distance(profile3)
        
        self.assertLess(distance_close, distance_far)
        self.assertLess(distance_close, 0.5)
        self.assertGreater(distance_far, 1.0)


class TestLocalEmotionAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Tests pour l'analyseur d'émotions local"""
    
    async def asyncSetUp(self):
        """Configuration des tests"""
        self.config = EmotionalConfig()
        self.analyzer = LocalEmotionAnalyzer(self.config)
        
        # Mock des modèles
        self.analyzer.emotion_tokenizer = MagicMock()
        self.analyzer.emotion_model = MagicMock()
        self.analyzer.sentiment_pipeline = MagicMock()
        self.analyzer.embedding_model = MagicMock()
    
    @patch('src.emotional.core.AutoTokenizer')
    @patch('src.emotional.core.AutoModelForSequenceClassification')
    @patch('src.emotional.core.pipeline')
    @patch('src.emotional.core.SentenceTransformer')
    async def test_initialize(self, mock_st, mock_pipeline, mock_model, mock_tokenizer):
        """Test de l'initialisation"""
        analyzer = LocalEmotionAnalyzer(self.config)
        await analyzer.initialize()
        
        mock_tokenizer.from_pretrained.assert_called()
        mock_model.from_pretrained.assert_called()
        mock_pipeline.assert_called()
        mock_st.assert_called()
    
    async def test_analyze_emotion_empty_text(self):
        """Test avec texte vide"""
        result = await self.analyzer.analyze_emotion("")
        
        self.assertIsInstance(result, EmotionalState)
        self.assertEqual(result.joy, 0.0)
        self.assertEqual(result.confidence, 0.0)
    
    async def test_analyze_emotion_with_cache(self):
        """Test du cache d'émotions"""
        # Première analyse
        self.analyzer._analyze_primary_emotions = AsyncMock(
            return_value={"joy": 0.8, "sadness": 0.2}
        )
        self.analyzer._analyze_sentiment = AsyncMock(
            return_value={"contentment": 0.6}
        )
        self.analyzer._analyze_context = AsyncMock(
            return_value={"curiosity": 0.3}
        )
        
        result1 = await self.analyzer.analyze_emotion("Je suis heureux!")
        
        # Deuxième analyse (devrait utiliser le cache)
        result2 = await self.analyzer.analyze_emotion("Je suis heureux!")
        
        # Les méthodes ne devraient être appelées qu'une fois
        self.analyzer._analyze_primary_emotions.assert_called_once()
        
        # Les résultats devraient être identiques
        self.assertEqual(result1.joy, result2.joy)
        self.assertEqual(result1.confidence, result2.confidence)


class TestPersonalityAnalyzer(unittest.IsolatedAsyncioTestCase):
    """Tests pour l'analyseur de personnalité"""
    
    async def asyncSetUp(self):
        """Configuration des tests"""
        self.config = EmotionalConfig()
        self.analyzer = PersonalityAnalyzer(self.config)
        
        # Mock des modèles
        self.analyzer.feature_extractor = MagicMock()
        self.analyzer.personality_model = MagicMock()
    
    async def test_analyze_personality_empty_texts(self):
        """Test avec liste vide"""
        result = await self.analyzer.analyze_personality([], [])
        
        self.assertIsInstance(result, PersonalityProfile)
        self.assertEqual(result.openness, 0.5)  # Valeur par défaut
    
    async def test_extract_linguistic_features(self):
        """Test d'extraction de features linguistiques"""
        texts = [
            "Bonjour! Comment allez-vous?",
            "Je suis très content :)",
            "Quelle belle journée!"
        ]
        
        features = self.analyzer._extract_linguistic_features(texts)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 10)
        # Vérifier quelques features
        self.assertGreater(features[1], 0)  # Questions
        self.assertGreater(features[2], 0)  # Exclamations
    
    async def test_calculate_empathy_level(self):
        """Test du calcul du niveau d'empathie"""
        # Historique avec émotions variées
        emotional_history = [
            EmotionalState(joy=0.8),
            EmotionalState(sadness=0.7),
            EmotionalState(fear=0.6),
            EmotionalState(love=0.8)
        ]
        
        empathy = self.analyzer._calculate_empathy_level(emotional_history)
        
        self.assertGreater(empathy, 0.3)
        self.assertLessEqual(empathy, 1.0)
    
    async def test_calculate_creativity_score(self):
        """Test du calcul du score de créativité"""
        texts_creative = [
            "J'adore explorer de nouvelles idées fascinantes",
            "Imaginons un monde différent et merveilleux",
            "Créons quelque chose d'unique ensemble"
        ]
        
        texts_repetitive = [
            "Ok ok ok",
            "Oui oui oui",
            "Non non non"
        ]
        
        creativity_high = self.analyzer._calculate_creativity_score(texts_creative)
        creativity_low = self.analyzer._calculate_creativity_score(texts_repetitive)
        
        self.assertGreater(creativity_high, creativity_low)
        self.assertGreater(creativity_high, 0.5)
        self.assertLess(creativity_low, 0.3)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests d'intégration entre composants"""
    
    async def test_emotion_to_personality_flow(self):
        """Test du flux émotion vers personnalité"""
        # Créer des états émotionnels
        emotional_states = [
            EmotionalState(joy=0.8, excitement=0.7),
            EmotionalState(curiosity=0.9, contentment=0.6),
            EmotionalState(love=0.8, joy=0.7)
        ]
        
        # Analyser la personnalité basée sur ces émotions
        config = EmotionalConfig()
        analyzer = PersonalityAnalyzer(config)
        analyzer.feature_extractor = MagicMock()
        analyzer.feature_extractor.encode = MagicMock(
            return_value=[np.random.random(384)]
        )
        
        texts = ["Je suis heureux!", "Quelle découverte!", "J'adore ça!"]
        
        profile = await analyzer.analyze_personality(texts, emotional_states)
        
        # Avec des émotions positives, on s'attend à un optimisme élevé
        self.assertGreater(profile.optimism, 0.5)


if __name__ == "__main__":
    unittest.main()