"""
Moteur d'Intelligence Émotionnelle Avancé pour EmoIA
Système complet d'analyse, génération et adaptation émotionnelle.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from ..config import Config, EmotionalConfig


logger = logging.getLogger(__name__)


@dataclass
class EmotionalState:
    """État émotionnel complet à un moment donné"""
    
    # Émotions primaires (intensité 0-1)
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    love: float = 0.0
    excitement: float = 0.0
    anxiety: float = 0.0
    contentment: float = 0.0
    curiosity: float = 0.0
    
    # Métadonnées
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    context: str = ""
    
    def dominant_emotion(self) -> Tuple[str, float]:
        """Retourne l'émotion dominante et son intensité"""
        emotions = {
            "joy": self.joy, "sadness": self.sadness, "anger": self.anger,
            "fear": self.fear, "surprise": self.surprise, "disgust": self.disgust,
            "love": self.love, "excitement": self.excitement, "anxiety": self.anxiety,
            "contentment": self.contentment, "curiosity": self.curiosity
        }
        dominant = max(emotions.keys(), key=lambda k: emotions[k])
        return dominant, emotions[dominant]
    
    def emotional_vector(self) -> np.ndarray:
        """Retourne l'état émotionnel sous forme de vecteur"""
        return np.array([
            self.joy, self.sadness, self.anger, self.fear, self.surprise,
            self.disgust, self.love, self.excitement, self.anxiety,
            self.contentment, self.curiosity
        ])
    
    def similarity(self, other: "EmotionalState") -> float:
        """Calcule la similarité avec un autre état émotionnel"""
        vec1 = self.emotional_vector()
        vec2 = other.emotional_vector()
        return cosine_similarity([vec1], [vec2])[0][0]

    def to_dict(self) -> dict:
        """Convertit l'état émotionnel en dictionnaire sérialisable en JSON"""
        return {
            'joy': self.joy,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust,
            'love': self.love,
            'excitement': self.excitement,
            'anxiety': self.anxiety,
            'contentment': self.contentment,
            'curiosity': self.curiosity,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'confidence': self.confidence,
            'context': self.context
        }


@dataclass 
class PersonalityProfile:
    """Profil de personnalité Big Five avec extensions émotionnelles"""
    
    # Big Five
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    # Extensions émotionnelles
    emotional_intelligence: float = 0.7
    empathy_level: float = 0.8
    creativity: float = 0.6
    humor_appreciation: float = 0.7
    optimism: float = 0.6
    
    # Métadonnées
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    adaptation_count: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convertit le profil en vecteur numérique"""
        return np.array([
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism, self.emotional_intelligence,
            self.empathy_level, self.creativity, self.humor_appreciation,
            self.optimism
        ])
    
    def distance(self, other: "PersonalityProfile") -> float:
        """Calcule la distance euclidienne avec un autre profil"""
        vec1 = self.to_vector()
        vec2 = other.to_vector()
        return np.linalg.norm(vec1 - vec2)


class LocalEmotionAnalyzer:
    """Analyseur d'émotions utilisant des modèles locaux"""
    
    def __init__(self, config: EmotionalConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modèles d'analyse émotionnelle
        self.emotion_tokenizer = None
        self.emotion_model = None
        self.sentiment_pipeline = None
        self.embedding_model = None
        
        # Cache des résultats
        self._emotion_cache = {}
        self._cache_ttl = 3600  # 1 heure
        
    async def initialize(self):
        """Initialise les modèles d'analyse émotionnelle"""
        try:
            logger.info("Initialisation des modèles d'analyse émotionnelle...")
            
            # Modèle d'émotion principal
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            )
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            ).to(self.device)
            
            # Pipeline de sentiment
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modèle d'embedding pour analyse sémantique
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.info("Modèles d'analyse émotionnelle initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles: {e}")
            raise
    
    async def analyze_emotion(self, text: str, context: str = "") -> EmotionalState:
        """Analyse complète des émotions dans un texte"""
        if not text.strip():
            return EmotionalState()
        
        # Vérifier le cache
        cache_key = hash(text + context)
        if cache_key in self._emotion_cache:
            cached_result, cached_time = self._emotion_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self._cache_ttl):
                return cached_result
        
        try:
            # Analyse des émotions principales
            emotion_scores = await self._analyze_primary_emotions(text)
            
            # Analyse du sentiment
            sentiment_scores = await self._analyze_sentiment(text)
            
            # Analyse contextuelle
            contextual_boost = await self._analyze_context(text, context)
            
            # Fusion des résultats
            emotional_state = self._fuse_emotion_results(
                emotion_scores, sentiment_scores, contextual_boost
            )
            emotional_state.context = context
            
            # Mise en cache
            self._emotion_cache[cache_key] = (emotional_state, datetime.now())
            
            return emotional_state
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse émotionnelle: {e}")
            return EmotionalState()
    
    async def _analyze_primary_emotions(self, text: str) -> Dict[str, float]:
        """Analyse les émotions primaires avec le modèle local"""
        inputs = self.emotion_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Mapping des labels du modèle vers nos émotions
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        emotion_scores = {}
        
        for i, label in enumerate(emotion_labels):
            emotion_scores[label] = float(predictions[0][i])
        
        return emotion_scores
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyse du sentiment avec pipeline local"""
        result = self.sentiment_pipeline(text)[0]
        
        # Conversion en scores émotionnels
        sentiment_map = {
            'POSITIVE': {'joy': 0.7, 'contentment': 0.6, 'love': 0.3},
            'NEGATIVE': {'sadness': 0.6, 'anger': 0.4, 'anxiety': 0.5},
            'NEUTRAL': {'contentment': 0.4}
        }
        
        label = result['label']
        confidence = result['score']
        
        scores = {}
        if label in sentiment_map:
            for emotion, base_score in sentiment_map[label].items():
                scores[emotion] = base_score * confidence
        
        return scores
    
    async def _analyze_context(self, text: str, context: str) -> Dict[str, float]:
        """Analyse contextuelle pour ajuster les émotions"""
        if not context:
            return {}
        
        # Similarité sémantique entre texte et contexte
        embeddings = self.embedding_model.encode([text, context])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Boost contextuel basé sur la similarité
        context_boost = {}
        if similarity > 0.7:
            # Contexte très pertinent - amplifier les émotions
            context_boost = {'excitement': 0.2, 'curiosity': 0.3}
        elif similarity > 0.4:
            # Contexte moyennement pertinent
            context_boost = {'curiosity': 0.1}
        
        return context_boost
    
    def _fuse_emotion_results(
        self, 
        emotion_scores: Dict[str, float], 
        sentiment_scores: Dict[str, float],
        contextual_boost: Dict[str, float]
    ) -> EmotionalState:
        """Fusionne les différents scores d'émotion"""
        
        # Combinaison pondérée des scores
        final_scores = {}
        
        # Émotions de base
        for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']:
            score = emotion_scores.get(emotion, 0.0)
            score += sentiment_scores.get(emotion, 0.0) * 0.5
            score += contextual_boost.get(emotion, 0.0)
            final_scores[emotion] = min(score, 1.0)
        
        # Émotions étendues
        extended_emotions = ['excitement', 'anxiety', 'contentment', 'curiosity', 'disgust']
        for emotion in extended_emotions:
            score = sentiment_scores.get(emotion, 0.0)
            score += contextual_boost.get(emotion, 0.0)
            final_scores[emotion] = min(score, 1.0)
        
        # Calcul de la confiance globale
        confidence = np.mean(list(final_scores.values()))
        
        return EmotionalState(
            joy=final_scores.get('joy', 0.0),
            sadness=final_scores.get('sadness', 0.0),
            anger=final_scores.get('anger', 0.0),
            fear=final_scores.get('fear', 0.0),
            surprise=final_scores.get('surprise', 0.0),
            disgust=final_scores.get('disgust', 0.0),
            love=final_scores.get('love', 0.0),
            excitement=final_scores.get('excitement', 0.0),
            anxiety=final_scores.get('anxiety', 0.0),
            contentment=final_scores.get('contentment', 0.0),
            curiosity=final_scores.get('curiosity', 0.0),
            confidence=confidence
        )


class PersonalityAnalyzer:
    """Analyseur de personnalité basé sur les interactions"""
    
    def __init__(self, config: EmotionalConfig):
        self.config = config
        self.personality_model = None
        self.feature_extractor = None
        
    async def initialize(self):
        """Initialise l'analyseur de personnalité"""
        try:
            # Modèle d'extraction de features linguistiques
            self.feature_extractor = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Charger ou créer le modèle de personnalité
            model_path = Path("models/personality_model.joblib")
            if model_path.exists():
                self.personality_model = joblib.load(model_path)
            else:
                await self._create_personality_model()
                
            logger.info("Analyseur de personnalité initialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'analyseur de personnalité: {e}")
            raise
    
    async def analyze_personality(
        self, 
        texts: List[str], 
        emotional_history: List[EmotionalState]
    ) -> PersonalityProfile:
        """Analyse la personnalité basée sur les textes et l'historique émotionnel"""
        
        if not texts:
            return PersonalityProfile()
        
        try:
            # Extraction des features textuelles
            text_features = await self._extract_text_features(texts)
            
            # Features émotionnelles
            emotion_features = self._extract_emotion_features(emotional_history)
            
            # Features linguistiques
            linguistic_features = self._extract_linguistic_features(texts)
            
            # Combinaison des features
            all_features = np.concatenate([
                text_features, emotion_features, linguistic_features
            ])
            
            # Prédiction de personnalité
            if self.personality_model:
                personality_scores = self.personality_model.predict([all_features])[0]
            else:
                # Fallback vers une analyse heuristique
                personality_scores = self._heuristic_personality_analysis(texts)
            
            return PersonalityProfile(
                openness=personality_scores[0],
                conscientiousness=personality_scores[1],
                extraversion=personality_scores[2],
                agreeableness=personality_scores[3],
                neuroticism=personality_scores[4],
                emotional_intelligence=np.mean([e.confidence for e in emotional_history[-10:]]),
                empathy_level=self._calculate_empathy_level(emotional_history),
                creativity=self._calculate_creativity_score(texts),
                humor_appreciation=self._detect_humor_appreciation(texts),
                optimism=self._calculate_optimism_score(emotional_history)
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de personnalité: {e}")
            return PersonalityProfile()
    
    async def _extract_text_features(self, texts: List[str]) -> np.ndarray:
        """Extrait les features sémantiques des textes"""
        combined_text = " ".join(texts[-20:])  # Prendre les 20 derniers textes
        embedding = self.feature_extractor.encode([combined_text])
        return embedding[0]
    
    def _extract_emotion_features(self, emotional_history: List[EmotionalState]) -> np.ndarray:
        """Extrait les features basées sur l'historique émotionnel"""
        if not emotional_history:
            return np.zeros(15)
        
        recent_emotions = emotional_history[-50:]  # 50 dernières émotions
        
        # Statistiques émotionnelles
        emotion_vectors = [e.emotional_vector() for e in recent_emotions]
        emotion_matrix = np.array(emotion_vectors)
        
        features = []
        features.extend(np.mean(emotion_matrix, axis=0))  # Moyennes
        features.append(np.std(emotion_matrix))  # Variance émotionnelle
        features.append(len(set([e.dominant_emotion()[0] for e in recent_emotions])))  # Diversité
        features.append(np.mean([e.confidence for e in recent_emotions]))  # Confiance moyenne
        
        return np.array(features)
    
    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extrait les features linguistiques"""
        if not texts:
            return np.zeros(10)
        
        combined_text = " ".join(texts)
        
        features = []
        features.append(len(combined_text.split()) / len(texts))  # Mots par message
        features.append(combined_text.count('?') / len(texts))  # Questions
        features.append(combined_text.count('!') / len(texts))  # Exclamations
        features.append(len([w for w in combined_text.split() if w.isupper()]) / len(combined_text.split()))  # Majuscules
        features.append(combined_text.count(':)') + combined_text.count('😊'))  # Émojis positifs
        features.append(combined_text.count(':(') + combined_text.count('😢'))  # Émojis négatifs
        features.append(len(set(combined_text.lower().split())) / len(combined_text.split()))  # Diversité lexicale
        features.append(np.mean([len(sentence.split()) for sentence in combined_text.split('.')]))  # Longueur phrase
        features.append(combined_text.lower().count('je') / len(combined_text.split()))  # Auto-référence
        features.append(combined_text.lower().count('nous') + combined_text.lower().count('on'))  # Références groupe
        
        return np.array(features)
    
    async def _create_personality_model(self):
        """Crée un modèle de personnalité basique (à entraîner avec des données réelles)"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Modèle de placeholder - devrait être entraîné avec des données réelles
        self.personality_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Données synthétiques pour l'initialisation
        X_dummy = np.random.random((1000, 409))  # 384 (embedding) + 15 (émotions) + 10 (linguistique)
        y_dummy = np.random.random((1000, 5))   # Big Five scores
        
        self.personality_model.fit(X_dummy, y_dummy)
        
        # Sauvegarder le modèle
        model_path = Path("models/personality_model.joblib")
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.personality_model, model_path)
    
    def _heuristic_personality_analysis(self, texts: List[str]) -> np.ndarray:
        """Analyse heuristique de la personnalité en fallback"""
        combined_text = " ".join(texts).lower()
        
        # Heuristiques simples pour Big Five
        openness = min(1.0, len(set(combined_text.split())) / max(len(combined_text.split()), 1) * 2)
        conscientiousness = min(1.0, (combined_text.count('plan') + combined_text.count('organis')) / 10)
        extraversion = min(1.0, (combined_text.count('ami') + combined_text.count('social')) / 10)
        agreeableness = min(1.0, (combined_text.count('merci') + combined_text.count('s\'il')) / 10)
        neuroticism = min(1.0, (combined_text.count('stress') + combined_text.count('inquiet')) / 10)
        
        return np.array([openness, conscientiousness, extraversion, agreeableness, neuroticism])
    
    def _calculate_empathy_level(self, emotional_history: List[EmotionalState]) -> float:
        """Calcule le niveau d'empathie basé sur la diversité émotionnelle"""
        if not emotional_history:
            return 0.5
        
        emotions = [e.dominant_emotion()[0] for e in emotional_history[-20:]]
        unique_emotions = len(set(emotions))
        return min(1.0, unique_emotions / 8.0)  # 8 émotions principales
    
    def _calculate_creativity_score(self, texts: List[str]) -> float:
        """Calcule un score de créativité basé sur la diversité lexicale"""
        if not texts:
            return 0.5
        
        combined = " ".join(texts)
        words = combined.lower().split()
        unique_words = len(set(words))
        return min(1.0, unique_words / max(len(words), 1) * 3)
    
    def _detect_humor_appreciation(self, texts: List[str]) -> float:
        """Détecte l'appréciation de l'humour"""
        humor_indicators = ['haha', 'lol', 'mdr', '😂', '😄', 'drôle', 'blague']
        combined = " ".join(texts).lower()
        
        humor_count = sum(combined.count(indicator) for indicator in humor_indicators)
        return min(1.0, humor_count / max(len(texts), 1))
    
    def _calculate_optimism_score(self, emotional_history: List[EmotionalState]) -> float:
        """Calcule un score d'optimisme basé sur l'historique émotionnel"""
        if not emotional_history:
            return 0.5
        
        positive_emotions = ['joy', 'love', 'excitement', 'contentment']
        recent_states = emotional_history[-20:]
        
        positive_ratio = 0
        for state in recent_states:
            positive_score = sum(getattr(state, emotion) for emotion in positive_emotions)
            negative_score = state.sadness + state.anger + state.fear + state.anxiety
            if positive_score > negative_score:
                positive_ratio += 1
        
        return positive_ratio / len(recent_states)