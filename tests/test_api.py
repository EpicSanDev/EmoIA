"""
Tests pour l'API FastAPI d'EmoIA
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.core.api import app
from src.emotional import EmotionalState, PersonalityProfile


@pytest.fixture
def client():
    """Client de test FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_emoia():
    """Mock de la classe EmoIA"""
    with patch('src.core.api.emoia') as mock:
        # Configuration des méthodes mockées
        mock.process_message = AsyncMock(return_value={
            "response": "Bonjour ! Je suis heureux de vous aider.",
            "emotional_analysis": {
                "detected_emotion": "joy",
                "emotion_intensity": 0.8,
                "emotional_state": {"joy": 0.8, "confidence": 0.9},
                "confidence": 0.9
            },
            "personality_insights": {
                "profile": {"openness": 0.7},
                "dominant_traits": ["ouvert d'esprit"]
            },
            "interaction_metadata": {
                "importance": 0.7,
                "response_type": "conversational",
                "memories_used": 2,
                "conversation_depth": 5
            },
            "system_info": {
                "processing_timestamp": "2024-01-01T10:00:00",
                "total_interactions": 100
            }
        })
        
        mock.generate_suggestions = AsyncMock(return_value=[
            {"text": "Comment allez-vous ?", "type": "general", "confidence": 0.8},
            {"text": "Parlons de vos émotions", "type": "emotional", "confidence": 0.7}
        ])
        
        mock.get_emotional_insights = AsyncMock(return_value={
            "period_analyzed": "30 derniers jours",
            "total_interactions": 50,
            "emotional_timeline": {},
            "trends": {
                "most_frequent_emotion": "joy",
                "emotional_stability": 0.8,
                "positive_ratio": 0.7
            },
            "recommendations": ["Continuez ainsi !"]
        })
        
        mock.get_personality_profile = AsyncMock(return_value={
            "user_id": "test_user",
            "profile": {
                "big_five": {
                    "openness": 0.8,
                    "conscientiousness": 0.7,
                    "extraversion": 0.6,
                    "agreeableness": 0.9,
                    "neuroticism": 0.3
                }
            },
            "dominant_traits": ["ouvert d'esprit", "bienveillant"],
            "insights": ["Vous êtes très ouvert aux nouvelles expériences."],
            "recommendations": ["Continuez à cultiver vos forces."]
        })
        
        mock.get_mood_history = AsyncMock(return_value=[
            {
                "timestamp": "2024-01-01T10:00:00",
                "emotion": "joy",
                "intensity": 0.8,
                "valence": 0.8,
                "arousal": 0.7,
                "confidence": 0.9
            }
        ])
        
        mock.get_conversation_insights = AsyncMock(return_value={
            "user_id": "test_user",
            "conversation_stats": {
                "total_exchanges": 10,
                "conversation_depth": 5,
                "average_message_length": 15.5
            },
            "topics_discussed": [
                {"topic": "travail", "relevance": 0.8}
            ],
            "emotional_journey": [],
            "recommendations": ["Continuez la conversation"]
        })
        
        yield mock


class TestHealthEndpoint:
    """Tests pour l'endpoint de santé"""
    
    def test_health_check(self, client):
        """Test de l'endpoint /health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "3.0"
        assert "timestamp" in data


class TestLanguagesEndpoint:
    """Tests pour l'endpoint des langues"""
    
    def test_get_languages(self, client):
        """Test de l'endpoint /langues"""
        response = client.get("/langues")
        assert response.status_code == 200
        data = response.json()
        assert "fr" in data
        assert "en" in data
        assert "es" in data
        assert data["fr"] == "Français"


class TestUserPreferencesEndpoints:
    """Tests pour les endpoints de préférences utilisateur"""
    
    @patch('src.core.api.get_db')
    def test_update_preferences(self, mock_get_db, client):
        """Test de mise à jour des préférences"""
        # Mock de la session DB
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock de la requête DB
        mock_db.query().filter().first.return_value = None
        
        response = client.post(
            "/utilisateur/preferences?user_id=test_user",
            json={
                "language": "en",
                "theme": "dark",
                "notification_settings": {"email": False}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert "preferences" in data
    
    @patch('src.core.api.get_db')
    def test_get_preferences(self, mock_get_db, client):
        """Test de récupération des préférences"""
        # Mock de la session DB
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock des préférences existantes
        mock_prefs = MagicMock()
        mock_prefs.user_id = "test_user"
        mock_prefs.language = "fr"
        mock_prefs.theme = "light"
        mock_prefs.notification_settings = {"email": True}
        mock_prefs.ai_settings = {"personality_adaptation": True}
        
        mock_db.query().filter().first.return_value = mock_prefs
        
        response = client.get("/utilisateur/preferences/test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["language"] == "fr"
        assert data["theme"] == "light"


class TestChatEndpoint:
    """Tests pour l'endpoint de chat"""
    
    def test_chat_success(self, client, mock_emoia):
        """Test d'un chat réussi"""
        response = client.post("/chat", json={
            "user_id": "test_user",
            "message": "Bonjour !",
            "preferences": {"language": "fr"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "emotional_analysis" in data
        assert "personality_insights" in data
        assert data["emotional_analysis"]["detected_emotion"] == "joy"
    
    def test_chat_missing_fields(self, client):
        """Test avec des champs manquants"""
        response = client.post("/chat", json={
            "user_id": "test_user"
            # message manquant
        })
        
        assert response.status_code == 422  # Validation error


class TestAnalyticsEndpoints:
    """Tests pour les endpoints analytics"""
    
    def test_get_analytics(self, client, mock_emoia):
        """Test de l'endpoint analytics"""
        response = client.get("/analytics/test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert "trends" in data
        assert data["trends"]["most_frequent_emotion"] == "joy"
    
    def test_get_personality(self, client, mock_emoia):
        """Test du profil de personnalité"""
        response = client.get("/personality/test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        assert "big_five" in data["profile"]
        assert data["profile"]["big_five"]["openness"] == 0.8


class TestSuggestionsEndpoint:
    """Tests pour l'endpoint de suggestions"""
    
    def test_get_suggestions(self, client, mock_emoia):
        """Test de génération de suggestions"""
        response = client.post("/suggestions", json={
            "context": "Je me sens stressé",
            "emotional_state": {"dominant_emotion": "anxiety"},
            "max_suggestions": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0
        assert "text" in data["suggestions"][0]
        assert "confidence" in data["suggestions"][0]


class TestInsightsEndpoint:
    """Tests pour l'endpoint insights"""
    
    def test_get_conversation_insights(self, client, mock_emoia):
        """Test des insights de conversation"""
        response = client.get("/insights/test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert "conversation_stats" in data
        assert "topics_discussed" in data
        assert data["conversation_stats"]["total_exchanges"] == 10


class TestMoodHistoryEndpoint:
    """Tests pour l'historique d'humeur"""
    
    def test_get_mood_history_default(self, client, mock_emoia):
        """Test de l'historique d'humeur par défaut"""
        response = client.get("/mood/history/test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]) > 0
        assert "emotion" in data["history"][0]
        assert "valence" in data["history"][0]
    
    def test_get_mood_history_with_period(self, client, mock_emoia):
        """Test avec période spécifique"""
        response = client.get("/mood/history/test_user?period=month")
        
        assert response.status_code == 200
        data = response.json()
        assert "history" in data


class TestWebSocketChat:
    """Tests pour le WebSocket de chat"""
    
    def test_websocket_connection(self, client):
        """Test de connexion WebSocket"""
        with client.websocket_connect("/ws/chat") as websocket:
            # Identification
            websocket.send_json({
                "type": "identify",
                "user_id": "test_user"
            })
            
            data = websocket.receive_json()
            assert data["type"] == "identified"
            assert data["user_id"] == "test_user"
    
    @patch('src.core.api.emoia')
    def test_websocket_chat_message(self, mock_emoia_instance, client):
        """Test d'envoi de message via WebSocket"""
        # Configuration du mock
        mock_emoia_instance.process_message = AsyncMock(return_value={
            "response": "Test response",
            "emotional_analysis": {
                "detected_emotion": "joy",
                "emotion_intensity": 0.8,
                "confidence": 0.9
            },
            "personality_insights": {},
            "interaction_metadata": {},
            "system_info": {}
        })
        
        mock_emoia_instance.get_current_emotions = AsyncMock(return_value={
            "joy": 0.8
        })
        
        with client.websocket_connect("/ws/chat") as websocket:
            # Envoyer un message
            websocket.send_json({
                "type": "chat_message",
                "user_id": "test_user",
                "message": "Test message"
            })
            
            # Recevoir la réponse
            response = websocket.receive_json()
            assert response["type"] == "chat_response"
            assert "response" in response
            
            # Recevoir la mise à jour émotionnelle
            emotion_update = websocket.receive_json()
            assert emotion_update["type"] == "emotional_update"


class TestErrorHandling:
    """Tests de gestion d'erreur"""
    
    def test_invalid_endpoint(self, client):
        """Test d'un endpoint invalide"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """Test avec JSON invalide"""
        response = client.post(
            "/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @patch('src.core.api.emoia.process_message')
    async def test_internal_server_error(self, mock_process, client):
        """Test d'erreur serveur interne"""
        mock_process.side_effect = Exception("Test error")
        
        response = client.post("/chat", json={
            "user_id": "test_user",
            "message": "Test"
        })
        
        # L'API devrait gérer l'erreur gracieusement
        assert response.status_code == 200  # Réponse de fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])