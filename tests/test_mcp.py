"""
Tests pour le système MCP (Model Context Protocol)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.mcp import MCPManager, MCPClient, MCPProvider
from src.mcp.mcp_manager import MCPContext, MCPMessage

@pytest.fixture
async def mcp_manager():
    """Fixture pour le gestionnaire MCP"""
    manager = MCPManager()
    # Ne pas initialiser les providers par défaut pour les tests
    manager._initialized = True
    return manager

@pytest.fixture
async def mock_provider():
    """Fixture pour un provider mock"""
    provider = Mock(spec=MCPProvider)
    provider.name = "test_provider"
    provider.capabilities = ["chat", "text-generation"]
    provider.default_model = "test-model"
    provider.initialize = AsyncMock()
    provider.send_completion = AsyncMock(return_value={
        "content": "Test response",
        "metadata": {"test": True}
    })
    provider.list_models = AsyncMock(return_value=["test-model-1", "test-model-2"])
    provider.is_active = Mock(return_value=True)
    return provider

class TestMCPManager:
    """Tests pour MCPManager"""
    
    @pytest.mark.asyncio
    async def test_register_provider(self, mcp_manager, mock_provider):
        """Test l'enregistrement d'un provider"""
        await mcp_manager.register_provider("test", mock_provider)
        
        assert "test" in mcp_manager.providers
        assert mcp_manager.providers["test"] == mock_provider
        mock_provider.initialize.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_create_context(self, mcp_manager, mock_provider):
        """Test la création d'un contexte MCP"""
        await mcp_manager.register_provider("test", mock_provider)
        
        context = await mcp_manager.create_context(
            user_id="user123",
            provider="test",
            model="test-model-1",
            temperature=0.8
        )
        
        assert isinstance(context, MCPContext)
        assert context.provider == "test"
        assert context.model_id == "test-model-1"
        assert context.temperature == 0.8
        assert context.metadata["user_id"] == "user123"
        
    @pytest.mark.asyncio
    async def test_send_message(self, mcp_manager, mock_provider):
        """Test l'envoi d'un message via MCP"""
        await mcp_manager.register_provider("test", mock_provider)
        
        context = await mcp_manager.create_context("user123", "test")
        messages = [
            MCPMessage(role="user", content="Hello, test!")
        ]
        
        response = await mcp_manager.send_message(context, messages)
        
        assert isinstance(response, MCPMessage)
        assert response.role == "assistant"
        assert response.content == "Test response"
        assert response.metadata["test"] is True
        
        mock_provider.send_completion.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_list_models(self, mcp_manager, mock_provider):
        """Test la liste des modèles disponibles"""
        await mcp_manager.register_provider("test", mock_provider)
        
        models = await mcp_manager.list_models()
        
        assert "test" in models
        assert models["test"] == ["test-model-1", "test-model-2"]
        
    @pytest.mark.asyncio
    async def test_list_models_specific_provider(self, mcp_manager, mock_provider):
        """Test la liste des modèles pour un provider spécifique"""
        await mcp_manager.register_provider("test", mock_provider)
        
        models = await mcp_manager.list_models("test")
        
        assert "test" in models
        assert len(models) == 1
        
    @pytest.mark.asyncio
    async def test_get_provider_info(self, mcp_manager, mock_provider):
        """Test l'obtention des informations d'un provider"""
        await mcp_manager.register_provider("test", mock_provider)
        
        info = await mcp_manager.get_provider_info("test")
        
        assert info["name"] == "test"
        assert info["capabilities"] == ["chat", "text-generation"]
        assert info["default_model"] == "test-model"
        assert info["status"] == "active"
        
    @pytest.mark.asyncio
    async def test_cleanup(self, mcp_manager, mock_provider):
        """Test le nettoyage des ressources"""
        mock_provider.cleanup = AsyncMock()
        await mcp_manager.register_provider("test", mock_provider)
        
        await mcp_manager.cleanup()
        
        assert len(mcp_manager.providers) == 0
        assert len(mcp_manager.active_contexts) == 0
        assert mcp_manager._initialized is False
        mock_provider.cleanup.assert_called_once()

class TestMCPClient:
    """Tests pour MCPClient"""
    
    @pytest.fixture
    async def mcp_client(self, mcp_manager):
        """Fixture pour le client MCP"""
        client = MCPClient(mcp_manager)
        return client
        
    @pytest.mark.asyncio
    async def test_chat_simple(self, mcp_client, mcp_manager, mock_provider):
        """Test une conversation simple"""
        await mcp_manager.register_provider("test", mock_provider)
        
        response = await mcp_client.chat(
            user_id="user123",
            message="Hello!",
            provider="test"
        )
        
        assert response == "Test response"
        
    @pytest.mark.asyncio
    async def test_chat_with_history(self, mcp_client, mcp_manager, mock_provider):
        """Test une conversation avec historique"""
        await mcp_manager.register_provider("test", mock_provider)
        
        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        response = await mcp_client.chat_with_history(
            user_id="user123",
            message="New message",
            history=history
        )
        
        assert response == "Test response"
        
    @pytest.mark.asyncio
    async def test_complete(self, mcp_client, mcp_manager, mock_provider):
        """Test la complétion simple"""
        await mcp_manager.register_provider("test", mock_provider)
        mcp_manager.default_provider = "test"
        
        response = await mcp_client.complete(
            prompt="Complete this:",
            provider="test"
        )
        
        assert response == "Test response"
        
    @pytest.mark.asyncio
    async def test_list_available_models(self, mcp_client, mcp_manager, mock_provider):
        """Test la liste des modèles disponibles"""
        await mcp_manager.register_provider("test", mock_provider)
        
        models = await mcp_client.list_available_models()
        
        assert "test" in models
        assert models["test"] == ["test-model-1", "test-model-2"]
        
    @pytest.mark.asyncio
    async def test_switch_model(self, mcp_client, mcp_manager, mock_provider):
        """Test le changement de modèle"""
        await mcp_manager.register_provider("test", mock_provider)
        
        success = await mcp_client.switch_model(
            user_id="user123",
            provider="test",
            model="test-model-2"
        )
        
        assert success is True
        
    @pytest.mark.asyncio
    async def test_get_model_info(self, mcp_client, mcp_manager, mock_provider):
        """Test l'obtention d'informations sur un modèle"""
        await mcp_manager.register_provider("test", mock_provider)
        
        info = await mcp_client.get_model_info("test", "test-model-1")
        
        assert info["provider"] == "test"
        assert info["model"] == "test-model-1"
        assert info["available"] is True
        assert info["capabilities"] == ["chat", "text-generation"]
        
    def test_clear_context(self, mcp_client):
        """Test l'effacement du contexte"""
        # Ajouter des contextes de test
        mcp_client._contexts["user123_test"] = Mock()
        mcp_client._contexts["user123_other"] = Mock()
        mcp_client._contexts["user456_test"] = Mock()
        
        # Effacer un contexte spécifique
        mcp_client.clear_context("user123", "test")
        assert "user123_test" not in mcp_client._contexts
        assert "user123_other" in mcp_client._contexts
        
        # Effacer tous les contextes d'un utilisateur
        mcp_client.clear_context("user123")
        assert "user123_other" not in mcp_client._contexts
        assert "user456_test" in mcp_client._contexts

class TestMCPIntegration:
    """Tests d'intégration MCP"""
    
    @pytest.mark.asyncio
    @patch('src.mcp.providers.ollama_provider.OllamaProvider')
    async def test_ollama_integration(self, mock_ollama_class):
        """Test l'intégration avec Ollama"""
        # Créer un mock pour le provider Ollama
        mock_ollama = Mock()
        mock_ollama.initialize = AsyncMock()
        mock_ollama.capabilities = ["chat", "embeddings"]
        mock_ollama.default_model = "llama2"
        mock_ollama_class.return_value = mock_ollama
        
        # Initialiser le manager
        manager = MCPManager()
        manager._initialized = False
        
        # Le provider devrait être chargé automatiquement
        with patch.object(manager, '_load_default_providers') as mock_load:
            await manager.initialize()
            mock_load.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_manager):
        """Test la gestion des erreurs"""
        # Test avec un provider inexistant
        with pytest.raises(ValueError):
            await mcp_manager.create_context("user123", "nonexistent")
            
        # Test send_message sans provider
        context = MCPContext(
            model_id="test",
            provider="nonexistent",
            capabilities=[],
            max_tokens=100,
            temperature=0.7,
            metadata={}
        )
        
        with pytest.raises(ValueError):
            await mcp_manager.send_message(context, [])