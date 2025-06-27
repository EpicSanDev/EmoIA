"""
Module MCP (Model Context Protocol) pour EmoIA
Permet d'intégrer différents modèles et protocoles de contexte
"""

from .mcp_manager import MCPManager
from .mcp_provider import MCPProvider
from .mcp_client import MCPClient

__all__ = ['MCPManager', 'MCPProvider', 'MCPClient']