"""
Module WebSocket pour les analytics en temps réel
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from typing import Dict, Set
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Gestionnaire de connexions WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"WebSocket connecté pour l'utilisateur {user_id}")
        
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"WebSocket déconnecté pour l'utilisateur {user_id}")
        
    async def send_analytics(self, user_id: str, data: dict):
        """Envoie des données analytics à un utilisateur spécifique"""
        if user_id in self.active_connections:
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Erreur envoi WebSocket: {e}")
                    
    async def broadcast_to_all(self, data: dict):
        """Diffuse des données à tous les utilisateurs connectés"""
        for user_id, connections in self.active_connections.items():
            for websocket in connections:
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Erreur broadcast WebSocket: {e}")

# Instance globale du gestionnaire
manager = ConnectionManager()

@router.websocket("/ws/analytics/{user_id}")
async def analytics_websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    Endpoint WebSocket pour les analytics en temps réel
    
    Args:
        websocket: Connexion WebSocket
        user_id: ID de l'utilisateur
    """
    await manager.connect(websocket, user_id)
    
    try:
        # Envoyer les données initiales
        initial_data = {
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        }
        await websocket.send_json(initial_data)
        
        # Boucle de réception des messages
        while True:
            data = await websocket.receive_json()
            
            # Traiter les différents types de messages
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            
            elif data.get("type") == "request_analytics":
                # Envoyer les analytics demandées
                analytics_data = {
                    "type": "analytics_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "emotions": {"joy": 0.7, "sadness": 0.1, "neutral": 0.2},
                        "interaction_count": 42,
                        "sentiment_trend": "positive"
                    }
                }
                await websocket.send_json(analytics_data)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"Client {user_id} déconnecté")
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
        manager.disconnect(websocket, user_id)

# Fonction utilitaire pour envoyer des mises à jour analytics
async def send_analytics_update(user_id: str, analytics_data: dict):
    """
    Envoie une mise à jour analytics à un utilisateur via WebSocket
    
    Args:
        user_id: ID de l'utilisateur
        analytics_data: Données analytics à envoyer
    """
    data = {
        "type": "analytics_update",
        "timestamp": datetime.now().isoformat(),
        "data": analytics_data
    }
    await manager.send_analytics(user_id, data)