from fastapi import WebSocket, APIRouter
from fastapi.websockets import WebSocketDisconnect
import asyncio
import json
import random
from datetime import datetime
from src.config import Config

router = APIRouter()

# Données simulées pour les analytics
analytics_data = {
    "active_users": 0,
    "emotions": {"joy": 0, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0},
    "avg_response_time": 0.5
}

# Stockage des connexions WebSocket
active_connections = []

@router.websocket("/ws/analytics")
async def websocket_analytics(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    analytics_data["active_users"] = len(active_connections)
    
    try:
        while True:
            # Mettre à jour les données analytiques (simulation)
            analytics_data["emotions"] = {
                "joy": random.randint(0, 100),
                "sadness": random.randint(0, 50),
                "anger": random.randint(0, 30),
                "fear": random.randint(0, 20),
                "surprise": random.randint(0, 10)
            }
            analytics_data["avg_response_time"] = round(random.uniform(0.1, 1.0), 2)
            analytics_data["timestamp"] = datetime.utcnow().isoformat()
            
            # Envoyer à tous les connexions actives
            for connection in active_connections:
                await connection.send_json(analytics_data)
            
            await asyncio.sleep(1)  # Envoyer des mises à jour chaque seconde
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        analytics_data["active_users"] = len(active_connections)