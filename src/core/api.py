from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import asyncio
import uvicorn
from typing import List, Optional
from datetime import datetime, timedelta

from ..config import Config
from .emoia_main import EmoIA
from src.models.user_preferences import UserPreferences  # Nouvelle importation

# Modèles de requête/réponse
class ChatRequest(BaseModel):
    user_id: str
    message: str
    preferences: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    emotional_analysis: dict
    personality_insights: dict
    interaction_metadata: dict
    system_info: dict

class SuggestionsRequest(BaseModel):
    context: str
    user_input: Optional[str] = None
    emotional_state: Optional[dict] = None
    max_suggestions: int = 5

class InsightRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None

# Initialisation
config = Config()
emoia = EmoIA(config)

app = FastAPI(
    title="EmoIA API",
    version="3.0",
    description="API principale pour le système EmoIA",
    openapi_tags=[
        {
            "name": "Utilisateur",
            "description": "Gestion des préférences utilisateur"
        },
        {
            "name": "Chat",
            "description": "Interactions conversationnelles"
        },
        {
            "name": "Analytics",
            "description": "Données analytiques en temps réel"
        },
        {
            "name": "Intelligence",
            "description": "Endpoints d'intelligence artificielle"
        },
        {
            "name": "Système",
            "description": "Endpoints système"
        }
    ]
)

# CORS pour le frontend (à adapter en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation de la base de données
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.user_preferences import Base, UserPreferencesDB

engine = create_engine("sqlite:///emoia_memory.db")
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency pour les sessions DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    await emoia.initialize()

# Importer le routeur WebSocket pour les analytics
from src.analytics.websocket import router as analytics_router
app.include_router(analytics_router)

@app.get("/health", tags=["Système"])
async def health():
    """Vérifie l'état de santé de l'API"""
    return {"status": "ok", "version": "3.0", "timestamp": datetime.now().isoformat()}

@app.get("/langues", tags=["Utilisateur"])
async def get_languages():
    """Retourne les langues supportées
    
    Returns:
        dict: Dictionnaire des langues avec leur code et nom
    """
    return {
        "fr": "Français",
        "en": "English",
        "es": "Español"
    }

@app.post("/utilisateur/preferences", tags=["Utilisateur"])
async def update_preferences(prefs: UserPreferences.PreferencesUpdate, user_id: str, db: Session = Depends(get_db)):
    """Met à jour les préférences utilisateur
    
    Args:
        prefs (PreferencesUpdate): Objet contenant les préférences à mettre à jour
        user_id (str): ID de l'utilisateur
    
    Returns:
        dict: Statut de mise à jour et préférences actuelles
    """
    # Récupérer ou créer les préférences
    db_prefs = db.query(UserPreferencesDB).filter(UserPreferencesDB.user_id == user_id).first()
    if not db_prefs:
        db_prefs = UserPreferencesDB(user_id=user_id)
        db.add(db_prefs)
    
    # Mettre à jour les champs
    if prefs.language:
        db_prefs.language = prefs.language
    if prefs.theme:
        db_prefs.theme = prefs.theme
    if prefs.notification_settings:
        db_prefs.notification_settings = prefs.notification_settings
    if prefs.ai_settings:
        db_prefs.ai_settings = prefs.ai_settings
    
    db.commit()
    db.refresh(db_prefs)
    
    return {"status": "updated", "preferences": {
        "user_id": db_prefs.user_id,
        "language": db_prefs.language,
        "theme": db_prefs.theme,
        "notification_settings": db_prefs.notification_settings,
        "ai_settings": db_prefs.ai_settings
    }}

@app.get("/utilisateur/preferences/{user_id}", tags=["Utilisateur"])
async def get_preferences(user_id: str, db: Session = Depends(get_db)):
    """Récupère les préférences utilisateur
    
    Args:
        user_id (str): ID de l'utilisateur
    
    Returns:
        dict: Préférences de l'utilisateur
    """
    prefs = db.query(UserPreferencesDB).filter(UserPreferencesDB.user_id == user_id).first()
    if not prefs:
        # Créer des préférences par défaut
        prefs = UserPreferencesDB(user_id=user_id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    
    return {
        "user_id": prefs.user_id,
        "language": prefs.language,
        "theme": prefs.theme,
        "notification_settings": prefs.notification_settings,
        "ai_settings": prefs.ai_settings
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(req: ChatRequest):
    """Endpoint pour les interactions conversationnelles
    
    Args:
        req (ChatRequest): Requête contenant le message de l'utilisateur
    
    Returns:
        ChatResponse: Réponse avec analyse émotionnelle et insights
    """
    # Récupérer la langue des préférences si disponible
    language = "fr"  # Par défaut
    # Récupérer la langue depuis la base de données
    db = next(get_db())
    prefs = db.query(UserPreferencesDB).filter(UserPreferencesDB.user_id == req.user_id).first()
    if prefs:
        language = prefs.language
        ai_settings = prefs.ai_settings
    else:
        ai_settings = None
    db.close()
    
    result = await emoia.process_message(
        user_input=req.message,
        user_id=req.user_id,
        context_data={
            "language": language,
            "ai_settings": ai_settings or req.preferences
        }
    )
    return result

@app.get("/analytics/{user_id}", tags=["Analytics"])
async def analytics_endpoint(user_id: str):
    """Récupère les insights émotionnels pour un utilisateur
    
    Args:
        user_id (str): ID de l'utilisateur
    
    Returns:
        dict: Insights émotionnels et de personnalité
    """
    insights = await emoia.get_emotional_insights(user_id)
    return insights

@app.get("/personality/{user_id}", tags=["Intelligence"])
async def get_personality_profile(user_id: str):
    """Récupère le profil de personnalité de l'utilisateur
    
    Args:
        user_id (str): ID de l'utilisateur
    
    Returns:
        dict: Profil de personnalité détaillé
    """
    profile = await emoia.get_personality_profile(user_id)
    return profile

@app.post("/suggestions", tags=["Intelligence"])
async def get_suggestions(req: SuggestionsRequest):
    """Génère des suggestions intelligentes basées sur le contexte
    
    Args:
        req (SuggestionsRequest): Contexte et paramètres
    
    Returns:
        dict: Liste de suggestions avec confiance
    """
    suggestions = await emoia.generate_suggestions(
        context=req.context,
        user_input=req.user_input,
        emotional_state=req.emotional_state,
        max_suggestions=req.max_suggestions
    )
    return {"suggestions": suggestions}

@app.get("/insights/{user_id}", tags=["Intelligence"])
async def get_conversation_insights(user_id: str, conversation_id: Optional[str] = None):
    """Récupère les insights de conversation
    
    Args:
        user_id (str): ID de l'utilisateur
        conversation_id (str, optional): ID de conversation spécifique
    
    Returns:
        dict: Insights détaillés de la conversation
    """
    insights = await emoia.get_conversation_insights(
        user_id=user_id,
        conversation_id=conversation_id
    )
    return insights

@app.get("/mood/history/{user_id}", tags=["Analytics"])
async def get_mood_history(user_id: str, period: str = "week"):
    """Récupère l'historique d'humeur
    
    Args:
        user_id (str): ID de l'utilisateur
        period (str): Période (day, week, month)
    
    Returns:
        dict: Historique des points d'humeur
    """
    history = await emoia.get_mood_history(user_id, period)
    return {"history": history}

@app.post("/voice/transcribe", tags=["Intelligence"])
async def transcribe_audio(audio_file: bytes):
    """Transcrit un fichier audio en texte
    
    Args:
        audio_file (bytes): Fichier audio
    
    Returns:
        dict: Transcription et métadonnées
    """
    # Implémenter la transcription audio
    # Pour l'instant, retourner un placeholder
    return {
        "transcript": "Transcription audio non implémentée",
        "confidence": 0.0,
        "language": "fr"
    }

# WebSocket endpoints
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    user_id = None
    
    try:
        while True:
            data = await ws.receive_json()
            
            if data.get("type") == "identify":
                user_id = data.get("user_id")
                await ws.send_json({"type": "identified", "user_id": user_id})
                continue
            
            if data.get("type") == "chat_message":
                user_id = data.get("user_id", user_id)
                message = data.get("message", "")
                context = data.get("context", {})
                
                # Récupérer les préférences
                db = next(get_db())
                prefs = db.query(UserPreferencesDB).filter(UserPreferencesDB.user_id == user_id).first()
                if prefs:
                    language = prefs.language
                    ai_settings = prefs.ai_settings
                else:
                    language = context.get("language", "fr")
                    ai_settings = context.get("ai_settings")
                db.close()
                
                # Traiter le message
                result = await emoia.process_message(
                    user_input=message,
                    user_id=user_id,
                    context_data={
                        "language": language,
                        "ai_settings": ai_settings
                    }
                )
                
                # Envoyer la réponse
                await ws.send_json({
                    "type": "chat_response",
                    "response": result["response"],
                    "emotional_analysis": result["emotional_analysis"],
                    "personality_insights": result["personality_insights"],
                    "interaction_metadata": result["interaction_metadata"],
                    "system_info": result["system_info"],
                    "confidence": result.get("confidence", 0.9)
                })
                
                # Envoyer des mises à jour émotionnelles
                await ws.send_json({
                    "type": "emotional_update",
                    "current_emotions": await emoia.get_current_emotions(user_id),
                    "mood_point": {
                        "timestamp": datetime.now().isoformat(),
                        "valence": result["emotional_analysis"].get("valence", 0),
                        "arousal": result["emotional_analysis"].get("arousal", 0.5),
                        "dominantEmotion": result["emotional_analysis"].get("dominant_emotion", "neutral"),
                        "emotionIntensity": result["emotional_analysis"].get("confidence", 0.5)
                    }
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket déconnecté pour l'utilisateur {user_id}")
    except Exception as e:
        print(f"Erreur WebSocket: {e}")
        await ws.send_json({"type": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run("src.core.api:app", host="0.0.0.0", port=8000, reload=True)