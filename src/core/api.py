from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uvicorn

from ..config import Config
from .emoia_main import EmoIA
from src.models.user_preferences import UserPreferences  # Nouvelle importation

# Modèle de requête pour le chat
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Modèle de réponse pour le chat
class ChatResponse(BaseModel):
    response: str
    emotional_analysis: dict
    personality_insights: dict
    interaction_metadata: dict
    system_info: dict

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
    return {"status": "ok"}

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
    
    db.commit()
    db.refresh(db_prefs)
    
    return {"status": "updated", "preferences": {
        "user_id": db_prefs.user_id,
        "language": db_prefs.language,
        "theme": db_prefs.theme,
        "notification_settings": db_prefs.notification_settings
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
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": prefs.user_id,
        "language": prefs.language,
        "theme": prefs.theme,
        "notification_settings": prefs.notification_settings
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
    db.close()
    
    result = await emoia.process_message(
        user_input=req.message,
        user_id=req.user_id,
        context_data={"language": language}  # Passer la langue au traitement
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

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            user_id = data.get("user_id", "ws-user")
            message = data.get("message", "")
            
            # Récupérer la langue des préférences
            # Récupérer la langue depuis la base de données
            db = next(get_db())
            prefs = db.query(UserPreferencesDB).filter(UserPreferencesDB.user_id == user_id).first()
            if prefs:
                language = prefs.language
            db.close()
            
            result = await emoia.process_message(
                user_input=message,
                user_id=user_id,
                context_data={"language": language}
            )
            
            await ws.send_json({
                "response": result["response"],
                "emotional_analysis": result["emotional_analysis"],
                "personality_insights": result["personality_insights"],
                "interaction_metadata": result["interaction_metadata"],
                "system_info": result["system_info"]
            })
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("src.core.api:app", host="0.0.0.0", port=8000, reload=True)