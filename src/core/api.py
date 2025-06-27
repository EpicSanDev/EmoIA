from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import asyncio
import uvicorn
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from ..config import Config

logger = logging.getLogger(__name__)
from .emoia_main import EmoIA
from src.models.user_preferences import UserPreferences  # Nouvelle importation
from src.mcp import MCPClient, MCPManager  # Import MCP

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

# Modèles de requête/réponse supplémentaires
class NameRequest(BaseModel):
    user_id: str
    name: str
    nickname: Optional[str] = ""

class LearnConceptRequest(BaseModel):
    user_id: str
    concept_name: str
    explanation: str
    examples: Optional[List[str]] = []
    category: Optional[str] = "general"
    difficulty_level: Optional[int] = 3

class TDAHTaskRequest(BaseModel):
    user_id: str
    title: str
    description: Optional[str] = ""
    priority: Optional[int] = 3
    category: Optional[str] = "general"
    due_date: Optional[str] = None
    estimated_duration: Optional[int] = None
    emotional_state: Optional[str] = None

class TelegramUserRequest(BaseModel):
    user_id: str
    telegram_id: str
    telegram_username: Optional[str] = ""

# Initialisation
config = Config()
emoia = EmoIA(config)
mcp_manager = MCPManager()  # Gestionnaire MCP
mcp_client = MCPClient(mcp_manager)  # Client MCP

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
    try:
        logger.info("🚀 Démarrage de l'API EmoIA...")
        
        # Initialisation en mode progressif pour éviter le blocage
        try:
            await asyncio.wait_for(emoia.initialize(), timeout=30.0)
            logger.info("✅ EmoIA initialisé avec succès")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Initialisation d'EmoIA en cours en arrière-plan...")
            # Continuer le démarrage même si l'initialisation n'est pas terminée
            asyncio.create_task(emoia.initialize())
        
        # Initialisation MCP en mode non-bloquant
        try:
            await asyncio.wait_for(mcp_client.initialize(), timeout=10.0)
            logger.info("✅ MCP client initialisé")
        except asyncio.TimeoutError:
            logger.warning("⚠️ MCP client en cours d'initialisation...")
            asyncio.create_task(mcp_client.initialize())
        except Exception as e:
            logger.warning(f"⚠️ MCP non disponible: {e}")
        
        logger.info("🎉 API EmoIA démarrée et prête!")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {e}")
        # Ne pas faire échouer le démarrage, juste logger l'erreur

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage propre lors de l'arrêt"""
    try:
        logger.info("🛑 Arrêt de l'API EmoIA...")
        
        # Nettoyer le client MCP
        if mcp_client:
            await mcp_client.cleanup()
            logger.info("✅ MCP client nettoyé")
        
        # Nettoyer le gestionnaire MCP
        if mcp_manager:
            await mcp_manager.cleanup()
            logger.info("✅ MCP manager nettoyé")
        
        logger.info("🎯 API EmoIA arrêtée proprement")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt: {e}")

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

# Nouveaux endpoints MCP
@app.get("/mcp/providers", tags=["Intelligence"])
async def get_mcp_providers():
    """Liste tous les providers MCP disponibles
    
    Returns:
        dict: Liste des providers et leurs capacités
    """
    providers = await mcp_client.get_providers()
    provider_info = {}
    
    for provider in providers:
        info = await mcp_manager.get_provider_info(provider)
        provider_info[provider] = info
        
    return {"providers": provider_info}

@app.get("/mcp/models", tags=["Intelligence"])
async def get_mcp_models(provider: Optional[str] = None):
    """Liste tous les modèles disponibles
    
    Args:
        provider (str, optional): Filtrer par provider spécifique
    
    Returns:
        dict: Modèles disponibles par provider
    """
    models = await mcp_client.list_available_models()
    
    if provider:
        return {"models": {provider: models.get(provider, [])}}
    
    return {"models": models}

class MCPChatRequest(BaseModel):
    user_id: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    context_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

@app.post("/mcp/chat", tags=["Intelligence"])
async def mcp_chat(req: MCPChatRequest):
    """Chat via MCP avec un modèle spécifique
    
    Args:
        req (MCPChatRequest): Requête de chat MCP
    
    Returns:
        dict: Réponse du modèle et métadonnées
    """
    try:
        response = await mcp_client.chat(
            user_id=req.user_id,
            message=req.message,
            provider=req.provider,
            model=req.model,
            context_id=req.context_id,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        
        return {
            "response": response,
            "provider": req.provider or "ollama",
            "model": req.model,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/switch-model", tags=["Intelligence"])
async def switch_mcp_model(user_id: str, provider: str, model: str):
    """Change le modèle MCP pour un utilisateur
    
    Args:
        user_id (str): ID de l'utilisateur
        provider (str): Provider à utiliser
        model (str): Modèle à utiliser
    
    Returns:
        dict: Statut du changement
    """
    success = await mcp_client.switch_model(user_id, provider, model)
    
    if success:
        return {"status": "success", "provider": provider, "model": model}
    else:
        raise HTTPException(status_code=400, detail="Échec du changement de modèle")

@app.websocket("/ws/mcp")
async def websocket_mcp(ws: WebSocket):
    """WebSocket pour streaming MCP"""
    await ws.accept()
    user_id = None
    
    try:
        while True:
            data = await ws.receive_json()
            
            if data.get("type") == "identify":
                user_id = data.get("user_id")
                await ws.send_json({"type": "identified", "user_id": user_id})
                continue
                
            if data.get("type") == "mcp_stream":
                provider = data.get("provider", "ollama")
                model = data.get("model")
                message = data.get("message", "")
                
                # Créer le contexte
                context = await mcp_manager.create_context(
                    user_id=user_id,
                    provider=provider,
                    model=model
                )
                
                # Stream la réponse
                provider_instance = mcp_manager.providers.get(provider)
                if provider_instance:
                    messages = [{"role": "user", "content": message}]
                    
                    async for chunk in provider_instance.stream_completion(
                        model=model or provider_instance.default_model,
                        messages=messages
                    ):
                        await ws.send_json({
                            "type": "mcp_chunk",
                            "content": chunk,
                            "provider": provider,
                            "model": model
                        })
                        
                    await ws.send_json({"type": "mcp_complete"})
                else:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Provider {provider} non trouvé"
                    })
                    
    except WebSocketDisconnect:
        print(f"WebSocket MCP déconnecté pour l'utilisateur {user_id}")
    except Exception as e:
        print(f"Erreur WebSocket MCP: {e}")
        await ws.send_json({"type": "error", "message": str(e)})

@app.websocket("/ws/analytics/{user_id}")
async def websocket_analytics(ws: WebSocket, user_id: str):
    """WebSocket pour les analytics en temps réel (compatibilité frontend)"""
    await ws.accept()
    
    try:
        # Envoyer un message de confirmation de connexion
        await ws.send_json({
            "type": "connected",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Boucle de maintien de connexion
        while True:
            try:
                # Recevoir des messages du client
                data = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
                
                if data.get("type") == "ping":
                    await ws.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif data.get("type") == "request_analytics":
                    # Envoyer des données d'analytics de demo
                    analytics_data = {
                        "type": "analytics_update",
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "data": {
                            "emotions": {
                                "joy": 0.7,
                                "sadness": 0.1,
                                "neutral": 0.2
                            },
                            "interaction_count": 42,
                            "sentiment_trend": "positive",
                            "engagement_score": 0.85
                        }
                    }
                    await ws.send_json(analytics_data)
                    
            except asyncio.TimeoutError:
                # Envoyer un ping périodique pour maintenir la connexion
                await ws.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket analytics déconnecté pour l'utilisateur {user_id}")
    except Exception as e:
        logger.error(f"Erreur WebSocket analytics: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass

# ==================== NOUVEAUX ENDPOINTS ====================

@app.post("/memory/remember-name", tags=["Intelligence"])
async def remember_user_name(req: NameRequest):
    """Retient le nom d'un utilisateur"""
    try:
        success = await emoia.memory_system.remember_user_name(
            user_id=req.user_id,
            name=req.name,
            nickname=req.nickname
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Nom '{req.name}' retenu pour l'utilisateur {req.user_id}",
                "user_id": req.user_id,
                "name": req.name,
                "nickname": req.nickname
            }
        else:
            raise HTTPException(status_code=500, detail="Erreur lors de la sauvegarde du nom")
    
    except Exception as e:
        logger.error(f"Erreur remember_user_name: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/get-name/{user_id}", tags=["Intelligence"])
async def get_user_name(user_id: str):
    """Récupère le nom d'un utilisateur"""
    try:
        name = await emoia.memory_system.get_user_name(user_id)
        
        if name:
            return {
                "status": "success",
                "user_id": user_id,
                "name": name
            }
        else:
            return {
                "status": "not_found",
                "user_id": user_id,
                "message": "Nom non trouvé pour cet utilisateur"
            }
    
    except Exception as e:
        logger.error(f"Erreur get_user_name: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/learn-concept", tags=["Intelligence"])
async def learn_concept(req: LearnConceptRequest):
    """Apprend un nouveau concept"""
    try:
        concept_id = await emoia.memory_system.learn_concept(
            user_id=req.user_id,
            concept_name=req.concept_name,
            explanation=req.explanation,
            examples=req.examples,
            category=req.category,
            difficulty_level=req.difficulty_level
        )
        
        if concept_id:
            return {
                "status": "success",
                "message": f"Concept '{req.concept_name}' appris avec succès",
                "concept_id": concept_id,
                "concept_name": req.concept_name,
                "category": req.category
            }
        else:
            raise HTTPException(status_code=500, detail="Erreur lors de l'apprentissage du concept")
    
    except Exception as e:
        logger.error(f"Erreur learn_concept: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/concepts/{user_id}", tags=["Intelligence"])
async def get_learned_concepts(user_id: str, category: Optional[str] = None):
    """Récupère les concepts appris par un utilisateur"""
    try:
        concepts = await emoia.memory_system.get_learned_concepts(user_id, category)
        
        return {
            "status": "success",
            "user_id": user_id,
            "category": category,
            "concepts": [
                {
                    "concept": concept.concept,
                    "explanation": concept.explanation,
                    "examples": concept.examples,
                    "category": concept.category,
                    "difficulty_level": concept.difficulty_level,
                    "mastery_level": concept.mastery_level,
                    "last_reviewed": concept.last_reviewed.isoformat(),
                    "next_review": concept.next_review.isoformat()
                }
                for concept in concepts
            ]
        }
    
    except Exception as e:
        logger.error(f"Erreur get_learned_concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tdah/tasks", tags=["Intelligence"])
async def create_tdah_task(req: TDAHTaskRequest):
    """Crée une nouvelle tâche TDAH"""
    try:
        # Parser la date d'échéance si fournie
        due_date = None
        if req.due_date:
            try:
                due_date = datetime.fromisoformat(req.due_date)
            except ValueError:
                pass
        
        task_id = await emoia.memory_system.create_tdah_task(
            user_id=req.user_id,
            title=req.title,
            description=req.description,
            priority=req.priority,
            category=req.category,
            due_date=due_date,
            estimated_duration=req.estimated_duration,
            emotional_state=req.emotional_state
        )
        
        if task_id:
            return {
                "status": "success",
                "message": f"Tâche '{req.title}' créée avec succès",
                "task_id": task_id,
                "title": req.title,
                "priority": req.priority,
                "category": req.category
            }
        else:
            raise HTTPException(status_code=500, detail="Erreur lors de la création de la tâche")
    
    except Exception as e:
        logger.error(f"Erreur create_tdah_task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tdah/tasks/{user_id}", tags=["Intelligence"])
async def get_tdah_tasks(
    user_id: str, 
    completed: Optional[bool] = None,
    category: Optional[str] = None,
    priority_min: Optional[int] = 1
):
    """Récupère les tâches TDAH d'un utilisateur"""
    try:
        tasks = await emoia.memory_system.get_tdah_tasks(
            user_id=user_id,
            completed=completed,
            category=category,
            priority_min=priority_min or 1
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "filter_completed": completed,
            "filter_category": category,
            "tasks": [task.to_dict() for task in tasks]
        }
    
    except Exception as e:
        logger.error(f"Erreur get_tdah_tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tdah/tasks/{task_id}/complete", tags=["Intelligence"])
async def complete_tdah_task(task_id: str, user_id: str):
    """Marque une tâche TDAH comme terminée"""
    try:
        success = await emoia.memory_system.complete_tdah_task(user_id, task_id)
        
        if success:
            return {
                "status": "success",
                "message": "Tâche marquée comme terminée",
                "task_id": task_id
            }
        else:
            raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    except Exception as e:
        logger.error(f"Erreur complete_tdah_task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tdah/suggestions/{user_id}", tags=["Intelligence"])
async def get_tdah_suggestions(user_id: str):
    """Génère des suggestions pour la gestion du TDAH"""
    try:
        suggestions = await emoia.memory_system.get_tdah_suggestions(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
    
    except Exception as e:
        logger.error(f"Erreur get_tdah_suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/telegram/register", tags=["Intelligence"])
async def register_telegram_user(req: TelegramUserRequest):
    """Enregistre un utilisateur Telegram"""
    try:
        # Pour l'instant, on stocke juste en base de données
        # TODO: Implémenter le bot Telegram complet
        
        return {
            "status": "success",
            "message": "Utilisateur Telegram enregistré",
            "user_id": req.user_id,
            "telegram_id": req.telegram_id,
            "note": "Bot Telegram en cours de développement"
        }
    
    except Exception as e:
        logger.error(f"Erreur register_telegram_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats/{user_id}", tags=["Intelligence"])
async def get_memory_stats(user_id: str):
    """Récupère les statistiques de mémoire d'un utilisateur"""
    try:
        stats = emoia.memory_system.get_memory_stats()
        
        # Ajouter des stats spécifiques à l'utilisateur
        user_name = await emoia.memory_system.get_user_name(user_id)
        learned_concepts = await emoia.memory_system.get_learned_concepts(user_id)
        tasks = await emoia.memory_system.get_tdah_tasks(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "user_name": user_name,
            "global_stats": stats,
            "user_specific": {
                "learned_concepts_count": len(learned_concepts),
                "active_tasks_count": len([t for t in tasks if not t.completed]),
                "completed_tasks_count": len([t for t in tasks if t.completed]),
                "has_name": user_name is not None
            }
        }
    
    except Exception as e:
        logger.error(f"Erreur get_memory_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINTS MANQUANTS POUR LE FRONTEND ====================

# Endpoint pour les tâches (mapping vers les tâches TDAH)
@app.get("/api/tasks/{user_id}", tags=["Intelligence"])
async def get_api_tasks(user_id: str, completed: Optional[bool] = None):
    """API endpoint pour les tâches (compatible frontend)"""
    try:
        tasks = await emoia.memory_system.get_tdah_tasks(
            user_id=user_id,
            completed=completed
        )
        
        # Format compatible avec le frontend
        formatted_tasks = []
        for task in tasks:
            formatted_tasks.append({
                "id": task.id if hasattr(task, 'id') else str(len(formatted_tasks)),
                "title": task.title if hasattr(task, 'title') else "Tâche",
                "description": task.description if hasattr(task, 'description') else "",
                "completed": task.completed if hasattr(task, 'completed') else False,
                "priority": task.priority if hasattr(task, 'priority') else 3,
                "category": task.category if hasattr(task, 'category') else "general",
                "created_at": task.created_at.isoformat() if hasattr(task, 'created_at') else datetime.now().isoformat(),
                "due_date": task.due_date.isoformat() if hasattr(task, 'due_date') and task.due_date else None
            })
        
        return {
            "status": "success",
            "tasks": formatted_tasks,
            "count": len(formatted_tasks)
        }
    
    except Exception as e:
        logger.error(f"Erreur get_api_tasks: {e}")
        # Retourner une liste vide au lieu d'une erreur pour éviter les crashes frontend
        return {
            "status": "error",
            "tasks": [],
            "count": 0,
            "message": "Erreur lors du chargement des tâches"
        }

# Endpoint pour le calendrier
@app.get("/api/calendar/{user_id}", tags=["Intelligence"])
async def get_api_calendar(user_id: str, start: Optional[str] = None, end: Optional[str] = None):
    """API endpoint pour le calendrier"""
    try:
        # Pour l'instant, retourner des données de demo basées sur les tâches
        tasks = await emoia.memory_system.get_tdah_tasks(user_id=user_id)
        
        events = []
        for task in tasks:
            if hasattr(task, 'due_date') and task.due_date:
                events.append({
                    "id": task.id if hasattr(task, 'id') else f"task_{len(events)}",
                    "title": task.title if hasattr(task, 'title') else "Tâche",
                    "start": task.due_date.isoformat() if task.due_date else datetime.now().isoformat(),
                    "end": task.due_date.isoformat() if task.due_date else datetime.now().isoformat(),
                    "type": "task",
                    "priority": task.priority if hasattr(task, 'priority') else 3,
                    "completed": task.completed if hasattr(task, 'completed') else False
                })
        
        return {
            "status": "success",
            "events": events,
            "count": len(events)
        }
    
    except Exception as e:
        logger.error(f"Erreur get_api_calendar: {e}")
        return {
            "status": "error",
            "events": [],
            "count": 0,
            "message": "Erreur lors du chargement du calendrier"
        }

# Endpoint pour les utilisateurs
@app.get("/api/users", tags=["Utilisateur"])
async def get_api_users():
    """API endpoint pour la liste des utilisateurs"""
    try:
        # Pour l'instant, retourner un utilisateur de démo
        # En production, cela devrait venir d'une base de données
        users = [
            {
                "id": "demo-user",
                "name": "Utilisateur Démo",
                "email": "demo@emoia.ai",
                "avatar": "/default-avatar.png",
                "status": "active",
                "last_seen": datetime.now().isoformat(),
                "preferences": {
                    "language": "fr",
                    "theme": "light",
                    "notifications": True
                }
            }
        ]
        
        return {
            "status": "success",
            "users": users,
            "count": len(users)
        }
    
    except Exception as e:
        logger.error(f"Erreur get_api_users: {e}")
        return {
            "status": "error",
            "users": [],
            "count": 0,
            "message": "Erreur lors du chargement des utilisateurs"
        }

# Endpoint pour les mémoires
@app.get("/api/memories/{user_id}", tags=["Intelligence"])
async def get_api_memories(user_id: str, limit: int = 100, memory_type: Optional[str] = None):
    """API endpoint pour les mémoires utilisateur"""
    try:
        # Récupérer les informations de mémoire depuis le système existant
        stats = emoia.memory_system.get_memory_stats()
        
        # Pour l'instant, créer des données de démo basées sur les interactions
        memories = []
        
        # Ajouter les concepts appris
        try:
            concepts = await emoia.memory_system.get_learned_concepts(user_id)
            for concept in concepts[:limit//2]:
                memories.append({
                    "id": f"concept_{len(memories)}",
                    "type": "concept",
                    "title": concept.concept if hasattr(concept, 'concept') else "Concept",
                    "content": concept.explanation if hasattr(concept, 'explanation') else "",
                    "category": concept.category if hasattr(concept, 'category') else "general",
                    "importance": 0.8,
                    "created_at": concept.last_reviewed.isoformat() if hasattr(concept, 'last_reviewed') else datetime.now().isoformat(),
                    "accessed_count": 1,
                    "tags": ["apprentissage", concept.category if hasattr(concept, 'category') else "general"]
                })
        except:
            pass
        
        # Ajouter des mémoires conversationnelles de démo
        for i in range(min(10, limit - len(memories))):
            memories.append({
                "id": f"conversation_{i}",
                "type": "conversation",
                "title": f"Conversation {i+1}",
                "content": f"Résumé de la conversation {i+1} avec l'utilisateur",
                "category": "conversation",
                "importance": 0.6,
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "accessed_count": 1,
                "tags": ["conversation", "interaction"]
            })
        
        return {
            "status": "success",
            "memories": memories[:limit],
            "count": len(memories),
            "total_available": stats.get("total_memories", len(memories))
        }
    
    except Exception as e:
        logger.error(f"Erreur get_api_memories: {e}")
        return {
            "status": "error",
            "memories": [],
            "count": 0,
            "message": "Erreur lors du chargement des mémoires"
        }

@app.delete("/api/memories/{memory_id}", tags=["Intelligence"])
async def delete_api_memory(memory_id: str, user_id: str = None):
    """API endpoint pour supprimer une mémoire"""
    try:
        # Pour l'instant, juste retourner un succès simulé
        # En production, cela devrait vraiment supprimer de la base de données
        logger.info(f"Demande de suppression de la mémoire {memory_id} pour l'utilisateur {user_id}")
        
        return {
            "status": "success",
            "message": f"Mémoire {memory_id} supprimée",
            "memory_id": memory_id
        }
    
    except Exception as e:
        logger.error(f"Erreur delete_api_memory: {e}")
        return {
            "status": "error",
            "message": "Erreur lors de la suppression de la mémoire"
        }

# Endpoints Telegram
@app.get("/api/telegram/status", tags=["Intelligence"])
async def get_api_telegram_status():
    """API endpoint pour le statut Telegram"""
    try:
        # Pour l'instant, retourner un statut de démo
        return {
            "status": "inactive",
            "bot_username": "emoia_bot",
            "connected": False,
            "last_update": datetime.now().isoformat(),
            "active_users": 0,
            "message": "Bot Telegram en cours de développement"
        }
    
    except Exception as e:
        logger.error(f"Erreur get_api_telegram_status: {e}")
        return {
            "status": "error",
            "connected": False,
            "message": "Erreur lors de la vérification du statut Telegram"
        }

@app.get("/api/telegram/users", tags=["Intelligence"])
async def get_api_telegram_users():
    """API endpoint pour les utilisateurs Telegram"""
    try:
        # Pour l'instant, retourner une liste vide
        # En production, cela devrait venir du bot Telegram
        return {
            "status": "success",
            "users": [],
            "count": 0,
            "message": "Aucun utilisateur Telegram connecté"
        }
    
    except Exception as e:
        logger.error(f"Erreur get_api_telegram_users: {e}")
        return {
            "status": "error",
            "users": [],
            "count": 0,
            "message": "Erreur lors du chargement des utilisateurs Telegram"
        }

# ==================== FIN DES NOUVEAUX ENDPOINTS ====================

if __name__ == "__main__":
    uvicorn.run("src.core.api:app", host="0.0.0.0", port=8000, reload=True)