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

if __name__ == "__main__":
    uvicorn.run("src.core.api:app", host="0.0.0.0", port=8000, reload=True)