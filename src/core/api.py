from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import asyncio
import uvicorn
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import logging

from ..config import Config
from .emoia_main import EmoIA
from src.models.user_preferences import UserPreferences  # Nouvelle importation
from src.mcp import MCPClient, MCPManager  # Import MCP
from .advanced_api import register_advanced_endpoints

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    await mcp_client.initialize()  # Initialiser MCP

# Importer le routeur WebSocket pour les analytics
from src.analytics.websocket import router as analytics_router
app.include_router(analytics_router)

# Register advanced endpoints for task management and calendar
register_advanced_endpoints(app)

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

# === NEW ENDPOINTS FOR TASK MANAGEMENT ===

@app.post("/api/tasks")
async def create_task(request: dict):
    """Create a new task with AI suggestions."""
    try:
        user_id = request.get('userId')
        task = request.get('task')
        
        if not user_id or not task:
            raise HTTPException(status_code=400, detail="Missing user_id or task")
        
        # Add AI suggestions to the task
        task_title = task.get('title', '')
        task_description = task.get('description', '')
        
        # Generate AI suggestions for the task
        ai_suggestions = await generate_task_suggestions(task_title, task_description, user_id)
        task['aiSuggestions'] = ai_suggestions
        
        # Store task in memory/database
        task['id'] = str(uuid.uuid4())
        task['createdAt'] = datetime.now().isoformat()
        task['updatedAt'] = datetime.now().isoformat()
        
        # Here you would typically save to database
        # For now, we'll use in-memory storage
        if user_id not in user_tasks:
            user_tasks[user_id] = []
        user_tasks[user_id].append(task)
        
        return JSONResponse({"success": True, "task": task})
        
    except Exception as e:
        logging.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{user_id}")
async def get_user_tasks(user_id: str):
    """Get all tasks for a user."""
    try:
        tasks = user_tasks.get(user_id, [])
        return {"tasks": tasks}
    except Exception as e:
        logging.error(f"Error fetching tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/tasks/{task_id}")
async def update_task(task_id: str, request: dict):
    """Update a task status or other properties."""
    try:
        # Find and update task
        for user_id, tasks in user_tasks.items():
            for task in tasks:
                if task['id'] == task_id:
                    task.update(request)
                    task['updatedAt'] = datetime.now().isoformat()
                    return {"success": True, "task": task}
        
        raise HTTPException(status_code=404, detail="Task not found")
        
    except Exception as e:
        logging.error(f"Error updating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/task-suggestions")
async def get_task_suggestions(request: dict):
    """Generate AI suggestions for a task."""
    try:
        title = request.get('title', '')
        existing_tasks = request.get('existingTasks', [])
        user_id = request.get('userId', '')
        
        suggestions = await generate_task_suggestions(title, '', user_id, existing_tasks)
        
        return JSONResponse({"suggestions": suggestions})
        
    except Exception as e:
        logging.error(f"Error generating task suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === NEW ENDPOINTS FOR CALENDAR MANAGEMENT ===

@app.get("/api/calendar/{user_id}")
async def get_calendar_events(user_id: str, start: Optional[str] = None, end: Optional[str] = None):
    """Get calendar events for a user within a date range."""
    try:
        events = user_events.get(user_id, [])
        
        # Filter by date range if provided
        if start and end:
            start_date = datetime.fromisoformat(start.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(end.replace('Z', '+00:00'))
            
            filtered_events = []
            for event in events:
                event_start = datetime.fromisoformat(event['startTime'].replace('Z', '+00:00'))
                if start_date <= event_start <= end_date:
                    filtered_events.append(event)
            events = filtered_events
        
        return JSONResponse({"events": events})
        
    except Exception as e:
        logging.error(f"Error fetching calendar events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/calendar/events")
async def create_calendar_event(request: dict):
    """Create a new calendar event."""
    try:
        user_id = request.get('userId')
        event = request.get('event')
        
        if not user_id or not event:
            raise HTTPException(status_code=400, detail="Missing user_id or event")
        
        # Add AI optimization suggestions
        ai_suggestions = await generate_calendar_suggestions(event, user_id)
        event['aiSuggestions'] = ai_suggestions
        
        # Store event
        event['id'] = str(uuid.uuid4())
        event['createdAt'] = datetime.now().isoformat()
        event['updatedAt'] = datetime.now().isoformat()
        
        if user_id not in user_events:
            user_events[user_id] = []
        user_events[user_id].append(event)
        
        return JSONResponse({"success": True, "event": event})
        
    except Exception as e:
        logging.error(f"Error creating calendar event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/calendar-optimization")
async def optimize_calendar(request: dict):
    """Provide AI-powered calendar optimization suggestions."""
    try:
        events = request.get('events', [])
        tasks = request.get('tasks', [])
        user_id = request.get('userId', '')
        date_range = request.get('dateRange', {})
        
        suggestions = await generate_calendar_optimization(events, tasks, user_id, date_range)
        
        return JSONResponse({"suggestions": suggestions})
        
    except Exception as e:
        logging.error(f"Error generating calendar optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/parse-event")
async def parse_natural_language_event(request: dict):
    """Parse natural language text into calendar event."""
    try:
        text = request.get('text', '')
        user_id = request.get('userId', '')
        context = request.get('context', {})
        
        event = await parse_event_from_text(text, user_id, context)
        
        return JSONResponse({"event": event})
        
    except Exception as e:
        logging.error(f"Error parsing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/optimize-schedule")
async def optimize_user_schedule(request: dict):
    """Optimize user's complete schedule using AI."""
    try:
        events = request.get('events', [])
        tasks = request.get('tasks', [])
        user_id = request.get('userId', '')
        preferences = request.get('preferences', {})
        
        optimized_events = await optimize_schedule_ai(events, tasks, user_id, preferences)
        
        return JSONResponse({"optimizedEvents": optimized_events})
        
    except Exception as e:
        logging.error(f"Error optimizing schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === AI HELPER FUNCTIONS ===

async def generate_task_suggestions(title: str, description: str, user_id: str, existing_tasks: List = None) -> List[str]:
    """Generate AI suggestions for task optimization."""
    try:
        if not existing_tasks:
            existing_tasks = []
        
        # Create context from existing tasks
        context = f"User has {len(existing_tasks)} existing tasks. New task: '{title}'"
        if description:
            context += f" - {description}"
        
        # Generate suggestions using LLM
        suggestions = [
            f"Break down '{title}' into smaller, manageable subtasks",
            f"Estimate 2-3 hours for completion of '{title}'",
            f"Schedule '{title}' during your most productive hours",
            f"Consider delegating parts of '{title}' if possible",
            f"Set up reminders 30 minutes before starting '{title}'"
        ]
        
        return suggestions[:3]  # Return top 3 suggestions
        
    except Exception as e:
        logging.error(f"Error generating task suggestions: {e}")
        return []

async def generate_calendar_suggestions(event: dict, user_id: str) -> dict:
    """Generate AI suggestions for calendar event optimization."""
    try:
        title = event.get('title', '')
        start_time = event.get('startTime', '')
        
        suggestions = {
            'optimalTime': 'Consider scheduling during your peak energy hours (9-11 AM)',
            'preparationTime': 15,  # minutes
            'relatedTasks': [
                f"Prepare materials for {title}",
                f"Review agenda for {title}",
                f"Set up meeting space for {title}"
            ]
        }
        
        return suggestions
        
    except Exception as e:
        logging.error(f"Error generating calendar suggestions: {e}")
        return {}

async def generate_calendar_optimization(events: List, tasks: List, user_id: str, date_range: dict) -> List[str]:
    """Generate optimization suggestions for calendar."""
    try:
        suggestions = [
            "Consider grouping similar meetings together to minimize context switching",
            "Schedule focused work time in 2-3 hour blocks",
            "Add 15-minute buffers between meetings for transitions",
            "Block calendar time for urgent tasks",
            "Schedule breaks every 90 minutes for optimal productivity"
        ]
        
        # Analyze current calendar for specific suggestions
        if len(events) > 5:
            suggestions.append("Your calendar looks busy - consider moving non-urgent meetings")
        
        if len(tasks) > 10:
            suggestions.append("You have many pending tasks - consider time-blocking for task completion")
        
        return suggestions[:5]
        
    except Exception as e:
        logging.error(f"Error generating calendar optimization: {e}")
        return []

async def parse_event_from_text(text: str, user_id: str, context: dict) -> dict:
    """Parse natural language into calendar event structure."""
    try:
        # Simple NLP parsing - in production, use more sophisticated NLP
        current_date = datetime.fromisoformat(context.get('currentDate', datetime.now().isoformat()))
        
        # Default event structure
        event = {
            'title': text.strip(),
            'description': '',
            'startTime': current_date.replace(hour=9, minute=0).isoformat(),
            'endTime': current_date.replace(hour=10, minute=0).isoformat(),
            'category': 'other',
            'priority': 'medium',
            'status': 'scheduled'
        }
        
        # Basic parsing logic
        text_lower = text.lower()
        
        # Extract time information
        if 'tomorrow' in text_lower:
            tomorrow = current_date + timedelta(days=1)
            event['startTime'] = tomorrow.replace(hour=9, minute=0).isoformat()
            event['endTime'] = tomorrow.replace(hour=10, minute=0).isoformat()
        
        if 'meeting' in text_lower:
            event['category'] = 'meeting'
            event['priority'] = 'high'
        elif 'lunch' in text_lower:
            event['category'] = 'personal'
            event['startTime'] = current_date.replace(hour=12, minute=0).isoformat()
            event['endTime'] = current_date.replace(hour=13, minute=0).isoformat()
        elif 'workout' in text_lower or 'gym' in text_lower:
            event['category'] = 'health'
            event['startTime'] = current_date.replace(hour=18, minute=0).isoformat()
            event['endTime'] = current_date.replace(hour=19, minute=0).isoformat()
        
        return event
        
    except Exception as e:
        logging.error(f"Error parsing event: {e}")
        return {}

async def optimize_schedule_ai(events: List, tasks: List, user_id: str, preferences: dict) -> List:
    """Optimize entire schedule using AI algorithms."""
    try:
        # Simple optimization logic - in production, use more sophisticated algorithms
        optimized_events = events.copy()
        
        # Sort events by priority and time
        optimized_events.sort(key=lambda x: (
            datetime.fromisoformat(x['startTime']),
            {'urgent': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x.get('priority', 'medium'), 2)
        ))
        
        # Add buffer times between meetings
        for i in range(len(optimized_events) - 1):
            current_end = datetime.fromisoformat(optimized_events[i]['endTime'])
            next_start = datetime.fromisoformat(optimized_events[i + 1]['startTime'])
            
            # If events are too close, add buffer
            if (next_start - current_end).total_seconds() < 15 * 60:  # Less than 15 minutes
                new_start = current_end + timedelta(minutes=15)
                optimized_events[i + 1]['startTime'] = new_start.isoformat()
                
                # Adjust end time if duration is preserved
                original_duration = datetime.fromisoformat(optimized_events[i + 1]['endTime']) - next_start
                optimized_events[i + 1]['endTime'] = (new_start + original_duration).isoformat()
        
        return optimized_events
        
    except Exception as e:
        logging.error(f"Error optimizing schedule: {e}")
        return events

# Global storage (in production, use proper database)
user_tasks: Dict[str, List] = {}
user_events: Dict[str, List] = {}

if __name__ == "__main__":
    uvicorn.run("src.core.api:app", host="0.0.0.0", port=8000, reload=True)