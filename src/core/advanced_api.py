"""
Advanced API endpoints for EmoIA v3.0
Task Management and Calendar functionality
"""
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import logging

# Global storage (in production, use proper database)
user_tasks: Dict[str, List] = {}
user_events: Dict[str, List] = {}

def register_advanced_endpoints(app: FastAPI):
    """Register all advanced API endpoints."""
    
    # === TASK MANAGEMENT ENDPOINTS ===
    
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
            
            if user_id not in user_tasks:
                user_tasks[user_id] = []
            user_tasks[user_id].append(task)
            
            return {"success": True, "task": task}
            
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
            
            return {"suggestions": suggestions}
            
        except Exception as e:
            logging.error(f"Error generating task suggestions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # === CALENDAR MANAGEMENT ENDPOINTS ===

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
            
            return {"events": events}
            
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
            
            return {"success": True, "event": event}
            
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
            
            return {"suggestions": suggestions}
            
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
            
            return {"event": event}
            
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
            
            return {"optimizedEvents": optimized_events}
            
        except Exception as e:
            logging.error(f"Error optimizing schedule: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# === AI HELPER FUNCTIONS ===

async def generate_task_suggestions(title: str, description: str, user_id: str, existing_tasks: Optional[List] = None) -> List[str]:
    """Generate AI suggestions for task optimization."""
    try:
        if not existing_tasks:
            existing_tasks = []
        
        # Create context from existing tasks
        context = f"User has {len(existing_tasks)} existing tasks. New task: '{title}'"
        if description:
            context += f" - {description}"
        
        # Generate suggestions using simple logic (in production, use LLM)
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