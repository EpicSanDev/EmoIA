import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './SmartCalendar.css';

interface CalendarEvent {
  id: string;
  title: string;
  description?: string;
  startTime: string;
  endTime: string;
  category: 'work' | 'personal' | 'health' | 'meeting' | 'reminder' | 'other';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  location?: string;
  attendees?: string[];
  reminder?: number; // minutes before
  recurring?: {
    type: 'daily' | 'weekly' | 'monthly' | 'yearly';
    interval: number;
    endDate?: string;
  };
  aiGenerated?: boolean;
  aiSuggestions?: {
    optimalTime?: string;
    conflictWarning?: string;
    preparationTime?: number;
    relatedTasks?: string[];
  };
  color?: string;
  status: 'scheduled' | 'confirmed' | 'cancelled' | 'completed';
  createdAt: string;
  updatedAt: string;
}

interface SmartCalendarProps {
  userId: string;
  onEventAction?: (action: string, event: CalendarEvent) => void;
  tasks?: any[];
}

const SmartCalendar: React.FC<SmartCalendarProps> = ({ userId, onEventAction, tasks = [] }) => {
  const { t } = useTranslation();
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [viewMode, setViewMode] = useState<'month' | 'week' | 'day' | 'agenda'>('month');
  const [showEventModal, setShowEventModal] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null);
  const [newEvent, setNewEvent] = useState<Partial<CalendarEvent>>({
    title: '',
    description: '',
    category: 'work',
    priority: 'medium',
    status: 'scheduled'
  });
  const [aiSuggestions, setAiSuggestions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [quickAdd, setQuickAdd] = useState('');
  const [showConflicts, setShowConflicts] = useState(false);

  // Charger les événements
  useEffect(() => {
    fetchEvents();
  }, [userId, currentDate]);

  const fetchEvents = async () => {
    try {
      const startDate = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
      const endDate = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);
      
      const response = await fetch(`/api/calendar/${userId}?start=${startDate.toISOString()}&end=${endDate.toISOString()}`);
      if (response.ok) {
        const data = await response.json();
        setEvents(data.events || []);
      }
    } catch (error) {
      console.error('Erreur lors du chargement des événements:', error);
    }
  };

  // Suggestions IA pour l'optimisation du calendrier
  const generateCalendarSuggestions = useCallback(async () => {
    if (events.length === 0) return;

    setLoading(true);
    try {
      const response = await fetch('/api/ai/calendar-optimization', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          events: events,
          tasks: tasks,
          userId: userId,
          dateRange: {
            start: new Date(currentDate.getFullYear(), currentDate.getMonth(), 1).toISOString(),
            end: new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0).toISOString()
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        setAiSuggestions(data.suggestions || []);
      }
    } catch (error) {
      console.error('Erreur suggestions IA:', error);
    } finally {
      setLoading(false);
    }
  }, [events, tasks, userId, currentDate]);

  // Analyse de texte naturel pour création rapide d'événements
  const parseQuickAdd = async (text: string) => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('/api/ai/parse-event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          userId: userId,
          context: { currentDate: currentDate.toISOString() }
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.event) {
          setNewEvent({
            ...data.event,
            id: Date.now().toString(),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          });
          setShowEventModal(true);
        }
      }
    } catch (error) {
      console.error('Erreur parsing événement:', error);
    } finally {
      setLoading(false);
      setQuickAdd('');
    }
  };

  // Ajouter un nouvel événement
  const handleAddEvent = async () => {
    if (!newEvent.title?.trim() || !newEvent.startTime || !newEvent.endTime) return;

    const event: CalendarEvent = {
      id: newEvent.id || Date.now().toString(),
      title: newEvent.title,
      description: newEvent.description || '',
      startTime: newEvent.startTime!,
      endTime: newEvent.endTime!,
      category: newEvent.category || 'work',
      priority: newEvent.priority || 'medium',
      location: newEvent.location,
      attendees: newEvent.attendees || [],
      reminder: newEvent.reminder,
      recurring: newEvent.recurring,
      color: getCategoryColor(newEvent.category || 'work'),
      status: 'scheduled',
      createdAt: newEvent.createdAt || new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    // Vérifier les conflits
    const conflicts = checkTimeConflicts(event);
    if (conflicts.length > 0 && !showConflicts) {
      setShowConflicts(true);
      return;
    }

    try {
      const response = await fetch('/api/calendar/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, event })
      });

      if (response.ok) {
        setEvents(prev => [...prev, event]);
        setNewEvent({ title: '', description: '', category: 'work', priority: 'medium', status: 'scheduled' });
        setShowEventModal(false);
        setShowConflicts(false);
        onEventAction?.('created', event);
      }
    } catch (error) {
      console.error('Erreur création événement:', error);
    }
  };

  // Vérifier les conflits d'horaires
  const checkTimeConflicts = (event: CalendarEvent): CalendarEvent[] => {
    const startTime = new Date(event.startTime);
    const endTime = new Date(event.endTime);

    return events.filter(existingEvent => {
      if (existingEvent.id === event.id) return false;
      
      const existingStart = new Date(existingEvent.startTime);
      const existingEnd = new Date(existingEvent.endTime);

      return (startTime < existingEnd && endTime > existingStart);
    });
  };

  // Optimisation automatique du calendrier
  const optimizeSchedule = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/ai/optimize-schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          events: events,
          tasks: tasks,
          userId: userId,
          preferences: {
            workingHours: { start: '09:00', end: '18:00' },
            breaks: { duration: 15, frequency: 120 },
            focusTime: { morning: true, afternoon: false }
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.optimizedEvents) {
          setEvents(data.optimizedEvents);
                     onEventAction?.('optimized', events[0]);
        }
      }
    } catch (error) {
      console.error('Erreur optimisation:', error);
    } finally {
      setLoading(false);
    }
  };

  // Obtenir la couleur de catégorie
  const getCategoryColor = (category: CalendarEvent['category']): string => {
    const colors = {
      work: '#3B82F6',
      personal: '#10B981',
      health: '#F59E0B',
      meeting: '#8B5CF6',
      reminder: '#EF4444',
      other: '#6B7280'
    };
    return colors[category];
  };

  // Obtenir l'icône de catégorie
  const getCategoryIcon = (category: CalendarEvent['category']): string => {
    const icons = {
      work: '💼',
      personal: '🏠',
      health: '🏃‍♂️',
      meeting: '👥',
      reminder: '⏰',
      other: '📅'
    };
    return icons[category];
  };

  // Navigation dans le calendrier
  const navigateDate = (direction: 'prev' | 'next') => {
    const newDate = new Date(currentDate);
    
    switch (viewMode) {
      case 'month':
        newDate.setMonth(newDate.getMonth() + (direction === 'next' ? 1 : -1));
        break;
      case 'week':
        newDate.setDate(newDate.getDate() + (direction === 'next' ? 7 : -7));
        break;
      case 'day':
        newDate.setDate(newDate.getDate() + (direction === 'next' ? 1 : -1));
        break;
    }
    
    setCurrentDate(newDate);
  };

  // Rendu du calendrier selon le mode de vue
  const renderCalendarView = () => {
    switch (viewMode) {
      case 'month':
        return renderMonthView();
      case 'week':
        return renderWeekView();
      case 'day':
        return renderDayView();
      case 'agenda':
        return renderAgendaView();
      default:
        return renderMonthView();
    }
  };

  // Vue mensuelle
  const renderMonthView = () => {
    const firstDay = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
    const lastDay = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);
    const startDate = new Date(firstDay);
    startDate.setDate(startDate.getDate() - firstDay.getDay());

    const days = [];
    const currentDay = new Date(startDate);

    for (let i = 0; i < 42; i++) {
      const dayEvents = events.filter(event => {
        const eventDate = new Date(event.startTime);
        return eventDate.toDateString() === currentDay.toDateString();
      });

      days.push(
        <div 
          key={currentDay.toISOString()}
          className={`calendar-day ${currentDay.getMonth() !== currentDate.getMonth() ? 'other-month' : ''} ${currentDay.toDateString() === new Date().toDateString() ? 'today' : ''}`}
          onClick={() => {
            setCurrentDate(new Date(currentDay));
            setViewMode('day');
          }}
        >
          <div className="day-number">{currentDay.getDate()}</div>
          <div className="day-events">
            {dayEvents.slice(0, 3).map(event => (
              <div 
                key={event.id}
                className="day-event"
                style={{ backgroundColor: event.color }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedEvent(event);
                  setShowEventModal(true);
                }}
              >
                <span className="event-time">
                  {new Date(event.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                <span className="event-title">{event.title}</span>
              </div>
            ))}
            {dayEvents.length > 3 && (
              <div className="more-events">+{dayEvents.length - 3} more</div>
            )}
          </div>
        </div>
      );

      currentDay.setDate(currentDay.getDate() + 1);
    }

    return (
      <div className="month-view">
        <div className="weekday-headers">
          {['Dim', 'Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam'].map(day => (
            <div key={day} className="weekday-header">{day}</div>
          ))}
        </div>
        <div className="calendar-grid">
          {days}
        </div>
      </div>
    );
  };

  // Vue hebdomadaire
  const renderWeekView = () => {
    const startOfWeek = new Date(currentDate);
    startOfWeek.setDate(currentDate.getDate() - currentDate.getDay());

    const weekDays = [];
    for (let i = 0; i < 7; i++) {
      const day = new Date(startOfWeek);
      day.setDate(startOfWeek.getDate() + i);
      
      const dayEvents = events.filter(event => {
        const eventDate = new Date(event.startTime);
        return eventDate.toDateString() === day.toDateString();
      }).sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime());

      weekDays.push(
        <div key={day.toISOString()} className="week-day">
          <div className="week-day-header">
            <div className="day-name">{day.toLocaleDateString([], { weekday: 'short' })}</div>
            <div className={`day-number ${day.toDateString() === new Date().toDateString() ? 'today' : ''}`}>
              {day.getDate()}
            </div>
          </div>
          <div className="week-day-events">
            {dayEvents.map(event => (
              <div 
                key={event.id}
                className="week-event"
                style={{ backgroundColor: event.color }}
                onClick={() => {
                  setSelectedEvent(event);
                  setShowEventModal(true);
                }}
              >
                <div className="event-time">
                  {new Date(event.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - 
                  {new Date(event.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
                <div className="event-title">{event.title}</div>
                {event.location && (
                  <div className="event-location">📍 {event.location}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }

    return <div className="week-view">{weekDays}</div>;
  };

  // Vue journalière
  const renderDayView = () => {
    const dayEvents = events.filter(event => {
      const eventDate = new Date(event.startTime);
      return eventDate.toDateString() === currentDate.toDateString();
    }).sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime());

    const hours = [];
    for (let hour = 0; hour < 24; hour++) {
      const hourEvents = dayEvents.filter(event => {
        const eventHour = new Date(event.startTime).getHours();
        return eventHour === hour;
      });

      hours.push(
        <div key={hour} className="day-hour">
          <div className="hour-label">
            {hour.toString().padStart(2, '0')}:00
          </div>
          <div className="hour-events">
            {hourEvents.map(event => (
              <div 
                key={event.id}
                className="hour-event"
                style={{ backgroundColor: event.color }}
                onClick={() => {
                  setSelectedEvent(event);
                  setShowEventModal(true);
                }}
              >
                <div className="event-header">
                  <span className="event-icon">{getCategoryIcon(event.category)}</span>
                  <span className="event-title">{event.title}</span>
                  <span className="event-priority priority-{event.priority}">●</span>
                </div>
                <div className="event-time">
                  {new Date(event.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - 
                  {new Date(event.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
                {event.description && (
                  <div className="event-description">{event.description}</div>
                )}
                {event.location && (
                  <div className="event-location">📍 {event.location}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      );
    }

    return <div className="day-view">{hours}</div>;
  };

  // Vue agenda
  const renderAgendaView = () => {
    const upcomingEvents = events
      .filter(event => new Date(event.startTime) > new Date())
      .sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime())
      .slice(0, 20);

    return (
      <div className="agenda-view">
        {upcomingEvents.length === 0 ? (
          <div className="empty-agenda">
            <div className="empty-icon">📅</div>
            <h3>{t('calendar.noUpcomingEvents')}</h3>
            <p>{t('calendar.noUpcomingEventsDescription')}</p>
          </div>
        ) : (
          upcomingEvents.map(event => (
            <div 
              key={event.id}
              className="agenda-event"
              onClick={() => {
                setSelectedEvent(event);
                setShowEventModal(true);
              }}
            >
              <div className="agenda-event-date">
                <div className="event-day">
                  {new Date(event.startTime).toLocaleDateString([], { weekday: 'short' })}
                </div>
                <div className="event-date-number">
                  {new Date(event.startTime).getDate()}
                </div>
                <div className="event-month">
                  {new Date(event.startTime).toLocaleDateString([], { month: 'short' })}
                </div>
              </div>
              <div className="agenda-event-content">
                <div className="agenda-event-header">
                  <span className="event-icon">{getCategoryIcon(event.category)}</span>
                  <h4 className="event-title">{event.title}</h4>
                  <div className="event-badges">
                    <span className={`priority-badge priority-${event.priority}`}>
                      {t(`calendar.priorities.${event.priority}`)}
                    </span>
                    <span className="category-badge">
                      {t(`calendar.categories.${event.category}`)}
                    </span>
                  </div>
                </div>
                <div className="agenda-event-details">
                  <div className="event-time">
                    ⏰ {new Date(event.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - 
                    {new Date(event.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  {event.location && (
                    <div className="event-location">📍 {event.location}</div>
                  )}
                  {event.description && (
                    <div className="event-description">{event.description}</div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    );
  };

  return (
    <div className="smart-calendar">
      {/* Header */}
      <div className="calendar-header">
        <div className="header-top">
          <div className="calendar-title">
            <span className="title-icon">📅</span>
            <h2>{t('calendar.title')}</h2>
          </div>
          
          <div className="calendar-actions">
            <button 
              className="btn btn-secondary"
              onClick={generateCalendarSuggestions}
              disabled={loading}
            >
              {loading ? '⏳' : '🤖'} {t('calendar.optimize')}
            </button>
            <button 
              className="btn btn-primary"
              onClick={() => setShowEventModal(true)}
            >
              ➕ {t('calendar.addEvent')}
            </button>
          </div>
        </div>

        {/* Quick Add */}
        <div className="quick-add-container">
          <input
            type="text"
            placeholder={t('calendar.quickAddPlaceholder')}
            value={quickAdd}
            onChange={(e) => setQuickAdd(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                parseQuickAdd(quickAdd);
              }
            }}
            className="quick-add-input"
          />
          <button 
            className="btn-quick-add"
            onClick={() => parseQuickAdd(quickAdd)}
            disabled={!quickAdd.trim() || loading}
          >
            {loading ? '⏳' : '🪄'} {t('calendar.parseEvent')}
          </button>
        </div>

        {/* Navigation et vues */}
        <div className="calendar-controls">
          <div className="date-navigation">
            <button className="btn-nav" onClick={() => navigateDate('prev')}>‹</button>
            <h3 className="current-date">
              {currentDate.toLocaleDateString([], { 
                month: 'long', 
                year: 'numeric',
                ...(viewMode === 'day' && { day: 'numeric', weekday: 'long' })
              })}
            </h3>
            <button className="btn-nav" onClick={() => navigateDate('next')}>›</button>
            <button 
              className="btn btn-ghost btn-today"
              onClick={() => setCurrentDate(new Date())}
            >
              {t('calendar.today')}
            </button>
          </div>

          <div className="view-selector">
            {(['month', 'week', 'day', 'agenda'] as const).map(mode => (
              <button
                key={mode}
                className={`view-btn ${viewMode === mode ? 'active' : ''}`}
                onClick={() => setViewMode(mode)}
              >
                {t(`calendar.views.${mode}`)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Calendar View */}
      <div className="calendar-content">
        {renderCalendarView()}
      </div>

      {/* AI Suggestions Sidebar */}
      {aiSuggestions.length > 0 && (
        <div className="ai-suggestions-panel">
          <h3>🤖 {t('calendar.aiSuggestions')}</h3>
          <div className="suggestions-list">
            {aiSuggestions.map((suggestion, index) => (
              <div key={index} className="suggestion-card">
                <div className="suggestion-header">
                  <span className="suggestion-type">{suggestion.type}</span>
                  <span className="suggestion-priority">{suggestion.priority}</span>
                </div>
                <div className="suggestion-content">
                  <h4>{suggestion.title}</h4>
                  <p>{suggestion.description}</p>
                  {suggestion.action && (
                    <button 
                      className="btn btn-primary btn-suggestion"
                      onClick={() => suggestion.action()}
                    >
                      {suggestion.actionLabel}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modal d'événement */}
      {showEventModal && (
        <div className="event-modal">
          <div className="modal-content card-glass">
            <div className="modal-header">
              <h3>{selectedEvent ? t('calendar.editEvent') : t('calendar.newEvent')}</h3>
              <button 
                className="btn-close"
                onClick={() => {
                  setShowEventModal(false);
                  setSelectedEvent(null);
                  setNewEvent({ title: '', description: '', category: 'work', priority: 'medium', status: 'scheduled' });
                  setShowConflicts(false);
                }}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              {/* Formulaire événement */}
              <div className="form-group">
                <input
                  type="text"
                  placeholder={t('calendar.eventTitle')}
                  value={newEvent.title || ''}
                  onChange={(e) => setNewEvent(prev => ({ ...prev, title: e.target.value }))}
                  className="event-input"
                  autoFocus
                />
              </div>

              <div className="form-group">
                <textarea
                  placeholder={t('calendar.eventDescription')}
                  value={newEvent.description || ''}
                  onChange={(e) => setNewEvent(prev => ({ ...prev, description: e.target.value }))}
                  className="event-textarea"
                  rows={3}
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>{t('calendar.startTime')}</label>
                  <input
                    type="datetime-local"
                    value={newEvent.startTime || ''}
                    onChange={(e) => setNewEvent(prev => ({ ...prev, startTime: e.target.value }))}
                    className="event-input"
                  />
                </div>

                <div className="form-group">
                  <label>{t('calendar.endTime')}</label>
                  <input
                    type="datetime-local"
                    value={newEvent.endTime || ''}
                    onChange={(e) => setNewEvent(prev => ({ ...prev, endTime: e.target.value }))}
                    className="event-input"
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>{t('calendar.category')}</label>
                  <select
                    value={newEvent.category || 'work'}
                    onChange={(e) => setNewEvent(prev => ({ ...prev, category: e.target.value as any }))}
                    className="event-select"
                  >
                    <option value="work">{t('calendar.categories.work')}</option>
                    <option value="personal">{t('calendar.categories.personal')}</option>
                    <option value="health">{t('calendar.categories.health')}</option>
                    <option value="meeting">{t('calendar.categories.meeting')}</option>
                    <option value="reminder">{t('calendar.categories.reminder')}</option>
                    <option value="other">{t('calendar.categories.other')}</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>{t('calendar.priority')}</label>
                  <select
                    value={newEvent.priority || 'medium'}
                    onChange={(e) => setNewEvent(prev => ({ ...prev, priority: e.target.value as any }))}
                    className="event-select"
                  >
                    <option value="low">{t('calendar.priorities.low')}</option>
                    <option value="medium">{t('calendar.priorities.medium')}</option>
                    <option value="high">{t('calendar.priorities.high')}</option>
                    <option value="urgent">{t('calendar.priorities.urgent')}</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <input
                  type="text"
                  placeholder={t('calendar.location')}
                  value={newEvent.location || ''}
                  onChange={(e) => setNewEvent(prev => ({ ...prev, location: e.target.value }))}
                  className="event-input"
                />
              </div>

              {/* Avertissement de conflit */}
              {showConflicts && newEvent.startTime && newEvent.endTime && (
                <div className="conflict-warning">
                  <div className="warning-header">
                    ⚠️ {t('calendar.conflictDetected')}
                  </div>
                  <div className="conflicts-list">
                    {checkTimeConflicts(newEvent as CalendarEvent).map(conflict => (
                      <div key={conflict.id} className="conflict-item">
                        <strong>{conflict.title}</strong>
                        <span>
                          {new Date(conflict.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - 
                          {new Date(conflict.endTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="modal-actions">
              <button 
                className="btn btn-secondary"
                onClick={() => {
                  setShowEventModal(false);
                  setSelectedEvent(null);
                  setShowConflicts(false);
                }}
              >
                {t('common.cancel')}
              </button>
              <button 
                className="btn btn-primary"
                onClick={handleAddEvent}
                disabled={!newEvent.title?.trim() || !newEvent.startTime || !newEvent.endTime}
              >
                {selectedEvent ? t('calendar.updateEvent') : t('calendar.createEvent')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SmartCalendar;