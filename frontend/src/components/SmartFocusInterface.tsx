import React, { useState, useEffect } from 'react';
import './SmartFocusInterface.css';

interface FocusSession {
  session_id: string;
  elapsed_minutes: number;
  planned_duration: number;
  current_productivity: number;
  flow_state: boolean;
}

interface FocusAnalytics {
  active_session?: FocusSession;
  statistics: {
    total_sessions: number;
    total_focus_time_hours: number;
    average_productivity_score: number;
    flow_state_sessions: number;
    productivity_trend: string;
  };
  profile: {
    preferred_session_duration: number;
    optimal_focus_times: Array<[number, number]>;
    most_effective_music?: string;
    energy_peaks: number[];
  };
  recommendations: string[];
}

enum FocusLevel {
  LIGHT = 'light',
  MEDIUM = 'medium',
  DEEP = 'deep',
  FLOW = 'flow'
}

const SmartFocusInterface: React.FC = () => {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [currentSession, setCurrentSession] = useState<FocusSession | null>(null);
  const [analytics, setAnalytics] = useState<FocusAnalytics | null>(null);
  const [sessionForm, setSessionForm] = useState({
    duration: 45,
    task_description: '',
    focus_level: FocusLevel.MEDIUM,
    custom_music: ''
  });

  const userId = 'demo-user'; // In real app, get from auth context

  useEffect(() => {
    loadAnalytics();
    const interval = setInterval(() => {
      if (isSessionActive) {
        updateSessionStatus();
      }
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [isSessionActive]);

  const loadAnalytics = async () => {
    try {
      const response = await fetch(`/api/focus/analytics/${userId}`);
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data);
        if (data.active_session) {
          setCurrentSession(data.active_session);
          setIsSessionActive(true);
        }
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const updateSessionStatus = async () => {
    if (currentSession) {
      try {
        const response = await fetch(`/api/focus/analytics/${userId}`);
        if (response.ok) {
          const data = await response.json();
          if (data.active_session) {
            setCurrentSession(data.active_session);
          } else {
            setIsSessionActive(false);
            setCurrentSession(null);
            loadAnalytics(); // Reload to get updated stats
          }
        }
      } catch (error) {
        console.error('Failed to update session status:', error);
      }
    }
  };

  const startFocusSession = async () => {
    try {
      const response = await fetch('/api/focus/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...sessionForm,
          user_id: userId
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setIsSessionActive(true);
        // The session will be updated by the next analytics call
        setTimeout(loadAnalytics, 1000);
      } else {
        console.error('Failed to start focus session');
      }
    } catch (error) {
      console.error('Error starting focus session:', error);
    }
  };

  const endFocusSession = async (completionRate: number = 1.0) => {
    if (!currentSession) return;

    try {
      const response = await fetch(`/api/focus/end/${currentSession.session_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ completion_rate: completionRate }),
      });

      if (response.ok) {
        const result = await response.json();
        setIsSessionActive(false);
        setCurrentSession(null);
        loadAnalytics();
        
        // Show session summary
        alert(`Session terminée!\nProductivité: ${(result.final_productivity_score * 100).toFixed(0)}%\nFlow state: ${result.flow_state_achieved ? 'Oui' : 'Non'}`);
      }
    } catch (error) {
      console.error('Error ending focus session:', error);
    }
  };

  const getProductivityColor = (score: number) => {
    if (score >= 0.8) return '#4CAF50';
    if (score >= 0.6) return '#FF9800';
    return '#F44336';
  };

  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  const renderSessionControls = () => (
    <div className="session-controls">
      <h3>🎯 Démarrer une session de concentration</h3>
      
      <div className="form-group">
        <label>Durée (minutes):</label>
        <input
          type="range"
          min="15"
          max="240"
          value={sessionForm.duration}
          onChange={(e) => setSessionForm({...sessionForm, duration: parseInt(e.target.value)})}
        />
        <span className="duration-display">{formatTime(sessionForm.duration)}</span>
      </div>

      <div className="form-group">
        <label>Description de la tâche:</label>
        <input
          type="text"
          placeholder="Ex: Rédaction rapport mensuel, Développement feature X..."
          value={sessionForm.task_description}
          onChange={(e) => setSessionForm({...sessionForm, task_description: e.target.value})}
        />
      </div>

      <div className="form-group">
        <label>Niveau de concentration:</label>
        <select
          value={sessionForm.focus_level}
          onChange={(e) => setSessionForm({...sessionForm, focus_level: e.target.value as FocusLevel})}
        >
          <option value={FocusLevel.LIGHT}>💡 Léger - Notifications réduites</option>
          <option value={FocusLevel.MEDIUM}>🎯 Moyen - Musique + blocages partiels</option>
          <option value={FocusLevel.DEEP}>🔒 Profond - Blocage total, environnement optimal</option>
          <option value={FocusLevel.FLOW}>🌊 Flow - Adaptation dynamique maximale</option>
        </select>
      </div>

      <div className="form-group">
        <label>Musique personnalisée (optionnel):</label>
        <select
          value={sessionForm.custom_music}
          onChange={(e) => setSessionForm({...sessionForm, custom_music: e.target.value})}
        >
          <option value="">🤖 Choix IA automatique</option>
          <option value="binaural_beats_40hz">🧠 Battements binauraux 40Hz</option>
          <option value="classical_baroque">🎼 Classique baroque</option>
          <option value="ambient_nature">🌿 Ambiance nature</option>
          <option value="white_noise">⚪ Bruit blanc</option>
          <option value="brown_noise">🤎 Bruit brun</option>
        </select>
      </div>

      <button
        className="start-session-btn"
        onClick={startFocusSession}
        disabled={!sessionForm.task_description.trim()}
      >
        🚀 Démarrer la session
      </button>
    </div>
  );

  const renderActiveSession = () => {
    if (!currentSession) return null;

    const progressPercentage = (currentSession.elapsed_minutes / currentSession.planned_duration) * 100;
    const timeRemaining = currentSession.planned_duration - currentSession.elapsed_minutes;

    return (
      <div className="active-session">
        <div className="session-header">
          <h3>🎯 Session active</h3>
          {currentSession.flow_state && <div className="flow-indicator">🌊 État de Flow détecté!</div>}
        </div>

        <div className="session-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ 
                width: `${Math.min(progressPercentage, 100)}%`,
                backgroundColor: getProductivityColor(currentSession.current_productivity)
              }}
            />
          </div>
          <div className="time-info">
            <span>{formatTime(currentSession.elapsed_minutes)} / {formatTime(currentSession.planned_duration)}</span>
            <span>Reste: {formatTime(Math.max(0, timeRemaining))}</span>
          </div>
        </div>

        <div className="productivity-metrics">
          <div className="metric">
            <span className="metric-label">Productivité:</span>
            <span 
              className="metric-value"
              style={{ color: getProductivityColor(currentSession.current_productivity) }}
            >
              {(currentSession.current_productivity * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="session-actions">
          <button
            className="end-session-btn completed"
            onClick={() => endFocusSession(1.0)}
          >
            ✅ Terminer (Complété)
          </button>
          <button
            className="end-session-btn partial"
            onClick={() => endFocusSession(0.5)}
          >
            ⏸️ Terminer (Partiel)
          </button>
          <button
            className="end-session-btn cancelled"
            onClick={() => endFocusSession(0.1)}
          >
            ❌ Annuler
          </button>
        </div>
      </div>
    );
  };

  const renderAnalytics = () => {
    if (!analytics) return <div className="loading">Chargement des analytics...</div>;

    const { statistics, profile, recommendations } = analytics;

    return (
      <div className="analytics-dashboard">
        <h3>📊 Vos statistiques de concentration</h3>
        
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-number">{statistics.total_sessions}</div>
            <div className="stat-label">Sessions totales</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{statistics.total_focus_time_hours}h</div>
            <div className="stat-label">Temps de focus</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{(statistics.average_productivity_score * 100).toFixed(0)}%</div>
            <div className="stat-label">Productivité moyenne</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{statistics.flow_state_sessions}</div>
            <div className="stat-label">Sessions Flow</div>
          </div>
        </div>

        <div className="profile-insights">
          <h4>🧠 Votre profil de concentration</h4>
          <div className="insights-grid">
            <div className="insight">
              <strong>Durée optimale:</strong> {formatTime(profile.preferred_session_duration)}
            </div>
            <div className="insight">
              <strong>Heures de pointe:</strong> 
              {profile.energy_peaks.length > 0 
                ? profile.energy_peaks.map(hour => `${hour}h`).join(', ')
                : 'En analyse...'
              }
            </div>
            {profile.most_effective_music && (
              <div className="insight">
                <strong>Musique efficace:</strong> {profile.most_effective_music}
              </div>
            )}
          </div>
        </div>

        {recommendations.length > 0 && (
          <div className="recommendations">
            <h4>💡 Recommandations IA</h4>
            <ul>
              {recommendations.map((rec, index) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="smart-focus-interface">
      <div className="focus-header">
        <h2>🧠 Smart Focus Mode</h2>
        <p>Mode de concentration intelligente avec IA adaptative</p>
      </div>

      {isSessionActive ? renderActiveSession() : renderSessionControls()}
      
      <div className="analytics-section">
        {renderAnalytics()}
      </div>
    </div>
  );
};

export default SmartFocusInterface;