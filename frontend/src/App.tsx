import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, RadialLinearScale, BarElement, Filler
} from 'chart.js';
import LanguageSwitcher from './components/LanguageSwitcher.tsx';
import AnalyticsDashboard from './components/AnalyticsDashboard.tsx';
import EmotionWheel from './components/EmotionWheel.tsx';
import PersonalityRadar from './components/PersonalityRadar.tsx';
import MoodHistory from './components/MoodHistory.tsx';
import VoiceInput from './components/VoiceInput.tsx';
import ConversationInsights from './components/ConversationInsights.tsx';
import SmartSuggestions from './components/SmartSuggestions.tsx';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, RadialLinearScale, BarElement, Filler);

// Interfaces
interface Message {
  id: string;
  sender: 'user' | 'emoia';
  text: string;
  emotion?: any;
  timestamp: Date;
  audioUrl?: string;
  confidence?: number;
}

interface Preferences {
  language: string;
  theme: string;
  notification_settings: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
  ai_settings?: {
    personality_style: 'professional' | 'friendly' | 'casual' | 'empathetic';
    response_length: 'concise' | 'detailed' | 'balanced';
    emotional_intelligence_level: number;
  };
}

interface EmotionalAnalysis {
  dominant_emotion: string;
  emotion_scores: { [key: string]: number };
  valence: number;
  arousal: number;
  confidence: number;
}

// Constants
const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/chat';

function App() {
  const { t, i18n } = useTranslation();
  const [tab, setTab] = useState<'chat' | 'dashboard' | 'preferences' | 'insights'>('chat');
  const [userId, setUserId] = useState<string>('demo-user');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [analytics, setAnalytics] = useState<any>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [currentEmotions, setCurrentEmotions] = useState<any[]>([]);
  const [personalityProfile, setPersonalityProfile] = useState<any>(null);
  const [moodHistory, setMoodHistory] = useState<any[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [showInsights, setShowInsights] = useState(true);
  const [showSuggestions, setShowSuggestions] = useState(true);
  
  const [preferences, setPreferences] = useState<Preferences>({
    language: 'fr',
    theme: 'light',
    notification_settings: {
      email: true,
      push: false,
      sound: true
    },
    ai_settings: {
      personality_style: 'empathetic',
      response_length: 'balanced',
      emotional_intelligence_level: 0.8
    }
  });
  
  const [prefsStatus, setPrefsStatus] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  // Appliquer le thème
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', preferences.theme);
  }, [preferences.theme]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  // Charger les préférences et initialiser
  useEffect(() => {
    const fetchPreferences = async () => {
      try {
        const res = await fetch(`${API_URL}/utilisateur/preferences/${userId}`);
        if (res.ok) {
          const data = await res.json();
          setPreferences(prev => ({ ...prev, ...data }));
          // Mettre à jour la langue
          i18n.changeLanguage(data.language);
        }
      } catch (e) {
        console.error("Erreur chargement préférences", e);
      }
    };

    fetchPreferences();
    setupWebSocket();
    fetchPersonalityProfile();
    
    // Message de bienvenue intelligent
    const welcomeMessage: Message = {
      id: Date.now().toString(),
      sender: 'emoia',
      text: t('welcome'),
      timestamp: new Date(),
      emotion: {
        dominant_emotion: 'joy',
        confidence: 0.9
      }
    };
    setMessages([welcomeMessage]);
  }, [userId, t]);

  const setupWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
      setWsConnected(true);
      ws.send(JSON.stringify({ type: 'identify', user_id: userId }));
    };
    
    ws.onclose = () => {
      setWsConnected(false);
      setTimeout(setupWebSocket, 3000);
    };
    
    ws.onerror = () => setWsConnected(false);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'chat_response':
          handleChatResponse(data);
          break;
        case 'emotional_update':
          updateEmotionalState(data);
          break;
        case 'insight_update':
          // Les insights sont gérés par le composant ConversationInsights
          break;
      }
    };
    
    wsRef.current = ws;
  }, [userId]);

  const handleChatResponse = (data: any) => {
    setLoading(false);
    const message: Message = {
      id: Date.now().toString(),
      sender: 'emoia',
      text: data.response,
      emotion: data.emotional_analysis,
      timestamp: new Date(),
      confidence: data.confidence
    };
    setMessages(msgs => [...msgs, message]);
    
    // Mettre à jour l'état émotionnel
    if (data.emotional_analysis) {
      updateEmotionalVisualization(data.emotional_analysis);
    }
  };

  const updateEmotionalState = (data: any) => {
    // Mise à jour des émotions actuelles pour la visualisation
    if (data.current_emotions) {
      setCurrentEmotions(data.current_emotions);
    }
    
    // Mise à jour de l'historique d'humeur
    if (data.mood_point) {
      setMoodHistory(prev => [...prev, data.mood_point].slice(-50)); // Garder les 50 derniers
    }
  };

  const updateEmotionalVisualization = (emotionalAnalysis: EmotionalAnalysis) => {
    const emotions = Object.entries(emotionalAnalysis.emotion_scores || {})
      .map(([emotion, value]) => ({
        emotion,
        value: value as number,
        color: getEmotionColor(emotion),
        icon: getEmotionIcon(emotion)
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8); // Top 8 émotions
    
    setCurrentEmotions(emotions);
  };

  const getEmotionColor = (emotion: string): string => {
    const colors: { [key: string]: string } = {
      joy: '#FFD93D',
      sadness: '#6495ED',
      anger: '#FF6B6B',
      fear: '#9370DB',
      surprise: '#FFB6C1',
      love: '#FF69B4',
      excitement: '#FFA500',
      anxiety: '#DDA0DD',
      contentment: '#98FB98',
      curiosity: '#87CEEB',
      disgust: '#8B4513'
    };
    return colors[emotion] || '#808080';
  };

  const getEmotionIcon = (emotion: string): string => {
    const icons: { [key: string]: string } = {
      joy: '😊',
      sadness: '😢',
      anger: '😠',
      fear: '😨',
      surprise: '😮',
      love: '❤️',
      excitement: '🎉',
      anxiety: '😰',
      contentment: '😌',
      curiosity: '🤔',
      disgust: '🤢'
    };
    return icons[emotion] || '🎭';
  };

  const sendMessage = async (text?: string) => {
    const messageText = text || input;
    if (!messageText.trim()) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: messageText,
      timestamp: new Date()
    };
    setMessages(msgs => [...msgs, userMessage]);
    setInput('');
    setLoading(true);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'chat_message',
        user_id: userId,
        message: messageText,
        context: {
          language: preferences.language,
          ai_settings: preferences.ai_settings
        }
      }));
    } else {
      // Fallback HTTP
      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            message: messageText,
            preferences: preferences.ai_settings
          }),
        });
        const data = await res.json();
        handleChatResponse(data);
      } catch (e) {
        const errorMessage: Message = {
          id: Date.now().toString(),
          sender: 'emoia',
          text: t('errorMessage'),
          timestamp: new Date()
        };
        setMessages(msgs => [...msgs, errorMessage]);
        setLoading(false);
      }
    }
  };

  const handleVoiceTranscript = (transcript: string) => {
    setInput(transcript);
    // Option pour envoyer automatiquement
    if (preferences.ai_settings?.response_length === 'concise') {
      sendMessage(transcript);
    }
  };

  const handleVoiceAudio = async (audioBlob: Blob) => {
    // Ici on pourrait envoyer l'audio au backend pour analyse vocale
    console.log('Audio reçu:', audioBlob);
  };

  const fetchAnalytics = async () => {
    setAnalytics(null);
    try {
      const res = await fetch(`${API_URL}/analytics/${userId}`);
      const data = await res.json();
      setAnalytics(data);
    } catch (e) {
      setAnalytics({ error: t('analyticsError') });
    }
  };

  const fetchPersonalityProfile = async () => {
    try {
      const res = await fetch(`${API_URL}/personality/${userId}`);
      if (res.ok) {
        const data = await res.json();
        setPersonalityProfile(data);
      }
    } catch (e) {
      console.error('Erreur lors de la récupération du profil de personnalité:', e);
    }
  };

  const savePreferences = async () => {
    try {
      const res = await fetch(`${API_URL}/utilisateur/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          ...preferences
        })
      });
      
      if (res.ok) {
        setPrefsStatus(t('preferencesSaved'));
        i18n.changeLanguage(preferences.language);
        setTimeout(() => setPrefsStatus(null), 3000);
      } else {
        setPrefsStatus(t('preferencesError'));
      }
    } catch (e) {
      setPrefsStatus(t('preferencesError'));
    }
  };

  const handleSuggestionSelect = (suggestion: any) => {
    if (suggestion.type === 'response' || suggestion.type === 'question') {
      setInput(suggestion.text);
    } else if (suggestion.type === 'action') {
      // Exécuter l'action suggérée
      sendMessage(suggestion.text);
    }
  };

  const handleEmotionClick = (emotion: string) => {
    // Envoyer un message contextuel basé sur l'émotion
    const emotionMessages: { [key: string]: string } = {
      joy: t('emotionClickJoy'),
      sadness: t('emotionClickSadness'),
      anger: t('emotionClickAnger'),
      fear: t('emotionClickFear'),
      love: t('emotionClickLove')
    };
    
    const message = emotionMessages[emotion] || t('emotionClickDefault', { emotion });
    sendMessage(message);
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="emoji-logo">🧠</span>
            {t('title')}
            <span className="version-badge">v3.0</span>
          </h1>
          <div className="header-status">
            <div className={`connection-status ${wsConnected ? 'connected' : 'disconnected'}`}>
              <span className="status-dot"></span>
              {wsConnected ? t('connected') : t('reconnecting')}
            </div>
          </div>
        </div>
        <div className="controls">
          <LanguageSwitcher />
          <nav className="main-nav">
            <button
              onClick={() => setTab('chat')}
              className={`nav-btn ${tab === 'chat' ? 'active' : ''}`}
            >
              <span className="nav-icon">💬</span>
              {t('chatTab')}
            </button>
            <button
              onClick={() => { setTab('insights'); }}
              className={`nav-btn ${tab === 'insights' ? 'active' : ''}`}
            >
              <span className="nav-icon">🧠</span>
              {t('insightsTab')}
            </button>
            <button
              onClick={() => { setTab('dashboard'); fetchAnalytics(); }}
              className={`nav-btn ${tab === 'dashboard' ? 'active' : ''}`}
            >
              <span className="nav-icon">📊</span>
              {t('dashboardTab')}
            </button>
            <button
              onClick={() => setTab('preferences')}
              className={`nav-btn ${tab === 'preferences' ? 'active' : ''}`}
            >
              <span className="nav-icon">⚙️</span>
              {t('preferencesTab')}
            </button>
          </nav>
        </div>
      </header>

      <main className="main-content">
        {tab === 'chat' && (
          <div className="chat-layout">
            <div className="chat-main">
              <div className="chat-container" ref={chatContainerRef}>
                <div className="messages-area">
                  {messages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.sender}`}>
                      <div className="message-content">
                        <div className="bubble">
                          {msg.text}
                          {msg.emotion && msg.sender === 'emoia' && (
                            <div className="emotion-indicator">
                              <span className="emotion-icon">
                                {getEmotionIcon(msg.emotion.dominant_emotion)}
                              </span>
                              <span className="emotion-label">
                                {t(msg.emotion.dominant_emotion)}
                              </span>
                            </div>
                          )}
                        </div>
                        <div className="message-meta">
                          <span className="timestamp">
                            {msg.timestamp.toLocaleTimeString()}
                          </span>
                          {msg.confidence && (
                            <span className="confidence">
                              {Math.round(msg.confidence * 100)}%
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                  {loading && (
                    <div className="message emoia">
                      <div className="bubble typing">
                        <div className="typing-dots">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
                
                {showSuggestions && messages.length > 0 && (
                  <div className="suggestions-wrapper">
                    <SmartSuggestions
                      context={messages.slice(-5).map(m => m.text).join(' ')}
                      userInput={input}
                      emotionalState={messages[messages.length - 1]?.emotion}
                      onSuggestionSelect={handleSuggestionSelect}
                    />
                  </div>
                )}
                
                <div className="input-area">
                  <div className="input-controls">
                    <VoiceInput
                      onTranscript={handleVoiceTranscript}
                      onAudioData={handleVoiceAudio}
                      language={preferences.language === 'fr' ? 'fr-FR' : preferences.language === 'es' ? 'es-ES' : 'en-US'}
                    />
                    <input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                      placeholder={t('inputPlaceholder')}
                      disabled={loading}
                      className="message-input"
                    />
                    <button
                      onClick={() => sendMessage()}
                      disabled={loading || !input.trim()}
                      className="send-button"
                    >
                      <span className="send-icon">➤</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            <aside className="chat-sidebar">
              {currentEmotions.length > 0 && (
                <div className="emotion-panel">
                  <h3>{t('currentEmotions')}</h3>
                  <EmotionWheel
                    emotions={currentEmotions}
                    size={250}
                    onEmotionClick={handleEmotionClick}
                  />
                </div>
              )}
              
              {showInsights && (
                <div className="insights-panel">
                  <ConversationInsights
                    userId={userId}
                    onInsightAction={(insight) => console.log('Insight action:', insight)}
                  />
                </div>
              )}
            </aside>
          </div>
        )}

        {tab === 'insights' && (
          <div className="insights-container">
            <div className="insights-grid">
              {personalityProfile && (
                <div className="insight-card personality">
                  <h2>{t('personalityProfile')}</h2>
                  <PersonalityRadar data={personalityProfile} />
                </div>
              )}
              
              {moodHistory.length > 0 && (
                <div className="insight-card mood">
                  <h2>{t('moodHistory')}</h2>
                  <MoodHistory history={moodHistory} period="week" />
                </div>
              )}
              
              <div className="insight-card emotions">
                <h2>{t('emotionalBalance')}</h2>
                {currentEmotions.length > 0 && (
                  <EmotionWheel
                    emotions={currentEmotions}
                    size={300}
                    onEmotionClick={handleEmotionClick}
                  />
                )}
              </div>
            </div>
          </div>
        )}

        {tab === 'dashboard' && (
          <AnalyticsDashboard userId={userId} />
        )}

        {tab === 'preferences' && (
          <div className="preferences-container">
            <h2>{t('preferencesTab')}</h2>
            <div className="preferences-grid">
              <div className="pref-section">
                <h3>{t('generalSettings')}</h3>
                <div className="form-group">
                  <label>{t('languageLabel')}</label>
                  <select
                    value={preferences.language}
                    onChange={(e) => setPreferences({...preferences, language: e.target.value})}
                  >
                    <option value="fr">Français</option>
                    <option value="en">English</option>
                    <option value="es">Español</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label>{t('themeLabel')}</label>
                  <div className="theme-options">
                    <button
                      className={`theme-btn ${preferences.theme === 'light' ? 'active' : ''}`}
                      onClick={() => setPreferences({...preferences, theme: 'light'})}
                    >
                      ☀️ {t('lightTheme')}
                    </button>
                    <button
                      className={`theme-btn ${preferences.theme === 'dark' ? 'active' : ''}`}
                      onClick={() => setPreferences({...preferences, theme: 'dark'})}
                    >
                      🌙 {t('darkTheme')}
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="pref-section">
                <h3>{t('aiSettings')}</h3>
                <div className="form-group">
                  <label>{t('personalityStyle')}</label>
                  <select
                    value={preferences.ai_settings?.personality_style || 'empathetic'}
                    onChange={(e) => setPreferences({
                      ...preferences,
                      ai_settings: {
                        ...preferences.ai_settings!,
                        personality_style: e.target.value as any
                      }
                    })}
                  >
                    <option value="professional">{t('professional')}</option>
                    <option value="friendly">{t('friendly')}</option>
                    <option value="casual">{t('casual')}</option>
                    <option value="empathetic">{t('empathetic')}</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label>{t('responseLength')}</label>
                  <select
                    value={preferences.ai_settings?.response_length || 'balanced'}
                    onChange={(e) => setPreferences({
                      ...preferences,
                      ai_settings: {
                        ...preferences.ai_settings!,
                        response_length: e.target.value as any
                      }
                    })}
                  >
                    <option value="concise">{t('concise')}</option>
                    <option value="balanced">{t('balanced')}</option>
                    <option value="detailed">{t('detailed')}</option>
                  </select>
                </div>
                
                <div className="form-group">
                  <label>{t('emotionalIntelligence')}</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={(preferences.ai_settings?.emotional_intelligence_level || 0.8) * 100}
                    onChange={(e) => setPreferences({
                      ...preferences,
                      ai_settings: {
                        ...preferences.ai_settings!,
                        emotional_intelligence_level: Number(e.target.value) / 100
                      }
                    })}
                  />
                  <span>{Math.round((preferences.ai_settings?.emotional_intelligence_level || 0.8) * 100)}%</span>
                </div>
              </div>
              
              <div className="pref-section">
                <h3>{t('notifications')}</h3>
                <div className="notification-settings">
                  <label className="switch-label">
                    <input
                      type="checkbox"
                      checked={preferences.notification_settings.email}
                      onChange={(e) => setPreferences({
                        ...preferences,
                        notification_settings: {
                          ...preferences.notification_settings,
                          email: e.target.checked
                        }
                      })}
                    />
                    <span className="switch"></span>
                    {t('emailNotifications')}
                  </label>
                  
                  <label className="switch-label">
                    <input
                      type="checkbox"
                      checked={preferences.notification_settings.push}
                      onChange={(e) => setPreferences({
                        ...preferences,
                        notification_settings: {
                          ...preferences.notification_settings,
                          push: e.target.checked
                        }
                      })}
                    />
                    <span className="switch"></span>
                    {t('pushNotifications')}
                  </label>
                  
                  <label className="switch-label">
                    <input
                      type="checkbox"
                      checked={preferences.notification_settings.sound}
                      onChange={(e) => setPreferences({
                        ...preferences,
                        notification_settings: {
                          ...preferences.notification_settings,
                          sound: e.target.checked
                        }
                      })}
                    />
                    <span className="switch"></span>
                    {t('soundNotifications')}
                  </label>
                </div>
              </div>
            </div>
            
            <div className="preferences-actions">
              <button onClick={savePreferences} className="save-btn primary">
                💾 {t('savePreferences')}
              </button>
              
              {prefsStatus && (
                <div className={`status-message ${prefsStatus.includes('Error') ? 'error' : 'success'}`}>
                  {prefsStatus}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
      
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-info">
            <small>EmoIA V3 &copy; 2024 - {t('aiWithHeart')}</small>
          </div>
          <div className="footer-actions">
            <button
              className="toggle-btn"
              onClick={() => setShowInsights(!showInsights)}
            >
              {showInsights ? '🙈' : '👁️'} {t('toggleInsights')}
            </button>
            <button
              className="toggle-btn"
              onClick={() => setShowSuggestions(!showSuggestions)}
            >
              {showSuggestions ? '🚫' : '💡'} {t('toggleSuggestions')}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
