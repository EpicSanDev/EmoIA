import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement
} from 'chart.js';
import LanguageSwitcher from './components/LanguageSwitcher';
import AnalyticsDashboard from './components/AnalyticsDashboard';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement);

// Interfaces
interface Message {
  sender: 'user' | 'emoia';
  text: string;
  emotion?: any;
}

interface Preferences {
  language: string;
  theme: string;
  notification_settings: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
}

// Constants
const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8001/ws/chat';

function App() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<'chat' | 'dashboard' | 'preferences'>('chat');
  const [userId, setUserId] = useState<string>('demo-user');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [analytics, setAnalytics] = useState<any>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [preferences, setPreferences] = useState<Preferences>({
    language: 'fr',
    theme: 'light',
    notification_settings: {
      email: true,
      push: false,
      sound: true
    }
  });
  const [prefsStatus, setPrefsStatus] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const [lang, setLang] = useState<'en' | 'fr' | 'es'>(localStorage.getItem('emoia-lang') || 'fr');

  // Appliquer le thème
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', preferences.theme);
  }, [preferences.theme]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  // Charger les préférences au démarrage
  useEffect(() => {
    const fetchPreferences = async () => {
      try {
        const res = await fetch(`${API_URL}/utilisateur/preferences/${userId}`);
        if (res.ok) {
          const data = await res.json();
          setPreferences(data);
          // Mettre à jour la langue dans localStorage et i18next
          localStorage.setItem('emoia-lang', data.language);
        }
      } catch (e) {
        console.error("Erreur chargement préférences", e);
      }
    };

    fetchPreferences();
    setupWebSocket();
    setMessages([{ sender: 'emoia', text: t('welcome') }]);
  }, [userId, t]);

  const setupWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setTimeout(setupWebSocket, 3000); // Retry connection
    ws.onerror = () => setWsConnected(false);
    ws.onmessage = (event) => {
      setLoading(false);
      const data = JSON.parse(event.data);
      setMessages((msgs) => [...msgs, { sender: 'emoia', text: data.response, emotion: data.emotional_analysis }]);
    };
    wsRef.current = ws;
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage: Message = { sender: 'user', text: input };
    setMessages((msgs) => [...msgs, userMessage]);
    setInput('');
    setLoading(true);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ user_id: userId, message: input }));
    } else {
      // Fallback to HTTP if WS is not open
      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, message: input }),
        });
        const data = await res.json();
        setMessages((msgs) => [...msgs, { sender: 'emoia', text: data.response, emotion: data.emotional_analysis }]);
      } catch (e) {
        setMessages((msgs) => [...msgs, { sender: 'emoia', text: `[${t('analyticsError')}]` }]);
      }
      setLoading(false);
    }
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

  const savePreferences = async () => {
    try {
      const res = await fetch(`${API_URL}/utilisateur/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          language: preferences.language,
          theme: preferences.theme,
          notification_settings: preferences.notification_settings
        })
      });
      
      if (res.ok) {
        setPrefsStatus(t('preferencesSaved'));
        // Mettre à jour la langue dans localStorage et i18next
        localStorage.setItem('emoia-lang', preferences.language);
        setTimeout(() => setPrefsStatus(null), 3000);
      } else {
        setPrefsStatus(t('preferencesError'));
      }
    } catch (e) {
      setPrefsStatus(t('preferencesError'));
    }
  };

  // Chart data preparation
  let lineData, pieData;
  if (analytics && analytics.trends) {
    // ... (Chart data logic can be improved or kept as is)
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>{t('title')}</h1>
        <div className="controls">
          <LanguageSwitcher />
          <nav>
            <button onClick={() => setTab('chat')} className={tab === 'chat' ? 'active' : ''}>{t('chatTab')}</button>
            <button onClick={() => { setTab('dashboard'); fetchAnalytics(); }} className={tab === 'dashboard' ? 'active' : ''}>{t('dashboardTab')}</button>
            <button onClick={() => setTab('preferences')} className={tab === 'preferences' ? 'active' : ''}>{t('preferencesTab')}</button>
          </nav>
        </div>
      </header>

      <main>
        {tab === 'chat' && (
          <div className="chat-container">
            <div className="messages-area">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.sender}`}>
                  <div className="bubble">{msg.text}</div>
                </div>
              ))}
              {loading && (
                <div className="message emoia">
                  <div className="bubble"><i>{t('typing')}</i></div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <div className="input-area">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                placeholder={t('inputPlaceholder')}
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !input.trim()}>{t('sendButton')}</button>
            </div>
          </div>
        )}

        {tab === 'dashboard' && (
          <AnalyticsDashboard userId={userId} />
        )}

        {tab === 'preferences' && (
          <div className="preferences-container">
            <h2>{t('preferencesTab')}</h2>
            <div className="preferences-form">
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
                    className={preferences.theme === 'light' ? 'active' : ''}
                    onClick={() => setPreferences({...preferences, theme: 'light'})}
                  >
                    {t('lightTheme')}
                  </button>
                  <button
                    className={preferences.theme === 'dark' ? 'active' : ''}
                    onClick={() => setPreferences({...preferences, theme: 'dark'})}
                  >
                    {t('darkTheme')}
                  </button>
                </div>
              </div>
              
              <button onClick={savePreferences} className="save-btn">
                {t('savePreferences')}
              </button>
              
              {prefsStatus && <div className="status-message">{prefsStatus}</div>}
            </div>
          </div>
        )}
      </main>
      <footer>
        <small>EmoIA V3 &copy; 2024</small>
      </footer>
    </div>
  );
}

export default App;
