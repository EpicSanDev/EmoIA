import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import './TelegramConfig.css';

interface TelegramStatus {
  enabled: boolean;
  connected_users: number;
  bot_username?: string;
}

interface TelegramUser {
  telegram_id: string;
  emoia_id: string;
  display_name: string;
  last_active?: string;
}

const TelegramConfig: React.FC = () => {
  const { t } = useTranslation();
  const [config, setConfig] = useState({
    bot_token: '',
    enabled: false,
    notification_types: ['proactive', 'reminders', 'insights'],
    auto_respond: false
  });
  
  const [status, setStatus] = useState<TelegramStatus>({
    enabled: false,
    connected_users: 0
  });
  
  const [users, setUsers] = useState<TelegramUser[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info', text: string } | null>(null);
  const [showToken, setShowToken] = useState(false);

  useEffect(() => {
    fetchStatus();
    fetchUsers();
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/telegram/status');
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Error fetching Telegram status:', error);
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await fetch('/api/telegram/users');
      const data = await response.json();
      setUsers(data.users || []);
    } catch (error) {
      console.error('Error fetching Telegram users:', error);
    }
  };

  const handleConfigSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch('/api/telegram/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessage({ type: 'success', text: result.message });
        await fetchStatus();
        await fetchUsers();
      } else {
        setMessage({ type: 'error', text: result.message });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Erreur lors de la configuration' });
    } finally {
      setLoading(false);
    }
  };

  const handleNotificationTypeToggle = (type: string) => {
    setConfig(prev => ({
      ...prev,
      notification_types: prev.notification_types.includes(type)
        ? prev.notification_types.filter(t => t !== type)
        : [...prev.notification_types, type]
    }));
  };

  const getStatusIcon = () => {
    if (status.enabled) {
      return <span className="status-icon connected">🟢</span>;
    }
    return <span className="status-icon disconnected">🔴</span>;
  };

  const formatLastActive = (dateString?: string) => {
    if (!dateString) return 'Jamais';
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Il y a moins d\'une heure';
    if (diffInHours < 24) return `Il y a ${diffInHours}h`;
    return `Il y a ${Math.floor(diffInHours / 24)} jours`;
  };

  return (
    <div className="telegram-config">
      <div className="config-header">
        <h2>
          <span className="telegram-icon">📱</span>
          Configuration Telegram
        </h2>
        <div className="status-indicator">
          {getStatusIcon()}
          <span className="status-text">
            {status.enabled ? `Connecté (${status.connected_users} utilisateurs)` : 'Déconnecté'}
          </span>
          {status.bot_username && (
            <span className="bot-username">@{status.bot_username}</span>
          )}
        </div>
      </div>

      {message && (
        <div className={`message ${message.type}`}>
          <span className="message-icon">
            {message.type === 'success' ? '✅' : message.type === 'error' ? '❌' : 'ℹ️'}
          </span>
          {message.text}
        </div>
      )}

      <div className="config-content">
        <div className="config-section">
          <h3>Configuration du Bot</h3>
          <form onSubmit={handleConfigSubmit} className="config-form">
            <div className="form-group">
              <label htmlFor="bot_token">
                Token du Bot Telegram
                <span className="required">*</span>
              </label>
              <div className="token-input-group">
                <input
                  type={showToken ? "text" : "password"}
                  id="bot_token"
                  value={config.bot_token}
                  onChange={(e) => setConfig(prev => ({ ...prev, bot_token: e.target.value }))}
                  placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
                  required
                />
                <button
                  type="button"
                  className="show-token-btn"
                  onClick={() => setShowToken(!showToken)}
                >
                  {showToken ? '🙈' : '👁️'}
                </button>
              </div>
              <small className="form-help">
                Obtenez votre token auprès de @BotFather sur Telegram
              </small>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={config.enabled}
                  onChange={(e) => setConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                />
                <span className="checkbox-custom"></span>
                Activer le bot Telegram
              </label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={config.auto_respond}
                  onChange={(e) => setConfig(prev => ({ ...prev, auto_respond: e.target.checked }))}
                />
                <span className="checkbox-custom"></span>
                Réponses automatiques
              </label>
              <small className="form-help">
                EmoIA répondra automatiquement aux messages sans attendre votre validation
              </small>
            </div>

            <div className="form-group">
              <label>Types de notifications</label>
              <div className="notification-types">
                {[
                  { id: 'proactive', label: 'Messages proactifs', icon: '🤖' },
                  { id: 'reminders', label: 'Rappels', icon: '⏰' },
                  { id: 'insights', label: 'Insights émotionnels', icon: '💡' }
                ].map(type => (
                  <label key={type.id} className="checkbox-label notification-type">
                    <input
                      type="checkbox"
                      checked={config.notification_types.includes(type.id)}
                      onChange={() => handleNotificationTypeToggle(type.id)}
                    />
                    <span className="checkbox-custom"></span>
                    <span className="type-icon">{type.icon}</span>
                    {type.label}
                  </label>
                ))}
              </div>
            </div>

            <div className="form-actions">
              <button
                type="submit"
                className="submit-btn"
                disabled={loading || !config.bot_token}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner">⏳</span>
                    Configuration en cours...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🚀</span>
                    {config.enabled ? 'Mettre à jour' : 'Activer le bot'}
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        <div className="users-section">
          <h3>
            Utilisateurs connectés
            <span className="users-count">({users.length})</span>
          </h3>
          
          {users.length === 0 ? (
            <div className="no-users">
              <span className="no-users-icon">👥</span>
              <p>Aucun utilisateur connecté</p>
              <small>
                Les utilisateurs apparaîtront ici après avoir utilisé la commande /register sur Telegram
              </small>
            </div>
          ) : (
            <div className="users-list">
              {users.map(user => (
                <div key={user.telegram_id} className="user-card">
                  <div className="user-info">
                    <div className="user-avatar">
                      {user.display_name.charAt(0).toUpperCase()}
                    </div>
                    <div className="user-details">
                      <h4>{user.display_name}</h4>
                      <p className="user-id">ID: {user.emoia_id}</p>
                      <p className="last-active">
                        Dernière activité: {formatLastActive(user.last_active)}
                      </p>
                    </div>
                  </div>
                  <div className="user-actions">
                    <button className="action-btn message-btn" title="Envoyer un message">
                      💬
                    </button>
                    <button className="action-btn profile-btn" title="Voir le profil">
                      👤
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="config-instructions">
        <h3>Instructions d'utilisation</h3>
        <div className="instructions-grid">
          <div className="instruction-card">
            <span className="instruction-icon">1️⃣</span>
            <h4>Créer un bot</h4>
            <p>Contactez @BotFather sur Telegram et créez un nouveau bot avec /newbot</p>
          </div>
          <div className="instruction-card">
            <span className="instruction-icon">2️⃣</span>
            <h4>Configurer le token</h4>
            <p>Copiez le token fourni par BotFather et collez-le dans le champ ci-dessus</p>
          </div>
          <div className="instruction-card">
            <span className="instruction-icon">3️⃣</span>
            <h4>Activer le bot</h4>
            <p>Cochez "Activer le bot Telegram" et cliquez sur "Activer le bot"</p>
          </div>
          <div className="instruction-card">
            <span className="instruction-icon">4️⃣</span>
            <h4>Commencer à utiliser</h4>
            <p>Les utilisateurs peuvent maintenant utiliser /start sur votre bot</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TelegramConfig;