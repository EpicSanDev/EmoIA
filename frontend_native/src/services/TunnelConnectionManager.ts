/**
 * Gestionnaire de connexion tunnel pour EmoIA Native App
 * Gère la connexion sécurisée au serveur backend via tunnel
 */

import { io, Socket } from 'socket.io-client';

interface TunnelConfig {
  baseUrl: string;
  protocol: 'https' | 'http';
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  heartbeatInterval: number;
}

interface ApiEndpoints {
  chat: string;
  mcp: string;
  analytics: string;
  health: string;
  websocket: string;
}

class TunnelConnectionManager {
  private config: TunnelConfig;
  private socket: Socket | null = null;
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private statusChangeCallbacks: ((connected: boolean) => void)[] = [];
  private apiEndpoints: ApiEndpoints;

  constructor() {
    // Configuration par défaut
    this.config = {
      baseUrl: this.detectTunnelUrl(),
      protocol: 'https',
      timeout: 10000,
      retryAttempts: 5,
      retryDelay: 2000,
      heartbeatInterval: 30000
    };

    // Endpoints API
    this.apiEndpoints = {
      chat: '/chat',
      mcp: '/mcp/chat',
      analytics: '/analytics',
      health: '/health',
      websocket: '/ws/chat'
    };

    this.initializeEventListeners();
  }

  /**
   * Détecte automatiquement l'URL du tunnel depuis les variables d'environnement
   */
  private detectTunnelUrl(): string {
    // Vérifier les variables d'environnement
    if (process.env.REACT_APP_TUNNEL_URL) {
      return process.env.REACT_APP_TUNNEL_URL;
    }

    // URLs de tunnel par défaut pour développement
    const defaultUrls = [
      'https://emoia-ai.ngrok.io',
      'https://emoia-ai.loca.lt',
      'https://emoia-ai.cloudflare.com',
      'http://localhost:8000' // Fallback local
    ];

    // En production, utiliser l'URL configurée
    if (process.env.NODE_ENV === 'production') {
      return defaultUrls[0];
    }

    return defaultUrls[3]; // Local pour dev
  }

  /**
   * Initialise les listeners d'événements
   */
  private initializeEventListeners(): void {
    // Listener pour les changements de connectivité réseau
    window.addEventListener('online', () => {
      console.log('🌐 Réseau en ligne - Tentative de reconnexion');
      this.connect();
    });

    window.addEventListener('offline', () => {
      console.log('📱 Réseau hors ligne');
      this.disconnect();
    });

    // Listener pour la visibilité de l'app (mobile)
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        this.reconnectIfNeeded();
      }
    });
  }

  /**
   * Se connecte au serveur via tunnel
   */
  async connect(): Promise<boolean> {
    try {
      console.log(`🔌 Connexion au serveur EmoIA via ${this.config.baseUrl}`);

      // Vérifier d'abord si le serveur est accessible
      const isHealthy = await this.checkServerHealth();
      if (!isHealthy) {
        throw new Error('Serveur non accessible');
      }

      // Établir la connexion WebSocket
      await this.connectWebSocket();

      // Démarrer le heartbeat
      this.startHeartbeat();

      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.notifyStatusChange(true);

      console.log('✅ Connexion tunnel établie avec succès');
      return true;

    } catch (error) {
      console.error('❌ Erreur de connexion tunnel:', error);
      this.handleConnectionError();
      return false;
    }
  }

  /**
   * Vérifie la santé du serveur
   */
  private async checkServerHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.baseUrl}${this.apiEndpoints.health}`, {
        method: 'GET',
        timeout: this.config.timeout,
      });

      return response.ok;
    } catch (error) {
      console.error('Échec du health check:', error);
      return false;
    }
  }

  /**
   * Établit la connexion WebSocket
   */
  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `${this.config.baseUrl}${this.apiEndpoints.websocket}`;
      
      this.socket = io(wsUrl, {
        transports: ['websocket', 'polling'],
        timeout: this.config.timeout,
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: this.config.retryAttempts,
        reconnectionDelay: this.config.retryDelay
      });

      this.socket.on('connect', () => {
        console.log('🔗 WebSocket connecté');
        resolve();
      });

      this.socket.on('disconnect', (reason) => {
        console.log('🔌 WebSocket déconnecté:', reason);
        this.handleDisconnection();
      });

      this.socket.on('connect_error', (error) => {
        console.error('❌ Erreur WebSocket:', error);
        reject(error);
      });

      // Événements spécifiques à EmoIA
      this.socket.on('chat_response', (data) => {
        this.handleChatResponse(data);
      });

      this.socket.on('emotional_update', (data) => {
        this.handleEmotionalUpdate(data);
      });

      this.socket.on('system_notification', (data) => {
        this.handleSystemNotification(data);
      });
    });
  }

  /**
   * Déconnecte du serveur
   */
  disconnect(): void {
    console.log('🔌 Déconnexion du tunnel');

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.stopHeartbeat();
    this.isConnected = false;
    this.notifyStatusChange(false);
  }

  /**
   * Reconnecter si nécessaire
   */
  private async reconnectIfNeeded(): Promise<void> {
    if (!this.isConnected && navigator.onLine) {
      await this.connect();
    }
  }

  /**
   * Gère les erreurs de connexion
   */
  private handleConnectionError(): void {
    this.reconnectAttempts++;
    
    if (this.reconnectAttempts < this.config.retryAttempts) {
      console.log(`🔄 Tentative de reconnexion ${this.reconnectAttempts}/${this.config.retryAttempts} dans ${this.config.retryDelay}ms`);
      
      setTimeout(() => {
        this.connect();
      }, this.config.retryDelay * this.reconnectAttempts);
    } else {
      console.log('⚠️ Nombre maximum de tentatives de reconnexion atteint');
      this.isConnected = false;
      this.notifyStatusChange(false);
    }
  }

  /**
   * Gère la déconnexion
   */
  private handleDisconnection(): void {
    this.isConnected = false;
    this.stopHeartbeat();
    this.notifyStatusChange(false);
    
    // Tenter une reconnexion automatique
    setTimeout(() => {
      this.reconnectIfNeeded();
    }, this.config.retryDelay);
  }

  /**
   * Démarre le heartbeat pour maintenir la connexion
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.socket && this.socket.connected) {
        this.socket.emit('ping');
      } else {
        this.handleDisconnection();
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Arrête le heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Envoie un message chat via WebSocket
   */
  async sendChatMessage(message: string, userId: string, context?: any): Promise<void> {
    if (!this.socket || !this.isConnected) {
      throw new Error('Non connecté au serveur');
    }

    const messageData = {
      type: 'chat_message',
      message,
      user_id: userId,
      context: context || {},
      timestamp: new Date().toISOString()
    };

    this.socket.emit('chat_message', messageData);
  }

  /**
   * Envoie une requête MCP
   */
  async sendMCPRequest(message: string, provider: string, model: string, userId: string): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Non connecté au serveur');
    }

    try {
      const response = await fetch(`${this.config.baseUrl}${this.apiEndpoints.mcp}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          message,
          provider,
          model
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Erreur requête MCP:', error);
      throw error;
    }
  }

  /**
   * Gère les réponses de chat
   */
  private handleChatResponse(data: any): void {
    // Émettre l'événement pour les composants React
    window.dispatchEvent(new CustomEvent('emoia:chat_response', { detail: data }));
  }

  /**
   * Gère les mises à jour émotionnelles
   */
  private handleEmotionalUpdate(data: any): void {
    window.dispatchEvent(new CustomEvent('emoia:emotional_update', { detail: data }));
  }

  /**
   * Gère les notifications système
   */
  private handleSystemNotification(data: any): void {
    window.dispatchEvent(new CustomEvent('emoia:system_notification', { detail: data }));
  }

  /**
   * Ajoute un callback pour les changements de statut
   */
  onStatusChange(callback: (connected: boolean) => void): void {
    this.statusChangeCallbacks.push(callback);
  }

  /**
   * Notifie les changements de statut
   */
  private notifyStatusChange(connected: boolean): void {
    this.statusChangeCallbacks.forEach(callback => {
      try {
        callback(connected);
      } catch (error) {
        console.error('Erreur dans callback status change:', error);
      }
    });
  }

  /**
   * Obtient le statut de connexion actuel
   */
  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  /**
   * Obtient la configuration actuelle
   */
  getConfig(): TunnelConfig {
    return { ...this.config };
  }

  /**
   * Met à jour la configuration
   */
  updateConfig(newConfig: Partial<TunnelConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Nettoie les ressources
   */
  cleanup(): void {
    this.disconnect();
    this.statusChangeCallbacks = [];
    
    window.removeEventListener('online', this.connect);
    window.removeEventListener('offline', this.disconnect);
    document.removeEventListener('visibilitychange', this.reconnectIfNeeded);
  }
}

export default TunnelConnectionManager;