import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// Ressources de traduction
const resources = {
  en: {
    translation: {
      title: "EmoIA - Your Emotional AI",
      chatTab: "Chat",
      dashboardTab: "Dashboard",
      preferencesTab: "Preferences",
      userIdLabel: "User ID:",
      inputPlaceholder: "Type your message...",
      sendButton: "Send",
      wsConnected: "WebSocket Connected",
      wsDisconnected: "WebSocket Disconnected",
      typing: "EmoIA is typing...",
      fetchAnalytics: "Refresh",
      analyticsError: "Error connecting to API",
      you: "You",
      welcome: "Welcome to EmoIA! I'm here to listen.",
      languageLabel: "Language:",
      themeLabel: "Theme:",
      lightTheme: "Light",
      darkTheme: "Dark",
      savePreferences: "Save",
      preferencesSaved: "Preferences saved!",
      preferencesError: "Error saving preferences"
    }
  },
  fr: {
    translation: {
      title: "EmoIA - Votre IA Émotionnelle",
      chatTab: "Chat",
      dashboardTab: "Tableau de bord",
      preferencesTab: "Préférences",
      userIdLabel: "Identifiant:",
      inputPlaceholder: "Écrivez votre message...",
      sendButton: "Envoyer",
      wsConnected: "WebSocket Connecté",
      wsDisconnected: "WebSocket Déconnecté",
      typing: "EmoIA est en train d'écrire...",
      fetchAnalytics: "Rafraîchir",
      analyticsError: "Erreur de connexion à l'API",
      you: "Vous",
      welcome: "Bienvenue sur EmoIA ! Je suis là pour vous écouter.",
      languageLabel: "Langue:",
      themeLabel: "Thème:",
      lightTheme: "Clair",
      darkTheme: "Sombre",
      savePreferences: "Enregistrer",
      preferencesSaved: "Préférences enregistrées !",
      preferencesError: "Erreur lors de l'enregistrement",
      // Nouvelles clés pour le tableau de bord
      emotionTrends: "Tendances Émotionnelles",
      emotionDistribution: "Distribution des Émotions",
      loadingData: "Chargement des données...",
      realTimeUpdates: "Mises à jour en temps réel"
    }
  },
  es: {
    translation: {
      title: "EmoIA - Tu IA Emocional",
      chatTab: "Chat",
      dashboardTab: "Panel",
      preferencesTab: "Preferencias",
      userIdLabel: "ID de Usuario:",
      inputPlaceholder: "Escribe tu mensaje...",
      sendButton: "Enviar",
      wsConnected: "WebSocket Conectado",
      wsDisconnected: "WebSocket Desconectado",
      typing: "EmoIA está escribiendo...",
      fetchAnalytics: "Actualizar",
      analyticsError: "Error al conectar con la API",
      you: "Tú",
      welcome: "¡Bienvenido a EmoIA! Estoy aquí para escucharte.",
      languageLabel: "Idioma:",
      themeLabel: "Tema:",
      lightTheme: "Claro",
      darkTheme: "Oscuro",
      savePreferences: "Guardar",
      preferencesSaved: "¡Preferencias guardadas!",
      preferencesError: "Error al guardar",
      // Nouvelles clés pour le tableau de bord
      emotionTrends: "Tendencias Emocionales",
      emotionDistribution: "Distribución de Emociones",
      loadingData: "Cargando datos...",
      realTimeUpdates: "Actualizaciones en tiempo real"
    }
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: localStorage.getItem('emoia-lang') || 'fr',
    fallbackLng: 'fr',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;