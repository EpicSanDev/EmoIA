import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Ressources de traduction
const resources = {
  en: {
    translation: {
      // Application
      title: "EmoIA - Emotional AI Assistant",
      welcome: "Hello! I'm EmoIA, your emotional AI companion. How can I help you today?",
      aiWithHeart: "AI with Heart",
      
      // Navigation
      chatTab: "Chat",
      dashboardTab: "Analytics",
      preferencesTab: "Settings",
      insightsTab: "Insights",
      
      // Chat
      inputPlaceholder: "Type your message...",
      sendButton: "Send",
      typing: "EmoIA is typing",
      recording: "Recording...",
      startRecording: "Start recording",
      stopRecording: "Stop recording",
      
      // Status
      connected: "Connected",
      disconnected: "Disconnected",
      reconnecting: "Reconnecting...",
      loading: "Loading",
      
      // Emotions
      joy: "Joy",
      sadness: "Sadness",
      anger: "Anger",
      fear: "Fear",
      surprise: "Surprise",
      love: "Love",
      excitement: "Excitement",
      anxiety: "Anxiety",
      contentment: "Contentment",
      curiosity: "Curiosity",
      disgust: "Disgust",
      
      // Dashboard
      status: "Status",
      totalInteractions: "Total Interactions",
      period: "Period",
      emotionalState: "Emotional State",
      dominantEmotion: "Dominant Emotion",
      emotionalStability: "Emotional Stability",
      positivityRatio: "Positivity Ratio",
      emotionDistribution: "Emotion Distribution",
      emotionalTrend: "Emotional Trend",
      interactionFrequency: "Interaction Frequency",
      recommendations: "Recommendations",
      quickActions: "Quick Actions",
      refresh: "Refresh",
      exportData: "Export Data",
      notifications: "Notifications",
      positiveEmotions: "Positive Emotions",
      negativeEmotions: "Negative Emotions",
      interactions: "Interactions",
      
      // Insights
      currentEmotions: "Current Emotions",
      personalityProfile: "Personality Profile",
      moodHistory: "Mood History",
      emotionalBalance: "Emotional Balance",
      conversationInsights: "Conversation Insights",
      emotionalStateInsight: "Emotional State",
      emotionalStateDescription: "You seem to be feeling {{emotion}}",
      conversationPattern: "Conversation Pattern",
      suggestion: "Suggestion",
      applySuggestion: "Apply",
      detectedTopics: "Detected Topics",
      emotionalWarning: "Emotional Warning",
      analyzingConversation: "Analyzing conversation...",
      type: "Type",
      confidence: "Confidence",
      close: "Close",
      
      // Mood History
      emotionalValence: "Emotional Valence",
      arousalLevel: "Arousal Level",
             emotionIntensity: "Emotion Intensity",
       moodHistoryTitle: "Mood Evolution",
       emotionLegend: "Emotion Legend",
      
      // Smart Suggestions
      smartSuggestions: "Smart Suggestions",
      category_all: "All",
      category_response: "Responses",
      category_question: "Questions",
      category_action: "Actions",
      category_topic: "Topics",
      noSuggestions: "No suggestions available",
      refreshSuggestions: "Refresh",
      
      // Suggestion content
      suggestionComfort1: "Would you like to talk about what's bothering you?",
      suggestionActivity1: "Maybe a short walk or some music could help?",
      suggestionCelebrate1: "That's wonderful! Tell me more about what makes you happy!",
      suggestionCalm1: "Let's try a breathing exercise together",
      suggestionWork1: "How's your work-life balance lately?",
      suggestionStress1: "Have you tried any stress management techniques?",
      suggestionExplore1: "What else is on your mind?",
      
      // Emotion interactions
      emotionClickJoy: "I noticed you're feeling joyful! That's wonderful!",
      emotionClickSadness: "I see you're feeling sad. I'm here to listen.",
      emotionClickAnger: "I understand you're feeling angry. Would you like to talk about it?",
      emotionClickFear: "I notice some fear. You're safe here to express yourself.",
      emotionClickLove: "Love is such a beautiful emotion! Tell me more.",
      emotionClickDefault: "I see you're interested in {{emotion}}. How does it relate to you?",
      
      // Settings
      generalSettings: "General Settings",
      aiSettings: "AI Settings",
      languageLabel: "Language",
      themeLabel: "Theme",
      lightTheme: "Light",
      darkTheme: "Dark",
      personalityStyle: "Personality Style",
      professional: "Professional",
      friendly: "Friendly",
      casual: "Casual",
      empathetic: "Empathetic",
      responseLength: "Response Length",
      concise: "Concise",
      balanced: "Balanced",
      detailed: "Detailed",
      emotionalIntelligence: "Emotional Intelligence Level",
      emailNotifications: "Email Notifications",
      pushNotifications: "Push Notifications",
      soundNotifications: "Sound Notifications",
      savePreferences: "Save Settings",
      preferencesSaved: "Settings saved successfully!",
      preferencesError: "Error saving settings",
      
      // Toggle buttons
      toggleInsights: "Toggle Insights",
      toggleSuggestions: "Toggle Suggestions",
      
      // Errors
      errorMessage: "Sorry, an error occurred. Please try again.",
      analyticsError: "Error loading analytics"
    }
  },
  fr: {
    translation: {
      // Application
      title: "EmoIA - Assistant IA Émotionnel",
      welcome: "Bonjour ! Je suis EmoIA, votre compagnon IA émotionnel. Comment puis-je vous aider aujourd'hui ?",
      aiWithHeart: "L'IA avec du Cœur",
      
      // Navigation
      chatTab: "Discussion",
      dashboardTab: "Analytiques",
      preferencesTab: "Paramètres",
      insightsTab: "Aperçus",
      
      // Chat
      inputPlaceholder: "Tapez votre message...",
      sendButton: "Envoyer",
      typing: "EmoIA écrit",
      recording: "Enregistrement...",
      startRecording: "Commencer l'enregistrement",
      stopRecording: "Arrêter l'enregistrement",
      
      // Status
      connected: "Connecté",
      disconnected: "Déconnecté",
      reconnecting: "Reconnexion...",
      loading: "Chargement",
      
      // Emotions
      joy: "Joie",
      sadness: "Tristesse",
      anger: "Colère",
      fear: "Peur",
      surprise: "Surprise",
      love: "Amour",
      excitement: "Excitation",
      anxiety: "Anxiété",
      contentment: "Contentement",
      curiosity: "Curiosité",
      disgust: "Dégoût",
      
      // Dashboard
      status: "Statut",
      totalInteractions: "Total des interactions",
      period: "Période",
      emotionalState: "État émotionnel",
      dominantEmotion: "Émotion dominante",
      emotionalStability: "Stabilité émotionnelle",
      positivityRatio: "Ratio de positivité",
      emotionDistribution: "Distribution des émotions",
      emotionalTrend: "Tendance émotionnelle",
      interactionFrequency: "Fréquence d'interaction",
      recommendations: "Recommandations",
      quickActions: "Actions rapides",
      refresh: "Actualiser",
      exportData: "Exporter les données",
      notifications: "Notifications",
      positiveEmotions: "Émotions positives",
      negativeEmotions: "Émotions négatives",
      interactions: "Interactions",
      
      // Insights
      currentEmotions: "Émotions actuelles",
      personalityProfile: "Profil de personnalité",
      moodHistory: "Historique d'humeur",
      emotionalBalance: "Équilibre émotionnel",
      conversationInsights: "Aperçus de conversation",
      emotionalStateInsight: "État émotionnel",
      emotionalStateDescription: "Vous semblez ressentir de la {{emotion}}",
      conversationPattern: "Modèle de conversation",
      suggestion: "Suggestion",
      applySuggestion: "Appliquer",
      detectedTopics: "Sujets détectés",
      emotionalWarning: "Alerte émotionnelle",
      analyzingConversation: "Analyse de la conversation...",
      type: "Type",
      confidence: "Confiance",
      close: "Fermer",
      
      // Mood History
      emotionalValence: "Valence émotionnelle",
      arousalLevel: "Niveau d'activation",
             emotionIntensity: "Intensité émotionnelle",
       moodHistoryTitle: "Évolution de l'humeur",
       emotionLegend: "Légende des émotions",
      
      // Smart Suggestions
      smartSuggestions: "Suggestions intelligentes",
      category_all: "Tout",
      category_response: "Réponses",
      category_question: "Questions",
      category_action: "Actions",
      category_topic: "Sujets",
      noSuggestions: "Aucune suggestion disponible",
      refreshSuggestions: "Actualiser",
      
      // Suggestion content
      suggestionComfort1: "Voulez-vous parler de ce qui vous préoccupe ?",
      suggestionActivity1: "Peut-être qu'une courte promenade ou de la musique pourrait aider ?",
      suggestionCelebrate1: "C'est merveilleux ! Dites-m'en plus sur ce qui vous rend heureux !",
      suggestionCalm1: "Essayons un exercice de respiration ensemble",
      suggestionWork1: "Comment va votre équilibre travail-vie personnelle ?",
      suggestionStress1: "Avez-vous essayé des techniques de gestion du stress ?",
      suggestionExplore1: "Qu'est-ce qui vous préoccupe d'autre ?",
      
      // Emotion interactions
      emotionClickJoy: "Je remarque que vous vous sentez joyeux ! C'est merveilleux !",
      emotionClickSadness: "Je vois que vous êtes triste. Je suis là pour vous écouter.",
      emotionClickAnger: "Je comprends que vous êtes en colère. Voulez-vous en parler ?",
      emotionClickFear: "Je remarque de la peur. Vous êtes en sécurité ici pour vous exprimer.",
      emotionClickLove: "L'amour est une si belle émotion ! Dites-m'en plus.",
      emotionClickDefault: "Je vois que vous vous intéressez à {{emotion}}. Comment cela vous concerne-t-il ?",
      
      // Settings
      generalSettings: "Paramètres généraux",
      aiSettings: "Paramètres IA",
      languageLabel: "Langue",
      themeLabel: "Thème",
      lightTheme: "Clair",
      darkTheme: "Sombre",
      personalityStyle: "Style de personnalité",
      professional: "Professionnel",
      friendly: "Amical",
      casual: "Décontracté",
      empathetic: "Empathique",
      responseLength: "Longueur des réponses",
      concise: "Concis",
      balanced: "Équilibré",
      detailed: "Détaillé",
      emotionalIntelligence: "Niveau d'intelligence émotionnelle",
      emailNotifications: "Notifications par email",
      pushNotifications: "Notifications push",
      soundNotifications: "Notifications sonores",
      savePreferences: "Enregistrer les paramètres",
      preferencesSaved: "Paramètres enregistrés avec succès !",
      preferencesError: "Erreur lors de l'enregistrement des paramètres",
      
      // Toggle buttons
      toggleInsights: "Afficher/Masquer les aperçus",
      toggleSuggestions: "Afficher/Masquer les suggestions",
      
      // Errors
      errorMessage: "Désolé, une erreur s'est produite. Veuillez réessayer.",
      analyticsError: "Erreur lors du chargement des analytiques"
    }
  },
  es: {
    translation: {
      // Application
      title: "EmoIA - Asistente IA Emocional",
      welcome: "¡Hola! Soy EmoIA, tu compañero de IA emocional. ¿Cómo puedo ayudarte hoy?",
      aiWithHeart: "IA con Corazón",
      
      // Navigation
      chatTab: "Chat",
      dashboardTab: "Analíticas",
      preferencesTab: "Configuración",
      insightsTab: "Perspectivas",
      
      // Chat
      inputPlaceholder: "Escribe tu mensaje...",
      sendButton: "Enviar",
      typing: "EmoIA está escribiendo",
      recording: "Grabando...",
      startRecording: "Iniciar grabación",
      stopRecording: "Detener grabación",
      
      // Status
      connected: "Conectado",
      disconnected: "Desconectado",
      reconnecting: "Reconectando...",
      loading: "Cargando",
      
      // Emotions
      joy: "Alegría",
      sadness: "Tristeza",
      anger: "Ira",
      fear: "Miedo",
      surprise: "Sorpresa",
      love: "Amor",
      excitement: "Emoción",
      anxiety: "Ansiedad",
      contentment: "Satisfacción",
      curiosity: "Curiosidad",
      disgust: "Disgusto",
      
      // Dashboard
      status: "Estado",
      totalInteractions: "Total de interacciones",
      period: "Período",
      emotionalState: "Estado emocional",
      dominantEmotion: "Emoción dominante",
      emotionalStability: "Estabilidad emocional",
      positivityRatio: "Ratio de positividad",
      emotionDistribution: "Distribución de emociones",
      emotionalTrend: "Tendencia emocional",
      interactionFrequency: "Frecuencia de interacción",
      recommendations: "Recomendaciones",
      quickActions: "Acciones rápidas",
      refresh: "Actualizar",
      exportData: "Exportar datos",
      notifications: "Notificaciones",
      positiveEmotions: "Emociones positivas",
      negativeEmotions: "Emociones negativas",
      interactions: "Interacciones",
      
      // Insights
      currentEmotions: "Emociones actuales",
      personalityProfile: "Perfil de personalidad",
      moodHistory: "Historial de ánimo",
      emotionalBalance: "Balance emocional",
      conversationInsights: "Perspectivas de conversación",
      emotionalStateInsight: "Estado emocional",
      emotionalStateDescription: "Pareces estar sintiendo {{emotion}}",
      conversationPattern: "Patrón de conversación",
      suggestion: "Sugerencia",
      applySuggestion: "Aplicar",
      detectedTopics: "Temas detectados",
      emotionalWarning: "Alerta emocional",
      analyzingConversation: "Analizando conversación...",
      type: "Tipo",
      confidence: "Confianza",
      close: "Cerrar",
      
      // Mood History
      emotionalValence: "Valencia emocional",
      arousalLevel: "Nivel de activación",
             emotionIntensity: "Intensidad emocional",
       moodHistoryTitle: "Evolución del ánimo",
       emotionLegend: "Leyenda de emociones",
      
      // Smart Suggestions
      smartSuggestions: "Sugerencias inteligentes",
      category_all: "Todo",
      category_response: "Respuestas",
      category_question: "Preguntas",
      category_action: "Acciones",
      category_topic: "Temas",
      noSuggestions: "No hay sugerencias disponibles",
      refreshSuggestions: "Actualizar",
      
      // Suggestion content
      suggestionComfort1: "¿Te gustaría hablar sobre lo que te preocupa?",
      suggestionActivity1: "¿Tal vez un paseo corto o algo de música podría ayudar?",
      suggestionCelebrate1: "¡Eso es maravilloso! ¡Cuéntame más sobre lo que te hace feliz!",
      suggestionCalm1: "Intentemos un ejercicio de respiración juntos",
      suggestionWork1: "¿Cómo está tu equilibrio trabajo-vida últimamente?",
      suggestionStress1: "¿Has probado alguna técnica de manejo del estrés?",
      suggestionExplore1: "¿Qué más tienes en mente?",
      
      // Emotion interactions
      emotionClickJoy: "¡Noto que te sientes alegre! ¡Eso es maravilloso!",
      emotionClickSadness: "Veo que estás triste. Estoy aquí para escucharte.",
      emotionClickAnger: "Entiendo que estás enojado. ¿Quieres hablar sobre ello?",
      emotionClickFear: "Noto algo de miedo. Estás seguro aquí para expresarte.",
      emotionClickLove: "¡El amor es una emoción tan hermosa! Cuéntame más.",
      emotionClickDefault: "Veo que estás interesado en {{emotion}}. ¿Cómo se relaciona contigo?",
      
      // Settings
      generalSettings: "Configuración general",
      aiSettings: "Configuración IA",
      languageLabel: "Idioma",
      themeLabel: "Tema",
      lightTheme: "Claro",
      darkTheme: "Oscuro",
      personalityStyle: "Estilo de personalidad",
      professional: "Profesional",
      friendly: "Amigable",
      casual: "Casual",
      empathetic: "Empático",
      responseLength: "Longitud de respuesta",
      concise: "Conciso",
      balanced: "Equilibrado",
      detailed: "Detallado",
      emotionalIntelligence: "Nivel de inteligencia emocional",
      emailNotifications: "Notificaciones por email",
      pushNotifications: "Notificaciones push",
      soundNotifications: "Notificaciones sonoras",
      savePreferences: "Guardar configuración",
      preferencesSaved: "¡Configuración guardada exitosamente!",
      preferencesError: "Error al guardar la configuración",
      
      // Toggle buttons
      toggleInsights: "Mostrar/Ocultar perspectivas",
      toggleSuggestions: "Mostrar/Ocultar sugerencias",
      
      // Errors
      errorMessage: "Lo siento, ocurrió un error. Por favor intenta de nuevo.",
      analyticsError: "Error al cargar las analíticas"
    }
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'fr',
    lng: localStorage.getItem('emoia-lang') || 'fr',
    
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage']
    },

    interpolation: {
      escapeValue: false
    }
  });

export default i18n;