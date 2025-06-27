// Types pour EmoIA Frontend

export interface Message {
  id: string;
  sender: 'user' | 'emoia';
  text: string;
  emotion?: EmotionalAnalysis;
  timestamp: Date;
  audioUrl?: string;
  confidence?: number;
}

export interface EmotionalAnalysis {
  dominant_emotion: EmotionType;
  emotion_scores: { [key in EmotionType]?: number };
  valence: number; // -1 to 1 (negative to positive)
  arousal: number; // 0 to 1 (calm to excited)
  confidence: number;
}

export type EmotionType = 
  | 'joy'
  | 'sadness'
  | 'anger'
  | 'fear'
  | 'surprise'
  | 'disgust'
  | 'love'
  | 'excitement'
  | 'anxiety'
  | 'contentment'
  | 'curiosity';

export interface EmotionData {
  emotion: string;
  value: number;
  color: string;
  icon: string;
}

export interface PersonalityProfile {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
  emotional_intelligence: number;
  empathy_level: number;
  creativity: number;
  humor_appreciation?: number;
  optimism?: number;
}

export interface MoodPoint {
  timestamp: string;
  valence: number;
  arousal: number;
  dominantEmotion: EmotionType;
  emotionIntensity: number;
}

export interface Preferences {
  language: 'fr' | 'en' | 'es';
  theme: 'light' | 'dark';
  notification_settings: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
  ai_settings?: {
    personality_style: PersonalityStyle;
    response_length: ResponseLength;
    emotional_intelligence_level: number;
  };
}

export type PersonalityStyle = 'professional' | 'friendly' | 'casual' | 'empathetic';
export type ResponseLength = 'concise' | 'detailed' | 'balanced';

export interface Suggestion {
  id: string;
  text: string;
  type: SuggestionType;
  confidence: number;
  metadata?: {
    emotion?: EmotionType;
    intent?: string;
    context?: string;
  };
}

export type SuggestionType = 'response' | 'question' | 'action' | 'topic';

export interface InsightData {
  type: InsightType;
  title: string;
  description: string;
  confidence: number;
  actionable?: {
    text: string;
    action: () => void;
  };
  icon?: string;
  color?: string;
}

export type InsightType = 'emotion' | 'topic' | 'suggestion' | 'pattern' | 'warning';

export interface AnalyticsData {
  trends?: {
    most_frequent_emotion: EmotionType;
    emotional_stability: number;
    positive_ratio: number;
    emotional_timeline?: Array<[string, any]>;
  };
  total_interactions?: number;
  period_analyzed?: string;
  recommendations?: string[];
  error?: string;
}

export interface WebSocketMessage {
  type: WSMessageType;
  user_id?: string;
  message?: string;
  context?: any;
  data?: any;
}

export type WSMessageType = 
  | 'identify'
  | 'chat_message'
  | 'chat_response'
  | 'emotional_update'
  | 'insight_update'
  | 'error';

export interface ChatResponse {
  response: string;
  emotional_analysis: EmotionalAnalysis;
  personality_insights: PersonalityProfile;
  interaction_metadata: {
    response_time: number;
    model_used: string;
    context_length: number;
  };
  system_info: {
    version: string;
    language: string;
  };
  confidence?: number;
}