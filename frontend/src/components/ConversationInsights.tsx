import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';

interface InsightData {
  type: 'emotion' | 'topic' | 'suggestion' | 'pattern' | 'warning';
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

interface Props {
  userId: string;
  conversationId?: string;
  onInsightAction?: (insight: InsightData) => void;
}

const ConversationInsights: React.FC<Props> = ({ userId, conversationId, onInsightAction }) => {
  const { t } = useTranslation();
  const [insights, setInsights] = useState<InsightData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedInsight, setSelectedInsight] = useState<InsightData | null>(null);

  useEffect(() => {
    fetchInsights();
    const interval = setInterval(fetchInsights, 30000); // Mise à jour toutes les 30 secondes
    return () => clearInterval(interval);
  }, [userId, conversationId]);

  const fetchInsights = async () => {
    try {
      const response = await fetch(`http://localhost:8000/insights/${userId}${conversationId ? `/${conversationId}` : ''}`);
      if (response.ok) {
        const data = await response.json();
        setInsights(generateInsights(data));
      }
    } catch (error) {
      console.error('Erreur lors de la récupération des insights:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateInsights = (data: any): InsightData[] => {
    const insights: InsightData[] = [];

    // Insight émotionnel
    if (data.emotional_state) {
      const emotion = data.emotional_state.dominant_emotion;
      insights.push({
        type: 'emotion',
        title: t('emotionalStateInsight'),
        description: t('emotionalStateDescription', { emotion: t(emotion) }),
        confidence: data.emotional_state.confidence,
        icon: getEmotionIcon(emotion),
        color: getEmotionColor(emotion)
      });
    }

    // Patterns de conversation
    if (data.conversation_patterns) {
      data.conversation_patterns.forEach((pattern: any) => {
        insights.push({
          type: 'pattern',
          title: t('conversationPattern'),
          description: pattern.description,
          confidence: pattern.confidence,
          icon: '🔄',
          color: '#9b59b6'
        });
      });
    }

    // Suggestions intelligentes
    if (data.suggestions) {
      data.suggestions.forEach((suggestion: any) => {
        insights.push({
          type: 'suggestion',
          title: t('suggestion'),
          description: suggestion.text,
          confidence: suggestion.relevance,
          actionable: {
            text: t('applySuggestion'),
            action: () => onInsightAction?.(suggestion)
          },
          icon: '💡',
          color: '#f39c12'
        });
      });
    }

    // Topics détectés
    if (data.detected_topics) {
      insights.push({
        type: 'topic',
        title: t('detectedTopics'),
        description: data.detected_topics.join(', '),
        confidence: 0.8,
        icon: '🏷️',
        color: '#3498db'
      });
    }

    // Avertissements émotionnels
    if (data.emotional_warnings) {
      data.emotional_warnings.forEach((warning: any) => {
        insights.push({
          type: 'warning',
          title: t('emotionalWarning'),
          description: warning.message,
          confidence: warning.severity,
          icon: '⚠️',
          color: '#e74c3c'
        });
      });
    }

    return insights.sort((a, b) => b.confidence - a.confidence);
  };

  const getEmotionIcon = (emotion: string): string => {
    const icons: { [key: string]: string } = {
      joy: '😊',
      sadness: '😢',
      anger: '😠',
      fear: '😨',
      surprise: '😮',
      love: '❤️',
      anxiety: '😰',
      contentment: '😌',
      curiosity: '🤔',
      excitement: '🎉'
    };
    return icons[emotion] || '🎭';
  };

  const getEmotionColor = (emotion: string): string => {
    const colors: { [key: string]: string } = {
      joy: '#FFD93D',
      sadness: '#6495ED',
      anger: '#FF6B6B',
      fear: '#9370DB',
      surprise: '#FFB6C1',
      love: '#FF69B4',
      anxiety: '#DDA0DD',
      contentment: '#98FB98',
      curiosity: '#87CEEB',
      excitement: '#FFA500'
    };
    return colors[emotion] || '#808080';
  };

  const insightVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    exit: { opacity: 0, x: -100 }
  };

  return (
    <div className="conversation-insights">
      <h3>{t('conversationInsights')}</h3>
      
      {loading ? (
        <div className="insights-loading">
          <div className="loading-spinner"></div>
          <p>{t('analyzingConversation')}</p>
        </div>
      ) : (
        <AnimatePresence>
          <div className="insights-grid">
            {insights.map((insight, index) => (
              <motion.div
                key={`${insight.type}-${index}`}
                className={`insight-card ${insight.type}`}
                variants={insightVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedInsight(insight)}
                style={{
                  borderLeft: `4px solid ${insight.color || '#3498db'}`
                }}
              >
                <div className="insight-header">
                  <span className="insight-icon">{insight.icon}</span>
                  <h4>{insight.title}</h4>
                  <div className="confidence-badge">
                    {Math.round(insight.confidence * 100)}%
                  </div>
                </div>
                
                <p className="insight-description">{insight.description}</p>
                
                {insight.actionable && (
                  <button
                    className="insight-action"
                    onClick={(e) => {
                      e.stopPropagation();
                      insight.actionable!.action();
                    }}
                  >
                    {insight.actionable.text}
                  </button>
                )}
              </motion.div>
            ))}
          </div>
        </AnimatePresence>
      )}

      {/* Modal de détails */}
      <AnimatePresence>
        {selectedInsight && (
          <motion.div
            className="insight-modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedInsight(null)}
          >
            <motion.div
              className="insight-modal"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <h3>{selectedInsight.title}</h3>
              <p>{selectedInsight.description}</p>
              
              <div className="insight-details">
                <div className="detail-item">
                  <span className="label">{t('type')}:</span>
                  <span className="value">{t(selectedInsight.type)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">{t('confidence')}:</span>
                  <span className="value">{Math.round(selectedInsight.confidence * 100)}%</span>
                </div>
              </div>
              
              <button
                className="close-button"
                onClick={() => setSelectedInsight(null)}
              >
                {t('close')}
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ConversationInsights;