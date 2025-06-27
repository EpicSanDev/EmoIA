import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';

interface Suggestion {
  id: string;
  text: string;
  type: 'response' | 'question' | 'action' | 'topic';
  confidence: number;
  metadata?: {
    emotion?: string;
    intent?: string;
    context?: string;
  };
}

interface Props {
  context: string;
  userInput?: string;
  emotionalState?: any;
  onSuggestionSelect: (suggestion: Suggestion) => void;
  maxSuggestions?: number;
}

const SmartSuggestions: React.FC<Props> = ({
  context,
  userInput,
  emotionalState,
  onSuggestionSelect,
  maxSuggestions = 5
}) => {
  const { t } = useTranslation();
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    if (context || userInput) {
      generateSuggestions();
    }
  }, [context, userInput, emotionalState]);

  const generateSuggestions = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          context,
          user_input: userInput,
          emotional_state: emotionalState,
          max_suggestions: maxSuggestions
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSuggestions(data.suggestions || generateLocalSuggestions());
      } else {
        setSuggestions(generateLocalSuggestions());
      }
    } catch (error) {
      console.error('Erreur lors de la génération des suggestions:', error);
      setSuggestions(generateLocalSuggestions());
    } finally {
      setLoading(false);
    }
  };

  const generateLocalSuggestions = (): Suggestion[] => {
    const suggestions: Suggestion[] = [];
    
    // Suggestions basées sur l'état émotionnel
    if (emotionalState) {
      const emotion = emotionalState.dominant_emotion;
      if (emotion === 'sadness') {
        suggestions.push({
          id: 'comfort-1',
          text: t('suggestionComfort1'),
          type: 'response',
          confidence: 0.9,
          metadata: { emotion: 'sadness', intent: 'comfort' }
        });
        suggestions.push({
          id: 'activity-1',
          text: t('suggestionActivity1'),
          type: 'action',
          confidence: 0.8,
          metadata: { emotion: 'sadness', intent: 'distract' }
        });
      } else if (emotion === 'joy') {
        suggestions.push({
          id: 'celebrate-1',
          text: t('suggestionCelebrate1'),
          type: 'response',
          confidence: 0.9,
          metadata: { emotion: 'joy', intent: 'celebrate' }
        });
      } else if (emotion === 'anxiety') {
        suggestions.push({
          id: 'calm-1',
          text: t('suggestionCalm1'),
          type: 'action',
          confidence: 0.9,
          metadata: { emotion: 'anxiety', intent: 'calm' }
        });
      }
    }

    // Suggestions contextuelles
    if (context) {
      if (context.toLowerCase().includes('travail') || context.toLowerCase().includes('work')) {
        suggestions.push({
          id: 'work-1',
          text: t('suggestionWork1'),
          type: 'topic',
          confidence: 0.7,
          metadata: { context: 'work' }
        });
      }
      
      if (context.toLowerCase().includes('stress')) {
        suggestions.push({
          id: 'stress-1',
          text: t('suggestionStress1'),
          type: 'action',
          confidence: 0.8,
          metadata: { context: 'stress' }
        });
      }
    }

    // Questions exploratoires
    suggestions.push({
      id: 'explore-1',
      text: t('suggestionExplore1'),
      type: 'question',
      confidence: 0.6,
      metadata: { intent: 'explore' }
    });

    return suggestions.sort((a, b) => b.confidence - a.confidence).slice(0, maxSuggestions);
  };

  const getTypeIcon = (type: string): string => {
    const icons: { [key: string]: string } = {
      response: '💬',
      question: '❓',
      action: '🎯',
      topic: '📌'
    };
    return icons[type] || '💡';
  };

  const getTypeColor = (type: string): string => {
    const colors: { [key: string]: string } = {
      response: '#3498db',
      question: '#9b59b6',
      action: '#2ecc71',
      topic: '#f39c12'
    };
    return colors[type] || '#95a5a6';
  };

  const categories = ['all', 'response', 'question', 'action', 'topic'];
  const filteredSuggestions = selectedCategory === 'all' 
    ? suggestions 
    : suggestions.filter(s => s.type === selectedCategory);

  return (
    <div className="smart-suggestions">
      <div className="suggestions-header">
        <h4>{t('smartSuggestions')}</h4>
        <div className="category-filters">
          {categories.map(category => (
            <button
              key={category}
              className={`category-btn ${selectedCategory === category ? 'active' : ''}`}
              onClick={() => setSelectedCategory(category)}
            >
              {t(`category_${category}`)}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="suggestions-loading">
          <div className="shimmer-wrapper">
            {[1, 2, 3].map(i => (
              <div key={i} className="shimmer-line" />
            ))}
          </div>
        </div>
      ) : (
        <AnimatePresence mode="wait">
          <motion.div
            className="suggestions-list"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {filteredSuggestions.length > 0 ? (
              filteredSuggestions.map((suggestion, index) => (
                <motion.div
                  key={suggestion.id}
                  className="suggestion-item"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => onSuggestionSelect(suggestion)}
                  style={{
                    borderLeft: `3px solid ${getTypeColor(suggestion.type)}`
                  }}
                >
                  <div className="suggestion-content">
                    <span className="suggestion-icon">{getTypeIcon(suggestion.type)}</span>
                    <span className="suggestion-text">{suggestion.text}</span>
                  </div>
                  <div className="suggestion-confidence">
                    <div
                      className="confidence-bar"
                      style={{
                        width: `${suggestion.confidence * 100}%`,
                        backgroundColor: getTypeColor(suggestion.type)
                      }}
                    />
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="no-suggestions">
                <p>{t('noSuggestions')}</p>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      )}

      <div className="suggestions-footer">
        <button
          className="refresh-btn"
          onClick={generateSuggestions}
          disabled={loading}
        >
          🔄 {t('refreshSuggestions')}
        </button>
      </div>
    </div>
  );
};

export default SmartSuggestions;