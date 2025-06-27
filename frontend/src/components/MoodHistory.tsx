import React from 'react';
import { Line } from 'react-chartjs-2';
import { format, parseISO } from 'date-fns';
import { fr, enUS, es } from 'date-fns/locale';
import { useTranslation } from 'react-i18next';

interface MoodPoint {
  timestamp: string;
  valence: number; // -1 à 1 (négatif à positif)
  arousal: number; // 0 à 1 (calme à excité)
  dominantEmotion: string;
  emotionIntensity: number;
}

interface Props {
  history: MoodPoint[];
  period?: 'day' | 'week' | 'month';
}

const MoodHistory: React.FC<Props> = ({ history, period = 'week' }) => {
  const { t, i18n } = useTranslation();

  const getLocale = () => {
    switch (i18n.language) {
      case 'fr': return fr;
      case 'es': return es;
      default: return enUS;
    }
  };

  const formatDate = (timestamp: string) => {
    const date = parseISO(timestamp);
    switch (period) {
      case 'day':
        return format(date, 'HH:mm', { locale: getLocale() });
      case 'week':
        return format(date, 'EEE dd', { locale: getLocale() });
      case 'month':
        return format(date, 'dd MMM', { locale: getLocale() });
      default:
        return format(date, 'dd/MM', { locale: getLocale() });
    }
  };

  const getEmotionColor = (emotion: string) => {
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
      curiosity: '#87CEEB'
    };
    return colors[emotion] || '#808080';
  };

  const data = {
    labels: history.map(point => formatDate(point.timestamp)),
    datasets: [
      {
        label: t('emotionalValence'),
        data: history.map(point => point.valence * 50 + 50), // Convertir -1/1 en 0/100
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.4,
        borderWidth: 2
      },
      {
        label: t('arousalLevel'),
        data: history.map(point => point.arousal * 100),
        borderColor: 'rgba(255, 159, 64, 1)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        tension: 0.4,
        borderWidth: 2
      },
      {
        label: t('emotionIntensity'),
        data: history.map(point => point.emotionIntensity * 100),
        borderColor: 'rgba(153, 102, 255, 1)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        tension: 0.4,
        borderWidth: 2
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false
    },
    plugins: {
      title: {
        display: true,
        text: t('moodHistoryTitle'),
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          afterLabel: (context: any) => {
            const index = context.dataIndex;
            const point = history[index];
            if (point) {
              return `${t('dominantEmotion')}: ${t(point.dominantEmotion)}`;
            }
            return '';
          }
        }
      },
      legend: {
        position: 'bottom' as const
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (value: any) => `${value}%`
        }
      }
    }
  };

  // Créer les annotations pour les émotions dominantes
  const emotionAnnotations = history.map((point, index) => ({
    type: 'point',
    xValue: index,
    yValue: point.valence * 50 + 50,
    backgroundColor: getEmotionColor(point.dominantEmotion),
    radius: 6,
    borderColor: 'white',
    borderWidth: 2
  }));

  return (
    <div className="mood-history-container">
      <div style={{ height: '300px' }}>
        <Line data={data} options={options} />
      </div>
      
      <div className="emotion-legend">
        <h4>{t('emotionLegend')}</h4>
        <div className="emotion-tags">
          {Array.from(new Set(history.map(h => h.dominantEmotion))).map(emotion => (
            <span
              key={emotion}
              className="emotion-tag"
              style={{ backgroundColor: getEmotionColor(emotion) }}
            >
              {t(emotion)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MoodHistory;