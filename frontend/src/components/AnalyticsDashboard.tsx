import React, { useState, useEffect } from 'react';
import { Line, Pie, Bar } from 'react-chartjs-2';
import { useTranslation } from 'react-i18next';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
} from 'chart.js';
import '../App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
);

interface AnalyticsData {
  trends?: {
    most_frequent_emotion: string;
    emotional_stability: number;
    positive_ratio: number;
    emotional_timeline?: Array<[string, any]>;
  };
  total_interactions?: number;
  period_analyzed?: string;
  recommendations?: string[];
  error?: string;
}

interface Props {
  userId: string;
}

const AnalyticsDashboard: React.FC<Props> = ({ userId }) => {
  const { t } = useTranslation();
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    fetchAnalytics();
    connectWebSocket();

    return () => {
      // Cleanup WebSocket on unmount
    };
  }, [userId]);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/analytics/${userId}`);
      const data = await response.json();
      setAnalytics(data);
    } catch (error) {
      setAnalytics({ error: t('analyticsError') });
    } finally {
      setLoading(false);
    }
  };

  const connectWebSocket = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/analytics/${userId}`);
    
    ws.onopen = () => {
      setWsConnected(true);
      ws.send(JSON.stringify({ type: 'request_analytics' }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'analytics_update') {
        // Update analytics with real-time data
        setAnalytics(prev => ({
          ...prev,
          ...data.data
        }));
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
      // Reconnect after 5 seconds
      setTimeout(connectWebSocket, 5000);
    };
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="dashboard-card">
          <div className="loading-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <p>{t('loading')}...</p>
        </div>
      </div>
    );
  }

  if (analytics?.error) {
    return (
      <div className="dashboard-container">
        <div className="dashboard-card">
          <h3>❌ {t('error')}</h3>
          <p>{analytics.error}</p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const emotionDistribution = {
    labels: ['😊 Joy', '😢 Sadness', '😠 Anger', '😨 Fear', '😮 Surprise', '❤️ Love'],
    datasets: [{
      data: [30, 15, 10, 5, 20, 20],
      backgroundColor: [
        '#FFD93D',
        '#6495ED',
        '#FF6B6B',
        '#9370DB',
        '#FFB6C1',
        '#FF69B4'
      ],
      borderWidth: 2,
      borderColor: '#fff'
    }]
  };

  const emotionalTrend = {
    labels: ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'],
    datasets: [
      {
        label: t('positiveEmotions'),
        data: [65, 59, 80, 81, 56, 55, 70],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.4
      },
      {
        label: t('negativeEmotions'),
        data: [35, 41, 20, 19, 44, 45, 30],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.4
      }
    ]
  };

  const interactionFrequency = {
    labels: ['00h', '04h', '08h', '12h', '16h', '20h'],
    datasets: [{
      label: t('interactions'),
      data: [2, 1, 5, 12, 8, 15],
      backgroundColor: 'rgba(99, 102, 241, 0.6)',
      borderColor: 'rgba(99, 102, 241, 1)',
      borderWidth: 2
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            family: 'var(--font-primary)',
            size: 12
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        cornerRadius: 8,
        titleFont: {
          size: 14,
          weight: 'bold' as const
        },
        bodyFont: {
          size: 12
        }
      }
    }
  };

  return (
    <div className="dashboard-container">
      {/* Status Card */}
      <div className="dashboard-card">
        <h3>📊 {t('status')}</h3>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <p><strong>{t('totalInteractions')}:</strong> {analytics?.total_interactions || 0}</p>
            <p><strong>{t('period')}:</strong> {analytics?.period_analyzed || '30 jours'}</p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span className={`status-indicator ${wsConnected ? 'connected' : 'disconnected'}`}>
              {wsConnected ? '🟢 ' + t('connected') : '🔴 ' + t('disconnected')}
            </span>
          </div>
        </div>
      </div>

      {/* Emotional State Card */}
      <div className="dashboard-card">
        <h3>🎭 {t('emotionalState')}</h3>
        {analytics?.trends && (
          <>
            <p><strong>{t('dominantEmotion')}:</strong> {analytics.trends.most_frequent_emotion}</p>
            <p><strong>{t('emotionalStability')}:</strong> {(analytics.trends.emotional_stability * 100).toFixed(0)}%</p>
            <p><strong>{t('positivityRatio')}:</strong> {(analytics.trends.positive_ratio * 100).toFixed(0)}%</p>
          </>
        )}
      </div>

      {/* Emotion Distribution Chart */}
      <div className="dashboard-card" style={{ gridColumn: 'span 2' }}>
        <h3>🎨 {t('emotionDistribution')}</h3>
        <div style={{ height: '300px' }}>
          <Pie data={emotionDistribution} options={chartOptions} />
        </div>
      </div>

      {/* Emotional Trend Chart */}
      <div className="dashboard-card" style={{ gridColumn: 'span 3' }}>
        <h3>📈 {t('emotionalTrend')}</h3>
        <div style={{ height: '300px' }}>
          <Line data={emotionalTrend} options={chartOptions} />
        </div>
      </div>

      {/* Interaction Frequency Chart */}
      <div className="dashboard-card" style={{ gridColumn: 'span 2' }}>
        <h3>⏰ {t('interactionFrequency')}</h3>
        <div style={{ height: '300px' }}>
          <Bar data={interactionFrequency} options={chartOptions} />
        </div>
      </div>

      {/* Recommendations Card */}
      {analytics?.recommendations && analytics.recommendations.length > 0 && (
        <div className="dashboard-card" style={{ gridColumn: 'span 3' }}>
          <h3>💡 {t('recommendations')}</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {analytics.recommendations.map((rec, index) => (
              <li key={index} style={{ marginBottom: '10px', display: 'flex', alignItems: 'flex-start' }}>
                <span style={{ marginRight: '10px' }}>🔸</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Quick Actions Card */}
      <div className="dashboard-card">
        <h3>⚡ {t('quickActions')}</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <button onClick={fetchAnalytics} className="action-button">
            🔄 {t('refresh')}
          </button>
          <button className="action-button">
            📥 {t('exportData')}
          </button>
          <button className="action-button">
            🔔 {t('notifications')}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;