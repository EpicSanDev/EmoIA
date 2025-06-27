import React, { useEffect, useState } from 'react';
import { Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement
} from 'chart.js';
import { useTranslation } from 'react-i18next';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement);

interface AnalyticsData {
  emotionTrends: { [emotion: string]: number[] };
  emotionDistribution: { [emotion: string]: number };
  timestamps: string[];
}

const AnalyticsDashboard: React.FC<{ userId: string }> = ({ userId }) => {
  const { t } = useTranslation();
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = React.useRef<WebSocket | null>(null);

  // Configuration WebSocket pour les données en temps réel
  useEffect(() => {
    const setupWebSocket = () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      const ws = new WebSocket(`ws://localhost:8001/ws/analytics/${userId}`);
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setTimeout(setupWebSocket, 3000);
      ws.onerror = () => setWsConnected(false);
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setAnalytics(data);
      };
      
      wsRef.current = ws;
    };

    setupWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [userId]);

  // Préparation des données pour les graphiques
  let lineData, pieData;
  if (analytics) {
    // Données pour le graphique linéaire (tendances émotionnelles)
    lineData = {
      labels: analytics.timestamps,
      datasets: Object.entries(analytics.emotionTrends).map(([emotion, values], index) => ({
        label: emotion,
        data: values,
        borderColor: `hsl(${index * 60}, 70%, 50%)`,
        backgroundColor: 0.2,
        tension: 0.3
      }))
    };

    // Données pour le graphique circulaire (distribution des émotions)
    pieData = {
      labels: Object.keys(analytics.emotionDistribution),
      datasets: [{
        data: Object.values(analytics.emotionDistribution),
        backgroundColor: Object.keys(analytics.emotionDistribution).map((_, i) => 
          `hsl(${i * 60}, 70%, 50%)`
        )
      }]
    };
  }

  return (
    <div className="dashboard-container">
      <h2>{t('dashboardTab')}</h2>
      <div className="dashboard-grid">
        <div className="chart-container">
          <h3>{t('emotionTrends')}</h3>
          {lineData ? (
            <Line data={lineData} options={{ responsive: true }} />
          ) : (
            <p>{t('loadingData')}</p>
          )}
        </div>
        
        <div className="chart-container">
          <h3>{t('emotionDistribution')}</h3>
          {pieData ? (
            <Pie data={pieData} options={{ responsive: true }} />
          ) : (
            <p>{t('loadingData')}</p>
          )}
        </div>
      </div>
      
      <div className="status-indicator">
        {wsConnected ? t('wsConnected') : t('wsDisconnected')}
      </div>
    </div>
  );
};

export default AnalyticsDashboard;