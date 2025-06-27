import React from 'react';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

interface PersonalityData {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
  emotional_intelligence: number;
  empathy_level: number;
  creativity: number;
}

interface Props {
  data: PersonalityData;
  title?: string;
}

const PersonalityRadar: React.FC<Props> = ({ data, title = "Profil de Personnalité" }) => {
  if (!data) {
    return (
      <div style={{ height: '400px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p>Données de personnalité non disponibles</p>
      </div>
    );
  }

  const safeData = {
    openness: data.openness ?? 0.5,
    conscientiousness: data.conscientiousness ?? 0.5,
    extraversion: data.extraversion ?? 0.5,
    agreeableness: data.agreeableness ?? 0.5,
    neuroticism: data.neuroticism ?? 0.5,
    emotional_intelligence: data.emotional_intelligence ?? 0.5,
    empathy_level: data.empathy_level ?? 0.5,
    creativity: data.creativity ?? 0.5
  };

  const chartData = {
    labels: [
      '🌟 Ouverture',
      '📋 Conscienciosité',
      '🎉 Extraversion',
      '🤝 Agréabilité',
      '😰 Neuroticisme',
      '🧠 Intelligence Émotionnelle',
      '❤️ Empathie',
      '🎨 Créativité'
    ],
    datasets: [
      {
        label: 'Votre Profil',
        data: [
          safeData.openness * 100,
          safeData.conscientiousness * 100,
          safeData.extraversion * 100,
          safeData.agreeableness * 100,
          safeData.neuroticism * 100,
          safeData.emotional_intelligence * 100,
          safeData.empathy_level * 100,
          safeData.creativity * 100
        ],
        backgroundColor: 'rgba(99, 102, 241, 0.2)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(99, 102, 241, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(99, 102, 241, 1)'
      },
      {
        label: 'Moyenne',
        data: [50, 50, 50, 50, 50, 50, 50, 50],
        backgroundColor: 'rgba(128, 128, 128, 0.1)',
        borderColor: 'rgba(128, 128, 128, 0.5)',
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: title,
        font: {
          size: 18,
          weight: 'bold'
        }
      },
      legend: {
        position: 'bottom' as const
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            return `${context.dataset.label}: ${context.parsed.r}%`;
          }
        }
      }
    },
    scales: {
      r: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 20,
          callback: (value: any) => `${value}%`
        },
        pointLabels: {
          font: {
            size: 12
          }
        }
      }
    }
  };

  return (
    <div style={{ height: '400px', width: '100%' }}>
      <Radar data={chartData} options={options} />
    </div>
  );
};

export default PersonalityRadar;