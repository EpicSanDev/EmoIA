import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import './ModelSelector.css';

interface Model {
  name: string;
  provider: string;
  description?: string;
  capabilities?: string[];
  status?: string;
}

interface ModelSelectorProps {
  userId: string;
  onModelChange?: (provider: string, model: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ userId, onModelChange }) => {
  const { t } = useTranslation();
  const [providers, setProviders] = useState<any>({});
  const [models, setModels] = useState<any>({});
  const [selectedProvider, setSelectedProvider] = useState<string>('ollama');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    fetchProviders();
    fetchModels();
  }, []);

  const fetchProviders = async () => {
    try {
      const response = await fetch('http://localhost:8000/mcp/providers');
      const data = await response.json();
      setProviders(data.providers);
      
      // Sélectionner le premier provider par défaut
      const firstProvider = Object.keys(data.providers)[0];
      if (firstProvider) {
        setSelectedProvider(firstProvider);
      }
    } catch (error) {
      console.error('Erreur lors de la récupération des providers:', error);
    }
  };

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/mcp/models');
      const data = await response.json();
      setModels(data.models);
      
      // Sélectionner le premier modèle du provider par défaut
      if (data.models[selectedProvider]?.length > 0) {
        setSelectedModel(data.models[selectedProvider][0]);
      }
    } catch (error) {
      console.error('Erreur lors de la récupération des modèles:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleProviderChange = (provider: string) => {
    setSelectedProvider(provider);
    
    // Sélectionner le premier modèle de ce provider
    if (models[provider]?.length > 0) {
      setSelectedModel(models[provider][0]);
    } else {
      setSelectedModel('');
    }
  };

  const handleModelChange = async (model: string) => {
    setSelectedModel(model);
    
    if (onModelChange) {
      onModelChange(selectedProvider, model);
    }
    
    // Envoyer la mise à jour au backend
    try {
      const response = await fetch('http://localhost:8000/mcp/switch-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          provider: selectedProvider,
          model: model,
        }),
      });
      
      if (response.ok) {
        console.log('Modèle changé avec succès');
      }
    } catch (error) {
      console.error('Erreur lors du changement de modèle:', error);
    }
  };

  const getProviderIcon = (provider: string) => {
    const icons: { [key: string]: string } = {
      ollama: '🦙',
      openai: '🤖',
      anthropic: '🧠',
      mistral: '🌊',
    };
    return icons[provider] || '🔮';
  };

  const getModelDescription = (model: string) => {
    const descriptions: { [key: string]: string } = {
      'llama2': 'Modèle open-source performant pour le dialogue',
      'mistral': 'Modèle rapide et efficace',
      'phi': 'Modèle compact optimisé pour les tâches courantes',
      'gpt-4': 'Modèle avancé avec capacités étendues',
    };
    
    for (const [key, desc] of Object.entries(descriptions)) {
      if (model.toLowerCase().includes(key)) {
        return desc;
      }
    }
    return 'Modèle IA généraliste';
  };

  return (
    <div className="model-selector">
      <div className="model-selector-header">
        <h3>{t('modelSelection')}</h3>
        <button 
          className="details-toggle"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? '📊' : '📈'}
        </button>
      </div>
      
      <div className="provider-tabs">
        {Object.keys(providers).map((provider) => (
          <button
            key={provider}
            className={`provider-tab ${selectedProvider === provider ? 'active' : ''}`}
            onClick={() => handleProviderChange(provider)}
          >
            <span className="provider-icon">{getProviderIcon(provider)}</span>
            <span className="provider-name">{provider}</span>
            {providers[provider]?.status === 'active' && (
              <span className="status-indicator active" />
            )}
          </button>
        ))}
      </div>
      
      <div className="model-list">
        {loading ? (
          <div className="loading-spinner">
            <div className="spinner" />
            <p>{t('loadingModels')}</p>
          </div>
        ) : (
          <>
            {models[selectedProvider]?.map((model: string) => (
              <div
                key={model}
                className={`model-item ${selectedModel === model ? 'selected' : ''}`}
                onClick={() => handleModelChange(model)}
              >
                <div className="model-info">
                  <h4 className="model-name">{model}</h4>
                  <p className="model-description">{getModelDescription(model)}</p>
                </div>
                {selectedModel === model && (
                  <span className="check-icon">✓</span>
                )}
              </div>
            ))}
            
            {(!models[selectedProvider] || models[selectedProvider].length === 0) && (
              <div className="no-models">
                <p>{t('noModelsAvailable')}</p>
                <button className="refresh-btn" onClick={fetchModels}>
                  {t('refresh')}
                </button>
              </div>
            )}
          </>
        )}
      </div>
      
      {showDetails && selectedProvider && providers[selectedProvider] && (
        <div className="provider-details">
          <h4>{t('providerDetails')}</h4>
          <div className="details-grid">
            <div className="detail-item">
              <span className="detail-label">{t('capabilities')}:</span>
              <div className="capabilities-list">
                {providers[selectedProvider].capabilities?.map((cap: string) => (
                  <span key={cap} className="capability-badge">{cap}</span>
                ))}
              </div>
            </div>
            <div className="detail-item">
              <span className="detail-label">{t('defaultModel')}:</span>
              <span className="detail-value">{providers[selectedProvider].default_model}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">{t('status')}:</span>
              <span className={`status-badge ${providers[selectedProvider].status}`}>
                {providers[selectedProvider].status}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;