import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import './MemoryManager.css';

interface Memory {
  id: string;
  content: string;
  title: string;
  tags: string[];
  importance: number;
  created_at: string;
  accessed_count: number;
  emotional_context?: {
    dominant_emotion: string;
    confidence: number;
  };
  similarity?: number;
}

const MemoryManager: React.FC<{ userId: string }> = ({ userId }) => {
  const { t } = useTranslation();
  const [memories, setMemories] = useState<Memory[]>([]);
  const [filteredMemories, setFilteredMemories] = useState<Memory[]>([]);
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'important' | 'recent' | 'emotion'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'importance' | 'access'>('date');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info', text: string } | null>(null);

  const [newMemory, setNewMemory] = useState({
    title: '',
    content: '',
    tags: '',
    importance: 5,
    type: 'personal'
  });

  useEffect(() => {
    fetchMemories();
  }, [userId]);

  useEffect(() => {
    filterAndSortMemories();
  }, [memories, searchQuery, filterType, sortBy]);

  const fetchMemories = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/memories/${userId}?limit=100`);
      const data = await response.json();
      setMemories(data.memories || []);
    } catch (error) {
      console.error('Error fetching memories:', error);
      setMessage({ type: 'error', text: 'Erreur lors du chargement des souvenirs' });
    } finally {
      setLoading(false);
    }
  };

  const searchMemories = async (query: string) => {
    if (!query.trim()) {
      fetchMemories();
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`/api/memories/${userId}/search?query=${encodeURIComponent(query)}&limit=50`);
      const data = await response.json();
      setMemories(data.memories || []);
    } catch (error) {
      console.error('Error searching memories:', error);
      setMessage({ type: 'error', text: 'Erreur lors de la recherche' });
    } finally {
      setLoading(false);
    }
  };

  const filterAndSortMemories = () => {
    let filtered = [...memories];

    // Filtrage
    switch (filterType) {
      case 'important':
        filtered = filtered.filter(m => m.importance >= 7);
        break;
      case 'recent':
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
        filtered = filtered.filter(m => new Date(m.created_at) > oneWeekAgo);
        break;
      case 'emotion':
        filtered = filtered.filter(m => m.emotional_context);
        break;
    }

    // Tri
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'importance':
          return b.importance - a.importance;
        case 'access':
          return b.accessed_count - a.accessed_count;
        default: // date
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
    });

    setFilteredMemories(filtered);
  };

  const handleCreateMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const tags = newMemory.tags.split(',').map(tag => tag.trim()).filter(tag => tag.length > 0);
      
      const response = await fetch(`/api/memories/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: newMemory.content,
          title: newMemory.title,
          tags,
          importance: newMemory.importance / 10,
          type: newMemory.type,
          context: `Créé via l'interface web`
        }),
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessage({ type: 'success', text: 'Souvenir créé avec succès' });
        setShowCreateForm(false);
        setNewMemory({
          title: '',
          content: '',
          tags: '',
          importance: 5,
          type: 'personal'
        });
        await fetchMemories();
      } else {
        setMessage({ type: 'error', text: 'Erreur lors de la création du souvenir' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Erreur lors de la création du souvenir' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteMemory = async (memoryId: string) => {
    if (!confirm('Êtes-vous sûr de vouloir supprimer ce souvenir ?')) return;

    try {
      const response = await fetch(`/api/memories/${memoryId}`, {
        method: 'DELETE',
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessage({ type: 'success', text: 'Souvenir supprimé avec succès' });
        setSelectedMemory(null);
        await fetchMemories();
      } else {
        setMessage({ type: 'error', text: 'Erreur lors de la suppression' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Erreur lors de la suppression' });
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('fr-FR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return '#ef4444';
    if (importance >= 0.6) return '#f59e0b';
    if (importance >= 0.4) return '#10b981';
    return '#6b7280';
  };

  const getEmotionEmoji = (emotion: string) => {
    const emojis: { [key: string]: string } = {
      joy: '😊',
      sadness: '😢',
      anger: '😠',
      fear: '😨',
      surprise: '😮',
      love: '❤️',
      excitement: '🎉',
      anxiety: '😰',
      contentment: '😌',
      curiosity: '🤔'
    };
    return emojis[emotion] || '🎭';
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    searchMemories(searchQuery);
  };

  return (
    <div className="memory-manager">
      <div className="manager-header">
        <h2>
          <span className="header-icon">🧠</span>
          Gestionnaire de Souvenirs
        </h2>
        <div className="header-actions">
          <button 
            className="create-btn"
            onClick={() => setShowCreateForm(true)}
          >
            <span className="btn-icon">➕</span>
            Nouveau Souvenir
          </button>
        </div>
      </div>

      {message && (
        <div className={`message ${message.type}`}>
          <span className="message-icon">
            {message.type === 'success' ? '✅' : message.type === 'error' ? '❌' : 'ℹ️'}
          </span>
          {message.text}
        </div>
      )}

      <div className="controls-panel">
        <form onSubmit={handleSearchSubmit} className="search-form">
          <div className="search-input-group">
            <input
              type="text"
              placeholder="Rechercher dans vos souvenirs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button type="submit" className="search-btn">
              🔍
            </button>
          </div>
        </form>

        <div className="filter-controls">
          <select 
            value={filterType} 
            onChange={(e) => setFilterType(e.target.value as any)}
            className="filter-select"
          >
            <option value="all">Tous les souvenirs</option>
            <option value="important">Importants</option>
            <option value="recent">Récents (7 jours)</option>
            <option value="emotion">Avec émotion</option>
          </select>

          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value as any)}
            className="sort-select"
          >
            <option value="date">Date</option>
            <option value="importance">Importance</option>
            <option value="access">Accès</option>
          </select>
        </div>
      </div>

      <div className="manager-content">
        <div className="memories-sidebar">
          {loading && <div className="loading">Chargement...</div>}
          
          <div className="memories-stats">
            <div className="stat">
              <span className="stat-number">{filteredMemories.length}</span>
              <span className="stat-label">Souvenirs</span>
            </div>
            <div className="stat">
              <span className="stat-number">
                {filteredMemories.filter(m => m.importance >= 0.7).length}
              </span>
              <span className="stat-label">Importants</span>
            </div>
          </div>

          <div className="memories-list">
            {filteredMemories.map(memory => (
              <div
                key={memory.id}
                className={`memory-item ${selectedMemory?.id === memory.id ? 'selected' : ''}`}
                onClick={() => setSelectedMemory(memory)}
              >
                <div className="memory-header">
                  <h4 className="memory-title">
                    {memory.title || memory.content.substring(0, 50) + '...'}
                  </h4>
                  <div className="memory-indicators">
                    <div 
                      className="importance-indicator"
                      style={{ backgroundColor: getImportanceColor(memory.importance) }}
                      title={`Importance: ${Math.round(memory.importance * 10)}/10`}
                    ></div>
                    {memory.emotional_context && (
                      <span className="emotion-indicator">
                        {getEmotionEmoji(memory.emotional_context.dominant_emotion)}
                      </span>
                    )}
                  </div>
                </div>
                
                <p className="memory-preview">
                  {memory.content.substring(0, 100)}
                  {memory.content.length > 100 && '...'}
                </p>
                
                <div className="memory-meta">
                  <span className="memory-date">
                    {formatDate(memory.created_at)}
                  </span>
                  <span className="memory-access">
                    Consulté {memory.accessed_count} fois
                  </span>
                </div>
                
                {memory.tags.length > 0 && (
                  <div className="memory-tags">
                    {memory.tags.slice(0, 3).map(tag => (
                      <span key={tag} className="tag">#{tag}</span>
                    ))}
                    {memory.tags.length > 3 && (
                      <span className="tag-more">+{memory.tags.length - 3}</span>
                    )}
                  </div>
                )}

                {memory.similarity && (
                  <div className="similarity-score">
                    Similarité: {Math.round(memory.similarity * 100)}%
                  </div>
                )}
              </div>
            ))}

            {filteredMemories.length === 0 && !loading && (
              <div className="no-memories">
                <span className="no-memories-icon">🧠</span>
                <p>Aucun souvenir trouvé</p>
                <small>
                  {searchQuery ? 
                    'Essayez une autre recherche' : 
                    'Créez votre premier souvenir'
                  }
                </small>
              </div>
            )}
          </div>
        </div>

        <div className="memory-details">
          {selectedMemory ? (
            <div className="memory-content">
              <div className="memory-detail-header">
                <div className="memory-title-section">
                  <h3>{selectedMemory.title || 'Sans titre'}</h3>
                  <div className="memory-meta-info">
                    <span className="created-date">
                      Créé le {formatDate(selectedMemory.created_at)}
                    </span>
                    <span className="access-count">
                      {selectedMemory.accessed_count} consultations
                    </span>
                  </div>
                </div>
                
                <div className="memory-actions">
                  <button 
                    className="delete-btn"
                    onClick={() => handleDeleteMemory(selectedMemory.id)}
                  >
                    🗑️ Supprimer
                  </button>
                </div>
              </div>

              <div className="memory-info-cards">
                <div className="info-card importance">
                  <h4>Importance</h4>
                  <div className="importance-display">
                    <div className="importance-bar">
                      <div 
                        className="importance-fill"
                        style={{ 
                          width: `${selectedMemory.importance * 100}%`,
                          backgroundColor: getImportanceColor(selectedMemory.importance)
                        }}
                      ></div>
                    </div>
                    <span>{Math.round(selectedMemory.importance * 10)}/10</span>
                  </div>
                </div>

                {selectedMemory.emotional_context && (
                  <div className="info-card emotion">
                    <h4>Contexte émotionnel</h4>
                    <div className="emotion-display">
                      <span className="emotion-emoji">
                        {getEmotionEmoji(selectedMemory.emotional_context.dominant_emotion)}
                      </span>
                      <div className="emotion-info">
                        <span className="emotion-name">
                          {selectedMemory.emotional_context.dominant_emotion}
                        </span>
                        <span className="emotion-confidence">
                          Confiance: {Math.round(selectedMemory.emotional_context.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="memory-body">
                <h4>Contenu</h4>
                <div className="memory-text">
                  {selectedMemory.content.split('\n').map((paragraph, index) => (
                    <p key={index}>{paragraph}</p>
                  ))}
                </div>
              </div>

              {selectedMemory.tags.length > 0 && (
                <div className="memory-tags-section">
                  <h4>Tags</h4>
                  <div className="tags-list">
                    {selectedMemory.tags.map(tag => (
                      <span key={tag} className="tag">#{tag}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="no-selection">
              <span className="no-selection-icon">🧠</span>
              <h3>Sélectionnez un souvenir</h3>
              <p>Choisissez un souvenir dans la liste pour voir ses détails</p>
            </div>
          )}
        </div>
      </div>

      {/* Modal de création de souvenir */}
      {showCreateForm && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h3>Créer un nouveau souvenir</h3>
              <button 
                className="close-btn"
                onClick={() => setShowCreateForm(false)}
              >
                ❌
              </button>
            </div>
            
            <form onSubmit={handleCreateMemory} className="create-form">
              <div className="form-group">
                <label htmlFor="memory_title">Titre (optionnel)</label>
                <input
                  type="text"
                  id="memory_title"
                  value={newMemory.title}
                  onChange={(e) => setNewMemory({
                    ...newMemory,
                    title: e.target.value
                  })}
                  placeholder="Un titre pour ce souvenir..."
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="memory_content">
                  Contenu <span className="required">*</span>
                </label>
                <textarea
                  id="memory_content"
                  value={newMemory.content}
                  onChange={(e) => setNewMemory({
                    ...newMemory,
                    content: e.target.value
                  })}
                  rows={6}
                  placeholder="Décrivez votre souvenir en détail..."
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="memory_tags">Tags (séparés par des virgules)</label>
                <input
                  type="text"
                  id="memory_tags"
                  value={newMemory.tags}
                  onChange={(e) => setNewMemory({
                    ...newMemory,
                    tags: e.target.value
                  })}
                  placeholder="travail, important, famille..."
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="memory_importance">
                  Importance ({newMemory.importance}/10)
                </label>
                <input
                  type="range"
                  id="memory_importance"
                  min="1"
                  max="10"
                  value={newMemory.importance}
                  onChange={(e) => setNewMemory({
                    ...newMemory,
                    importance: parseInt(e.target.value)
                  })}
                  className="importance-slider"
                />
                <div className="importance-labels">
                  <span>Faible</span>
                  <span>Moyenne</span>
                  <span>Élevée</span>
                </div>
              </div>
              
              <div className="form-group">
                <label htmlFor="memory_type">Type</label>
                <select
                  id="memory_type"
                  value={newMemory.type}
                  onChange={(e) => setNewMemory({
                    ...newMemory,
                    type: e.target.value
                  })}
                >
                  <option value="personal">Personnel</option>
                  <option value="work">Professionnel</option>
                  <option value="learning">Apprentissage</option>
                  <option value="emotion">Émotionnel</option>
                  <option value="goal">Objectif</option>
                </select>
              </div>
              
              <div className="form-actions">
                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? 'Création...' : 'Créer le souvenir'}
                </button>
                <button 
                  type="button" 
                  className="cancel-btn"
                  onClick={() => setShowCreateForm(false)}
                >
                  Annuler
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default MemoryManager;