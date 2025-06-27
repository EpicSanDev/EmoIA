import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import './UserProfileManager.css';

interface UserProfile {
  user_id: string;
  display_name: string;
  email?: string;
  telegram_id?: string;
  avatar_url?: string;
  bio?: string;
  preferences: { [key: string]: any };
  personality_traits: { [key: string]: number };
  created_at: string;
  last_active: string;
  settings: { [key: string]: any };
  privacy_settings: { [key: string]: boolean };
  notification_preferences: { [key: string]: boolean };
  language: string;
  timezone: string;
}

interface ProfileStats {
  profile_completion: number;
  days_since_creation: number;
  days_since_last_activity: number;
  has_telegram: boolean;
  has_email: boolean;
  personality_traits_count: number;
  preferences_count: number;
}

const UserProfileManager: React.FC = () => {
  const { t } = useTranslation();
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<UserProfile | null>(null);
  const [profileStats, setProfileStats] = useState<ProfileStats | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info', text: string } | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  const [newProfile, setNewProfile] = useState<Partial<UserProfile>>({
    display_name: '',
    email: '',
    bio: '',
    language: 'fr',
    timezone: 'Europe/Paris',
    preferences: {},
    personality_traits: {},
    privacy_settings: {
      share_emotions: true,
      share_memories: false,
      public_profile: false
    },
    notification_preferences: {
      email_notifications: true,
      telegram_notifications: true,
      proactive_messages: true,
      reminders: true
    }
  });

  useEffect(() => {
    fetchProfiles();
  }, []);

  const fetchProfiles = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/users');
      const data = await response.json();
      setProfiles(data.users || []);
    } catch (error) {
      console.error('Error fetching profiles:', error);
      setMessage({ type: 'error', text: 'Erreur lors du chargement des profils' });
    } finally {
      setLoading(false);
    }
  };

  const fetchProfileDetails = async (userId: string) => {
    try {
      const [profileResponse, statsResponse] = await Promise.all([
        fetch(`/api/users/profile/${userId}`),
        fetch(`/api/users/profile/${userId}/stats`)
      ]);

      if (profileResponse.ok) {
        const profileData = await profileResponse.json();
        setSelectedProfile(profileData.profile);
      }

      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setProfileStats(statsData);
      }
    } catch (error) {
      console.error('Error fetching profile details:', error);
    }
  };

  const handleCreateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('/api/users/profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newProfile),
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessage({ type: 'success', text: 'Profil créé avec succès' });
        setShowCreateForm(false);
        setNewProfile({
          display_name: '',
          email: '',
          bio: '',
          language: 'fr',
          timezone: 'Europe/Paris',
          preferences: {},
          personality_traits: {},
          privacy_settings: {
            share_emotions: true,
            share_memories: false,
            public_profile: false
          },
          notification_preferences: {
            email_notifications: true,
            telegram_notifications: true,
            proactive_messages: true,
            reminders: true
          }
        });
        await fetchProfiles();
      } else {
        setMessage({ type: 'error', text: 'Erreur lors de la création du profil' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Erreur lors de la création du profil' });
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateProfile = async () => {
    if (!selectedProfile) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/users/profile/${selectedProfile.user_id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(selectedProfile),
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessage({ type: 'success', text: 'Profil mis à jour avec succès' });
        setIsEditing(false);
        await fetchProfiles();
      } else {
        setMessage({ type: 'error', text: 'Erreur lors de la mise à jour' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Erreur lors de la mise à jour' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteProfile = async (userId: string) => {
    if (!window.confirm('Êtes-vous sûr de vouloir supprimer ce profil ?')) return;

    try {
      const response = await fetch(`/api/users/profile/${userId}`, {
        method: 'DELETE',
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessage({ type: 'success', text: 'Profil supprimé avec succès' });
        setSelectedProfile(null);
        await fetchProfiles();
      } else {
        setMessage({ type: 'error', text: 'Erreur lors de la suppression' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Erreur lors de la suppression' });
    }
  };

  const filteredProfiles = profiles.filter(profile =>
    profile.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (profile.email && profile.email.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('fr-FR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getProfileCompletionColor = (completion: number) => {
    if (completion < 0.3) return '#ef4444';
    if (completion < 0.7) return '#f59e0b';
    return '#10b981';
  };

  return (
    <div className="profile-manager">
      <div className="manager-header">
        <h2>
          <span className="header-icon">👥</span>
          Gestionnaire de Profils
        </h2>
        <div className="header-actions">
          <button 
            className="create-btn"
            onClick={() => setShowCreateForm(true)}
          >
            <span className="btn-icon">➕</span>
            Nouveau Profil
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

      <div className="manager-content">
        <div className="profiles-sidebar">
          <div className="search-bar">
            <input
              type="text"
              placeholder="Rechercher un profil..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <span className="search-icon">🔍</span>
          </div>

          <div className="profiles-list">
            {loading && <div className="loading">Chargement...</div>}
            
            {filteredProfiles.map(profile => (
              <div
                key={profile.user_id}
                className={`profile-item ${selectedProfile?.user_id === profile.user_id ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedProfile(profile);
                  fetchProfileDetails(profile.user_id);
                }}
              >
                <div className="profile-avatar">
                  {profile.avatar_url ? (
                    <img src={profile.avatar_url} alt={profile.display_name} />
                  ) : (
                    <div className="avatar-placeholder">
                      {profile.display_name.charAt(0).toUpperCase()}
                    </div>
                  )}
                </div>
                <div className="profile-info">
                  <h4>{profile.display_name}</h4>
                  <p className="profile-email">{profile.email || 'Pas d\'email'}</p>
                  <div className="profile-badges">
                    {profile.telegram_id && <span className="badge telegram">📱</span>}
                    {profile.email && <span className="badge email">✉️</span>}
                  </div>
                </div>
              </div>
            ))}

            {filteredProfiles.length === 0 && !loading && (
              <div className="no-profiles">
                <span className="no-profiles-icon">👤</span>
                <p>Aucun profil trouvé</p>
              </div>
            )}
          </div>
        </div>

        <div className="profile-details">
          {selectedProfile ? (
            <div className="profile-content">
              <div className="profile-header">
                <div className="profile-main-info">
                  <div className="profile-avatar-large">
                    {selectedProfile.avatar_url ? (
                      <img src={selectedProfile.avatar_url} alt={selectedProfile.display_name} />
                    ) : (
                      <div className="avatar-placeholder-large">
                        {selectedProfile.display_name.charAt(0).toUpperCase()}
                      </div>
                    )}
                  </div>
                  <div className="profile-details-info">
                    {isEditing ? (
                      <input
                        type="text"
                        value={selectedProfile.display_name}
                        onChange={(e) => setSelectedProfile({
                          ...selectedProfile,
                          display_name: e.target.value
                        })}
                        className="edit-input"
                      />
                    ) : (
                      <h3>{selectedProfile.display_name}</h3>
                    )}
                    <p className="profile-id">ID: {selectedProfile.user_id}</p>
                    <p className="profile-dates">
                      Créé le {formatDate(selectedProfile.created_at)}
                    </p>
                  </div>
                </div>

                <div className="profile-actions">
                  {isEditing ? (
                    <>
                      <button 
                        className="save-btn"
                        onClick={handleUpdateProfile}
                        disabled={loading}
                      >
                        💾 Sauvegarder
                      </button>
                      <button 
                        className="cancel-btn"
                        onClick={() => setIsEditing(false)}
                      >
                        ❌ Annuler
                      </button>
                    </>
                  ) : (
                    <>
                      <button 
                        className="edit-btn"
                        onClick={() => setIsEditing(true)}
                      >
                        ✏️ Modifier
                      </button>
                      <button 
                        className="delete-btn"
                        onClick={() => handleDeleteProfile(selectedProfile.user_id)}
                      >
                        🗑️ Supprimer
                      </button>
                    </>
                  )}
                </div>
              </div>

              {profileStats && (
                <div className="profile-stats">
                  <div className="stat-card completion">
                    <h4>Profil complété</h4>
                    <div className="completion-bar">
                      <div 
                        className="completion-fill"
                        style={{ 
                          width: `${profileStats.profile_completion * 100}%`,
                          backgroundColor: getProfileCompletionColor(profileStats.profile_completion)
                        }}
                      ></div>
                    </div>
                    <span>{Math.round(profileStats.profile_completion * 100)}%</span>
                  </div>
                  
                  <div className="stat-card">
                    <h4>Dernière activité</h4>
                    <p>{profileStats.days_since_last_activity} jours</p>
                  </div>
                  
                  <div className="stat-card">
                    <h4>Traits de personnalité</h4>
                    <p>{profileStats.personality_traits_count}</p>
                  </div>
                  
                  <div className="stat-card">
                    <h4>Préférences</h4>
                    <p>{profileStats.preferences_count}</p>
                  </div>
                </div>
              )}

              <div className="profile-sections">
                <div className="section">
                  <h4>Informations personnelles</h4>
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Email</label>
                      {isEditing ? (
                        <input
                          type="email"
                          value={selectedProfile.email || ''}
                          onChange={(e) => setSelectedProfile({
                            ...selectedProfile,
                            email: e.target.value
                          })}
                        />
                      ) : (
                        <p>{selectedProfile.email || 'Non renseigné'}</p>
                      )}
                    </div>
                    
                    <div className="form-group">
                      <label>Telegram ID</label>
                      <p>{selectedProfile.telegram_id || 'Non connecté'}</p>
                    </div>
                    
                    <div className="form-group">
                      <label>Langue</label>
                      {isEditing ? (
                        <select
                          value={selectedProfile.language}
                          onChange={(e) => setSelectedProfile({
                            ...selectedProfile,
                            language: e.target.value
                          })}
                        >
                          <option value="fr">Français</option>
                          <option value="en">English</option>
                          <option value="es">Español</option>
                        </select>
                      ) : (
                        <p>{selectedProfile.language}</p>
                      )}
                    </div>
                    
                    <div className="form-group">
                      <label>Fuseau horaire</label>
                      {isEditing ? (
                        <input
                          type="text"
                          value={selectedProfile.timezone}
                          onChange={(e) => setSelectedProfile({
                            ...selectedProfile,
                            timezone: e.target.value
                          })}
                        />
                      ) : (
                        <p>{selectedProfile.timezone}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="form-group bio">
                    <label>Biographie</label>
                    {isEditing ? (
                      <textarea
                        value={selectedProfile.bio || ''}
                        onChange={(e) => setSelectedProfile({
                          ...selectedProfile,
                          bio: e.target.value
                        })}
                        rows={3}
                        placeholder="Parlez-nous de vous..."
                      />
                    ) : (
                      <p>{selectedProfile.bio || 'Aucune biographie'}</p>
                    )}
                  </div>
                </div>

                <div className="section">
                  <h4>Paramètres de confidentialité</h4>
                  <div className="privacy-settings">
                    {Object.entries(selectedProfile.privacy_settings || {}).map(([key, value]) => (
                      <label key={key} className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={(e) => setSelectedProfile({
                            ...selectedProfile,
                            privacy_settings: {
                              ...selectedProfile.privacy_settings,
                              [key]: e.target.checked
                            }
                          })}
                          disabled={!isEditing}
                        />
                        <span className="checkbox-custom"></span>
                        {key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1)}
                      </label>
                    ))}
                  </div>
                </div>

                <div className="section">
                  <h4>Notifications</h4>
                  <div className="notification-settings">
                    {Object.entries(selectedProfile.notification_preferences || {}).map(([key, value]) => (
                      <label key={key} className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={(e) => setSelectedProfile({
                            ...selectedProfile,
                            notification_preferences: {
                              ...selectedProfile.notification_preferences,
                              [key]: e.target.checked
                            }
                          })}
                          disabled={!isEditing}
                        />
                        <span className="checkbox-custom"></span>
                        {key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1)}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-selection">
              <span className="no-selection-icon">👤</span>
              <h3>Sélectionnez un profil</h3>
              <p>Choisissez un profil dans la liste pour voir ses détails</p>
            </div>
          )}
        </div>
      </div>

      {/* Modal de création de profil */}
      {showCreateForm && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h3>Créer un nouveau profil</h3>
              <button 
                className="close-btn"
                onClick={() => setShowCreateForm(false)}
              >
                ❌
              </button>
            </div>
            
            <form onSubmit={handleCreateProfile} className="create-form">
              <div className="form-group">
                <label htmlFor="display_name">
                  Nom d'affichage <span className="required">*</span>
                </label>
                <input
                  type="text"
                  id="display_name"
                  value={newProfile.display_name}
                  onChange={(e) => setNewProfile({
                    ...newProfile,
                    display_name: e.target.value
                  })}
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  type="email"
                  id="email"
                  value={newProfile.email}
                  onChange={(e) => setNewProfile({
                    ...newProfile,
                    email: e.target.value
                  })}
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="bio">Biographie</label>
                <textarea
                  id="bio"
                  value={newProfile.bio}
                  onChange={(e) => setNewProfile({
                    ...newProfile,
                    bio: e.target.value
                  })}
                  rows={3}
                />
              </div>
              
              <div className="form-actions">
                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? 'Création...' : 'Créer le profil'}
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

export default UserProfileManager;