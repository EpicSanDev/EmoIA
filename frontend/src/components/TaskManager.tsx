import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './TaskManager.css';

interface Task {
  id: string;
  title: string;
  description?: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'pending' | 'in-progress' | 'completed' | 'cancelled';
  category: 'work' | 'personal' | 'health' | 'learning' | 'other';
  dueDate?: string;
  estimatedTime?: number; // en minutes
  aiSuggestions?: string[];
  emotionalContext?: {
    mood: string;
    energy: number;
    motivation: number;
  };
  dependencies?: string[];
  tags: string[];
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
}

interface TaskManagerProps {
  userId: string;
  onTaskAction?: (action: string, task: Task) => void;
}

const TaskManager: React.FC<TaskManagerProps> = ({ userId, onTaskAction }) => {
  const { t } = useTranslation();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [newTask, setNewTask] = useState<Partial<Task>>({
    title: '',
    description: '',
    priority: 'medium',
    category: 'work',
    tags: []
  });
  const [filter, setFilter] = useState<'all' | 'pending' | 'completed' | 'today'>('all');
  const [showNewTaskForm, setShowNewTaskForm] = useState(false);
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'priority' | 'dueDate' | 'created'>('priority');

  // Charger les tâches
  useEffect(() => {
    fetchTasks();
  }, [userId]);

  const fetchTasks = async () => {
    try {
      const response = await fetch(`/api/tasks/${userId}`);
      if (response.ok) {
        const data = await response.json();
        setTasks(data.tasks || []);
      }
    } catch (error) {
      console.error('Erreur lors du chargement des tâches:', error);
    }
  };

  // Suggestions IA basées sur le contexte
  const generateAiSuggestions = useCallback(async (taskTitle: string) => {
    if (!taskTitle.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/ai/task-suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: taskTitle,
          existingTasks: tasks,
          userId: userId
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAiSuggestions(data.suggestions || []);
      }
    } catch (error) {
      console.error('Erreur suggestions IA:', error);
    } finally {
      setLoading(false);
    }
  }, [tasks, userId]);

  // Ajouter une nouvelle tâche
  const handleAddTask = async () => {
    if (!newTask.title?.trim()) return;

    const task: Task = {
      id: Date.now().toString(),
      title: newTask.title,
      description: newTask.description || '',
      priority: newTask.priority || 'medium',
      status: 'pending',
      category: newTask.category || 'work',
      tags: newTask.tags || [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      aiSuggestions: aiSuggestions
    };

    try {
      const response = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, task })
      });

      if (response.ok) {
        setTasks(prev => [task, ...prev]);
        setNewTask({ title: '', description: '', priority: 'medium', category: 'work', tags: [] });
        setShowNewTaskForm(false);
        setAiSuggestions([]);
        onTaskAction?.('created', task);
      }
    } catch (error) {
      console.error('Erreur création tâche:', error);
    }
  };

  // Mettre à jour le statut d'une tâche
  const updateTaskStatus = async (taskId: string, status: Task['status']) => {
    const updatedTasks = tasks.map(task => 
      task.id === taskId 
        ? { 
            ...task, 
            status, 
            updatedAt: new Date().toISOString(),
            completedAt: status === 'completed' ? new Date().toISOString() : undefined
          }
        : task
    );
    setTasks(updatedTasks);

    const task = updatedTasks.find(t => t.id === taskId);
    if (task) {
      onTaskAction?.(status === 'completed' ? 'completed' : 'updated', task);
    }

    // Sauvegarder en base
    try {
      await fetch(`/api/tasks/${taskId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status, updatedAt: new Date().toISOString() })
      });
    } catch (error) {
      console.error('Erreur mise à jour tâche:', error);
    }
  };

  // Filtrer et trier les tâches
  const filteredTasks = tasks
    .filter(task => {
      if (filter === 'pending') return task.status === 'pending' || task.status === 'in-progress';
      if (filter === 'completed') return task.status === 'completed';
      if (filter === 'today') {
        const today = new Date().toDateString();
        return task.dueDate && new Date(task.dueDate).toDateString() === today;
      }
      return true;
    })
    .filter(task => 
      task.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      task.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      task.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    )
    .sort((a, b) => {
      if (sortBy === 'priority') {
        const priorityOrder = { urgent: 4, high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      }
      if (sortBy === 'dueDate') {
        if (!a.dueDate && !b.dueDate) return 0;
        if (!a.dueDate) return 1;
        if (!b.dueDate) return -1;
        return new Date(a.dueDate).getTime() - new Date(b.dueDate).getTime();
      }
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

  const getPriorityColor = (priority: Task['priority']) => {
    const colors = {
      low: '#10b981',
      medium: '#f59e0b',
      high: '#f97316',
      urgent: '#ef4444'
    };
    return colors[priority];
  };

  const getCategoryIcon = (category: Task['category']) => {
    const icons = {
      work: '💼',
      personal: '🏠',
      health: '🏃‍♂️',
      learning: '📚',
      other: '📋'
    };
    return icons[category];
  };

  const getStatusIcon = (status: Task['status']) => {
    const icons = {
      pending: '⏳',
      'in-progress': '⚡',
      completed: '✅',
      cancelled: '❌'
    };
    return icons[status];
  };

  return (
    <div className="task-manager">
      <div className="task-manager-header">
        <div className="header-top">
          <h2 className="task-manager-title">
            <span className="title-icon">📋</span>
            {t('taskManager.title')}
          </h2>
          <button 
            className="btn btn-primary btn-add-task"
            onClick={() => setShowNewTaskForm(true)}
          >
            <span>➕</span>
            {t('taskManager.addTask')}
          </button>
        </div>

        <div className="task-controls">
          <div className="search-container">
            <input
              type="text"
              placeholder={t('taskManager.searchPlaceholder')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            <span className="search-icon">🔍</span>
          </div>

          <div className="filter-controls">
            <select 
              value={filter} 
              onChange={(e) => setFilter(e.target.value as any)}
              className="filter-select"
            >
              <option value="all">{t('taskManager.filters.all')}</option>
              <option value="pending">{t('taskManager.filters.pending')}</option>
              <option value="completed">{t('taskManager.filters.completed')}</option>
              <option value="today">{t('taskManager.filters.today')}</option>
            </select>

            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value as any)}
              className="sort-select"
            >
              <option value="priority">{t('taskManager.sort.priority')}</option>
              <option value="dueDate">{t('taskManager.sort.dueDate')}</option>
              <option value="created">{t('taskManager.sort.created')}</option>
            </select>
          </div>
        </div>
      </div>

      {/* Formulaire nouvelle tâche */}
      {showNewTaskForm && (
        <div className="new-task-form card-glass">
          <div className="form-header">
            <h3>{t('taskManager.newTask')}</h3>
            <button 
              className="btn-close"
              onClick={() => {
                setShowNewTaskForm(false);
                setAiSuggestions([]);
              }}
            >
              ✕
            </button>
          </div>

          <div className="form-content">
            <div className="form-group">
              <input
                type="text"
                placeholder={t('taskManager.taskTitle')}
                value={newTask.title || ''}
                onChange={(e) => {
                  setNewTask(prev => ({ ...prev, title: e.target.value }));
                  generateAiSuggestions(e.target.value);
                }}
                className="task-input"
                autoFocus
              />
            </div>

            <div className="form-group">
              <textarea
                placeholder={t('taskManager.taskDescription')}
                value={newTask.description || ''}
                onChange={(e) => setNewTask(prev => ({ ...prev, description: e.target.value }))}
                className="task-textarea"
                rows={3}
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>{t('taskManager.priority')}</label>
                <select
                  value={newTask.priority || 'medium'}
                  onChange={(e) => setNewTask(prev => ({ ...prev, priority: e.target.value as any }))}
                  className="task-select"
                >
                  <option value="low">{t('taskManager.priorities.low')}</option>
                  <option value="medium">{t('taskManager.priorities.medium')}</option>
                  <option value="high">{t('taskManager.priorities.high')}</option>
                  <option value="urgent">{t('taskManager.priorities.urgent')}</option>
                </select>
              </div>

              <div className="form-group">
                <label>{t('taskManager.category')}</label>
                <select
                  value={newTask.category || 'work'}
                  onChange={(e) => setNewTask(prev => ({ ...prev, category: e.target.value as any }))}
                  className="task-select"
                >
                  <option value="work">{t('taskManager.categories.work')}</option>
                  <option value="personal">{t('taskManager.categories.personal')}</option>
                  <option value="health">{t('taskManager.categories.health')}</option>
                  <option value="learning">{t('taskManager.categories.learning')}</option>
                  <option value="other">{t('taskManager.categories.other')}</option>
                </select>
              </div>
            </div>

            {/* Suggestions IA */}
            {aiSuggestions.length > 0 && (
              <div className="ai-suggestions">
                <h4>{t('taskManager.aiSuggestions')}</h4>
                <div className="suggestions-list">
                  {aiSuggestions.map((suggestion, index) => (
                    <div 
                      key={index} 
                      className="suggestion-item"
                      onClick={() => setNewTask(prev => ({ ...prev, description: suggestion }))}
                    >
                      <span className="suggestion-icon">💡</span>
                      {suggestion}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="form-actions">
              <button 
                className="btn btn-secondary"
                onClick={() => {
                  setShowNewTaskForm(false);
                  setAiSuggestions([]);
                }}
              >
                {t('common.cancel')}
              </button>
              <button 
                className="btn btn-primary"
                onClick={handleAddTask}
                disabled={!newTask.title?.trim() || loading}
              >
                {loading ? '⏳' : '✅'} {t('taskManager.createTask')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Liste des tâches */}
      <div className="tasks-list">
        {filteredTasks.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📋</div>
            <h3>{t('taskManager.noTasks')}</h3>
            <p>{t('taskManager.noTasksDescription')}</p>
          </div>
        ) : (
          filteredTasks.map(task => (
            <div 
              key={task.id} 
              className={`task-item ${task.status} ${task.priority}-priority`}
            >
              <div className="task-main">
                <div className="task-checkbox">
                  <input
                    type="checkbox"
                    checked={task.status === 'completed'}
                    onChange={(e) => updateTaskStatus(task.id, e.target.checked ? 'completed' : 'pending')}
                    className="task-check"
                  />
                </div>

                <div className="task-content">
                  <div className="task-header">
                    <h4 className="task-title">{task.title}</h4>
                    <div className="task-badges">
                      <span className="priority-badge" style={{ backgroundColor: getPriorityColor(task.priority) }}>
                        {t(`taskManager.priorities.${task.priority}`)}
                      </span>
                      <span className="category-badge">
                        {getCategoryIcon(task.category)} {t(`taskManager.categories.${task.category}`)}
                      </span>
                      <span className="status-badge">
                        {getStatusIcon(task.status)} {t(`taskManager.status.${task.status}`)}
                      </span>
                    </div>
                  </div>

                  {task.description && (
                    <p className="task-description">{task.description}</p>
                  )}

                  <div className="task-meta">
                    <span className="task-date">
                      📅 {new Date(task.createdAt).toLocaleDateString()}
                    </span>
                    {task.dueDate && (
                      <span className="task-due">
                        ⏰ {new Date(task.dueDate).toLocaleDateString()}
                      </span>
                    )}
                    {task.estimatedTime && (
                      <span className="task-time">
                        ⏱️ {task.estimatedTime}min
                      </span>
                    )}
                  </div>

                  {task.tags.length > 0 && (
                    <div className="task-tags">
                      {task.tags.map(tag => (
                        <span key={tag} className="task-tag">#{tag}</span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="task-actions">
                  <button 
                    className="btn-action btn-edit"
                    title={t('taskManager.editTask')}
                  >
                    ✏️
                  </button>
                  <button 
                    className="btn-action btn-delete"
                    title={t('taskManager.deleteTask')}
                  >
                    🗑️
                  </button>
                </div>
              </div>

              {/* Suggestions IA pour cette tâche */}
              {task.aiSuggestions && task.aiSuggestions.length > 0 && (
                <div className="task-ai-suggestions">
                  <div className="ai-suggestions-header">
                    <span className="ai-icon">🤖</span>
                    {t('taskManager.aiRecommendations')}
                  </div>
                  <div className="ai-suggestions-content">
                    {task.aiSuggestions.slice(0, 2).map((suggestion, index) => (
                      <div key={index} className="ai-suggestion">
                        💡 {suggestion}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Statistiques rapides */}
      <div className="task-stats">
        <div className="stat-item">
          <span className="stat-number">{tasks.filter(t => t.status === 'pending').length}</span>
          <span className="stat-label">{t('taskManager.stats.pending')}</span>
        </div>
        <div className="stat-item">
          <span className="stat-number">{tasks.filter(t => t.status === 'completed').length}</span>
          <span className="stat-label">{t('taskManager.stats.completed')}</span>
        </div>
        <div className="stat-item">
          <span className="stat-number">
            {Math.round((tasks.filter(t => t.status === 'completed').length / tasks.length) * 100) || 0}%
          </span>
          <span className="stat-label">{t('taskManager.stats.completion')}</span>
        </div>
      </div>
    </div>
  );
};

export default TaskManager;