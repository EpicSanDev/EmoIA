# EmoIA - Intelligence Artificielle Émotionnelle 🤖❤️

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/emoia/emoia)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

EmoIA est une intelligence artificielle avancée dotée de capacités émotionnelles, conçue pour créer des interactions plus naturelles et empathiques entre humains et machines.

## 🌟 Fonctionnalités Principales

- **🎭 Analyse Émotionnelle Avancée** : Détection et analyse des émotions dans les conversations avec 11 émotions distinctes
- **🧠 Profil de Personnalité** : Analyse de personnalité basée sur le modèle Big Five avec extensions émotionnelles
- **💾 Mémoire Intelligente** : Système de mémoire hiérarchique avec consolidation automatique
- **🌍 Multilingue** : Support natif du français, anglais et espagnol avec détection automatique
- **📊 Analytics en Temps Réel** : Dashboard avec visualisations des émotions et insights
- **🤝 Suggestions Contextuelles** : Génération de suggestions intelligentes basées sur le contexte émotionnel
- **🔄 Apprentissage Continu** : Amélioration constante basée sur les interactions
- **🦙 Multi-Modèles IA** : Support de multiples modèles via MCP (Ollama, OpenAI, etc.)
- **🎨 Interface UI/UX Moderne** : Design professionnel et responsive

## 🆕 Nouvelles Fonctionnalités v3.0

### � Model Context Protocol (MCP)
- Architecture flexible pour intégrer différents modèles IA
- Changement de modèle en temps réel
- Support du streaming pour les réponses
- Gestion unifiée des contextes de conversation

### 🦙 Intégration Ollama
- Modèles IA locaux (Llama2, Mistral, Phi, etc.)
- Pas de dépendance cloud
- Performance optimisée avec support GPU
- Installation automatique des modèles

### 🎨 Interface Améliorée
- Sélecteur de modèles intégré
- Visualisations temps réel des émotions
- Design moderne avec thème clair/sombre
- Composants React optimisés

## �🚀 Démarrage Rapide

### Prérequis

- Docker et Docker Compose
- Python 3.11+ (pour le développement local)
- Node.js 18+ (pour le frontend)

### Installation avec Docker (Recommandé)

1. **Cloner le repository**
   ```bash
   git clone https://github.com/emoia/emoia.git
   cd emoia
   ```

2. **Démarrer avec le script amélioré**
   ```bash
   # Rendre le script exécutable
   chmod +x start_docker_enhanced.sh
   
   # Mode développement (recommandé pour commencer)
   ./start_docker_enhanced.sh development
   
   # Mode production avec toutes les fonctionnalités
   ./start_docker_enhanced.sh production
   
   # Mode avec monitoring (Prometheus + Grafana)
   ./start_docker_enhanced.sh monitoring
   ```

3. **Accéder à l'application**
   - 🌐 Frontend: http://localhost:3000
   - 📡 API: http://localhost:8000
   - 📚 Documentation API: http://localhost:8000/docs
   - 🦙 Ollama: http://localhost:11434

### Installation Manuelle

1. **Backend**
   ```bash
   # Créer un environnement virtuel
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows

   # Installer les dépendances
   pip install -r requirements.txt

   # Démarrer l'API
   python -m uvicorn src.core.api:app --reload
   ```

2. **Frontend**
   ```bash
   cd frontend
   npm install
   npm start
   ```

## 📖 Guide d'Utilisation

### API REST

#### Chat avec l'IA
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Bonjour, comment allez-vous?",
    "preferences": {
      "language": "fr"
    }
  }'
```

#### Chat avec MCP (Multi-Modèles)
```bash
curl -X POST "http://localhost:8000/mcp/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Raconte-moi une histoire",
    "provider": "ollama",
    "model": "llama2"
  }'
```

#### Obtenir des suggestions
```bash
curl -X POST "http://localhost:8000/suggestions" \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Je me sens un peu stressé par le travail",
    "emotional_state": {"dominant_emotion": "anxiety"},
    "max_suggestions": 5
  }'
```

### WebSocket pour le temps réel

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  // S'identifier
  ws.send(JSON.stringify({
    type: 'identify',
    user_id: 'user123'
  }));

  // Envoyer un message
  ws.send(JSON.stringify({
    type: 'chat_message',
    message: 'Bonjour!',
    user_id: 'user123'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Réponse:', data);
};
```

### WebSocket MCP (Streaming)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/mcp');

ws.send(JSON.stringify({
  type: 'mcp_stream',
  provider: 'ollama',
  model: 'mistral',
  message: 'Explique-moi la photosynthèse'
}));
```

## 🏗️ Architecture

```
EmoIA/
├── src/                    # Code source principal
│   ├── core/              # Logique métier principale
│   │   ├── api.py         # API FastAPI
│   │   └── emoia_main.py  # Classe principale EmoIA
│   ├── emotional/         # Module d'intelligence émotionnelle
│   ├── memory/            # Système de mémoire
│   ├── models/            # Modèles et LLM
│   └── mcp/               # Model Context Protocol
│       ├── mcp_manager.py # Gestionnaire MCP
│       └── providers/     # Providers de modèles
├── frontend/              # Application React
│   ├── src/
│   │   ├── components/    # Composants React
│   │   └── App.tsx        # Composant principal
├── tests/                 # Tests unitaires
├── config.yaml            # Configuration
└── docker-compose.yml     # Orchestration Docker
```

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest

# Tests avec couverture
pytest --cov=src tests/

# Tests spécifiques
pytest tests/test_emoia_main.py
pytest tests/test_mcp.py
```

## 📊 Analyse Émotionnelle

EmoIA analyse 11 émotions distinctes :
- **Émotions primaires** : joie, tristesse, colère, peur, surprise, dégoût
- **Émotions complexes** : amour, excitation, anxiété, contentement, curiosité

Chaque émotion est évaluée sur une échelle de 0 à 1 avec un score de confiance.

## 🔧 Configuration

Le fichier `config.yaml` permet de personnaliser :
- Modèles d'IA utilisés
- Paramètres émotionnels
- Configuration de la mémoire
- Paramètres d'apprentissage
- Configuration MCP

```yaml
emotional:
  empathy_threshold: 0.7
  emotional_intensity: 0.8
  base_personality:
    openness: 0.8
    conscientiousness: 0.7

mcp:
  default_provider: ollama
  providers:
    ollama:
      base_url: http://ollama:11434
      default_model: llama2
```

## 🎯 Gestion des Modèles

### Installer de nouveaux modèles Ollama

```bash
# Lister les modèles disponibles
docker exec emoia-ollama ollama list

# Installer un modèle
docker exec emoia-ollama ollama pull llama2:13b
docker exec emoia-ollama ollama pull mistral
docker exec emoia-ollama ollama pull codellama
```

### Changer de modèle via l'API

```bash
curl -X POST "http://localhost:8000/mcp/switch-model" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "provider": "ollama",
    "model": "mistral"
  }'
```

## 📚 Documentation Complète

- [Architecture MCP et Ollama](README_MCP_OLLAMA.md)
- [Guide d'installation détaillé](README_INSTALL.md)
- [Documentation Frontend](README_FRONTEND.md)
- [Guide de migration](MIGRATION_GUIDE.md)

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez consulter [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- Équipe Hugging Face pour les modèles de transformers
- Communauté FastAPI pour le framework web
- Équipe Ollama pour les modèles locaux
- Contributeurs open source

## 📞 Contact

- Site web : [emoia.ai](https://emoia.ai)
- Email : contact@emoia.ai
- GitHub : [@emoia](https://github.com/emoia)

---

Fait avec ❤️ par l'équipe EmoIA
