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

## 🚀 Démarrage Rapide

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

2. **Démarrer avec Docker**
   ```bash
   # Mode développement
   ./start_docker.sh

   # Mode production (avec PostgreSQL et Redis)
   ./start_docker.sh production
   ```

3. **Accéder à l'application**
   - 🌐 Frontend: http://localhost:3000
   - 📡 API: http://localhost:8000
   - 📚 Documentation API: http://localhost:8000/docs

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

## 🏗️ Architecture

```
EmoIA/
├── src/                    # Code source principal
│   ├── core/              # Logique métier principale
│   │   ├── api.py         # API FastAPI
│   │   └── emoia_main.py  # Classe principale EmoIA
│   ├── emotional/         # Module d'intelligence émotionnelle
│   ├── memory/            # Système de mémoire
│   └── models/            # Modèles et LLM
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

```yaml
emotional:
  empathy_threshold: 0.7
  emotional_intensity: 0.8
  base_personality:
    openness: 0.8
    conscientiousness: 0.7
```

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
- Contributeurs open source

## 📞 Contact

- Site web : [emoia.ai](https://emoia.ai)
- Email : contact@emoia.ai
- GitHub : [@emoia](https://github.com/emoia)

---

Fait avec ❤️ par l'équipe EmoIA
