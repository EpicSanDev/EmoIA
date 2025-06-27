# 🤖 EmoIA - Intelligence Artificielle Émotionnelle

<div align="center">
  <img src="https://img.shields.io/badge/Version-3.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.8+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/React-18.2+-61DAFB.svg" alt="React">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

## 📋 Table des matières

- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [API Documentation](#api-documentation)
- [Développement](#développement)
- [Dépannage](#dépannage)

## 🌟 Introduction

EmoIA est une intelligence artificielle émotionnelle avancée qui comprend et s'adapte aux émotions humaines. Elle offre une expérience conversationnelle empathique et personnalisée grâce à l'analyse émotionnelle en temps réel et l'apprentissage continu.

### Points forts
- 🎭 **Analyse émotionnelle avancée** : Détection et compréhension des émotions
- 🧠 **Mémoire intelligente** : Apprentissage et mémorisation des interactions
- 🌍 **Multilingue** : Support français, anglais et espagnol
- 🎨 **Interface moderne** : Design épuré avec thèmes clair/sombre
- 📊 **Analytics en temps réel** : Tableaux de bord et insights émotionnels
- 🔒 **Respect de la vie privée** : Données stockées localement

## 🏗️ Architecture

```
EmoIA/
├── frontend/          # Interface React
│   ├── src/
│   │   ├── components/
│   │   ├── App.tsx
│   │   └── i18n.ts
│   └── public/
├── src/              # Backend Python
│   ├── core/         # Logique principale
│   ├── emotional/    # Analyse émotionnelle
│   ├── memory/       # Système de mémoire
│   ├── models/       # Modèles IA
│   └── analytics/    # WebSocket temps réel
├── config.yaml       # Configuration
└── main.py          # Point d'entrée CLI
```

## 🚀 Installation

### Prérequis

- **Python 3.8+**
- **Node.js 16+** et npm
- **Git**
- 8GB RAM minimum
- 10GB d'espace disque (pour les modèles IA)

### Installation rapide (Linux/macOS)

```bash
# Cloner le projet
git clone https://github.com/votre-repo/EmoIA.git
cd EmoIA

# Lancer l'installation automatique
./start.sh
```

### Installation manuelle

#### 1. Backend Python

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

#### 2. Frontend React

```bash
cd frontend
npm install
```

#### 3. Configuration Ollama (optionnel)

Pour utiliser Mistral en local :

```bash
# Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Télécharger Mistral
ollama pull mistral
```

## ⚙️ Configuration

### Configuration de base

Éditez `config.yaml` selon vos besoins :

```yaml
# Modèle de langage
models:
  language_model: "mistralai/Mistral-7B-Instruct-v0.2"
  language_model_device: "auto"  # "cpu", "cuda", ou "auto"

# Paramètres émotionnels
emotional:
  empathy_threshold: 0.7
  emotional_intensity: 0.8

# API
communication:
  api_host: "localhost"
  api_port: 8000
```

### Variables d'environnement (optionnel)

Créez un fichier `.env` :

```env
# Pour utiliser OpenAI au lieu de Mistral
OPENAI_API_KEY=your_api_key

# Pour Telegram Bot (optionnel)
TELEGRAM_TOKEN=your_telegram_token
```

## 📱 Utilisation

### Démarrage rapide

```bash
./start.sh
```

Ou manuellement :

```bash
# Terminal 1 - Backend
python -m uvicorn src.core.api:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend && npm start
```

### Accès aux services

- 🌐 **Interface Web** : http://localhost:3000
- 🔧 **API REST** : http://localhost:8000
- 📚 **Documentation API** : http://localhost:8000/docs
- 🔌 **WebSocket** : ws://localhost:8001/ws/chat

### Interface en ligne de commande

```bash
python main.py
```

Commandes disponibles :
- `help` : Afficher l'aide
- `stats` : Statistiques système
- `insights` : Insights émotionnels
- `quit` : Quitter

## 📡 API Documentation

### Endpoints principaux

#### Chat
```http
POST /chat
{
  "user_id": "string",
  "message": "string"
}
```

#### Analytics
```http
GET /analytics/{user_id}
```

#### Préférences
```http
GET /utilisateur/preferences/{user_id}
POST /utilisateur/preferences
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/chat');

ws.send(JSON.stringify({
  user_id: "demo-user",
  message: "Bonjour EmoIA!"
}));
```

## 🛠️ Développement

### Structure du code

```python
# Exemple d'utilisation de l'API Python
from src.core import EmoIA
from src.config import Config

async def main():
    config = Config()
    emoia = EmoIA(config)
    await emoia.initialize()
    
    response = await emoia.process_message(
        user_input="Comment vas-tu ?",
        user_id="dev-user"
    )
    print(response)
```

### Ajout de nouvelles fonctionnalités

1. **Nouvelles émotions** : Modifier `src/emotional/core.py`
2. **Nouveaux modèles** : Ajouter dans `src/models/`
3. **Nouvelles langues** : Éditer `frontend/src/i18n.ts`

### Tests

```bash
# Tests backend
pytest tests/

# Tests frontend
cd frontend && npm test
```

## 🔧 Dépannage

### Problèmes courants

#### Erreur de mémoire
```bash
# Augmenter la limite de mémoire Python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Port déjà utilisé
```bash
# Changer le port dans config.yaml
api_port: 8001  # au lieu de 8000
```

#### Modèles non trouvés
```bash
# Vérifier que les modèles sont téléchargés
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

### Logs

Les logs sont disponibles dans :
- `logs/emoia.log` : Logs principaux
- Console : Logs en temps réel

## 📈 Performance

### Optimisations recommandées

1. **GPU** : Utiliser CUDA pour de meilleures performances
2. **Cache Redis** : Activer pour les déploiements multi-utilisateurs
3. **Modèles quantifiés** : Réduire l'utilisation mémoire

### Métriques

- Temps de réponse moyen : < 500ms
- Utilisation mémoire : ~4GB (CPU) / ~6GB (GPU)
- Connexions simultanées : 100+

## 🤝 Contribution

Les contributions sont bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE)

## 🙏 Remerciements

- Hugging Face pour les modèles de transformers
- FastAPI pour le framework API
- React pour l'interface utilisateur
- La communauté open source

---

<div align="center">
  Fait avec ❤️ par l'équipe EmoIA
</div>