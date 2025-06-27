#!/bin/bash

# Script de démarrage EmoIA
echo "🚀 Démarrage d'EmoIA - Intelligence Artificielle Émotionnelle"
echo "============================================"

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Vérifier si Node.js est installé
if ! command -v node &> /dev/null; then
    echo "❌ Node.js n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Créer un environnement virtuel Python si nécessaire
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances Python
echo "📦 Installation des dépendances Python..."
pip install -r requirements.txt

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p logs data models cache

# Télécharger les modèles NLTK si nécessaire
echo "📥 Téléchargement des ressources NLTK..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Installer les dépendances du frontend
echo "📦 Installation des dépendances du frontend..."
cd frontend
npm install
cd ..

# Démarrer les services
echo "🚀 Démarrage des services..."

# Démarrer l'API backend en arrière-plan
echo "🔧 Démarrage de l'API backend..."
python3 -m uvicorn src.core.api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Attendre que le backend soit prêt
echo "⏳ Attente du démarrage du backend..."
sleep 5

# Démarrer le frontend
echo "🎨 Démarrage du frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ EmoIA est maintenant démarré !"
echo "============================================"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API Backend: http://localhost:8000"
echo "📚 Documentation API: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Pour arrêter l'application, appuyez sur Ctrl+C"

# Fonction pour nettoyer à la sortie
cleanup() {
    echo ""
    echo "🛑 Arrêt d'EmoIA..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Capturer Ctrl+C
trap cleanup INT

# Attendre
wait