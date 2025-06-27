#!/bin/bash

echo "🧹 Nettoyage d'EmoIA..."
echo "============================================"

# Arrêter les processus en cours
echo "🛑 Arrêt des processus en cours..."
pkill -f "uvicorn src.core.api:app" 2>/dev/null
pkill -f "npm start" 2>/dev/null

# Nettoyer l'environnement virtuel Python
echo "🐍 Nettoyage de l'environnement virtuel Python..."
if [ -d "venv" ]; then
    rm -rf venv
fi

# Nettoyer le cache npm
echo "📦 Nettoyage du cache npm..."
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
cd ..

# Nettoyer les caches Python
echo "🐍 Nettoyage des caches Python..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Nettoyer les logs et caches
echo "🗑️ Nettoyage des logs et caches..."
rm -rf logs/* cache/* 2>/dev/null

echo "✅ Nettoyage terminé !"
echo "============================================"
echo "Vous pouvez maintenant relancer start.sh" 