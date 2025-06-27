#!/bin/bash

# Script de démarrage d'EmoIA avec Docker

echo "🚀 Démarrage d'EmoIA..."

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez installer Docker Desktop."
    exit 1
fi

# Vérifier si docker-compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose n'est pas installé. Veuillez installer docker-compose."
    exit 1
fi

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p data logs models cache

# Arrêter les conteneurs existants
echo "🛑 Arrêt des conteneurs existants..."
docker-compose down

# Mode de démarrage
if [ "$1" = "production" ]; then
    echo "🏭 Démarrage en mode PRODUCTION avec PostgreSQL et Redis..."
    docker-compose --profile production up --build -d
else
    echo "🔧 Démarrage en mode DÉVELOPPEMENT..."
    docker-compose up --build -d
fi

# Attendre que les services soient prêts
echo "⏳ Attente du démarrage des services..."
sleep 10

# Vérifier l'état des services
echo "✅ Vérification des services..."
docker-compose ps

# Afficher les URLs
echo ""
echo "🎉 EmoIA est démarré!"
echo "📡 API Backend: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📊 Documentation API: http://localhost:8000/docs"
echo ""
echo "Pour voir les logs: docker-compose logs -f"
echo "Pour arrêter: docker-compose down"