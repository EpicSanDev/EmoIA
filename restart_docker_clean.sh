#!/bin/bash

echo "🧹 Nettoyage et redémarrage d'EmoIA avec Docker..."

# Arrêter tous les conteneurs
echo "🛑 Arrêt des conteneurs..."
docker-compose down

# Supprimer les images pour forcer la reconstruction
echo "🗑️  Suppression des images Docker EmoIA..."
docker rmi emoia-emoia-frontend emoia-emoia-api 2>/dev/null || true

# Nettoyer le cache Docker
echo "🧽 Nettoyage du cache Docker..."
docker system prune -f

# Redémarrer avec reconstruction complète
echo "🔨 Reconstruction et démarrage des services..."
./start_docker.sh

echo "✅ Redémarrage terminé!"