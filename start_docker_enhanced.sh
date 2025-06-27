#!/bin/bash

# EmoIA Enhanced Docker Startup Script
# Avec support Ollama et MCP intégré

set -e

echo "🚀 Démarrage d'EmoIA Enhanced avec Ollama et MCP..."

# Couleurs pour le terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker n'est pas installé!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose n'est pas installé!"
    exit 1
fi

# Créer les répertoires nécessaires
log_info "Création des répertoires nécessaires..."
mkdir -p data logs models cache models/ollama monitoring/prometheus monitoring/grafana/dashboards

# Vérifier le mode de démarrage
MODE=${1:-development}

log_info "Mode de démarrage: ${PURPLE}$MODE${NC}"

# Nettoyer les anciens conteneurs si demandé
if [ "$2" == "--clean" ]; then
    log_warning "Nettoyage des conteneurs existants..."
    docker-compose down -v
    docker system prune -f
fi

# Construire les images
log_info "Construction des images Docker..."
docker-compose build --parallel

# Fonction pour vérifier si un service est prêt
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log_info "En attente du service $service..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log_success "$service est prêt!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service n'a pas démarré après $max_attempts tentatives"
    return 1
}

# Démarrer les services selon le mode
case $MODE in
    development|dev)
        log_info "Démarrage en mode développement..."
        docker-compose up -d emoia-api emoia-frontend ollama
        ;;
    
    production|prod)
        log_info "Démarrage en mode production..."
        docker-compose --profile production up -d
        ;;
    
    monitoring|mon)
        log_info "Démarrage avec monitoring..."
        docker-compose --profile production --profile monitoring up -d
        ;;
    
    *)
        log_error "Mode inconnu: $MODE"
        echo "Usage: $0 [development|production|monitoring] [--clean]"
        exit 1
        ;;
esac

# Attendre que les services soient prêts
log_info "Vérification des services..."
wait_for_service "Ollama" "http://localhost:11434/api/tags"
wait_for_service "API Backend" "http://localhost:8000/health"
wait_for_service "Frontend" "http://localhost:3000"

# Afficher l'état des services
log_info "État des services:"
docker-compose ps

# Afficher les URLs d'accès
echo ""
log_success "🎉 EmoIA est maintenant opérationnel!"
echo ""
echo -e "${CYAN}URLs d'accès:${NC}"
echo -e "  📱 Frontend:        ${GREEN}http://localhost:3000${NC}"
echo -e "  🔌 API Backend:     ${GREEN}http://localhost:8000${NC}"
echo -e "  📚 API Docs:        ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  🦙 Ollama:          ${GREEN}http://localhost:11434${NC}"

if [ "$MODE" == "production" ] || [ "$MODE" == "monitoring" ]; then
    echo -e "  🗄️  PostgreSQL:      ${GREEN}localhost:5432${NC}"
    echo -e "  📦 Redis:           ${GREEN}localhost:6379${NC}"
fi

if [ "$MODE" == "monitoring" ]; then
    echo -e "  📊 Prometheus:      ${GREEN}http://localhost:9090${NC}"
    echo -e "  📈 Grafana:         ${GREEN}http://localhost:3001${NC} (admin/admin)"
fi

echo ""
echo -e "${PURPLE}Modèles Ollama disponibles:${NC}"
curl -s http://localhost:11434/api/tags | jq -r '.models[]?.name' 2>/dev/null || echo "En cours de chargement..."

echo ""
echo -e "${YELLOW}Commandes utiles:${NC}"
echo "  - Voir les logs:        docker-compose logs -f"
echo "  - Arrêter:              docker-compose down"
echo "  - Redémarrer:           docker-compose restart"
echo "  - Pull un modèle:       docker exec emoia-ollama ollama pull <model>"
echo ""

# Fonction pour suivre les logs si demandé
if [ "$3" == "--logs" ]; then
    log_info "Affichage des logs..."
    docker-compose logs -f
fi