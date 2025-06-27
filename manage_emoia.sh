#!/bin/bash

# ==============================================================================
# EmoIA - Script de gestion et maintenance
# ==============================================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables
INSTALL_DIR="/opt/emoia"
SERVICE_USER="emoia"
SERVICES=("emoia-backend" "emoia-frontend" "emoia-telegram" "postgresql" "redis-server" "nginx")

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Ce script doit être exécuté en tant que root"
        exit 1
    fi
}

# ==============================================================================
# FONCTIONS DE GESTION DES SERVICES
# ==============================================================================

status_services() {
    echo -e "${BLUE}=== Statut des services EmoIA ===${NC}\n"
    
    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            echo -e "✅ ${GREEN}$service${NC}: ACTIF"
        else
            echo -e "❌ ${RED}$service${NC}: INACTIF"
        fi
    done
    
    echo ""
    
    # Afficher l'utilisation des ressources
    echo -e "${BLUE}=== Utilisation des ressources ===${NC}"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')%"
    echo "RAM: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
    echo "Disque: $(df -h $INSTALL_DIR | awk 'NR==2{print $5}')"
    
    # GPU si disponible
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
        echo "VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf("%.1f%%", $1/$2 * 100.0)}')"
    fi
}

start_services() {
    echo -e "${BLUE}=== Démarrage des services EmoIA ===${NC}\n"
    
    for service in "${SERVICES[@]}"; do
        echo "Démarrage de $service..."
        if systemctl start "$service"; then
            log_info "$service démarré avec succès"
        else
            log_error "Échec du démarrage de $service"
        fi
        sleep 2
    done
}

stop_services() {
    echo -e "${BLUE}=== Arrêt des services EmoIA ===${NC}\n"
    
    # Arrêter dans l'ordre inverse
    for ((i=${#SERVICES[@]}-1; i>=0; i--)); do
        service="${SERVICES[$i]}"
        echo "Arrêt de $service..."
        if systemctl stop "$service"; then
            log_info "$service arrêté avec succès"
        else
            log_warn "Échec de l'arrêt de $service"
        fi
        sleep 1
    done
}

restart_services() {
    echo -e "${BLUE}=== Redémarrage des services EmoIA ===${NC}\n"
    
    stop_services
    echo ""
    sleep 3
    start_services
}

# ==============================================================================
# FONCTIONS DE MONITORING
# ==============================================================================

show_logs() {
    local service="$1"
    local lines="${2:-50}"
    
    if [[ -z "$service" ]]; then
        echo "Usage: $0 logs <service> [lines]"
        echo "Services disponibles: ${SERVICES[*]}"
        return 1
    fi
    
    echo -e "${BLUE}=== Logs de $service (dernières $lines lignes) ===${NC}\n"
    journalctl -u "$service" -n "$lines" --no-pager
}

follow_logs() {
    local service="$1"
    
    if [[ -z "$service" ]]; then
        echo "Usage: $0 follow <service>"
        echo "Services disponibles: ${SERVICES[*]}"
        return 1
    fi
    
    echo -e "${BLUE}=== Suivi en temps réel des logs de $service ===${NC}"
    echo "Appuyez sur Ctrl+C pour arrêter"
    echo ""
    
    journalctl -u "$service" -f
}

show_performance() {
    echo -e "${BLUE}=== Métriques de performance ===${NC}\n"
    
    # Métriques système
    echo -e "${YELLOW}Système:${NC}"
    echo "  Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo "  Uptime: $(uptime -p)"
    echo "  Processus: $(ps aux | wc -l)"
    echo ""
    
    # Métriques mémoire
    echo -e "${YELLOW}Mémoire:${NC}"
    free -h
    echo ""
    
    # Métriques disque
    echo -e "${YELLOW}Disque:${NC}"
    df -h "$INSTALL_DIR"
    echo ""
    
    # Métriques réseau
    echo -e "${YELLOW}Réseau:${NC}"
    ss -tuln | grep -E ":(3000|8000|5432|6379|80|443)"
    echo ""
    
    # GPU si disponible
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}GPU:${NC}"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
    fi
}

# ==============================================================================
# FONCTIONS DE MAINTENANCE
# ==============================================================================

update_models() {
    echo -e "${BLUE}=== Mise à jour des modèles ML ===${NC}\n"
    
    log_info "Mise à jour des modèles de machine learning..."
    
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        source venv/bin/activate
        python -c '
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

models_dir = \"$INSTALL_DIR/models\"

print(\"Mise à jour du modèle de langage...\")
tokenizer = AutoTokenizer.from_pretrained(\"microsoft/DialoGPT-large\", cache_dir=models_dir)
model = AutoModel.from_pretrained(\"microsoft/DialoGPT-large\", cache_dir=models_dir)

print(\"Mise à jour du modèle d embedding...\")
embedding_model = SentenceTransformer(\"sentence-transformers/all-mpnet-base-v2\", cache_folder=models_dir)

print(\"Nettoyage du cache GPU...\")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(\"Modèles mis à jour avec succès!\")
'
    "
    
    log_info "Modèles mis à jour avec succès"
}

clean_cache() {
    echo -e "${BLUE}=== Nettoyage du cache ===${NC}\n"
    
    log_info "Nettoyage du cache applicatif..."
    
    # Nettoyer le cache Redis
    redis-cli -a "$(grep REDIS_PASSWORD $INSTALL_DIR/.env | cut -d'=' -f2)" FLUSHALL
    
    # Nettoyer le cache EmoIA
    sudo -u "$SERVICE_USER" rm -rf "$INSTALL_DIR/cache"/*
    sudo -u "$SERVICE_USER" mkdir -p "$INSTALL_DIR/cache"
    
    # Nettoyer les logs anciens
    find "$INSTALL_DIR/logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Nettoyer le cache système
    apt autoremove -y
    apt autoclean
    
    log_info "Cache nettoyé avec succès"
}

backup_data() {
    local backup_dir="/var/backups/emoia"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/emoia_backup_$timestamp.tar.gz"
    
    echo -e "${BLUE}=== Sauvegarde des données ===${NC}\n"
    
    mkdir -p "$backup_dir"
    
    log_info "Création de la sauvegarde..."
    
    # Sauvegarder la base de données
    sudo -u postgres pg_dump emoia_prod > "$backup_dir/database_$timestamp.sql"
    
    # Sauvegarder les fichiers de configuration et données
    tar -czf "$backup_file" \
        --exclude="$INSTALL_DIR/venv" \
        --exclude="$INSTALL_DIR/node_modules" \
        --exclude="$INSTALL_DIR/frontend/node_modules" \
        --exclude="$INSTALL_DIR/cache" \
        --exclude="$INSTALL_DIR/logs" \
        "$INSTALL_DIR"
    
    # Nettoyer les anciennes sauvegardes (garder 7 jours)
    find "$backup_dir" -name "emoia_backup_*.tar.gz" -mtime +7 -delete 2>/dev/null || true
    find "$backup_dir" -name "database_*.sql" -mtime +7 -delete 2>/dev/null || true
    
    log_info "Sauvegarde créée: $backup_file"
    log_info "Base de données sauvegardée: $backup_dir/database_$timestamp.sql"
}

update_system() {
    echo -e "${BLUE}=== Mise à jour du système ===${NC}\n"
    
    log_info "Mise à jour des paquets système..."
    apt update && apt upgrade -y
    
    log_info "Mise à jour des dépendances Python..."
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        source venv/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install --upgrade -r requirements.txt
    "
    
    log_info "Mise à jour des dépendances Node.js..."
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR/frontend'
        npm update
        npm audit fix
    "
    
    log_info "Système mis à jour avec succès"
}

# ==============================================================================
# FONCTIONS DE DIAGNOSTIC
# ==============================================================================

diagnose() {
    echo -e "${BLUE}=== Diagnostic EmoIA ===${NC}\n"
    
    # Vérifier les services
    echo -e "${YELLOW}Services:${NC}"
    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            echo "  ✅ $service"
        else
            echo "  ❌ $service"
            # Afficher la dernière erreur
            echo "     $(journalctl -u $service -n 1 --no-pager | tail -1)"
        fi
    done
    echo ""
    
    # Vérifier les ports
    echo -e "${YELLOW}Ports:${NC}"
    ports=(3000 8000 5432 6379 80)
    for port in "${ports[@]}"; do
        if ss -tuln | grep -q ":$port "; then
            echo "  ✅ Port $port ouvert"
        else
            echo "  ❌ Port $port fermé"
        fi
    done
    echo ""
    
    # Vérifier l'espace disque
    echo -e "${YELLOW}Espace disque:${NC}"
    usage=$(df "$INSTALL_DIR" | awk 'NR==2{print int($5)}')
    if [[ $usage -lt 80 ]]; then
        echo "  ✅ Espace disque: ${usage}% utilisé"
    else
        echo "  ⚠️  Espace disque: ${usage}% utilisé (critique)"
    fi
    echo ""
    
    # Vérifier la mémoire
    echo -e "${YELLOW}Mémoire:${NC}"
    mem_usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
    if [[ $mem_usage -lt 90 ]]; then
        echo "  ✅ Utilisation mémoire: ${mem_usage}%"
    else
        echo "  ⚠️  Utilisation mémoire: ${mem_usage}% (élevée)"
    fi
    echo ""
    
    # Vérifier GPU si disponible
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}GPU:${NC}"
        if nvidia-smi &> /dev/null; then
            echo "  ✅ GPU NVIDIA disponible"
            echo "  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        else
            echo "  ❌ Problème avec le GPU NVIDIA"
        fi
    fi
}

# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

show_help() {
    echo -e "${BLUE}EmoIA - Script de gestion${NC}\n"
    
    echo "Usage: $0 <commande> [options]"
    echo ""
    echo "Commandes de service:"
    echo "  status          Afficher le statut de tous les services"
    echo "  start           Démarrer tous les services"
    echo "  stop            Arrêter tous les services"
    echo "  restart         Redémarrer tous les services"
    echo ""
    echo "Commandes de monitoring:"
    echo "  logs <service>  Afficher les logs d'un service"
    echo "  follow <service> Suivre les logs en temps réel"
    echo "  performance     Afficher les métriques de performance"
    echo ""
    echo "Commandes de maintenance:"
    echo "  update-models   Mettre à jour les modèles ML"
    echo "  clean-cache     Nettoyer le cache"
    echo "  backup          Sauvegarder les données"
    echo "  update-system   Mettre à jour le système"
    echo ""
    echo "Commandes de diagnostic:"
    echo "  diagnose        Exécuter un diagnostic complet"
    echo ""
    echo "Services disponibles: ${SERVICES[*]}"
}

main() {
    local command="$1"
    shift || true
    
    case "$command" in
        "status")
            status_services
            ;;
        "start")
            check_root
            start_services
            ;;
        "stop")
            check_root
            stop_services
            ;;
        "restart")
            check_root
            restart_services
            ;;
        "logs")
            show_logs "$@"
            ;;
        "follow")
            follow_logs "$@"
            ;;
        "performance")
            show_performance
            ;;
        "update-models")
            check_root
            update_models
            ;;
        "clean-cache")
            check_root
            clean_cache
            ;;
        "backup")
            check_root
            backup_data
            ;;
        "update-system")
            check_root
            update_system
            ;;
        "diagnose")
            diagnose
            ;;
        "help"|"-h"|"--help"|"")
            show_help
            ;;
        *)
            echo "Commande inconnue: $command"
            echo "Utilisez '$0 help' pour voir les commandes disponibles"
            exit 1
            ;;
    esac
}

main "$@"