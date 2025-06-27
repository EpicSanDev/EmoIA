#!/bin/bash

# ==============================================================================
# EmoIA - Vérification des prérequis système
# ==============================================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables
MIN_RAM_GB=16
RECOMMENDED_RAM_GB=64
MIN_DISK_GB=100
RECOMMENDED_DISK_GB=500

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

log_info() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_check() {
    echo -e "${BLUE}[?]${NC} $1"
}

# ==============================================================================
# VÉRIFICATIONS SYSTÈME
# ==============================================================================

check_os() {
    log_check "Vérification du système d'exploitation..."
    
    if [[ ! -f /etc/os-release ]]; then
        log_error "Impossible de détecter le système d'exploitation"
        return 1
    fi
    
    source /etc/os-release
    
    if [[ "$ID" != "ubuntu" ]]; then
        log_error "Système non supporté: $PRETTY_NAME"
        log_error "EmoIA nécessite Ubuntu 20.04 LTS ou 22.04 LTS"
        return 1
    fi
    
    version_id_major=$(echo "$VERSION_ID" | cut -d. -f1)
    
    if [[ "$version_id_major" -lt 20 ]]; then
        log_error "Version Ubuntu trop ancienne: $PRETTY_NAME"
        log_error "EmoIA nécessite Ubuntu 20.04 LTS ou supérieur"
        return 1
    fi
    
    log_info "Système d'exploitation compatible: $PRETTY_NAME"
    return 0
}

check_cpu() {
    log_check "Vérification du processeur..."
    
    cpu_cores=$(nproc)
    cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    
    if [[ $cpu_cores -lt 4 ]]; then
        log_error "Processeur insuffisant: $cpu_cores cœurs (minimum 4 cœurs)"
        return 1
    elif [[ $cpu_cores -lt 8 ]]; then
        log_warn "Processeur correct mais pas optimal: $cpu_cores cœurs (recommandé 8+ cœurs)"
    else
        log_info "Processeur compatible: $cpu_cores cœurs"
    fi
    
    log_info "Modèle CPU: $cpu_model"
    return 0
}

check_memory() {
    log_check "Vérification de la mémoire RAM..."
    
    total_ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    total_ram_gb=$((total_ram_kb / 1024 / 1024))
    
    if [[ $total_ram_gb -lt $MIN_RAM_GB ]]; then
        log_error "Mémoire RAM insuffisante: ${total_ram_gb}GB (minimum ${MIN_RAM_GB}GB)"
        return 1
    elif [[ $total_ram_gb -lt $RECOMMENDED_RAM_GB ]]; then
        log_warn "Mémoire RAM correcte mais pas optimale: ${total_ram_gb}GB (recommandé ${RECOMMENDED_RAM_GB}GB)"
    else
        log_info "Mémoire RAM optimale: ${total_ram_gb}GB"
    fi
    
    return 0
}

check_disk() {
    log_check "Vérification de l'espace disque..."
    
    available_gb=$(df / | awk 'NR==2{printf "%.0f", $4/1024/1024}')
    
    if [[ $available_gb -lt $MIN_DISK_GB ]]; then
        log_error "Espace disque insuffisant: ${available_gb}GB disponible (minimum ${MIN_DISK_GB}GB)"
        return 1
    elif [[ $available_gb -lt $RECOMMENDED_DISK_GB ]]; then
        log_warn "Espace disque correct mais pas optimal: ${available_gb}GB disponible (recommandé ${RECOMMENDED_DISK_GB}GB)"
    else
        log_info "Espace disque optimal: ${available_gb}GB disponible"
    fi
    
    # Vérifier le type de stockage
    if mount | grep -q "/ type ext4"; then
        log_info "Système de fichiers: ext4 (recommandé)"
    else
        log_warn "Système de fichiers non optimal (ext4 recommandé)"
    fi
    
    return 0
}

check_network() {
    log_check "Vérification de la connectivité réseau..."
    
    # Test connectivité internet
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        log_info "Connectivité internet: OK"
    else
        log_error "Pas de connectivité internet"
        return 1
    fi
    
    # Test résolution DNS
    if nslookup google.com > /dev/null 2>&1; then
        log_info "Résolution DNS: OK"
    else
        log_warn "Problème de résolution DNS"
    fi
    
    return 0
}

check_gpu() {
    log_check "Vérification du GPU NVIDIA (optionnel)..."
    
    if lspci | grep -i nvidia > /dev/null 2>&1; then
        gpu_info=$(lspci | grep -i nvidia | head -1 | cut -d: -f3 | xargs)
        log_info "GPU NVIDIA détecté: $gpu_info"
        
        # Vérifier si les drivers sont installés
        if command -v nvidia-smi > /dev/null 2>&1; then
            log_info "Drivers NVIDIA déjà installés"
            nvidia_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)
            log_info "Version driver: $nvidia_version"
        else
            log_warn "GPU NVIDIA détecté mais drivers non installés (seront installés automatiquement)"
        fi
        
        return 0
    else
        log_warn "Aucun GPU NVIDIA détecté (EmoIA fonctionnera en mode CPU)"
        return 0
    fi
}

check_privileges() {
    log_check "Vérification des privilèges..."
    
    if [[ $EUID -eq 0 ]]; then
        log_info "Exécuté en tant que root"
        return 0
    elif sudo -n true 2>/dev/null; then
        log_info "Privilèges sudo disponibles"
        return 0
    else
        log_error "Privilèges root/sudo requis"
        return 1
    fi
}

check_ports() {
    log_check "Vérification des ports requis..."
    
    ports=(80 443 3000 8000 5432 6379)
    ports_busy=()
    
    for port in "${ports[@]}"; do
        if ss -tuln | grep -q ":$port "; then
            ports_busy+=($port)
        fi
    done
    
    if [[ ${#ports_busy[@]} -eq 0 ]]; then
        log_info "Tous les ports requis sont disponibles"
        return 0
    else
        log_warn "Ports occupés: ${ports_busy[*]} (peuvent être libérés pendant l'installation)"
        return 0
    fi
}

check_dependencies() {
    log_check "Vérification des dépendances système de base..."
    
    required_commands=("curl" "wget" "git")
    missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" > /dev/null 2>&1; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [[ ${#missing_commands[@]} -eq 0 ]]; then
        log_info "Toutes les dépendances de base sont disponibles"
        return 0
    else
        log_warn "Dépendances manquantes: ${missing_commands[*]} (seront installées automatiquement)"
        return 0
    fi
}

# ==============================================================================
# ESTIMATION DES PERFORMANCES
# ==============================================================================

estimate_performance() {
    echo -e "\n${BLUE}=== Estimation des performances ===${NC}"
    
    cpu_cores=$(nproc)
    total_ram_gb=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
    
    # Score de performance basé sur CPU et RAM
    cpu_score=$((cpu_cores * 10))
    ram_score=$((total_ram_gb))
    
    if lspci | grep -i nvidia > /dev/null 2>&1; then
        gpu_score=50
        log_info "Mode GPU activé - Performance optimale"
    else
        gpu_score=0
        log_warn "Mode CPU uniquement - Performance réduite"
    fi
    
    total_score=$((cpu_score + ram_score + gpu_score))
    
    if [[ $total_score -gt 150 ]]; then
        log_info "Performance estimée: Excellente (score: $total_score)"
        log_info "Temps d'inférence: < 100ms"
        log_info "Modèles recommandés: Tous les modèles supportés"
    elif [[ $total_score -gt 100 ]]; then
        log_info "Performance estimée: Bonne (score: $total_score)"
        log_info "Temps d'inférence: 100-300ms"
        log_info "Modèles recommandés: Modèles standard"
    elif [[ $total_score -gt 60 ]]; then
        log_warn "Performance estimée: Correcte (score: $total_score)"
        log_warn "Temps d'inférence: 300-1000ms"
        log_warn "Modèles recommandés: Modèles légers"
    else
        log_error "Performance estimée: Insuffisante (score: $total_score)"
        log_error "Temps d'inférence: > 1000ms"
        log_error "Configuration non recommandée pour la production"
    fi
}

# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

main() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "           EmoIA - Vérification des prérequis système"
    echo "================================================================"
    echo -e "${NC}"
    
    failed_checks=0
    warning_checks=0
    
    # Exécuter toutes les vérifications
    checks=(
        "check_os"
        "check_privileges"
        "check_cpu"
        "check_memory"
        "check_disk"
        "check_network"
        "check_gpu"
        "check_ports"
        "check_dependencies"
    )
    
    for check in "${checks[@]}"; do
        if ! $check; then
            ((failed_checks++))
        fi
    done
    
    # Estimation des performances
    estimate_performance
    
    # Résumé final
    echo -e "\n${BLUE}=== Résumé de la vérification ===${NC}"
    
    if [[ $failed_checks -eq 0 ]]; then
        log_info "Toutes les vérifications sont passées avec succès !"
        log_info "Votre système est compatible avec EmoIA"
        echo -e "\n${GREEN}Vous pouvez procéder à l'installation avec:${NC}"
        echo -e "${GREEN}sudo ./install_ubuntu_server.sh${NC}"
    else
        log_error "$failed_checks vérification(s) ont échoué"
        log_error "Veuillez corriger ces problèmes avant l'installation"
        
        echo -e "\n${YELLOW}Actions recommandées:${NC}"
        if [[ $failed_checks -gt 0 ]]; then
            echo -e "  1. Vérifiez la configuration matérielle"
            echo -e "  2. Mettez à jour le système: sudo apt update && sudo apt upgrade"
            echo -e "  3. Libérez de l'espace disque si nécessaire"
            echo -e "  4. Vérifiez la connectivité réseau"
        fi
    fi
    
    echo -e "\n${BLUE}Configuration détectée:${NC}"
    echo -e "  OS: $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
    echo -e "  CPU: $(nproc) cœurs"
    echo -e "  RAM: $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))GB"
    echo -e "  Disque libre: $(df / | awk 'NR==2{printf "%.0f", $4/1024/1024}')GB"
    
    if lspci | grep -i nvidia > /dev/null 2>&1; then
        echo -e "  GPU: $(lspci | grep -i nvidia | head -1 | cut -d: -f3 | xargs)"
    else
        echo -e "  GPU: Aucun GPU NVIDIA détecté"
    fi
    
    return $failed_checks
}

# Exécuter la vérification
main "$@"