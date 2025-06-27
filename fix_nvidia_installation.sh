#!/bin/bash

# ==============================================================================
# EmoIA - Script de correction pour l'installation NVIDIA
# Résout les erreurs d'installation des drivers NVIDIA
# ==============================================================================

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

log_step() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Ce script doit être exécuté en tant que root"
        exit 1
    fi
}

# ==============================================================================
# NETTOYAGE DES INSTALLATIONS NVIDIA EXISTANTES
# ==============================================================================

cleanup_nvidia() {
    log_step "Nettoyage des installations NVIDIA existantes"
    
    log_info "Arrêt des services utilisant NVIDIA..."
    systemctl stop display-manager || true
    
    log_info "Suppression des packages NVIDIA existants..."
    apt remove --purge -y \
        nvidia-* \
        libnvidia-* \
        cuda-* \
        libcuda* \
        libcudnn* || true
    
    log_info "Nettoyage des dépendances orphelines..."
    apt autoremove -y
    apt autoclean
    
    # Nettoyer les fichiers de configuration
    log_info "Suppression des fichiers de configuration..."
    rm -rf /etc/modprobe.d/blacklist-nvidia* || true
    rm -rf /etc/modprobe.d/nvidia* || true
    
    # Nettoyer le cache apt
    log_info "Nettoyage du cache apt..."
    apt clean
    
    log_info "Nettoyage terminé"
}

# ==============================================================================
# INSTALLATION DES PRÉREQUIS
# ==============================================================================

install_prerequisites() {
    log_step "Installation des prérequis pour NVIDIA"
    
    log_info "Mise à jour de la liste des paquets..."
    apt update
    
    log_info "Installation des headers du kernel..."
    apt install -y \
        linux-headers-$(uname -r) \
        build-essential \
        dkms \
        software-properties-common \
        pkg-config \
        libvulkan1 \
        mesa-vulkan-drivers
    
    log_info "Prérequis installés"
}

# ==============================================================================
# INSTALLATION NVIDIA AMÉLIORÉE
# ==============================================================================

install_nvidia_improved() {
    log_step "Installation améliorée des drivers NVIDIA"
    
    # Vérifier la présence d'une carte NVIDIA
    if ! lspci | grep -i nvidia > /dev/null 2>&1; then
        log_warn "Aucune carte NVIDIA détectée. Installation ignorée."
        return 0
    fi
    
    log_info "Carte NVIDIA détectée:"
    lspci | grep -i nvidia
    
    # Désactiver nouveau (driver open source)
    log_info "Configuration des modules kernel..."
    cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF
    
    # Mise à jour de l'initramfs
    update-initramfs -u
    
    # Ajouter le repository NVIDIA avec gestion d'erreur
    log_info "Ajout du repository NVIDIA..."
    cd /tmp
    if [[ -f cuda-keyring_1.0-1_all.deb ]]; then
        rm -f cuda-keyring_1.0-1_all.deb
    fi
    
    wget -O cuda-keyring_1.0-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb || {
        log_error "Échec de l'installation du keyring NVIDIA"
        apt --fix-broken install -y
        dpkg -i cuda-keyring_1.0-1_all.deb
    }
    
    apt update
    
    # Installation progressive des drivers NVIDIA
    log_info "Installation des drivers NVIDIA avec gestion d'erreur..."
    
    # Méthode 1: Essayer l'installation recommandée
    if ! apt install -y nvidia-driver-535; then
        log_warn "Échec de l'installation du driver 535, essai avec une version alternative..."
        
        # Méthode 2: Essayer avec le meta-package
        if ! apt install -y nvidia-driver-525; then
            log_warn "Échec du driver 525, essai avec ubuntu-drivers..."
            
            # Méthode 3: Utiliser ubuntu-drivers
            apt install -y ubuntu-drivers-common
            ubuntu-drivers autoinstall || {
                log_error "Échec de toutes les méthodes d'installation NVIDIA"
                log_warn "Continuation sans GPU NVIDIA..."
                return 1
            }
        fi
    fi
    
    # Installation de CUDA si les drivers sont installés
    log_info "Installation de CUDA Toolkit..."
    if apt install -y cuda-toolkit-11-8; then
        # Ajouter CUDA au PATH
        echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> /etc/environment
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> /etc/environment
        
        # Installer cuDNN
        log_info "Installation de cuDNN..."
        apt install -y libcudnn8 libcudnn8-dev || log_warn "Échec de l'installation cuDNN"
    else
        log_warn "Échec de l'installation CUDA Toolkit"
    fi
    
    log_info "Installation NVIDIA terminée"
}

# ==============================================================================
# VÉRIFICATION DE L'INSTALLATION
# ==============================================================================

verify_nvidia() {
    log_step "Vérification de l'installation NVIDIA"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "nvidia-smi trouvé, test de fonctionnement après redémarrage requis"
    else
        log_warn "nvidia-smi non trouvé, installation partielle ou échouée"
    fi
    
    if [[ -d /usr/local/cuda-11.8 ]]; then
        log_info "CUDA Toolkit installé dans /usr/local/cuda-11.8"
    else
        log_warn "CUDA Toolkit non trouvé"
    fi
}

# ==============================================================================
# OPTION ALTERNATIVE SANS NVIDIA
# ==============================================================================

install_cpu_only() {
    log_step "Configuration pour fonctionnement CPU uniquement"
    
    log_info "Configuration de PyTorch pour CPU..."
    # Cette configuration sera utilisée lors de l'installation Python
    
    # Créer un marqueur pour l'installation sans GPU
    touch /tmp/emoia_cpu_only
    
    log_info "EmoIA sera configuré pour fonctionner en mode CPU uniquement"
    log_warn "Les performances seront réduites mais le système fonctionnera"
}

# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

main() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "               EmoIA - Correction installation NVIDIA"
    echo "================================================================"
    echo -e "${NC}"
    
    check_root
    
    echo -e "\nCe script va tenter de corriger l'installation NVIDIA."
    echo -e "Que souhaitez-vous faire ?"
    echo -e "1) Nettoyer et réinstaller NVIDIA (recommandé)"
    echo -e "2) Configurer pour fonctionnement CPU uniquement"
    echo -e "3) Quitter"
    echo -e "\nChoix (1-3): "
    read -r choice
    
    case $choice in
        1)
            log_info "Démarrage de la correction NVIDIA..."
            cleanup_nvidia
            install_prerequisites
            if install_nvidia_improved; then
                verify_nvidia
                echo -e "\n${GREEN}Correction terminée !${NC}"
                echo -e "${YELLOW}IMPORTANT: Redémarrez le serveur pour activer les drivers NVIDIA${NC}"
                echo -e "Après redémarrage, testez avec: nvidia-smi"
            else
                log_warn "Installation NVIDIA échouée, basculement en mode CPU"
                install_cpu_only
            fi
            ;;
        2)
            log_info "Configuration pour mode CPU uniquement..."
            install_cpu_only
            echo -e "\n${GREEN}Configuration CPU terminée !${NC}"
            echo -e "Vous pouvez maintenant continuer l'installation EmoIA"
            ;;
        3)
            log_info "Opération annulée"
            exit 0
            ;;
        *)
            log_error "Choix invalide"
            exit 1
            ;;
    esac
}

# Exécuter le script
main "$@" 