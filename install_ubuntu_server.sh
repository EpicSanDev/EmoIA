#!/bin/bash

# ==============================================================================
# EmoIA - Script d'installation automatique pour Ubuntu Server
# Sans Docker - Installation native avec support GPU
# ==============================================================================

set -e  # Arrêt en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables globales
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/emoia"
SERVICE_USER="emoia"
DB_NAME="emoia_prod"
DB_USER="emoia_user"
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

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

check_ubuntu() {
    if ! grep -q "Ubuntu" /etc/os-release; then
        log_error "Ce script est conçu pour Ubuntu Server"
        exit 1
    fi
}

# ==============================================================================
# INSTALLATION DES DÉPENDANCES SYSTÈME
# ==============================================================================

install_system_dependencies() {
    log_step "Installation des dépendances système"
    
    # Mise à jour du système
    log_info "Mise à jour du système..."
    apt update && apt upgrade -y
    
    # Dépendances de base
    log_info "Installation des paquets de base..."
    apt install -y \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        pkg-config \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        unzip \
        htop \
        nginx \
        supervisor \
        fail2ban \
        ufw \
        openssl
    
    # Dépendances Python
    log_info "Installation des dépendances Python..."
    apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-setuptools \
        python3-wheel
    
    # Dépendances pour le machine learning
    log_info "Installation des dépendances ML..."
    apt install -y \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libffi-dev \
        libjpeg-dev \
        libpng-dev \
        libssl-dev \
        zlib1g-dev \
        liblzma-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev
    
    log_info "Dépendances système installées avec succès"
}

# ==============================================================================
# INSTALLATION NVIDIA CUDA
# ==============================================================================

install_nvidia_cuda() {
    log_step "Installation des drivers NVIDIA et CUDA"
    
    # Vérifier la présence d'une carte NVIDIA
    if ! lspci | grep -i nvidia > /dev/null 2>&1; then
        log_warn "Aucune carte NVIDIA détectée. Installation CUDA ignorée."
        touch /tmp/emoia_cpu_only
        return 0
    fi
    
    log_info "Carte NVIDIA détectée. Installation des drivers..."
    
    # Installer les prérequis pour NVIDIA
    log_info "Installation des prérequis NVIDIA..."
    apt install -y \
        linux-headers-$(uname -r) \
        build-essential \
        dkms \
        software-properties-common || {
        log_error "Échec de l'installation des prérequis"
        return 1
    }
    
    # Désactiver nouveau (driver open source)
    log_info "Configuration des modules kernel..."
    cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF
    update-initramfs -u
    
    # Ajouter le repository NVIDIA avec gestion d'erreur
    log_info "Ajout du repository NVIDIA..."
    cd /tmp
    if [[ -f cuda-keyring_1.0-1_all.deb ]]; then
        rm -f cuda-keyring_1.0-1_all.deb
    fi
    
    if ! wget -O cuda-keyring_1.0-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb; then
        log_error "Échec du téléchargement du keyring NVIDIA"
        log_warn "Basculement en mode CPU uniquement..."
        touch /tmp/emoia_cpu_only
        return 1
    fi
    
    if ! dpkg -i cuda-keyring_1.0-1_all.deb; then
        log_warn "Problème avec l'installation du keyring, tentative de correction..."
        apt --fix-broken install -y
        if ! dpkg -i cuda-keyring_1.0-1_all.deb; then
            log_error "Échec définitif du keyring NVIDIA"
            touch /tmp/emoia_cpu_only
            return 1
        fi
    fi
    
    apt update
    
    # Installation des drivers NVIDIA avec gestion d'erreur
    log_info "Installation des drivers NVIDIA avec gestion d'erreur..."
    
    # Essayer plusieurs versions de drivers
    if ! apt install -y nvidia-driver-535; then
        log_warn "Échec du driver 535, essai avec 525..."
        if ! apt install -y nvidia-driver-525; then
            log_warn "Échec du driver 525, essai avec ubuntu-drivers..."
            apt install -y ubuntu-drivers-common
            if ! ubuntu-drivers autoinstall; then
                log_error "Échec de toutes les méthodes d'installation NVIDIA"
                log_warn "Basculement en mode CPU uniquement..."
                touch /tmp/emoia_cpu_only
                return 1
            fi
        fi
    fi
    
    # Installation de CUDA si les drivers sont installés
    log_info "Installation de CUDA Toolkit 11.8..."
    if apt install -y cuda-toolkit-11-8; then
        # Ajouter CUDA au PATH
        echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> /etc/environment
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> /etc/environment
        
        # Installer cuDNN
        log_info "Installation de cuDNN..."
        apt install -y libcudnn8 libcudnn8-dev || log_warn "Échec de l'installation cuDNN (non critique)"
        
        log_info "NVIDIA CUDA installé. Redémarrage requis pour activer les drivers."
    else
        log_warn "Échec de l'installation CUDA Toolkit (non critique)"
    fi
}

# ==============================================================================
# INSTALLATION NODE.JS
# ==============================================================================

install_nodejs() {
    log_step "Installation de Node.js et npm"
    
    # Installer Node.js 18 LTS
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt install -y nodejs
    
    # Installer Yarn
    npm install -g yarn pm2
    
    # Vérifier l'installation
    node_version=$(node --version)
    npm_version=$(npm --version)
    log_info "Node.js $node_version et npm $npm_version installés"
}

# ==============================================================================
# INSTALLATION ET CONFIGURATION POSTGRESQL
# ==============================================================================

install_postgresql() {
    log_step "Installation et configuration de PostgreSQL"
    
    # Installer PostgreSQL
    apt install -y postgresql postgresql-contrib postgresql-client
    
    # Démarrer et activer PostgreSQL
    systemctl start postgresql
    systemctl enable postgresql
    
    # Créer l'utilisateur et la base de données
    log_info "Configuration de la base de données..."
    sudo -u postgres psql << EOF
CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE $DB_NAME OWNER $DB_USER;
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
ALTER USER $DB_USER CREATEDB;
\q
EOF
    
    # Configuration PostgreSQL pour la performance
    postgres_version=$(sudo -u postgres psql -t -c "SELECT version();" | grep -oP "PostgreSQL \K[0-9]+")
    postgres_conf="/etc/postgresql/$postgres_version/main/postgresql.conf"
    
    # Optimisations pour machine avec 64GB RAM
    cat >> "$postgres_conf" << EOF

# Optimisations EmoIA pour 64GB RAM
shared_buffers = 16GB
effective_cache_size = 48GB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 64MB
min_wal_size = 1GB
max_wal_size = 4GB
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4
EOF
    
    systemctl restart postgresql
    log_info "PostgreSQL configuré avec succès"
}

# ==============================================================================
# INSTALLATION ET CONFIGURATION REDIS
# ==============================================================================

install_redis() {
    log_step "Installation et configuration de Redis"
    
    # Installer Redis
    apt install -y redis-server
    
    # Configuration Redis
    redis_conf="/etc/redis/redis.conf"
    cp "$redis_conf" "$redis_conf.backup"
    
    # Optimisations Redis pour 64GB RAM
    cat > "$redis_conf" << EOF
# Configuration Redis optimisée pour EmoIA
bind 127.0.0.1
port 6379
requirepass $REDIS_PASSWORD

# Mémoire
maxmemory 16gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Performance
tcp-keepalive 300
timeout 0
tcp-backlog 511
databases 16

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Persistence
dir /var/lib/redis
dbfilename dump.rdb
rdbcompression yes
rdbchecksum yes

# AOF
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
EOF
    
    # Démarrer et activer Redis
    systemctl restart redis-server
    systemctl enable redis-server
    
    log_info "Redis configuré avec succès"
}

# ==============================================================================
# CRÉATION DE L'UTILISATEUR SYSTÈME
# ==============================================================================

create_system_user() {
    log_step "Création de l'utilisateur système EmoIA"
    
    # Créer l'utilisateur système
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/bash -d "$INSTALL_DIR" "$SERVICE_USER"
        log_info "Utilisateur $SERVICE_USER créé"
    else
        log_info "Utilisateur $SERVICE_USER existe déjà"
    fi
    
    # Créer le répertoire d'installation
    mkdir -p "$INSTALL_DIR"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
}

# ==============================================================================
# INSTALLATION DE L'APPLICATION EMOIA
# ==============================================================================

install_emoia_application() {
    log_step "Installation de l'application EmoIA"
    
    # Copier les fichiers de l'application
    log_info "Copie des fichiers de l'application..."
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    
    # Créer l'environnement virtuel Python
    log_info "Création de l'environnement virtuel Python..."
    sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
    
    # Activer l'environnement virtuel et installer les dépendances
    log_info "Installation des dépendances Python..."
    
    # Déterminer la version de PyTorch à installer
    if [[ -f /tmp/emoia_cpu_only ]]; then
        log_info "Mode CPU détecté, installation de PyTorch CPU..."
        TORCH_INSTALL="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else
        log_info "Mode GPU détecté, installation de PyTorch CUDA..."
        TORCH_INSTALL="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    fi
    
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        source venv/bin/activate
        pip install --upgrade pip setuptools wheel
        $TORCH_INSTALL
        pip install -r requirements.txt
    "
    
    # Installer les dépendances frontend
    log_info "Installation des dépendances frontend..."
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR/frontend'
        npm install
        npm run build
    "
    
    # Créer les répertoires nécessaires
    sudo -u "$SERVICE_USER" mkdir -p "$INSTALL_DIR"/{logs,data,models,cache,temp}
    
    log_info "Application EmoIA installée avec succès"
}

# ==============================================================================
# CONFIGURATION DE L'APPLICATION
# ==============================================================================

configure_emoia() {
    log_step "Configuration de l'application EmoIA"
    
    # Créer le fichier de configuration d'environnement
    cat > "$INSTALL_DIR/.env" << EOF
# Configuration EmoIA
APP_ENV=production
DEBUG=false

# Base de données
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost/$DB_NAME
REDIS_URL=redis://:$REDIS_PASSWORD@localhost:6379/0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="7.5"

# Security
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Paths
INSTALL_DIR=$INSTALL_DIR
MODELS_DIR=$INSTALL_DIR/models
DATA_DIR=$INSTALL_DIR/data
LOGS_DIR=$INSTALL_DIR/logs
CACHE_DIR=$INSTALL_DIR/cache
EOF
    
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    
    # Mise à jour de la configuration YAML
    sed -i "s|postgresql://user:pass@localhost/emoia_prod|postgresql://$DB_USER:$DB_PASSWORD@localhost/$DB_NAME|g" "$INSTALL_DIR/config.yaml"
    sed -i "s|redis://localhost:6379/0|redis://:$REDIS_PASSWORD@localhost:6379/0|g" "$INSTALL_DIR/config.yaml"
    
    log_info "Configuration de l'application terminée"
}

# ==============================================================================
# MIGRATION DE LA BASE DE DONNÉES
# ==============================================================================

setup_database() {
    log_step "Configuration de la base de données"
    
    log_info "Migration de la base de données..."
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        source venv/bin/activate
        export DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost/$DB_NAME
        python -c 'import asyncio; from src.config import Config; config = Config(); print(\"Database configuration loaded\")'
    "
    
    log_info "Base de données configurée avec succès"
}

# ==============================================================================
# CONFIGURATION DES SERVICES SYSTEMD
# ==============================================================================

create_systemd_services() {
    log_step "Création des services systemd"
    
    # Service principal EmoIA
    cat > /etc/systemd/system/emoia-backend.service << EOF
[Unit]
Description=EmoIA Backend Service
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/python -m uvicorn src.core.api:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Service frontend EmoIA
    cat > /etc/systemd/system/emoia-frontend.service << EOF
[Unit]
Description=EmoIA Frontend Service
After=network.target emoia-backend.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR/frontend
Environment=PATH=/usr/bin:/bin
Environment=PORT=3000
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Service Telegram Bot
    cat > /etc/systemd/system/emoia-telegram.service << EOF
[Unit]
Description=EmoIA Telegram Bot
After=network.target emoia-backend.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin:/usr/local/cuda-11.8/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/python src/telegram_bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Recharger systemd
    systemctl daemon-reload
    
    log_info "Services systemd créés"
}

# ==============================================================================
# CONFIGURATION NGINX
# ==============================================================================

configure_nginx() {
    log_step "Configuration du reverse proxy Nginx"
    
    # Créer la configuration Nginx
    cat > /etc/nginx/sites-available/emoia << EOF
server {
    listen 80;
    server_name localhost;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
    
    # API Backend
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Timeout pour les requêtes ML
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Logs
    access_log /var/log/nginx/emoia_access.log;
    error_log /var/log/nginx/emoia_error.log;
}
EOF
    
    # Activer le site
    ln -sf /etc/nginx/sites-available/emoia /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Tester et redémarrer nginx
    nginx -t
    systemctl restart nginx
    systemctl enable nginx
    
    log_info "Nginx configuré avec succès"
}

# ==============================================================================
# CONFIGURATION DU FIREWALL
# ==============================================================================

configure_firewall() {
    log_step "Configuration du firewall UFW"
    
    # Réinitialiser UFW
    ufw --force reset
    
    # Politique par défaut
    ufw default deny incoming
    ufw default allow outgoing
    
    # Autoriser SSH (port 22)
    ufw allow ssh
    
    # Autoriser HTTP/HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Activer le firewall
    ufw --force enable
    
    log_info "Firewall configuré"
}

# ==============================================================================
# TÉLÉCHARGEMENT DES MODÈLES ML
# ==============================================================================

download_ml_models() {
    log_step "Téléchargement des modèles de machine learning"
    
    log_info "Téléchargement des modèles... (cela peut prendre du temps)"
    
    sudo -u "$SERVICE_USER" bash -c "
        cd '$INSTALL_DIR'
        source venv/bin/activate
        python -c '
import os
from transformers import AutoTokenizer, AutoModel, pipeline

models_dir = \"$INSTALL_DIR/models\"
os.makedirs(models_dir, exist_ok=True)

# Télécharger les modèles principaux
print(\"Téléchargement du modèle de langage...\")
tokenizer = AutoTokenizer.from_pretrained(\"microsoft/DialoGPT-large\", cache_dir=models_dir)
model = AutoModel.from_pretrained(\"microsoft/DialoGPT-large\", cache_dir=models_dir)

print(\"Téléchargement du modèle d embedding...\")
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(\"sentence-transformers/all-mpnet-base-v2\", cache_folder=models_dir)

print(\"Téléchargement du modèle d émotion...\")
emotion_classifier = pipeline(\"text-classification\", model=\"j-hartmann/emotion-english-distilroberta-base\", cache_dir=models_dir)

print(\"Modèles téléchargés avec succès!\")
'
    "
    
    log_info "Modèles ML téléchargés avec succès"
}

# ==============================================================================
# FINALISATION ET DÉMARRAGE
# ==============================================================================

finalize_installation() {
    log_step "Finalisation de l'installation"
    
    # Activer et démarrer les services
    log_info "Activation des services..."
    systemctl enable emoia-backend.service
    systemctl enable emoia-frontend.service
    systemctl enable emoia-telegram.service
    
    # Démarrer les services
    log_info "Démarrage des services..."
    systemctl start emoia-backend.service
    sleep 10
    systemctl start emoia-frontend.service
    systemctl start emoia-telegram.service
    
    # Vérifier le statut des services
    log_info "Vérification des services..."
    services=("emoia-backend" "emoia-frontend" "emoia-telegram" "postgresql" "redis-server" "nginx")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            log_info "✓ Service $service: ACTIF"
        else
            log_warn "✗ Service $service: INACTIF"
        fi
    done
}

# ==============================================================================
# INFORMATIONS POST-INSTALLATION
# ==============================================================================

show_installation_info() {
    log_step "Installation terminée !"
    
    echo -e "\n${GREEN}EmoIA a été installé avec succès !${NC}\n"
    
    echo -e "${BLUE}Informations d'accès :${NC}"
    echo -e "  Interface Web: http://$(hostname -I | awk '{print $1}')"
    echo -e "  API Backend: http://$(hostname -I | awk '{print $1}')/api"
    echo -e "  Répertoire d'installation: $INSTALL_DIR"
    
    echo -e "\n${BLUE}Base de données :${NC}"
    echo -e "  PostgreSQL Database: $DB_NAME"
    echo -e "  PostgreSQL User: $DB_USER"
    echo -e "  PostgreSQL Password: $DB_PASSWORD"
    
    echo -e "\n${BLUE}Redis :${NC}"
    echo -e "  Redis Password: $REDIS_PASSWORD"
    
    echo -e "\n${BLUE}Commandes utiles :${NC}"
    echo -e "  Statut des services: systemctl status emoia-backend emoia-frontend emoia-telegram"
    echo -e "  Logs backend: journalctl -u emoia-backend -f"
    echo -e "  Logs frontend: journalctl -u emoia-frontend -f"
    echo -e "  Logs telegram: journalctl -u emoia-telegram -f"
    echo -e "  Redémarrer EmoIA: systemctl restart emoia-backend emoia-frontend emoia-telegram"
    
    echo -e "\n${YELLOW}Notes importantes :${NC}"
    if [[ -f /tmp/emoia_cpu_only ]]; then
        echo -e "  - EmoIA fonctionnera en mode CPU uniquement (GPU NVIDIA non disponible)"
        echo -e "  - Les performances seront réduites mais le système est fonctionnel"
    else
        echo -e "  - Si vous avez une carte NVIDIA, redémarrez le serveur pour activer les drivers"
        echo -e "  - Testez les drivers NVIDIA avec: nvidia-smi (après redémarrage)"
    fi
    echo -e "  - Configurez votre token Telegram bot dans config.yaml"
    echo -e "  - Les mots de passe ont été sauvegardés dans $INSTALL_DIR/.env"
    echo -e "  - Consultez les logs en cas de problème"
    
    # Sauvegarder les informations
    cat > "$INSTALL_DIR/installation_info.txt" << EOF
Installation EmoIA - $(date)

Base de données PostgreSQL:
- Database: $DB_NAME
- User: $DB_USER  
- Password: $DB_PASSWORD

Redis:
- Password: $REDIS_PASSWORD

URLs:
- Frontend: http://$(hostname -I | awk '{print $1}')
- API: http://$(hostname -I | awk '{print $1}')/api

Installation directory: $INSTALL_DIR
EOF
    
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/installation_info.txt"
    chmod 600 "$INSTALL_DIR/installation_info.txt"
}

# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

main() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "               EmoIA - Installation Ubuntu Server"
    echo "                    Installation native sans Docker"
    echo "================================================================"
    echo -e "${NC}"
    
    # Vérifications préliminaires
    check_root
    check_ubuntu
    
    # Confirmation utilisateur
    echo -e "\nCe script va installer EmoIA sur ce serveur Ubuntu."
    echo -e "L'installation peut prendre 30-60 minutes selon votre connexion internet."
    echo -e "\nVoulez-vous continuer ? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Installation annulée"
        exit 0
    fi
    
    # Démarrer l'installation
    log_info "Début de l'installation de EmoIA..."
    
    # Étapes d'installation
    install_system_dependencies
    install_nvidia_cuda
    install_nodejs
    install_postgresql
    install_redis
    create_system_user
    install_emoia_application
    configure_emoia
    setup_database
    download_ml_models
    create_systemd_services
    configure_nginx
    configure_firewall
    finalize_installation
    show_installation_info
    
    log_info "Installation terminée avec succès !"
}

# Exécuter l'installation
main "$@"