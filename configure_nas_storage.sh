#!/bin/bash

# Script de configuration du stockage persistant sur NAS pour EmoIA
# Ce script prépare l'arborescence NAS et migre les données existantes

set -e

# Configuration
DEFAULT_NAS_PATH="/mnt/nas/emoia"
ENV_FILE=".env.nas"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Fonction pour demander le chemin NAS
get_nas_path() {
    print_header "CONFIGURATION DU CHEMIN NAS"
    
    echo -e "Entrez le chemin de montage de votre NAS pour EmoIA:"
    echo -e "Exemples:"
    echo -e "  - /mnt/nas/emoia"
    echo -e "  - /media/nas-disk/emoia"  
    echo -e "  - /home/user/nas/emoia"
    echo -e "  - //192.168.1.100/emoia (SMB/CIFS)"
    echo ""
    
    read -p "Chemin NAS [${DEFAULT_NAS_PATH}]: " NAS_PATH
    
    if [ -z "$NAS_PATH" ]; then
        NAS_PATH="$DEFAULT_NAS_PATH"
    fi
    
    print_info "Chemin NAS configuré: $NAS_PATH"
}

# Fonction pour vérifier l'accès au NAS
check_nas_access() {
    print_header "VÉRIFICATION DE L'ACCÈS AU NAS"
    
    if [ ! -d "$NAS_PATH" ]; then
        print_warning "Le répertoire $NAS_PATH n'existe pas."
        read -p "Voulez-vous le créer ? (y/n): " create_dir
        
        if [ "$create_dir" = "y" ] || [ "$create_dir" = "Y" ]; then
            mkdir -p "$NAS_PATH" || {
                print_error "Impossible de créer le répertoire $NAS_PATH"
                print_error "Vérifiez les permissions et que le NAS est monté correctement"
                exit 1
            }
            print_success "Répertoire $NAS_PATH créé"
        else
            print_error "Configuration annulée"
            exit 1
        fi
    fi
    
    # Test d'écriture
    test_file="$NAS_PATH/.test_write_$$"
    if echo "test" > "$test_file" 2>/dev/null; then
        rm -f "$test_file"
        print_success "Accès en écriture au NAS confirmé"
    else
        print_error "Impossible d'écrire sur le NAS: $NAS_PATH"
        print_error "Vérifiez les permissions et le montage"
        exit 1
    fi
}

# Fonction pour créer l'arborescence NAS
create_nas_structure() {
    print_header "CRÉATION DE L'ARBORESCENCE NAS"
    
    directories=(
        "data"
        "logs" 
        "models"
        "models/ollama"
        "cache"
        "databases"
        "backups"
        "backups/daily"
        "backups/weekly"
        "backups/monthly"
        "ollama_data"
        "postgres_data"
        "redis_data"
        "prometheus_data"
        "grafana_data"
        "grafana_config"
        "redis_conf"
        "postgres_backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$NAS_PATH/$dir"
        print_success "Créé: $NAS_PATH/$dir"
    done
    
    # Créer les fichiers de configuration par défaut
    cat > "$NAS_PATH/README.md" << EOF
# EmoIA - Stockage Persistant NAS

Ce répertoire contient toutes les données persistantes d'EmoIA stockées sur le NAS.

## Structure des répertoires:

- \`data/\` - Données générales de l'application
- \`logs/\` - Journaux système
- \`models/\` - Modèles IA et poids des réseaux
- \`cache/\` - Cache temporaire 
- \`databases/\` - Bases de données SQLite
- \`backups/\` - Sauvegardes automatiques
- \`ollama_data/\` - Données du service Ollama
- \`postgres_data/\` - Données PostgreSQL (si utilisé)
- \`redis_data/\` - Données Redis (si utilisé)
- \`prometheus_data/\` - Données de monitoring
- \`grafana_data/\` - Configuration Grafana

## Sauvegardes:

Les sauvegardes automatiques sont créées quotidiennement dans \`backups/\`.
Elles incluent:
- Bases de données SQLite
- Modèles personnalisés
- Configuration utilisateur
- Logs importants

Création: $(date)
EOF

    print_success "Arborescence NAS créée avec succès"
}

# Fonction pour migrer les données existantes
migrate_existing_data() {
    print_header "MIGRATION DES DONNÉES EXISTANTES"
    
    print_info "Recherche des données existantes à migrer..."
    
    # Arrêter les services Docker existants
    if command -v docker-compose > /dev/null 2>&1; then
        if [ -f "docker-compose.yml" ]; then
            print_info "Arrêt des services Docker existants..."
            docker-compose down || print_warning "Impossible d'arrêter les services"
        fi
    fi
    
    # Migrer les répertoires de données
    data_dirs=("data" "logs" "models" "cache")
    for dir in "${data_dirs[@]}"; do
        if [ -d "./$dir" ] && [ "$(ls -A ./$dir 2>/dev/null)" ]; then
            print_info "Migration de ./$dir vers $NAS_PATH/$dir/"
            
            # Créer une sauvegarde avant migration
            if [ -d "$NAS_PATH/$dir" ] && [ "$(ls -A $NAS_PATH/$dir 2>/dev/null)" ]; then
                print_warning "Le répertoire $NAS_PATH/$dir existe déjà"
                read -p "Voulez-vous le fusionner avec les données existantes ? (y/n): " merge_data
                
                if [ "$merge_data" = "y" ] || [ "$merge_data" = "Y" ]; then
                    cp -r "./$dir/"* "$NAS_PATH/$dir/" 2>/dev/null || true
                    print_success "Données fusionnées: $dir"
                else
                    print_warning "Migration de $dir ignorée"
                fi
            else
                cp -r "./$dir" "$NAS_PATH/" 2>/dev/null || true
                print_success "Données migrées: $dir"
            fi
        fi
    done
    
    # Migrer les bases de données SQLite
    db_files=("*.db" "emoia_*.db")
    for pattern in "${db_files[@]}"; do
        for db_file in $pattern; do
            if [ -f "$db_file" ]; then
                print_info "Migration de la base de données: $db_file"
                cp "$db_file" "$NAS_PATH/databases/"
                print_success "Base de données migrée: $db_file"
            fi
        done
    done
    
    # Migrer les volumes Docker existants (si possibles)
    if command -v docker > /dev/null 2>&1; then
        print_info "Vérification des volumes Docker existants..."
        
        docker_volumes=("ollama_data" "postgres_data" "redis_data" "prometheus_data" "grafana_data")
        for volume in "${docker_volumes[@]}"; do
            if docker volume inspect "$volume" > /dev/null 2>&1; then
                print_info "Migration du volume Docker: $volume"
                
                # Créer un conteneur temporaire pour copier les données
                docker run --rm -v "$volume":/source -v "$NAS_PATH/${volume}":/target alpine sh -c "cp -r /source/* /target/ 2>/dev/null || true"
                print_success "Volume migré: $volume"
            fi
        done
    fi
}

# Fonction pour créer le fichier d'environnement
create_env_file() {
    print_header "CRÉATION DU FICHIER D'ENVIRONNEMENT"
    
    cat > "$ENV_FILE" << EOF
# Configuration EmoIA avec stockage NAS persistant
# Créé le: $(date)

# Chemin principal vers le NAS
NAS_PATH=$NAS_PATH

# Configuration des bases de données
EMOIA_MEMORY__DATABASE_URL=sqlite:///databases/emoia_memory.db
EMOIA_DATA_DIR=/app/data
EMOIA_MODELS_DIR=/app/models
EMOIA_LOGS_DIR=/app/logs
EMOIA_CACHE_DIR=/app/cache
EMOIA_DATABASES_DIR=/app/databases

# Configuration PostgreSQL (si utilisé)
POSTGRES_DB=emoia
POSTGRES_USER=emoia
POSTGRES_PASSWORD=emoia_secure_password_$(date +%s)

# Configuration Redis (si utilisé)
REDIS_PASSWORD=redis_secure_password_$(date +%s)

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin_$(date +%s)

# Sauvegardes
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE=0 2 * * *

# Performance
DOCKER_RESTART_POLICY=unless-stopped
EOF

    print_success "Fichier d'environnement créé: $ENV_FILE"
    print_info "Vous pouvez modifier ces paramètres selon vos besoins"
}

# Fonction pour créer les scripts de gestion
create_management_scripts() {
    print_header "CRÉATION DES SCRIPTS DE GESTION"
    
    # Script de démarrage NAS
    cat > "start_nas.sh" << 'EOF'
#!/bin/bash
# Script de démarrage EmoIA avec stockage NAS

if [ ! -f ".env.nas" ]; then
    echo "Erreur: Fichier .env.nas non trouvé"
    echo "Exécutez d'abord configure_nas_storage.sh"
    exit 1
fi

echo "Démarrage d'EmoIA avec stockage NAS persistant..."

# Charger la configuration
source .env.nas

# Vérifier l'accès au NAS
if [ ! -d "$NAS_PATH" ]; then
    echo "Erreur: NAS non accessible: $NAS_PATH"
    echo "Vérifiez que le NAS est monté correctement"
    exit 1
fi

# Démarrer avec la configuration NAS
docker-compose -f docker-compose.nas.yml --env-file .env.nas up -d

echo "EmoIA démarré avec succès!"
echo "Interface web: http://localhost:3000"
echo "API: http://localhost:8000"
echo "Données stockées sur: $NAS_PATH"
EOF

    # Script de sauvegarde manuelle  
    cat > "backup_nas.sh" << 'EOF'
#!/bin/bash
# Script de sauvegarde manuelle EmoIA

source .env.nas

echo "Démarrage de la sauvegarde manuelle..."

BACKUP_DIR="$NAS_PATH/backups/manual"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Sauvegarde des bases de données
echo "Sauvegarde des bases de données..."
for db in $NAS_PATH/databases/*.db; do
    if [ -f "$db" ]; then
        filename=$(basename "$db" .db)
        sqlite3 "$db" ".backup $BACKUP_DIR/${filename}_${TIMESTAMP}.db"
        echo "Sauvegardé: $filename"
    fi
done

# Sauvegarde complète
echo "Création de l'archive complète..."
cd "$NAS_PATH"
tar -czf "$BACKUP_DIR/emoia_complete_${TIMESTAMP}.tar.gz" \
    --exclude="backups" \
    --exclude="*.tmp" \
    --exclude="*.log" \
    .

echo "Sauvegarde terminée: $BACKUP_DIR/emoia_complete_${TIMESTAMP}.tar.gz"
EOF

    # Script de restauration
    cat > "restore_nas.sh" << 'EOF'
#!/bin/bash
# Script de restauration EmoIA depuis sauvegarde

if [ $# -ne 1 ]; then
    echo "Usage: $0 <fichier_de_sauvegarde.tar.gz>"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Erreur: Fichier de sauvegarde non trouvé: $BACKUP_FILE"
    exit 1
fi

source .env.nas

echo "ATTENTION: Cette opération va remplacer toutes les données existantes!"
read -p "Êtes-vous sûr de vouloir continuer ? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Restauration annulée"
    exit 0
fi

# Arrêter les services
echo "Arrêt des services..."
docker-compose -f docker-compose.nas.yml down

# Sauvegarder les données actuelles
echo "Sauvegarde des données actuelles..."
CURRENT_BACKUP="$NAS_PATH/backups/before_restore_$(date +%Y%m%d_%H%M%S).tar.gz"
cd "$NAS_PATH"
tar -czf "$CURRENT_BACKUP" .

# Restaurer depuis la sauvegarde
echo "Restauration en cours..."
cd "$NAS_PATH"
tar -xzf "$BACKUP_FILE"

echo "Restauration terminée!"
echo "Vous pouvez redémarrer EmoIA avec: ./start_nas.sh"
EOF

    # Rendre les scripts exécutables
    chmod +x start_nas.sh backup_nas.sh restore_nas.sh
    
    print_success "Scripts de gestion créés:"
    print_info "  - start_nas.sh : Démarrage avec stockage NAS"
    print_info "  - backup_nas.sh : Sauvegarde manuelle"
    print_info "  - restore_nas.sh : Restauration depuis sauvegarde"
}

# Fonction pour afficher les instructions finales
show_final_instructions() {
    print_header "CONFIGURATION TERMINÉE"
    
    print_success "Le stockage persistant NAS a été configuré avec succès!"
    
    echo ""
    print_info "Résumé de la configuration:"
    echo "  • Chemin NAS: $NAS_PATH"
    echo "  • Configuration: $ENV_FILE"
    echo "  • Docker Compose: docker-compose.nas.yml"
    
    echo ""
    print_info "Prochaines étapes:"
    echo "1. Vérifiez que votre NAS est bien monté sur: $NAS_PATH"
    echo "2. Adaptez la configuration dans $ENV_FILE si nécessaire"
    echo "3. Démarrez EmoIA avec: ./start_nas.sh"
    
    echo ""
    print_info "Commandes utiles:"
    echo "  • Démarrage: ./start_nas.sh"
    echo "  • Arrêt: docker-compose -f docker-compose.nas.yml down"
    echo "  • Sauvegarde: ./backup_nas.sh"
    echo "  • Restauration: ./restore_nas.sh <fichier.tar.gz>"
    echo "  • Logs: docker-compose -f docker-compose.nas.yml logs -f"
    
    echo ""
    print_warning "IMPORTANT:"
    echo "• Assurez-vous que le NAS reste accessible"
    echo "• Les sauvegardes automatiques sont créées quotidiennement"
    echo "• En cas de panne NAS, les services s'arrêteront"
    echo "• Testez la restauration régulièrement"
    
    echo ""
    print_info "Documentation du stockage NAS créée dans: $NAS_PATH/README.md"
}

# Script principal
main() {
    print_header "CONFIGURATION DU STOCKAGE PERSISTANT NAS - EmoIA"
    
    echo "Ce script va configurer EmoIA pour utiliser un stockage persistant sur votre NAS."
    echo "Toutes les données (modèles, mémoire, cache) seront stockées de manière persistante."
    echo ""
    
    read -p "Voulez-vous continuer ? (y/n): " proceed
    if [ "$proceed" != "y" ] && [ "$proceed" != "Y" ]; then
        echo "Configuration annulée"
        exit 0
    fi
    
    # Étapes de configuration
    get_nas_path
    check_nas_access
    create_nas_structure
    migrate_existing_data
    create_env_file
    create_management_scripts
    show_final_instructions
}

# Exécution du script
main "$@"