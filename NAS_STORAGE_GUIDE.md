# Guide de Configuration du Stockage Persistant NAS pour EmoIA

Ce guide vous explique comment configurer EmoIA pour utiliser un stockage persistant sur votre serveur NAS, garantissant que toutes vos données (mémoire, modèles, configurations) soient préservées même en cas de réinstallation.

## 🎯 Objectifs

- **Persistance totale** : Toutes les données survivent aux réinstallations
- **Stockage centralisé** : Tout sur votre NAS pour une gestion simplifiée
- **Sauvegardes automatiques** : Protection contre la perte de données
- **Scalabilité** : Stockage extensible selon vos besoins

## 📋 Prérequis

### 1. Serveur NAS
- NAS accessible depuis votre machine
- Espace libre suffisant (recommandé : 50 GB minimum)
- Protocoles supportés : NFS, SMB/CIFS, ou montage direct

### 2. Montage du NAS
Le NAS doit être monté sur votre système. Exemples de configuration :

#### NFS (recommandé)
```bash
# Ajoutez dans /etc/fstab
192.168.1.100:/volume1/emoia /mnt/nas/emoia nfs defaults,auto,timeo=14,intr 0 0

# Montage manuel
sudo mount -t nfs 192.168.1.100:/volume1/emoia /mnt/nas/emoia
```

#### SMB/CIFS
```bash
# Ajoutez dans /etc/fstab
//192.168.1.100/emoia /mnt/nas/emoia cifs username=user,password=pass,uid=1000,gid=1000 0 0

# Montage manuel  
sudo mount -t cifs //192.168.1.100/emoia /mnt/nas/emoia -o username=user,password=pass
```

#### Répertoire local (test)
```bash
# Pour tester sans NAS
sudo mkdir -p /mnt/nas/emoia
sudo chown $USER:$USER /mnt/nas/emoia
```

## 🚀 Installation et Configuration

### 1. Configuration automatique

Exécutez le script de configuration :

```bash
chmod +x configure_nas_storage.sh
./configure_nas_storage.sh
```

Le script vous guidera pour :
- Configurer le chemin vers votre NAS
- Créer l'arborescence nécessaire
- Migrer vos données existantes
- Générer les fichiers de configuration

### 2. Configuration manuelle

Si vous préférez configurer manuellement :

#### a) Créer l'arborescence NAS
```bash
NAS_PATH="/mnt/nas/emoia"
mkdir -p "$NAS_PATH"/{data,logs,models,cache,databases,backups}
mkdir -p "$NAS_PATH"/{ollama_data,postgres_data,redis_data}
mkdir -p "$NAS_PATH"/models/ollama
```

#### b) Créer le fichier d'environnement
```bash
cat > .env.nas << EOF
NAS_PATH=/mnt/nas/emoia
EMOIA_MEMORY__DATABASE_URL=sqlite:///databases/emoia_memory.db
EOF
```

#### c) Copier les données existantes
```bash
# Si vous avez des données existantes
cp -r data/* "$NAS_PATH/data/" 2>/dev/null || true
cp -r models/* "$NAS_PATH/models/" 2>/dev/null || true
cp *.db "$NAS_PATH/databases/" 2>/dev/null || true
```

## 🗂️ Structure du Stockage NAS

```
/mnt/nas/emoia/
├── data/                    # Données applicatives
│   ├── user_profiles.db     # Profils utilisateurs
│   └── knowledge_graph.db   # Graphe de connaissance
├── databases/               # Bases de données centralisées
│   ├── emoia_memory.db      # Mémoire principale
│   └── emoia_advanced_memory.db
├── models/                  # Modèles IA
│   ├── ollama/             # Modèles Ollama
│   ├── personality_model.joblib
│   └── custom_models/      # Modèles personnalisés
├── cache/                  # Cache temporaire
├── logs/                   # Journaux système
├── backups/                # Sauvegardes automatiques
│   ├── daily/
│   ├── weekly/
│   └── manual/
├── ollama_data/           # Données Ollama
├── postgres_data/         # PostgreSQL (si utilisé)
├── redis_data/           # Redis (si utilisé)
├── grafana_data/         # Configuration Grafana
└── prometheus_data/      # Données Prometheus
```

## 🐳 Utilisation avec Docker

### Démarrage avec stockage NAS
```bash
# Démarrage simple
./start_nas.sh

# Ou manuellement
docker-compose -f docker-compose.nas.yml --env-file .env.nas up -d
```

### Arrêt des services
```bash
docker-compose -f docker-compose.nas.yml down
```

### Logs et monitoring
```bash
# Voir les logs
docker-compose -f docker-compose.nas.yml logs -f

# Status des services
docker-compose -f docker-compose.nas.yml ps
```

## 💾 Gestion des Sauvegardes

### Sauvegardes automatiques
- **Fréquence** : Quotidienne à 2h du matin
- **Rétention** : 30 jours
- **Contenu** : Bases de données + données complètes

### Sauvegarde manuelle
```bash
./backup_nas.sh
```

### Restauration
```bash
# Lister les sauvegardes disponibles
ls /mnt/nas/emoia/backups/

# Restaurer depuis une sauvegarde
./restore_nas.sh /mnt/nas/emoia/backups/emoia_complete_20240101_120000.tar.gz
```

### Configuration des sauvegardes

Activez le service de sauvegarde :
```bash
docker-compose -f docker-compose.nas.yml --profile backup up -d
```

## 🔧 Configuration Avancée

### Variables d'environnement importantes

```env
# Chemin principal
NAS_PATH=/mnt/nas/emoia

# Bases de données
EMOIA_MEMORY__DATABASE_URL=sqlite:///databases/emoia_memory.db

# Répertoires
EMOIA_DATA_DIR=/app/data
EMOIA_MODELS_DIR=/app/models
EMOIA_LOGS_DIR=/app/logs
EMOIA_CACHE_DIR=/app/cache

# Performance
DOCKER_RESTART_POLICY=unless-stopped
```

### Optimisation des performances

#### Pour les NAS lents
```yaml
# Ajoutez dans docker-compose.nas.yml
environment:
  - SQLITE_SYNCHRONOUS=NORMAL
  - SQLITE_CACHE_SIZE=10000
```

#### Pour les gros volumes de données
```yaml
volumes:
  - type: bind
    source: ${NAS_PATH}/data
    target: /app/data
    bind:
      propagation: cached
```

## 🔍 Surveillance et Maintenance

### Vérification de l'état du NAS
```bash
# Test d'accès
ls -la /mnt/nas/emoia/

# Test d'écriture
echo "test" > /mnt/nas/emoia/.test_$(date +%s)

# Espace disponible
df -h /mnt/nas/emoia/
```

### Monitoring automatique

Activez Prometheus et Grafana :
```bash
docker-compose -f docker-compose.nas.yml --profile monitoring up -d
```

Accès :
- Grafana : http://localhost:3001
- Prometheus : http://localhost:9090

### Maintenance périodique

```bash
# Nettoyage des logs anciens
find /mnt/nas/emoia/logs -name "*.log" -mtime +30 -delete

# Nettoyage du cache
rm -rf /mnt/nas/emoia/cache/tmp/*

# Vérification des bases de données
sqlite3 /mnt/nas/emoia/databases/emoia_memory.db "PRAGMA integrity_check;"
```

## 🚨 Dépannage

### Problèmes courants

#### Le NAS n'est pas accessible
```bash
# Vérifier le montage
mount | grep emoia

# Remonter si nécessaire
sudo umount /mnt/nas/emoia
sudo mount /mnt/nas/emoia
```

#### Permissions insuffisantes
```bash
# Corriger les permissions
sudo chown -R $USER:$USER /mnt/nas/emoia
chmod -R 755 /mnt/nas/emoia
```

#### Services qui ne démarrent pas
```bash
# Vérifier les logs
docker-compose -f docker-compose.nas.yml logs

# Vérifier l'espace disque
df -h /mnt/nas/emoia/

# Nettoyer et redémarrer
docker-compose -f docker-compose.nas.yml down
docker system prune -f
./start_nas.sh
```

#### Base de données corrompue
```bash
# Restaurer depuis la dernière sauvegarde
latest_backup=$(ls -t /mnt/nas/emoia/backups/*.db | head -1)
cp "$latest_backup" /mnt/nas/emoia/databases/emoia_memory.db
```

### Logs de diagnostic
```bash
# Logs des conteneurs
docker-compose -f docker-compose.nas.yml logs --tail=50

# Logs système
journalctl -u docker
dmesg | grep -i error
```

## 🔄 Migration et Mise à Jour

### Migration depuis installation existante
```bash
# Sauvegarder l'installation actuelle
docker-compose down
tar -czf emoia_backup_$(date +%Y%m%d).tar.gz data logs models *.db

# Configurer le NAS
./configure_nas_storage.sh

# Démarrer avec le nouveau stockage
./start_nas.sh
```

### Mise à jour d'EmoIA
```bash
# Sauvegarder avant la mise à jour
./backup_nas.sh

# Mettre à jour le code
git pull

# Reconstruire les images
docker-compose -f docker-compose.nas.yml build --no-cache

# Redémarrer
docker-compose -f docker-compose.nas.yml up -d
```

## 📊 Monitoring des Performances

### Métriques importantes
- Espace disque utilisé
- Latence d'accès au NAS
- Performance des bases de données
- Temps de sauvegarde

### Alertes recommandées
- Espace disque < 10%
- NAS inaccessible > 5 minutes
- Échec de sauvegarde
- Corruption de base de données

## 🔐 Sécurité

### Recommandations
- Chiffrement du trafic NAS (SMB3/NFSv4 avec Kerberos)
- Sauvegardes chiffrées
- Accès restreint au répertoire EmoIA
- Surveillance des accès

### Sauvegarde de la configuration
```bash
# Sauvegarder la configuration complète
tar -czf emoia_config_$(date +%Y%m%d).tar.gz \
  docker-compose.nas.yml \
  .env.nas \
  config.yaml \
  start_nas.sh \
  backup_nas.sh \
  restore_nas.sh
```

---

## 📞 Support

En cas de problème :
1. Vérifiez que le NAS est monté et accessible
2. Consultez les logs : `docker-compose -f docker-compose.nas.yml logs`
3. Testez une restauration depuis sauvegarde
4. Vérifiez l'espace disque disponible

**Remarque** : Ce guide assure une persistance complète de vos données EmoIA sur votre NAS, vous permettant de réinstaller ou migrer le système sans perdre aucune information.