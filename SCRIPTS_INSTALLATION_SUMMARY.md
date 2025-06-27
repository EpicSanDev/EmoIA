# 📋 EmoIA - Résumé des Scripts d'Installation

Ce document décrit tous les scripts d'installation et de gestion créés pour déployer EmoIA sur Ubuntu Server sans Docker.

## 🎯 Scripts Disponibles

### 1. `check_requirements.sh` - Vérification des Prérequis

**Description**: Script de vérification des prérequis système avant installation.

**Utilisation**:
```bash
./check_requirements.sh
```

**Fonctionnalités**:
- ✅ Vérification du système d'exploitation (Ubuntu 20.04/22.04 LTS)
- ✅ Vérification des ressources (CPU, RAM, disque)
- ✅ Test de connectivité réseau
- ✅ Détection GPU NVIDIA (optionnel)
- ✅ Vérification des privilèges root/sudo
- ✅ Test de disponibilité des ports
- ✅ Estimation des performances

**Résultats possibles**:
- **Succès**: Système compatible, installation possible
- **Avertissements**: Configuration correcte mais pas optimale
- **Erreurs**: Prérequis non satisfaits, installation impossible

---

### 2. `install_ubuntu_server.sh` - Installation Automatique

**Description**: Script d'installation automatique complète d'EmoIA.

**Utilisation**:
```bash
sudo ./install_ubuntu_server.sh
```

**⚠️ Permissions**: Doit être exécuté en tant que root

**Composants installés**:

#### Dépendances Système
- Python 3.8+ avec environnement virtuel
- Node.js 18 LTS et npm/yarn
- PostgreSQL 14+ avec optimisations
- Redis 6+ avec configuration personnalisée
- Nginx avec reverse proxy
- NVIDIA CUDA 11.8 + cuDNN (si GPU détecté)

#### Services EmoIA
- **emoia-backend**: API FastAPI (port 8000)
- **emoia-frontend**: Interface React (port 3000)
- **emoia-telegram**: Bot Telegram
- **nginx**: Reverse proxy (port 80)

#### Configuration Automatique
- Utilisateur système `emoia`
- Base de données PostgreSQL avec optimisations
- Cache Redis optimisé
- Services systemd
- Firewall UFW
- Téléchargement des modèles ML

**Durée**: 30-60 minutes selon la connexion internet

---

### 3. `manage_emoia.sh` - Gestion et Maintenance

**Description**: Script complet de gestion et maintenance d'EmoIA.

**Utilisation**:
```bash
./manage_emoia.sh <commande> [options]
```

#### Commandes de Service

```bash
# Statut des services
./manage_emoia.sh status

# Démarrer tous les services
sudo ./manage_emoia.sh start

# Arrêter tous les services
sudo ./manage_emoia.sh stop

# Redémarrer tous les services
sudo ./manage_emoia.sh restart
```

#### Commandes de Monitoring

```bash
# Afficher les logs d'un service
./manage_emoia.sh logs emoia-backend

# Suivre les logs en temps réel
./manage_emoia.sh follow emoia-backend

# Métriques de performance
./manage_emoia.sh performance
```

#### Commandes de Maintenance

```bash
# Diagnostic complet
./manage_emoia.sh diagnose

# Nettoyage du cache
sudo ./manage_emoia.sh clean-cache

# Sauvegarde des données
sudo ./manage_emoia.sh backup

# Mise à jour du système
sudo ./manage_emoia.sh update-system

# Mise à jour des modèles ML
sudo ./manage_emoia.sh update-models
```

---

## 🗂️ Documentation

### 4. `README_INSTALLATION_UBUNTU.md` - Guide Complet

**Description**: Guide détaillé d'installation et d'utilisation.

**Contenu**:
- Prérequis système détaillés
- Instructions d'installation pas à pas
- Configuration post-installation
- Utilisation et gestion des services
- Monitoring et métriques
- Sécurité et sauvegardes
- Dépannage et résolution de problèmes
- Mise à jour et maintenance

---

## 🚀 Processus d'Installation Recommandé

### Étape 1: Vérification des Prérequis
```bash
# Télécharger et vérifier
./check_requirements.sh
```

### Étape 2: Installation
```bash
# Si les prérequis sont satisfaits
sudo ./install_ubuntu_server.sh
```

### Étape 3: Vérification Post-Installation
```bash
# Vérifier que tout fonctionne
./manage_emoia.sh status
./manage_emoia.sh diagnose
```

### Étape 4: Configuration Optionnelle
```bash
# Configurer le bot Telegram (optionnel)
sudo nano /opt/emoia/config.yaml

# Configurer SSL/HTTPS (recommandé)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d votre-domaine.com
```

---

## 📁 Structure des Fichiers Installés

### Répertoire Principal: `/opt/emoia/`

```
/opt/emoia/
├── src/                    # Code source Python
├── frontend/               # Interface React
├── venv/                   # Environnement virtuel Python
├── models/                 # Modèles ML téléchargés
├── data/                   # Données applicatives
├── logs/                   # Fichiers de logs
├── cache/                  # Cache temporaire
├── config.yaml             # Configuration principale
├── .env                    # Variables d'environnement
└── installation_info.txt   # Informations d'installation
```

### Services Systemd: `/etc/systemd/system/`

```
/etc/systemd/system/
├── emoia-backend.service   # Service API backend
├── emoia-frontend.service  # Service interface web
└── emoia-telegram.service  # Service bot Telegram
```

### Configuration Nginx: `/etc/nginx/`

```
/etc/nginx/sites-available/emoia  # Configuration reverse proxy
```

### Sauvegardes: `/var/backups/emoia/`

```
/var/backups/emoia/
├── emoia_backup_YYYYMMDD_HHMMSS.tar.gz  # Sauvegardes applicatives
└── database_YYYYMMDD_HHMMSS.sql         # Sauvegardes base de données
```

---

## 🔧 Configuration Avancée

### Variables d'Environnement (`.env`)

```bash
# Configuration EmoIA
APP_ENV=production
DEBUG=false

# Base de données
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://:pass@localhost:6379/0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="7.5"

# Security
JWT_SECRET=generated_secret
ENCRYPTION_KEY=generated_key
```

### Configuration PostgreSQL

Optimisée pour machine avec 64GB RAM:
- `shared_buffers = 16GB`
- `effective_cache_size = 48GB`
- `maintenance_work_mem = 2GB`

### Configuration Redis

Optimisée pour cache haute performance:
- `maxmemory 16gb`
- `maxmemory-policy allkeys-lru`
- Persistence activée avec AOF

---

## 🔍 Surveillance et Monitoring

### Métriques Système
- Utilisation CPU, RAM, disque
- Utilisation GPU (si disponible)
- Load average et uptime
- Processus actifs

### Métriques Application
- Statut des services EmoIA
- Temps de réponse API
- Utilisation de la base de données
- Cache hit/miss ratio

### Logs Disponibles
- **Application**: `/opt/emoia/logs/`
- **Services**: `journalctl -u service-name`
- **Système**: `/var/log/`
- **Nginx**: `/var/log/nginx/`

---

## 🆘 Support et Dépannage

### Commandes de Diagnostic

```bash
# Diagnostic complet
./manage_emoia.sh diagnose

# Vérification des services
./manage_emoia.sh status

# Logs détaillés
./manage_emoia.sh logs emoia-backend
./manage_emoia.sh follow emoia-backend
```

### Problèmes Courants

1. **Services qui ne démarrent pas**
   - Vérifier les logs avec `manage_emoia.sh logs`
   - Vérifier la configuration dans `/opt/emoia/config.yaml`

2. **Problèmes GPU**
   - Vérifier avec `nvidia-smi`
   - Redémarrer après installation des drivers

3. **Problèmes de performance**
   - Nettoyer le cache avec `manage_emoia.sh clean-cache`
   - Vérifier l'utilisation mémoire

4. **Problèmes réseau**
   - Vérifier nginx avec `nginx -t`
   - Vérifier le firewall avec `ufw status`

---

## 🎯 Conclusion

Ces scripts fournissent une solution complète pour:

- ✅ **Installation automatisée** sans intervention manuelle
- ✅ **Configuration optimisée** pour votre matériel
- ✅ **Gestion simplifiée** des services
- ✅ **Monitoring intégré** des performances
- ✅ **Maintenance automatisée** avec sauvegardes
- ✅ **Support GPU** avec NVIDIA CUDA
- ✅ **Sécurité renforcée** avec firewall et chiffrement

**EmoIA est maintenant prêt pour la production ! 🚀**