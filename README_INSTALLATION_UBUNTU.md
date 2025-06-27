# 🚀 EmoIA - Installation Ubuntu Server (Sans Docker)

Guide d'installation complet pour EmoIA sur Ubuntu Server 20.04/22.04 LTS avec support GPU NVIDIA.

## 📋 Prérequis Système

### Configuration Matérielle Recommandée
- **CPU**: 8+ cœurs (Intel Core i7/i9 ou AMD Ryzen 7/9)
- **RAM**: 64 GB (minimum 32 GB)
- **GPU**: NVIDIA RTX 2070 Super ou supérieur (8+ GB VRAM)
- **Stockage**: 500 GB SSD (minimum 200 GB)
- **Réseau**: Connexion internet stable

### Configuration Logicielle
- **OS**: Ubuntu Server 20.04 LTS ou 22.04 LTS
- **Accès**: Privilèges root/sudo
- **Réseau**: Ports 80, 443, 22 accessibles

## 🎯 Installation Automatique

### Étape 1: Télécharger les Scripts

```bash
# Cloner le repository
git clone https://github.com/your-repo/EmoIA.git
cd EmoIA

# Rendre les scripts exécutables
chmod +x install_ubuntu_server.sh
chmod +x manage_emoia.sh
```

### Étape 2: Lancer l'Installation

```bash
# Exécuter l'installation (en tant que root)
sudo ./install_ubuntu_server.sh
```

⏱️ **Durée d'installation**: 30-60 minutes selon votre connexion internet

## 📦 Composants Installés

L'installation automatique configure les éléments suivants :

### Dépendances Système
- ✅ Python 3.8+ avec environnement virtuel
- ✅ Node.js 18 LTS et npm
- ✅ PostgreSQL 14+ avec optimisations
- ✅ Redis 6+ avec configuration personnalisée
- ✅ Nginx avec reverse proxy
- ✅ NVIDIA CUDA 11.8 + cuDNN (si GPU détecté)

### Services EmoIA
- ✅ **emoia-backend**: API FastAPI sur port 8000
- ✅ **emoia-frontend**: Interface React sur port 3000  
- ✅ **emoia-telegram**: Bot Telegram
- ✅ **nginx**: Reverse proxy sur port 80

### Modèles IA
- ✅ microsoft/DialoGPT-large (762M paramètres)
- ✅ sentence-transformers/all-mpnet-base-v2
- ✅ j-hartmann/emotion-english-distilroberta-base
- ✅ cardiffnlp/twitter-roberta-base-sentiment-latest

## 🔧 Configuration Post-Installation

### 1. Configuration Telegram Bot (Optionnel)

```bash
# Éditer la configuration
sudo nano /opt/emoia/config.yaml

# Ajouter votre token bot
telegram:
  bot_token: "YOUR_BOT_TOKEN_HERE"
  enabled: true
```

### 2. Configuration SSL/HTTPS (Recommandé)

```bash
# Installer certbot
sudo apt install certbot python3-certbot-nginx

# Obtenir un certificat SSL
sudo certbot --nginx -d votre-domaine.com

# Redémarrer nginx
sudo systemctl restart nginx
```

### 3. Configuration Firewall

```bash
# Le firewall est automatiquement configuré
# Vérifier le statut
sudo ufw status

# Autoriser des ports supplémentaires si nécessaire
sudo ufw allow 8080/tcp  # Exemple
```

## 🎮 Utilisation

### Accès à l'Interface

- **Interface Web**: `http://votre-ip-serveur`
- **API Documentation**: `http://votre-ip-serveur/api/docs`
- **Admin Panel**: `http://votre-ip-serveur/admin`

### Gestion des Services

```bash
# Statut des services
sudo ./manage_emoia.sh status

# Redémarrer tous les services
sudo ./manage_emoia.sh restart

# Voir les logs
sudo ./manage_emoia.sh logs emoia-backend

# Suivi en temps réel
sudo ./manage_emoia.sh follow emoia-backend
```

### Commandes de Maintenance

```bash
# Diagnostic complet
sudo ./manage_emoia.sh diagnose

# Nettoyage du cache
sudo ./manage_emoia.sh clean-cache

# Sauvegarde des données
sudo ./manage_emoia.sh backup

# Mise à jour du système
sudo ./manage_emoia.sh update-system

# Mise à jour des modèles ML
sudo ./manage_emoia.sh update-models
```

## 📊 Monitoring

### Métriques de Performance

```bash
# Afficher les métriques
sudo ./manage_emoia.sh performance

# Utilisation GPU (si disponible)
nvidia-smi

# Utilisation mémoire
free -h

# Utilisation disque
df -h /opt/emoia
```

### Logs Importants

```bash
# Logs des services
journalctl -u emoia-backend -f
journalctl -u emoia-frontend -f
journalctl -u emoia-telegram -f

# Logs système
tail -f /var/log/nginx/emoia_access.log
tail -f /var/log/nginx/emoia_error.log
```

## 🔐 Sécurité

### Informations d'Accès

Les mots de passe et clés sont automatiquement générés et stockés dans :
- `/opt/emoia/.env` (configuration application)
- `/opt/emoia/installation_info.txt` (informations complètes)

### Sauvegarde des Données

```bash
# Sauvegarde manuelle
sudo ./manage_emoia.sh backup

# Sauvegardes automatiques (cron)
sudo crontab -e
# Ajouter : 0 2 * * * /path/to/manage_emoia.sh backup
```

### Mise à Jour de Sécurité

```bash
# Mise à jour du système
sudo ./manage_emoia.sh update-system

# Mise à jour d'urgence
sudo apt update && sudo apt upgrade -y
sudo systemctl restart emoia-backend emoia-frontend
```

## 🚨 Dépannage

### Problèmes Courants

#### Services qui ne démarrent pas
```bash
# Vérifier le statut
sudo ./manage_emoia.sh diagnose

# Vérifier les logs
sudo ./manage_emoia.sh logs emoia-backend

# Redémarrer les services
sudo systemctl restart emoia-backend
```

#### Problèmes GPU
```bash
# Vérifier les drivers NVIDIA
nvidia-smi

# Réinstaller les drivers si nécessaire
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535

# Redémarrer le serveur
sudo reboot
```

#### Problème de mémoire
```bash
# Nettoyer le cache
sudo ./manage_emoia.sh clean-cache

# Optimiser PostgreSQL
sudo nano /etc/postgresql/14/main/postgresql.conf
# Ajuster shared_buffers selon votre RAM

# Redémarrer PostgreSQL
sudo systemctl restart postgresql
```

#### Problèmes réseau
```bash
# Vérifier les ports
sudo ss -tuln | grep -E ":(3000|8000|5432|6379|80|443)"

# Vérifier nginx
sudo nginx -t
sudo systemctl status nginx

# Vérifier le firewall
sudo ufw status
```

### Logs de Débogage

```bash
# Activer le mode debug
sudo nano /opt/emoia/.env
# Changer DEBUG=false à DEBUG=true

# Redémarrer les services
sudo systemctl restart emoia-backend

# Voir les logs détaillés
sudo journalctl -u emoia-backend -f
```

### Restauration d'une Sauvegarde

```bash
# Lister les sauvegardes
ls -la /var/backups/emoia/

# Arrêter les services
sudo ./manage_emoia.sh stop

# Restaurer la base de données
sudo -u postgres psql < /var/backups/emoia/database_YYYYMMDD_HHMMSS.sql

# Restaurer les fichiers
sudo tar -xzf /var/backups/emoia/emoia_backup_YYYYMMDD_HHMMSS.tar.gz -C /

# Redémarrer les services
sudo ./manage_emoia.sh start
```

## 🔄 Mise à Jour de EmoIA

### Mise à Jour Mineure

```bash
# Sauvegarder avant mise à jour
sudo ./manage_emoia.sh backup

# Mise à jour du système
sudo ./manage_emoia.sh update-system

# Mise à jour des modèles
sudo ./manage_emoia.sh update-models

# Redémarrer les services
sudo ./manage_emoia.sh restart
```

### Mise à Jour Majeure

```bash
# 1. Sauvegarder
sudo ./manage_emoia.sh backup

# 2. Arrêter les services
sudo ./manage_emoia.sh stop

# 3. Télécharger la nouvelle version
cd /tmp
git clone https://github.com/your-repo/EmoIA.git EmoIA-new

# 4. Copier les nouveaux fichiers
sudo cp -r EmoIA-new/src/* /opt/emoia/src/
sudo cp -r EmoIA-new/frontend/* /opt/emoia/frontend/

# 5. Installer les nouvelles dépendances
sudo -u emoia bash -c "
    cd /opt/emoia
    source venv/bin/activate
    pip install -r requirements.txt
"

sudo -u emoia bash -c "
    cd /opt/emoia/frontend
    npm install
    npm run build
"

# 6. Redémarrer les services
sudo ./manage_emoia.sh start
```

## 📞 Support

### Commandes d'Aide

```bash
# Aide du script de gestion
./manage_emoia.sh help

# Diagnostic complet
sudo ./manage_emoia.sh diagnose

# Statut détaillé
sudo ./manage_emoia.sh status
```

### Fichiers de Configuration

- **Configuration principale**: `/opt/emoia/config.yaml`
- **Variables d'environnement**: `/opt/emoia/.env`
- **Configuration nginx**: `/etc/nginx/sites-available/emoia`
- **Services systemd**: `/etc/systemd/system/emoia-*.service`

### Répertoires Importants

- **Installation**: `/opt/emoia/`
- **Logs**: `/opt/emoia/logs/` et `journalctl`
- **Données**: `/opt/emoia/data/`
- **Modèles**: `/opt/emoia/models/`
- **Sauvegardes**: `/var/backups/emoia/`

## 🏁 Conclusion

EmoIA est maintenant installé et configuré sur votre serveur Ubuntu. L'installation inclut :

- ✅ Tous les services configurés et démarrés automatiquement
- ✅ Base de données optimisée pour votre configuration
- ✅ Support GPU NVIDIA activé
- ✅ Interface web accessible
- ✅ Scripts de gestion et maintenance
- ✅ Sauvegardes automatiques
- ✅ Monitoring intégré

Pour toute question ou problème, consultez les logs et utilisez les outils de diagnostic fournis.

**Bon voyage avec EmoIA ! 🤖✨**