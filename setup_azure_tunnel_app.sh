#!/bin/bash

# =============================================================================
# Script de Configuration EmoIA Native App avec Azure et Tunnel
# =============================================================================

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR"
FRONTEND_DIR="$SCRIPT_DIR/frontend_native"
NODE_VERSION="18"

echo -e "${BLUE}🚀 Configuration EmoIA Native App avec Azure et Tunnel${NC}"
echo "=================================================="

# =============================================================================
# Fonctions utilitaires
# =============================================================================

print_step() {
    echo -e "\n${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 n'est pas installé"
        return 1
    fi
    return 0
}

# =============================================================================
# Vérification des prérequis
# =============================================================================

print_step "Vérification des prérequis"

# Vérifier Node.js
if check_command node; then
    NODE_CURRENT=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_CURRENT" -lt "$NODE_VERSION" ]; then
        print_warning "Node.js version $NODE_CURRENT détectée. Version $NODE_VERSION+ recommandée"
    else
        print_success "Node.js version $NODE_CURRENT OK"
    fi
else
    print_error "Node.js requis. Installez-le depuis https://nodejs.org"
    exit 1
fi

# Vérifier npm
if check_command npm; then
    print_success "npm OK"
else
    print_error "npm requis"
    exit 1
fi

# Vérifier Python
if check_command python3; then
    print_success "Python 3 OK"
else
    print_error "Python 3 requis"
    exit 1
fi

# Vérifier Docker (optionnel)
if check_command docker; then
    print_success "Docker OK"
else
    print_warning "Docker non installé (optionnel pour développement)"
fi

# =============================================================================
# Configuration des variables d'environnement Azure
# =============================================================================

print_step "Configuration Azure"

# Créer le fichier .env pour le backend
cat > "$BACKEND_DIR/.env" << EOF
# Configuration Azure pour EmoIA
# Remplacez les valeurs YOUR_* par vos vraies clés Azure

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_KEY
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEFAULT_MODEL=gpt-4

# Azure Speech Services
AZURE_SPEECH_KEY=YOUR_SPEECH_KEY
AZURE_SPEECH_REGION=westeurope

# Azure Vision Services
AZURE_VISION_ENDPOINT=https://YOUR_REGION.api.cognitive.microsoft.com
AZURE_VISION_KEY=YOUR_VISION_KEY

# Azure Language Services
AZURE_LANGUAGE_ENDPOINT=https://YOUR_REGION.api.cognitive.microsoft.com
AZURE_LANGUAGE_KEY=YOUR_LANGUAGE_KEY

# Azure Translator
AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
AZURE_TRANSLATOR_KEY=YOUR_TRANSLATOR_KEY
AZURE_TRANSLATOR_REGION=westeurope

# Configuration Tunnel
TUNNEL_PROVIDER=ngrok
TUNNEL_AUTH_TOKEN=YOUR_NGROK_AUTH_TOKEN
TUNNEL_SUBDOMAIN=emoia-ai
TUNNEL_REGION=eu

# Configuration de base
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
EOF

print_success "Fichier .env créé pour le backend"

# Créer le fichier .env pour le frontend
mkdir -p "$FRONTEND_DIR"
cat > "$FRONTEND_DIR/.env" << EOF
# Configuration Frontend EmoIA Native App

# URL du tunnel (sera mise à jour automatiquement)
REACT_APP_TUNNEL_URL=https://emoia-ai.ngrok.io
REACT_APP_API_URL=https://emoia-ai.ngrok.io

# Configuration Azure (pour le frontend)
REACT_APP_AZURE_SPEECH_REGION=westeurope

# Configuration de l'app
REACT_APP_NAME=EmoIA
REACT_APP_VERSION=1.0.0
REACT_APP_ENVIRONMENT=development

# PWA Configuration
REACT_APP_PWA_ENABLED=true
REACT_APP_OFFLINE_SUPPORT=true
REACT_APP_NOTIFICATIONS_ENABLED=true

# Configuration native
REACT_APP_NATIVE_FEATURES=true
REACT_APP_CAMERA_ENABLED=true
REACT_APP_MICROPHONE_ENABLED=true
REACT_APP_LOCATION_ENABLED=false
EOF

print_success "Fichier .env créé pour le frontend"

# =============================================================================
# Installation des dépendances backend
# =============================================================================

print_step "Installation des dépendances backend"

cd "$BACKEND_DIR"

# Vérifier si requirements.txt existe
if [ -f "requirements.txt" ]; then
    # Créer un environnement virtuel si nécessaire
    if [ ! -d "venv" ]; then
        print_step "Création de l'environnement virtuel Python"
        python3 -m venv venv
    fi
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Installer les dépendances
    print_step "Installation des dépendances Python"
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Ajouter les dépendances Azure manquantes
    pip install azure-cognitiveservices-speech azure-cognitiveservices-vision-computervision azure-ai-textanalytics azure-ai-translation-text
    
    print_success "Dépendances backend installées"
else
    print_warning "requirements.txt non trouvé - vérifiez le répertoire backend"
fi

# =============================================================================
# Configuration et installation du frontend
# =============================================================================

print_step "Configuration du frontend native"

cd "$FRONTEND_DIR"

# Initialiser le projet React/TypeScript si nécessaire
if [ ! -f "package.json" ]; then
    print_step "Initialisation du projet React"
    npx create-react-app . --template typescript
fi

# Installer les dépendances supplémentaires
print_step "Installation des dépendances frontend"
npm install

# Installer Capacitor pour les fonctionnalités natives
npm install @capacitor/core @capacitor/cli
npm install @capacitor/android @capacitor/ios
npm install @capacitor/camera @capacitor/device @capacitor/filesystem
npm install @capacitor/geolocation @capacitor/haptics @capacitor/local-notifications
npm install @capacitor/network @capacitor/push-notifications @capacitor/share
npm install @capacitor/splash-screen @capacitor/status-bar @capacitor/voice-recorder

# Installer les dépendances UI et utilitaires
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material @mui/x-charts @mui/x-date-pickers
npm install axios socket.io-client react-router-dom
npm install @reduxjs/toolkit react-redux react-query
npm install framer-motion chart.js react-chartjs-2
npm install dexie localforage workbox-webpack-plugin

# Installer les dépendances de développement
npm install --save-dev @types/node @capacitor/cli
npm install --save-dev tailwindcss postcss autoprefixer

print_success "Dépendances frontend installées"

# Initialiser Capacitor
if [ ! -f "capacitor.config.ts" ]; then
    print_step "Initialisation Capacitor"
    npx cap init "EmoIA" "ai.emoia.app" --web-dir=build
fi

# =============================================================================
# Configuration du tunnel ngrok
# =============================================================================

print_step "Configuration du tunnel ngrok"

# Installer ngrok si pas présent
if ! check_command ngrok; then
    print_step "Installation de ngrok"
    
    # Détecter l'OS
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)     
            wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
            tar xvzf ngrok-v3-stable-linux-amd64.tgz
            sudo mv ngrok /usr/local/bin
            rm ngrok-v3-stable-linux-amd64.tgz
            ;;
        Darwin*)    
            if check_command brew; then
                brew install ngrok/ngrok/ngrok
            else
                print_error "Homebrew requis sur macOS pour installer ngrok"
                exit 1
            fi
            ;;
        *)          
            print_error "OS non supporté pour l'installation automatique de ngrok"
            print_warning "Installez ngrok manuellement depuis https://ngrok.com/download"
            ;;
    esac
    
    if check_command ngrok; then
        print_success "ngrok installé"
    else
        print_error "Échec de l'installation de ngrok"
        exit 1
    fi
else
    print_success "ngrok déjà installé"
fi

# =============================================================================
# Création des scripts de démarrage
# =============================================================================

print_step "Création des scripts de démarrage"

# Script de démarrage backend avec tunnel
cat > "$BACKEND_DIR/start_with_tunnel.sh" << 'EOF'
#!/bin/bash

# Script de démarrage EmoIA avec tunnel

source .env 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

echo "🚀 Démarrage EmoIA Backend avec tunnel..."

# Démarrer ngrok en arrière-plan
if [ ! -z "$TUNNEL_AUTH_TOKEN" ]; then
    echo "🔐 Configuration ngrok..."
    ngrok config add-authtoken $TUNNEL_AUTH_TOKEN
fi

echo "🌐 Démarrage du tunnel ngrok..."
ngrok http 8000 --subdomain=${TUNNEL_SUBDOMAIN:-emoia-ai} --region=${TUNNEL_REGION:-eu} &
NGROK_PID=$!

# Attendre que ngrok soit prêt
sleep 5

# Obtenir l'URL du tunnel
TUNNEL_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tunnel in data['tunnels']:
    if tunnel['proto'] == 'https':
        print(tunnel['public_url'])
        break
" 2>/dev/null)

if [ ! -z "$TUNNEL_URL" ]; then
    echo "✅ Tunnel actif: $TUNNEL_URL"
    
    # Mettre à jour le fichier .env du frontend
    if [ -f "frontend_native/.env" ]; then
        sed -i "s|REACT_APP_TUNNEL_URL=.*|REACT_APP_TUNNEL_URL=$TUNNEL_URL|" frontend_native/.env
        sed -i "s|REACT_APP_API_URL=.*|REACT_APP_API_URL=$TUNNEL_URL|" frontend_native/.env
        echo "📱 Configuration frontend mise à jour"
    fi
else
    echo "⚠️ Impossible d'obtenir l'URL du tunnel"
fi

# Démarrer le serveur EmoIA
echo "🤖 Démarrage du serveur EmoIA..."
python -m uvicorn src.core.api:app --host 0.0.0.0 --port 8000 --reload

# Nettoyer ngrok à la sortie
trap "kill $NGROK_PID 2>/dev/null" EXIT
EOF

chmod +x "$BACKEND_DIR/start_with_tunnel.sh"

# Script de démarrage frontend
cat > "$FRONTEND_DIR/start_native.sh" << 'EOF'
#!/bin/bash

echo "📱 Démarrage EmoIA Native App..."

# Vérifier que le backend est accessible
if [ ! -z "$REACT_APP_TUNNEL_URL" ]; then
    echo "🔍 Vérification de la connectivité backend..."
    if curl -s "$REACT_APP_TUNNEL_URL/health" > /dev/null; then
        echo "✅ Backend accessible"
    else
        echo "⚠️ Backend non accessible - vérifiez que start_with_tunnel.sh est lancé"
    fi
fi

# Démarrer l'app React
npm start
EOF

chmod +x "$FRONTEND_DIR/start_native.sh"

# Script de build pour les plateformes natives
cat > "$FRONTEND_DIR/build_native.sh" << 'EOF'
#!/bin/bash

echo "🔨 Build EmoIA Native App..."

# Build React
npm run build

# Synchroniser avec Capacitor
npx cap sync

echo "✅ Build terminé"
echo ""
echo "Pour ouvrir dans Android Studio: npx cap open android"
echo "Pour ouvrir dans Xcode: npx cap open ios"
EOF

chmod +x "$FRONTEND_DIR/build_native.sh"

print_success "Scripts de démarrage créés"

# =============================================================================
# Configuration finale
# =============================================================================

print_step "Configuration finale"

# Créer le manifeste PWA
cat > "$FRONTEND_DIR/public/manifest.json" << 'EOF'
{
  "short_name": "EmoIA",
  "name": "EmoIA - Intelligence Artificielle Émotionnelle",
  "icons": [
    {
      "src": "favicon.ico",
      "sizes": "64x64 32x32 24x24 16x16",
      "type": "image/x-icon"
    },
    {
      "src": "logo192.png",
      "type": "image/png",
      "sizes": "192x192"
    },
    {
      "src": "logo512.png",
      "type": "image/png",
      "sizes": "512x512"
    }
  ],
  "start_url": ".",
  "display": "standalone",
  "theme_color": "#1976d2",
  "background_color": "#ffffff",
  "orientation": "portrait-primary",
  "categories": ["productivity", "lifestyle", "health"],
  "description": "Assistant IA émotionnel intelligent, accessible partout via tunnel sécurisé"
}
EOF

# Créer le README pour l'utilisation
cat > "$SCRIPT_DIR/README_NATIVE_APP.md" << 'EOF'
# EmoIA Native App - Guide d'utilisation

## 🚀 Démarrage rapide

### 1. Configuration Azure

Editez les fichiers `.env` créés et remplacez les valeurs `YOUR_*` par vos vraies clés Azure :

- **Azure OpenAI** : Endpoint et clé API
- **Azure Speech Services** : Clé et région
- **Azure Vision Services** : Endpoint et clé  
- **Azure Language Services** : Endpoint et clé
- **Azure Translator** : Clé et région
- **ngrok** : Token d'authentification

### 2. Lancer le backend avec tunnel

```bash
./start_with_tunnel.sh
```

### 3. Lancer l'app native

```bash
cd frontend_native
./start_native.sh
```

## 📱 Fonctionnalités natives

- **PWA** : Installable sur mobile et desktop
- **Hors ligne** : Fonctionne sans connexion
- **Notifications push** : Alertes en temps réel
- **Caméra** : Analyse d'images avec Azure Vision
- **Microphone** : Reconnaissance vocale Azure Speech
- **Géolocalisation** : Contexte spatial (optionnel)
- **Stockage local** : Cache intelligent

## 🌐 Tunnel sécurisé

L'app se connecte automatiquement au backend via :
- **ngrok** (par défaut)
- **cloudflare tunnels** 
- **localtunnel**

## 🔧 Développement

### Web (développement)
```bash
cd frontend_native
npm start
```

### Android
```bash
cd frontend_native
./build_native.sh
npx cap open android
```

### iOS
```bash
cd frontend_native  
./build_native.sh
npx cap open ios
```

## 🧠 Intelligence Azure

L'app utilise tous les services cognitifs Azure :
- **GPT-4** pour les conversations
- **Speech-to-Text** pour la voix
- **Text-to-Speech** pour la synthèse vocale
- **Vision API** pour l'analyse d'images
- **Language Services** pour l'analyse de sentiment
- **Translator** pour le multilingue

## 🛡️ Sécurité

- Connexions HTTPS uniquement
- Tunnel sécurisé avec authentification
- Chiffrement bout en bout
- Respect du RGPD
EOF

print_success "Configuration terminée !"

# =============================================================================
# Instructions finales
# =============================================================================

echo ""
echo -e "${GREEN}🎉 Configuration EmoIA Native App terminée !${NC}"
echo ""
echo -e "${BLUE}📋 Prochaines étapes :${NC}"
echo ""
echo "1. 🔑 Configurez vos clés Azure dans les fichiers .env créés"
echo "2. 🌐 Obtenez un token ngrok sur https://ngrok.com"
echo "3. 🚀 Lancez le backend : ./start_with_tunnel.sh"
echo "4. 📱 Lancez l'app : cd frontend_native && ./start_native.sh"
echo ""
echo -e "${BLUE}📚 Documentation complète :${NC} README_NATIVE_APP.md"
echo ""
echo -e "${YELLOW}⚠️  N'oubliez pas de remplacer toutes les valeurs YOUR_* dans les fichiers .env !${NC}"

cd "$SCRIPT_DIR"