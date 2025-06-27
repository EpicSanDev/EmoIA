# Correction de l'erreur 502 - Configuration Telegram

## 🔧 Problème Résolu

L'erreur `502 (Bad Gateway)` qui se produisait lors de la configuration Telegram à la ligne 68 de `TelegramConfig.tsx` a été corrigée.

## ✅ Corrections Apportées

### 1. Configuration du Proxy Frontend

**Ajout dans `frontend/package.json`:**
```json
"proxy": "http://localhost:8000"
```

**Création de `frontend/src/setupProxy.js`:**
```javascript
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use('/api', createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
    logLevel: 'debug',
    onError: (err, req, res) => {
      console.error('Proxy error:', err.message);
      res.writeHead(502, {
        'Content-Type': 'application/json',
      });
      res.end(JSON.stringify({
        error: 'Backend server not available',
        message: 'Vérifiez que le serveur EmoIA est démarré sur le port 8000'
      }));
    }
  }));
};
```

### 2. Amélioration de la Gestion d'Erreurs

**Dans `TelegramConfig.tsx`:**
- Gestion spécifique de l'erreur 502
- Messages d'erreur plus explicites
- Validation de la réponse HTTP avant traitement

```typescript
if (!response.ok) {
  if (response.status === 502) {
    setMessage({ 
      type: 'error', 
      text: 'Erreur de connexion au backend. Vérifiez que le serveur EmoIA est démarré sur le port 8000.' 
    });
    return;
  }
  // ... autres gestions d'erreur
}
```

### 3. Ajout de la Dépendance Nécessaire

**Installation de `http-proxy-middleware`:**
```bash
npm install http-proxy-middleware --save-dev --legacy-peer-deps
```

## 🚀 Comment Utiliser la Configuration Telegram

### Étape 1: Démarrer le Backend
```bash
# Dans le répertoire racine du projet
cd /Users/bastienjavaux/Desktop/EMOAI/EmoIA
python main.py
# ou
./start.sh
```

### Étape 2: Démarrer le Frontend
```bash
cd frontend
npm start
```

### Étape 3: Configurer le Bot Telegram

1. **Créer un bot sur Telegram :**
   - Contactez @BotFather sur Telegram
   - Tapez `/newbot`
   - Suivez les instructions pour nommer votre bot
   - Récupérez le token fourni

2. **Configurer dans l'interface :**
   - Ouvrez l'interface EmoIA
   - Allez dans la section Configuration Telegram
   - Collez votre token bot
   - Cochez "Activer le bot Telegram"
   - Cliquez sur "Activer le bot"

## 🔍 Diagnostics

### Vérifier que le Backend Fonctionne
```bash
curl http://localhost:8000/api/telegram/status
```

### Vérifier les Logs du Proxy
Les logs du proxy s'afficheront dans la console du frontend lors du développement.

### Messages d'Erreur Courants

**"Erreur de connexion au backend"** → Le serveur EmoIA n'est pas démarré sur le port 8000

**"Backend server not available"** → Problème de connexion réseau ou serveur arrêté

**"Erreur 500"** → Erreur interne du serveur, vérifiez les logs backend

## 📋 Endpoints API Telegram

- `POST /api/telegram/config` - Configure le bot Telegram
- `GET /api/telegram/status` - Statut du bot
- `GET /api/telegram/users` - Utilisateurs connectés

## 🛠️ Maintenance

### Redémarrer les Services
```bash
# Backend
cd /Users/bastienjavaux/Desktop/EMOAI/EmoIA
python main.py

# Frontend (nouveau terminal)
cd frontend
npm start
```

### Vérifier la Configuration
```bash
# Test de l'API
curl -X POST http://localhost:8000/api/telegram/config \
  -H "Content-Type: application/json" \
  -d '{"bot_token":"TEST", "enabled":false}'
```

## ✨ Nouvelles Fonctionnalités

- Messages d'erreur détaillés
- Logs de débogage du proxy
- Gestion robuste des erreurs réseau
- Configuration automatique du proxy en développement

La configuration Telegram devrait maintenant fonctionner correctement sans erreur 502 ! 