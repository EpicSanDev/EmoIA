# Instructions pour corriger le problème Docker EmoIA

## 🔧 Résumé du problème
Le build Docker échouait à cause d'un conflit de dépendances npm dans le frontend :
- TypeScript 5.x est installé
- react-scripts 5.0.1 nécessite TypeScript 4.x
- Conflits avec les plugins ESLint

## ✅ Solution appliquée
J'ai modifié le fichier `frontend/Dockerfile` pour utiliser `npm ci --legacy-peer-deps` au lieu de `npm ci`.

## 🚀 Instructions pour redémarrer

Sur votre machine locale (MacBook), exécutez les commandes suivantes dans le terminal :

```bash
# 1. Aller dans le dossier du projet
cd ~/Desktop/EMOAI/EmoIA

# 2. Arrêter les conteneurs existants
docker-compose down

# 3. Supprimer les anciennes images pour forcer la reconstruction
docker rmi emoia-emoia-frontend emoia-emoia-api

# 4. Redémarrer avec la nouvelle configuration
./start_docker.sh
```

Ou utilisez le script de nettoyage créé :

```bash
# Exécuter le script de redémarrage complet
./restart_docker_clean.sh
```

## 📝 Vérification
Après le redémarrage, vérifiez que les services sont bien démarrés :
- API Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Documentation API: http://localhost:8000/docs

## 🔍 En cas de problème
Si le problème persiste, vous pouvez :
1. Vérifier les logs : `docker-compose logs -f`
2. Nettoyer complètement Docker : `docker system prune -a`
3. Reconstruire sans cache : `docker-compose build --no-cache`