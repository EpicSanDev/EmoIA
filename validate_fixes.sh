#!/bin/bash

echo "🔧 Validation des Corrections EmoIA"
echo "======================================"

echo ""
echo "📦 Arrêt des services..."
docker-compose down

echo ""
echo "🚀 Redémarrage des services..."
docker-compose up -d

echo ""
echo "⏳ Attente du démarrage (30 secondes)..."
sleep 30

echo ""
echo "🧪 Tests de validation..."

# Test 1: API Health Check
echo "1️⃣ Test de santé de l'API..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$response" = "200" ]; then
    echo "   ✅ API accessible"
else
    echo "   ❌ API non accessible (code: $response)"
fi

# Test 2: API Tasks (TDAH)
echo "2️⃣ Test des tâches TDAH..."
response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/api/tasks/demo-user")
if [ "$response" = "200" ]; then
    echo "   ✅ Endpoint des tâches fonctionnel"
else
    echo "   ❌ Endpoint des tâches non fonctionnel (code: $response)"
fi

# Test 3: API Calendar
echo "3️⃣ Test du calendrier..."
response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/api/calendar/demo-user")
if [ "$response" = "200" ]; then
    echo "   ✅ Endpoint du calendrier fonctionnel"
else
    echo "   ❌ Endpoint du calendrier non fonctionnel (code: $response)"
fi

# Test 4: API Memories avec DELETE
echo "4️⃣ Test de l'endpoint DELETE des mémoires..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "http://localhost:8000/api/memories/test_memory?user_id=demo-user")
if [ "$response" = "200" ]; then
    echo "   ✅ Endpoint DELETE des mémoires fonctionnel"
else
    echo "   ❌ Endpoint DELETE des mémoires non fonctionnel (code: $response)"
fi

# Test 5: Frontend
echo "5️⃣ Test du frontend..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$response" = "200" ]; then
    echo "   ✅ Frontend accessible"
else
    echo "   ❌ Frontend non accessible (code: $response)"
fi

echo ""
echo "📊 Vérification des logs pour les erreurs corrigées..."
echo "Recherche des erreurs dans les logs récents..."

# Vérifier l'absence des erreurs spécifiques
echo "🔍 Vérification des erreurs TDAH..."
tdah_errors=$(docker-compose logs emoia-api 2>/dev/null | grep -c "has no attribute 'get_tdah_tasks'" || echo "0")
if [ "$tdah_errors" = "0" ]; then
    echo "   ✅ Plus d'erreurs TDAH trouvées"
else
    echo "   ❌ $tdah_errors erreur(s) TDAH encore présente(s)"
fi

echo "🔍 Vérification des erreurs WebSocket..."
ws_errors=$(docker-compose logs emoia-api 2>/dev/null | grep -c "WebSocket connection.*failed" || echo "0")
if [ "$ws_errors" = "0" ]; then
    echo "   ✅ Plus d'erreurs WebSocket dans les logs API"
else
    echo "   ⚠️  $ws_errors erreur(s) WebSocket trouvée(s) (normal si frontend pas encore connecté)"
fi

echo ""
echo "📋 Résumé des corrections appliquées:"
echo "✅ Méthodes TDAH ajoutées à IntelligentMemorySystem"
echo "✅ Gestion des sessions aiohttp améliorée"
echo "✅ WebSocket analytics ajouté"
echo "✅ Route DELETE pour les mémoires ajoutée"
echo "✅ Configuration nginx améliorée"
echo "✅ Handler de shutdown ajouté"

echo ""
echo "🎯 Validation terminée!"
echo ""
echo "Pour surveiller les logs en temps réel:"
echo "docker-compose logs -f emoia-api"
echo ""
echo "Pour accéder à l'interface:"
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8000"
echo "Docs API: http://localhost:8000/docs"