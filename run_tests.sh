#!/bin/bash

# Script pour exécuter les tests d'EmoIA

echo "🧪 Exécution des tests EmoIA..."

# Fonction pour afficher l'aide
show_help() {
    echo "Usage: ./run_tests.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  all        Exécuter tous les tests"
    echo "  unit       Exécuter uniquement les tests unitaires"
    echo "  api        Exécuter uniquement les tests API"
    echo "  coverage   Exécuter avec rapport de couverture"
    echo "  watch      Exécuter en mode watch (relance automatique)"
    echo "  help       Afficher cette aide"
    echo ""
    echo "Exemples:"
    echo "  ./run_tests.sh all"
    echo "  ./run_tests.sh coverage"
}

# Vérifier l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "⚠️  Environnement virtuel non trouvé. Création..."
    python -m venv venv
fi

# Activer l'environnement virtuel
source venv/bin/activate || source venv/Scripts/activate

# Installer pytest si nécessaire
if ! command -v pytest &> /dev/null; then
    echo "📦 Installation de pytest..."
    pip install pytest pytest-asyncio pytest-cov
fi

# Traiter les arguments
case "$1" in
    "all")
        echo "🔍 Exécution de tous les tests..."
        pytest -v
        ;;
    
    "unit")
        echo "🔍 Exécution des tests unitaires..."
        pytest tests/test_emoia_main.py tests/test_emotional_core.py -v
        ;;
    
    "api")
        echo "🔍 Exécution des tests API..."
        pytest tests/test_api.py -v
        ;;
    
    "coverage")
        echo "📊 Exécution des tests avec couverture..."
        pytest --cov=src --cov-report=html --cov-report=term -v
        echo "📈 Rapport HTML généré dans htmlcov/index.html"
        ;;
    
    "watch")
        echo "👁️  Mode watch activé..."
        if ! command -v pytest-watch &> /dev/null; then
            pip install pytest-watch
        fi
        pytest-watch
        ;;
    
    "help"|"--help"|"-h")
        show_help
        ;;
    
    *)
        echo "🔍 Exécution des tests par défaut..."
        pytest -v
        ;;
esac

# Afficher le résultat
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Tests terminés avec succès !"
else
    echo ""
    echo "❌ Des tests ont échoué."
    exit 1
fi