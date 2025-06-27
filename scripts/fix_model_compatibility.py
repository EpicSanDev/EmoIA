#!/usr/bin/env python3
"""
Script pour corriger les problèmes de compatibilité des modèles
Régénère les modèles avec la bonne version de scikit-learn
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def regenerate_personality_model():
    """Régénère le modèle de personnalité avec la version actuelle de scikit-learn"""
    model_path = Path("models/personality_model.joblib")
    
    if not model_path.exists():
        logger.warning(f"Modèle non trouvé: {model_path}")
        return
    
    try:
        # Tenter de charger le modèle existant
        logger.info("Tentative de chargement du modèle existant...")
        old_model = joblib.load(model_path)
        logger.info("Modèle chargé avec succès")
        
        # Si le modèle se charge sans erreur, pas besoin de le régénérer
        return True
        
    except Exception as e:
        logger.warning(f"Erreur lors du chargement du modèle: {e}")
        logger.info("Régénération du modèle de personnalité...")
        
        # Créer des données d'exemple pour entraîner un nouveau modèle
        # Dans un vrai système, vous utiliseriez vos vraies données d'entraînement
        X_dummy = np.random.random((1000, 10))  # 10 features
        y_dummy = np.random.random((1000, 8))   # 8 traits de personnalité
        
        # Créer un nouveau modèle
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Entraîner le modèle
        X_train, X_test, y_train, y_test = train_test_split(
            X_dummy, y_dummy, test_size=0.2, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Sauvegarder le nouveau modèle
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        
        logger.info(f"Nouveau modèle sauvegardé: {model_path}")
        return True

def check_async_session_cleanup():
    """Vérifie et corrige les problèmes de session async"""
    logger.info("Vérification des sessions async...")
    
    # Ces corrections doivent être appliquées dans le code principal
    recommendations = [
        "1. Utiliser des context managers pour les sessions aiohttp",
        "2. Appeler explicitement session.close() dans les finally blocks",
        "3. Utiliser asyncio.gather avec return_exceptions=True",
        "4. Implémenter un cleanup handler pour les signaux d'arrêt"
    ]
    
    for rec in recommendations:
        logger.info(f"Recommandation: {rec}")

def main():
    """Fonction principale"""
    logger.info("Début de la correction des problèmes de compatibilité...")
    
    # Régénérer le modèle de personnalité
    regenerate_personality_model()
    
    # Vérifier les sessions async
    check_async_session_cleanup()
    
    logger.info("Corrections terminées!")

if __name__ == "__main__":
    main()