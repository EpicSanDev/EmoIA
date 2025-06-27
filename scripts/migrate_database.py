#!/usr/bin/env python3
"""
Script de migration pour ajouter la colonne ai_settings à la table user_preferences
"""

import sqlite3
import json
import os
from pathlib import Path

def migrate_user_preferences():
    """Migre la table user_preferences pour ajouter la colonne ai_settings"""
    
    # Chemins possibles pour la base de données
    db_paths = [
        "emoia_memory.db",
        "/workspace/emoia_memory.db",
        "./emoia_memory.db"
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("Base de données non trouvée, création d'une nouvelle...")
        db_path = "emoia_memory.db"
    
    print(f"Migration de la base de données: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Vérifier si la table existe
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_preferences'
        """)
        
        if not cursor.fetchone():
            print("Table user_preferences n'existe pas, création...")
            cursor.execute("""
                CREATE TABLE user_preferences (
                    user_id TEXT PRIMARY KEY,
                    language TEXT DEFAULT 'fr',
                    theme TEXT DEFAULT 'light',
                    notification_settings TEXT DEFAULT '{"email": true, "push": false, "sound": true}',
                    ai_settings TEXT DEFAULT '{"personality_adaptation": true, "emotion_intensity": 0.8, "response_style": "balanced"}'
                )
            """)
            print("Table user_preferences créée avec succès!")
        else:
            print("Table user_preferences existe, vérification des colonnes...")
            
            # Vérifier si la colonne ai_settings existe
            cursor.execute("PRAGMA table_info(user_preferences)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'ai_settings' not in columns:
                print("Colonne ai_settings manquante, ajout...")
                
                # Ajouter la colonne ai_settings
                cursor.execute("""
                    ALTER TABLE user_preferences 
                    ADD COLUMN ai_settings TEXT DEFAULT '{"personality_adaptation": true, "emotion_intensity": 0.8, "response_style": "balanced"}'
                """)
                
                # Mettre à jour les enregistrements existants avec des valeurs par défaut
                default_ai_settings = json.dumps({
                    "personality_adaptation": True,
                    "emotion_intensity": 0.8,
                    "response_style": "balanced"
                })
                
                cursor.execute("""
                    UPDATE user_preferences 
                    SET ai_settings = ? 
                    WHERE ai_settings IS NULL
                """, (default_ai_settings,))
                
                print("Colonne ai_settings ajoutée avec succès!")
            else:
                print("Colonne ai_settings déjà présente")
        
        conn.commit()
        print("Migration terminée avec succès!")
        
        # Afficher le schéma final
        cursor.execute("PRAGMA table_info(user_preferences)")
        columns = cursor.fetchall()
        print("\nSchéma final de la table user_preferences:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            
    except Exception as e:
        print(f"Erreur lors de la migration: {e}")
        return False
    finally:
        if conn:
            conn.close()
    
    return True

if __name__ == "__main__":
    migrate_user_preferences()