"""
Configuration NAS pour EmoIA
Extension des paramètres pour le stockage persistant sur NAS
"""

import os
from pathlib import Path
from typing import Optional
from .settings import Config, MemoryConfig


class NASConfig:
    """Configuration pour le stockage persistant NAS"""
    
    def __init__(self, nas_path: Optional[str] = None):
        # Utiliser la variable d'environnement ou le chemin par défaut
        self.nas_path = Path(nas_path or os.getenv("NAS_PATH", "/mnt/nas/emoia"))
        
        # Vérifier que le NAS est accessible
        if not self.nas_path.exists():
            raise ValueError(f"Chemin NAS non accessible: {self.nas_path}")
    
    def get_nas_config(self) -> Config:
        """Retourne une configuration adaptée au stockage NAS"""
        
        # Créer les répertoires sur le NAS s'ils n'existent pas
        self._ensure_nas_directories()
        
        # Configuration de base
        config = Config()
        
        # Rediriger tous les répertoires vers le NAS
        config.data_dir = self.nas_path / "data"
        config.models_dir = self.nas_path / "models"
        config.logs_dir = self.nas_path / "logs"
        config.cache_dir = self.nas_path / "cache"
        
        # Répertoire spécial pour les bases de données
        databases_dir = self.nas_path / "databases"
        databases_dir.mkdir(exist_ok=True)
        
        # Configuration de la mémoire pour utiliser le NAS
        config.memory = MemoryConfig(
            database_url=f"sqlite:///{databases_dir}/emoia_memory.db",
            redis_url=None  # Redis sera configuré séparément si nécessaire
        )
        
        return config
    
    def _ensure_nas_directories(self):
        """Crée la structure de répertoires sur le NAS"""
        directories = [
            "data",
            "logs", 
            "models",
            "models/ollama",
            "cache",
            "databases",
            "backups",
            "backups/daily",
            "backups/weekly", 
            "backups/monthly",
            "ollama_data",
            "postgres_data",
            "redis_data",
            "prometheus_data",
            "grafana_data"
        ]
        
        for directory in directories:
            dir_path = self.nas_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_database_paths(self) -> dict:
        """Retourne les chemins vers toutes les bases de données sur le NAS"""
        databases_dir = self.nas_path / "databases"
        
        return {
            "main_memory": databases_dir / "emoia_memory.db",
            "advanced_memory": databases_dir / "emoia_advanced_memory.db", 
            "user_profiles": self.nas_path / "data" / "user_profiles.db",
            "knowledge_graph": self.nas_path / "data" / "knowledge_graph.db"
        }
    
    def get_backup_config(self) -> dict:
        """Configuration pour les sauvegardes automatiques"""
        backups_dir = self.nas_path / "backups"
        
        return {
            "backup_dir": backups_dir,
            "daily_dir": backups_dir / "daily",
            "weekly_dir": backups_dir / "weekly",
            "monthly_dir": backups_dir / "monthly",
            "manual_dir": backups_dir / "manual",
            "retention_days": 30,
            "schedule": "0 2 * * *"  # 2h du matin tous les jours
        }
    
    def migrate_existing_data(self, source_dir: Path = Path(".")):
        """Migre les données existantes vers le NAS"""
        print(f"Migration des données de {source_dir} vers {self.nas_path}")
        
        # Migrer les répertoires de données
        data_mappings = {
            "data": "data",
            "logs": "logs", 
            "models": "models",
            "cache": "cache"
        }
        
        for source_name, target_name in data_mappings.items():
            source_path = source_dir / source_name
            target_path = self.nas_path / target_name
            
            if source_path.exists() and any(source_path.iterdir()):
                print(f"  Migration de {source_name}...")
                self._copy_directory(source_path, target_path)
        
        # Migrer les bases de données SQLite
        databases_dir = self.nas_path / "databases"
        
        for db_file in source_dir.glob("*.db"):
            target_file = databases_dir / db_file.name
            print(f"  Migration de la base de données: {db_file.name}")
            
            if target_file.exists():
                # Créer une sauvegarde avant écrasement
                backup_file = databases_dir / f"{db_file.stem}_backup_{db_file.suffix}"
                target_file.rename(backup_file)
            
            db_file.rename(target_file)
    
    def _copy_directory(self, source: Path, target: Path):
        """Copie récursive d'un répertoire"""
        import shutil
        
        if target.exists():
            print(f"    Le répertoire {target} existe déjà, fusion...")
            for item in source.iterdir():
                target_item = target / item.name
                if item.is_dir():
                    shutil.copytree(item, target_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target_item)
        else:
            shutil.copytree(source, target)
    
    def create_symlinks(self, project_dir: Path = Path(".")):
        """Crée des liens symboliques vers le NAS pour compatibilité"""
        
        # Liens vers les répertoires principaux
        symlinks = {
            "data": self.nas_path / "data",
            "logs": self.nas_path / "logs",
            "models": self.nas_path / "models",
            "cache": self.nas_path / "cache"
        }
        
        for link_name, target_path in symlinks.items():
            link_path = project_dir / link_name
            
            # Supprimer le lien existant s'il existe
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    print(f"Attention: {link_path} est un répertoire, renommage en {link_path}.backup")
                    link_path.rename(f"{link_path}.backup")
            
            # Créer le lien symbolique
            link_path.symlink_to(target_path)
            print(f"Lien symbolique créé: {link_name} -> {target_path}")
        
        # Liens vers les bases de données
        databases_dir = self.nas_path / "databases"
        for db_file in databases_dir.glob("*.db"):
            link_path = project_dir / db_file.name
            
            if link_path.exists() and not link_path.is_symlink():
                link_path.rename(f"{link_path}.backup")
            elif link_path.is_symlink():
                link_path.unlink()
            
            link_path.symlink_to(db_file)
            print(f"Lien vers base de données: {db_file.name} -> {db_file}")
    
    def verify_nas_access(self) -> bool:
        """Vérifie que le NAS est accessible en lecture/écriture"""
        try:
            # Test de lecture
            if not self.nas_path.exists():
                return False
            
            # Test d'écriture
            test_file = self.nas_path / f".test_write_{os.getpid()}"
            test_file.write_text("test")
            test_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"Erreur d'accès au NAS: {e}")
            return False
    
    def get_status(self) -> dict:
        """Retourne le statut du stockage NAS"""
        import shutil
        
        status = {
            "nas_path": str(self.nas_path),
            "accessible": self.verify_nas_access(),
            "directories": {},
            "databases": {},
            "space": {}
        }
        
        # Vérifier les répertoires
        directories = ["data", "logs", "models", "cache", "databases", "backups"]
        for directory in directories:
            dir_path = self.nas_path / directory
            status["directories"][directory] = {
                "exists": dir_path.exists(),
                "size_mb": self._get_directory_size(dir_path) if dir_path.exists() else 0
            }
        
        # Vérifier les bases de données
        db_paths = self.get_database_paths()
        for name, db_path in db_paths.items():
            status["databases"][name] = {
                "exists": db_path.exists(),
                "size_mb": db_path.stat().st_size // (1024*1024) if db_path.exists() else 0
            }
        
        # Espace disque
        if self.nas_path.exists():
            total, used, free = shutil.disk_usage(self.nas_path)
            status["space"] = {
                "total_gb": total // (1024**3),
                "used_gb": used // (1024**3), 
                "free_gb": free // (1024**3),
                "usage_percent": (used / total) * 100
            }
        
        return status
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calcule la taille d'un répertoire en MB"""
        if not directory.exists():
            return 0
        
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size // (1024 * 1024)


def get_nas_config(nas_path: Optional[str] = None) -> Config:
    """Fonction helper pour obtenir la configuration NAS"""
    nas_config = NASConfig(nas_path)
    return nas_config.get_nas_config()


def setup_nas_storage(nas_path: Optional[str] = None, migrate_data: bool = True) -> NASConfig:
    """Configuration complète du stockage NAS"""
    
    print("Configuration du stockage persistant NAS pour EmoIA")
    print("=" * 50)
    
    # Initialiser la configuration NAS
    nas_config = NASConfig(nas_path)
    
    # Vérifier l'accès
    if not nas_config.verify_nas_access():
        raise RuntimeError(f"Impossible d'accéder au NAS: {nas_config.nas_path}")
    
    print(f"✓ Accès au NAS confirmé: {nas_config.nas_path}")
    
    # Créer la structure de répertoires
    nas_config._ensure_nas_directories()
    print("✓ Structure de répertoires créée")
    
    # Migrer les données existantes si demandé
    if migrate_data:
        nas_config.migrate_existing_data()
        print("✓ Données existantes migrées")
    
    # Créer les liens symboliques pour compatibilité
    nas_config.create_symlinks()
    print("✓ Liens symboliques créés")
    
    # Afficher le statut
    status = nas_config.get_status()
    print(f"\nStatut du stockage NAS:")
    print(f"  Espace libre: {status['space'].get('free_gb', 'N/A')} GB")
    print(f"  Utilisation: {status['space'].get('usage_percent', 'N/A'):.1f}%")
    
    print("\nConfiguration NAS terminée avec succès!")
    return nas_config