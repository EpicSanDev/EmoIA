"""
Système de Mémoire Intelligente pour EmoIA
Gestion avancée de la mémoire avec embeddings vectoriels et consolidation automatique.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from collections import deque, defaultdict
import threading
import hashlib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import faiss

from ..config import MemoryConfig
from ..emotional import EmotionalState, PersonalityProfile


logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Item de mémoire avec métadonnées complètes"""
    
    content: str
    timestamp: datetime
    importance: float
    emotional_state: Optional[EmotionalState] = None
    context: str = ""
    memory_type: str = "episodic"  # episodic, semantic, procedural, name, learning, tdah
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    consolidation_level: int = 0  # 0: working, 1: short-term, 2: long-term
    embedding: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    user_id: str = ""
    learning_category: str = ""  # Pour les apprentissages
    tdah_task_type: str = ""  # Pour la gestion TDAH
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour la sérialisation"""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "emotional_state": self.emotional_state.__dict__ if self.emotional_state else None,
            "context": self.context,
            "memory_type": self.memory_type,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "consolidation_level": self.consolidation_level,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "tags": self.tags,
            "user_id": self.user_id,
            "learning_category": self.learning_category,
            "tdah_task_type": self.tdah_task_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Crée un MemoryItem depuis un dictionnaire"""
        if data.get("emotional_state"):
            emotional_state = EmotionalState(**data["emotional_state"])
        else:
            emotional_state = None
        
        embedding = np.array(data["embedding"]) if data.get("embedding") else None
        
        return cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data["importance"],
            emotional_state=emotional_state,
            context=data.get("context", ""),
            memory_type=data.get("memory_type", "episodic"),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", datetime.now().isoformat())),
            consolidation_level=data.get("consolidation_level", 0),
            embedding=embedding,
            tags=data.get("tags", []),
            user_id=data.get("user_id", ""),
            learning_category=data.get("learning_category", ""),
            tdah_task_type=data.get("tdah_task_type", "")
        )


@dataclass
class TDAHTask:
    """Tâche pour la gestion TDAH"""
    
    id: str
    title: str
    description: str
    priority: int  # 1-5
    due_date: Optional[datetime] = None
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    category: str = ""  # travail, personnel, santé, etc.
    estimated_duration: Optional[int] = None  # en minutes
    emotional_state: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed": self.completed,
            "created_at": self.created_at.isoformat(),
            "category": self.category,
            "estimated_duration": self.estimated_duration,
            "emotional_state": self.emotional_state
        }


@dataclass
class LearningItem:
    """Item d'apprentissage"""
    
    concept: str
    explanation: str
    examples: List[str]
    difficulty_level: int  # 1-5
    mastery_level: float  # 0.0-1.0
    last_reviewed: datetime
    next_review: datetime
    category: str
    user_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "explanation": self.explanation,
            "examples": self.examples,
            "difficulty_level": self.difficulty_level,
            "mastery_level": self.mastery_level,
            "last_reviewed": self.last_reviewed.isoformat(),
            "next_review": self.next_review.isoformat(),
            "category": self.category,
            "user_id": self.user_id
        }


class VectorIndex:
    """Index vectoriel pour recherche sémantique rapide"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Index par produit scalaire
        self.memory_ids = []  # Mapping index -> memory_id
        self._lock = threading.Lock()
    
    def add_vectors(self, vectors: np.ndarray, memory_ids: List[str]):
        """Ajoute des vecteurs à l'index"""
        with self._lock:
            # Normaliser les vecteurs pour le produit scalaire
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            self.memory_ids.extend(memory_ids)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Recherche les k vecteurs les plus similaires"""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # Normaliser le vecteur de requête
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Recherche
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.memory_ids):
                    results.append((self.memory_ids[idx], float(score)))
            
            return results
    
    def remove_vector(self, memory_id: str):
        """Supprime un vecteur de l'index (reconstruction complète)"""
        # Note: FAISS ne supporte pas la suppression efficace
        # Dans un cas réel, on utiliserait IndexIDMap ou reconstruction périodique
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'index"""
        with self._lock:
            return {
                "total_vectors": self.index.ntotal,
                "dimension": self.dimension,
                "memory_ids_count": len(self.memory_ids)
            }


class IntelligentMemorySystem:
    """Système de mémoire intelligent avec consolidation automatique"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        
        # Modèle d'embeddings
        self.embedding_model = None
        
        # Stockage par niveaux
        self.working_memory = deque(maxlen=config.short_term_capacity)
        self.short_term_memory = {}  # id -> MemoryItem
        self.long_term_memory = {}   # id -> MemoryItem
        
        # Nouvelles structures pour les fonctionnalités avancées
        self.names_memory = {}  # name -> user_info
        self.learning_items = {}  # concept_id -> LearningItem
        self.tdah_tasks = {}  # task_id -> TDAHTask
        self.emotional_tracking = defaultdict(list)  # user_id -> emotional_states
        
        # Index vectoriels
        self.vector_index = VectorIndex()
        
        # Base de données pour persistance
        self.db_path = Path(config.database_url.replace("sqlite:///", ""))
        self.db_lock = threading.Lock()
        
        # Système de consolidation
        self.consolidation_thread = None
        self.should_consolidate = threading.Event()
        
        # Métriques et statistiques
        self.stats = defaultdict(int)
        
    async def initialize(self):
        """Initialise le système de mémoire"""
        try:
            logger.info("Initialisation du système de mémoire intelligent...")
            
            # Charger le modèle d'embeddings
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Initialiser la base de données
            await self._init_database()
            
            # Charger les mémoires existantes
            await self._load_existing_memories()
            
            # Démarrer le processus de consolidation
            self._start_consolidation_process()
            
            logger.info("Système de mémoire intelligent initialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la mémoire: {e}")
            raise
    
    async def _init_database(self):
        """Initialise la base de données SQLite"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    content TEXT,
                    timestamp TEXT,
                    importance REAL,
                    emotional_state TEXT,
                    context TEXT,
                    memory_type TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    consolidation_level INTEGER DEFAULT 0,
                    embedding BLOB,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_memories ON memories(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
            """)
            
            conn.commit()
    
    async def _load_existing_memories(self):
        """Charge les mémoires existantes depuis la base de données"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM memories ORDER BY timestamp DESC LIMIT 1000
                """)
                
                vectors_to_add = []
                memory_ids_to_add = []
                
                for row in cursor.fetchall():
                    memory_item = self._row_to_memory_item(row)
                    
                    # Ajouter à la mémoire appropriée selon le niveau de consolidation
                    if memory_item.consolidation_level == 0:
                        self.working_memory.append(memory_item)
                    elif memory_item.consolidation_level == 1:
                        self.short_term_memory[self._generate_memory_id(memory_item)] = memory_item
                    else:
                        memory_id = self._generate_memory_id(memory_item)
                        self.long_term_memory[memory_id] = memory_item
                        
                        # Préparer pour l'index vectoriel
                        if memory_item.embedding is not None:
                            vectors_to_add.append(memory_item.embedding)
                            memory_ids_to_add.append(memory_id)
                
                # Ajouter à l'index vectoriel
                if vectors_to_add:
                    vectors_array = np.array(vectors_to_add)
                    self.vector_index.add_vectors(vectors_array, memory_ids_to_add)
                
                logger.info(f"Chargé {len(self.long_term_memory)} mémoires long terme")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des mémoires: {e}")
    
    def _row_to_memory_item(self, row) -> MemoryItem:
        """Convertit une ligne de base de données en MemoryItem"""
        (memory_id, user_id, content, timestamp, importance, emotional_state_json,
         context, memory_type, access_count, last_accessed, consolidation_level,
         embedding_blob, tags_json, metadata_json) = row
        
        # Désérialiser l'état émotionnel
        emotional_state = None
        if emotional_state_json:
            try:
                emotional_state = EmotionalState(**json.loads(emotional_state_json))
            except Exception:
                pass
        
        # Désérialiser l'embedding
        embedding = None
        if embedding_blob:
            try:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            except Exception:
                pass
        
        # Désérialiser les tags
        tags = []
        if tags_json:
            try:
                tags = json.loads(tags_json)
            except Exception:
                pass
        
        return MemoryItem(
            content=content,
            timestamp=datetime.fromisoformat(timestamp),
            importance=importance,
            emotional_state=emotional_state,
            context=context,
            memory_type=memory_type,
            access_count=access_count,
            last_accessed=datetime.fromisoformat(last_accessed),
            consolidation_level=consolidation_level,
            embedding=embedding,
            tags=tags,
            user_id=user_id
        )
    
    def _generate_memory_id(self, memory_item: MemoryItem) -> str:
        """Génère un ID unique pour un item de mémoire"""
        content_hash = hashlib.md5(
            f"{memory_item.content}_{memory_item.timestamp}_{memory_item.user_id}".encode()
        ).hexdigest()
        return f"mem_{content_hash[:12]}"
    
    async def store_memory(
        self,
        content: str,
        user_id: str,
        importance: float,
        emotional_state: Optional[EmotionalState] = None,
        context: str = "",
        memory_type: str = "episodic",
        tags: List[str] = None
    ) -> str:
        """Stocke un nouveau souvenir"""
        
        # Créer l'item de mémoire
        memory_item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            emotional_state=emotional_state,
            context=context,
            memory_type=memory_type,
            tags=tags or [],
            user_id=user_id
        )
        
        # Générer l'embedding
        memory_item.embedding = self.embedding_model.encode([content])[0]
        
        # Générer l'ID
        memory_id = self._generate_memory_id(memory_item)
        
        # Stocker dans la mémoire de travail
        self.working_memory.append(memory_item)
        
        # Sauvegarder en base de données
        await self._save_memory_to_db(memory_id, memory_item)
        
        # Déclencher la consolidation si nécessaire
        if len(self.working_memory) >= self.config.short_term_capacity * 0.8:
            self.should_consolidate.set()
        
        self.stats['memories_stored'] += 1
        return memory_id
    
    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        memory_types: List[str] = None,
        importance_threshold: float = 0.0,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """Récupère les mémoires les plus pertinentes"""
        
        # Générer l'embedding de la requête
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Recherche vectorielle dans la mémoire long terme
        vector_results = self.vector_index.search(query_embedding, k * 2)
        
        # Collecter tous les candidats
        candidates = []
        
        # Mémoire long terme (vectorielle)
        for memory_id, similarity in vector_results:
            if memory_id in self.long_term_memory:
                memory_item = self.long_term_memory[memory_id]
                if self._matches_criteria(memory_item, user_id, memory_types, importance_threshold, time_range):
                    candidates.append((memory_item, similarity))
        
        # Mémoire court terme (force brute)
        for memory_item in self.short_term_memory.values():
            if self._matches_criteria(memory_item, user_id, memory_types, importance_threshold, time_range):
                similarity = cosine_similarity([query_embedding], [memory_item.embedding])[0][0]
                candidates.append((memory_item, similarity))
        
        # Mémoire de travail (force brute)
        for memory_item in self.working_memory:
            if self._matches_criteria(memory_item, user_id, memory_types, importance_threshold, time_range):
                if memory_item.embedding is not None:
                    similarity = cosine_similarity([query_embedding], [memory_item.embedding])[0][0]
                    candidates.append((memory_item, similarity))
        
        # Trier par similarité et prendre les k meilleurs
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = candidates[:k]
        
        # Mettre à jour les statistiques d'accès
        for memory_item, _ in results:
            memory_item.access_count += 1
            memory_item.last_accessed = datetime.now()
        
        self.stats['retrievals_performed'] += 1
        return results
    
    def _matches_criteria(
        self,
        memory_item: MemoryItem,
        user_id: str,
        memory_types: Optional[List[str]],
        importance_threshold: float,
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> bool:
        """Vérifie si un item de mémoire correspond aux critères"""
        
        if memory_item.user_id != user_id:
            return False
        
        if memory_item.importance < importance_threshold:
            return False
        
        if memory_types and memory_item.memory_type not in memory_types:
            return False
        
        if time_range:
            start_time, end_time = time_range
            if not (start_time <= memory_item.timestamp <= end_time):
                return False
        
        return True
    
    async def _save_memory_to_db(self, memory_id: str, memory_item: MemoryItem):
        """Sauvegarde un item de mémoire en base de données"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Sérialiser les données complexes
                emotional_state_json = None
                if memory_item.emotional_state:
                    emotional_state_json = json.dumps(memory_item.emotional_state.to_dict())
                
                embedding_blob = None
                if memory_item.embedding is not None:
                    embedding_blob = memory_item.embedding.astype(np.float32).tobytes()
                
                tags_json = json.dumps(memory_item.tags)
                
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, user_id, content, timestamp, importance, emotional_state, 
                     context, memory_type, access_count, last_accessed, 
                     consolidation_level, embedding, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id, memory_item.user_id, memory_item.content,
                    memory_item.timestamp.isoformat(), memory_item.importance,
                    emotional_state_json, memory_item.context, memory_item.memory_type,
                    memory_item.access_count, memory_item.last_accessed.isoformat(),
                    memory_item.consolidation_level, embedding_blob, tags_json, "{}"
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde en base: {e}")
    
    def _start_consolidation_process(self):
        """Démarre le processus de consolidation en arrière-plan"""
        if self.consolidation_thread is None or not self.consolidation_thread.is_alive():
            self.consolidation_thread = threading.Thread(
                target=self._consolidation_worker, 
                daemon=True
            )
            self.consolidation_thread.start()
    
    def _consolidation_worker(self):
        """Worker de consolidation qui s'exécute en arrière-plan"""
        while True:
            try:
                # Attendre le signal de consolidation ou timeout
                if self.should_consolidate.wait(timeout=self.config.memory_consolidation_interval):
                    self.should_consolidate.clear()
                
                asyncio.run(self._perform_consolidation())
                
            except Exception as e:
                logger.error(f"Erreur durant la consolidation: {e}")
    
    async def _perform_consolidation(self):
        """Effectue la consolidation des mémoires"""
        try:
            logger.info("Début de la consolidation des mémoires")
            
            # Consolider la mémoire de travail vers court terme
            await self._consolidate_working_to_short_term()
            
            # Consolider la mémoire court terme vers long terme
            await self._consolidate_short_to_long_term()
            
            # Nettoyer les mémoires anciennes peu importantes
            await self._cleanup_old_memories()
            
            self.stats['consolidations_performed'] += 1
            logger.info("Consolidation terminée")
            
        except Exception as e:
            logger.error(f"Erreur durant la consolidation: {e}")
    
    async def _consolidate_working_to_short_term(self):
        """Consolide la mémoire de travail vers la mémoire court terme"""
        items_to_move = []
        
        # Sélectionner les items importants de la mémoire de travail
        for memory_item in list(self.working_memory):
            if memory_item.importance >= self.config.short_term_relevance_threshold:
                items_to_move.append(memory_item)
        
        # Déplacer vers la mémoire court terme
        for memory_item in items_to_move:
            memory_item.consolidation_level = 1
            memory_id = self._generate_memory_id(memory_item)
            self.short_term_memory[memory_id] = memory_item
            
            # Mettre à jour en base
            await self._save_memory_to_db(memory_id, memory_item)
            
            # Retirer de la mémoire de travail
            try:
                self.working_memory.remove(memory_item)
            except ValueError:
                pass
        
        logger.info(f"Consolidé {len(items_to_move)} items vers la mémoire court terme")
    
    async def _consolidate_short_to_long_term(self):
        """Consolide la mémoire court terme vers la mémoire long terme"""
        items_to_move = []
        
        # Sélectionner les items très importants ou accédés fréquemment
        for memory_id, memory_item in list(self.short_term_memory.items()):
            age_days = (datetime.now() - memory_item.timestamp).days
            
            should_promote = (
                memory_item.importance >= self.config.long_term_importance_threshold or
                memory_item.access_count >= 3 or
                age_days >= 7  # Après une semaine, promouvoir automatiquement
            )
            
            if should_promote:
                items_to_move.append((memory_id, memory_item))
        
        # Préparer les vecteurs pour l'index
        vectors_to_add = []
        memory_ids_to_add = []
        
        # Déplacer vers la mémoire long terme
        for memory_id, memory_item in items_to_move:
            memory_item.consolidation_level = 2
            self.long_term_memory[memory_id] = memory_item
            
            # Ajouter à l'index vectoriel
            if memory_item.embedding is not None:
                vectors_to_add.append(memory_item.embedding)
                memory_ids_to_add.append(memory_id)
            
            # Mettre à jour en base
            await self._save_memory_to_db(memory_id, memory_item)
            
            # Retirer de la mémoire court terme
            del self.short_term_memory[memory_id]
        
        # Mettre à jour l'index vectoriel
        if vectors_to_add:
            vectors_array = np.array(vectors_to_add)
            self.vector_index.add_vectors(vectors_array, memory_ids_to_add)
        
        logger.info(f"Consolidé {len(items_to_move)} items vers la mémoire long terme")
    
    async def _cleanup_old_memories(self):
        """Nettoie les mémoires anciennes peu importantes"""
        cutoff_date = datetime.now() - timedelta(days=self.config.knowledge_retention_days)
        items_to_remove = []
        
        # Identifier les mémoires à supprimer
        for memory_id, memory_item in list(self.long_term_memory.items()):
            if (memory_item.timestamp < cutoff_date and 
                memory_item.importance < 0.3 and 
                memory_item.access_count < 2):
                items_to_remove.append(memory_id)
        
        # Supprimer les mémoires
        for memory_id in items_to_remove:
            del self.long_term_memory[memory_id]
            
            # Supprimer de la base de données
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
        
        logger.info(f"Supprimé {len(items_to_remove)} mémoires anciennes")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de mémoire"""
        return {
            "working_memory_size": len(self.working_memory),
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "vector_index_stats": self.vector_index.get_stats(),
            "operations_stats": dict(self.stats),
            "config": {
                "short_term_capacity": self.config.short_term_capacity,
                "long_term_capacity": self.config.long_term_capacity,
                "consolidation_interval": self.config.memory_consolidation_interval
            }
        }
    
    async def search_by_emotion(
        self,
        emotional_state: EmotionalState,
        user_id: str,
        k: int = 5
    ) -> List[Tuple[MemoryItem, float]]:
        """Recherche des mémoires par similarité émotionnelle"""
        target_vector = emotional_state.emotional_vector()
        
        candidates = []
        
        # Rechercher dans toutes les mémoires
        all_memories = list(self.long_term_memory.values()) + list(self.short_term_memory.values()) + list(self.working_memory)
        
        for memory_item in all_memories:
            if memory_item.user_id == user_id and memory_item.emotional_state:
                memory_vector = memory_item.emotional_state.emotional_vector()
                similarity = cosine_similarity([target_vector], [memory_vector])[0][0]
                candidates.append((memory_item, similarity))
        
        # Trier et retourner les k meilleurs
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    async def get_emotional_timeline(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Tuple[datetime, EmotionalState]]:
        """Retourne une timeline des états émotionnels"""
        cutoff_date = datetime.now() - timedelta(days=days)
        timeline = []
        
        # Collecter tous les états émotionnels
        all_memories = list(self.long_term_memory.values()) + list(self.short_term_memory.values()) + list(self.working_memory)
        
        for memory_item in all_memories:
            if (memory_item.user_id == user_id and 
                memory_item.emotional_state and 
                memory_item.timestamp >= cutoff_date):
                timeline.append((memory_item.timestamp, memory_item.emotional_state))
        
        # Trier par timestamp
        timeline.sort(key=lambda x: x[0])
        return timeline
    
    # ==================== NOUVELLES FONCTIONNALITÉS ====================
    
    async def remember_user_name(self, user_id: str, name: str, nickname: str = "") -> bool:
        """Retient le nom d'un utilisateur"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Créer la table si elle n'existe pas
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_names (
                        user_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        nickname TEXT,
                        created_at TEXT,
                        last_updated TEXT
                    )
                """)
                
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_names 
                    (user_id, name, nickname, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, name, nickname, now, now))
                
                conn.commit()
                conn.close()
                
                # Mettre à jour la mémoire en cache
                self.names_memory[user_id] = {
                    "name": name,
                    "nickname": nickname,
                    "created_at": now,
                    "last_updated": now
                }
                
                # Ajouter en mémoire long terme
                if self.embedding_model:
                    await self.store_memory(
                        content=f"L'utilisateur s'appelle {name}" + (f" (surnom: {nickname})" if nickname else ""),
                        user_id=user_id,
                        importance=1.0,
                        memory_type="name",
                        tags=["nom", "identité", "personnel"]
                    )
                
                logger.info(f"Nom retenu pour {user_id}: {name}")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du nom: {e}")
            return False
    
    async def get_user_name(self, user_id: str) -> Optional[str]:
        """Récupère le nom d'un utilisateur"""
        if user_id in self.names_memory:
            return self.names_memory[user_id]["name"]
        
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name, nickname FROM user_names WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    return result[0]
                    
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du nom: {e}")
        
        return None
    
    async def learn_concept(
        self,
        user_id: str,
        concept_name: str,
        explanation: str,
        examples: List[str] = None,
        category: str = "general",
        difficulty_level: int = 3
    ) -> str:
        """Apprend un nouveau concept enseigné par l'utilisateur"""
        try:
            concept_id = f"concept_{hashlib.md5(f'{user_id}_{concept_name}'.encode()).hexdigest()[:12]}"
            
            learning_item = LearningItem(
                concept=concept_name,
                explanation=explanation,
                examples=examples or [],
                difficulty_level=difficulty_level,
                mastery_level=0.8,  # Niveau initial de maîtrise
                last_reviewed=datetime.now(),
                next_review=datetime.now() + timedelta(days=7),
                category=category,
                user_id=user_id
            )
            
            # Stocker en mémoire
            self.learning_items[concept_id] = learning_item
            
            # Sauvegarder en base de données
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learned_concepts (
                        concept_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        concept_name TEXT,
                        explanation TEXT,
                        examples TEXT,
                        category TEXT,
                        difficulty_level INTEGER,
                        mastery_level REAL,
                        created_at TEXT,
                        last_reviewed TEXT
                    )
                """)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO learned_concepts 
                    (concept_id, user_id, concept_name, explanation, examples, category, 
                     difficulty_level, mastery_level, created_at, last_reviewed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    concept_id, user_id, concept_name, explanation,
                    json.dumps(examples or []), category, difficulty_level,
                    learning_item.mastery_level, datetime.now().isoformat(),
                    learning_item.last_reviewed.isoformat()
                ))
                
                conn.commit()
                conn.close()
            
            # Ajouter en mémoire long terme
            if self.embedding_model:
                await self.store_memory(
                    content=f"Concept appris: {concept_name} - {explanation}",
                    user_id=user_id,
                    importance=0.9,
                    memory_type="learning",
                    tags=["apprentissage", "concept", category]
                )
            
            logger.info(f"Concept appris pour {user_id}: {concept_name}")
            return concept_id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'apprentissage du concept: {e}")
            return ""
    
    async def get_learned_concepts(self, user_id: str, category: str = None) -> List[LearningItem]:
        """Récupère les concepts appris par un utilisateur"""
        concepts = []
        
        for learning_item in self.learning_items.values():
            if learning_item.user_id == user_id:
                if category is None or learning_item.category == category:
                    concepts.append(learning_item)
        
        return sorted(concepts, key=lambda x: x.last_reviewed, reverse=True)

    # ==================== GESTION TDAH ====================
    
    async def create_tdah_task(
        self,
        user_id: str,
        title: str,
        description: str = "",
        priority: int = 3,
        category: str = "general",
        due_date: Optional[datetime] = None,
        estimated_duration: Optional[int] = None,
        emotional_state: Optional[str] = None
    ) -> str:
        """Crée une nouvelle tâche TDAH"""
        try:
            task_id = f"task_{hashlib.md5(f'{user_id}_{title}_{datetime.now()}'.encode()).hexdigest()[:12]}"
            
            task = TDAHTask(
                id=task_id,
                title=title,
                description=description,
                priority=priority,
                due_date=due_date,
                completed=False,
                created_at=datetime.now(),
                category=category,
                estimated_duration=estimated_duration,
                emotional_state=emotional_state
            )
            
            # Stocker en mémoire
            self.tdah_tasks[task_id] = task
            
            # Sauvegarder en base de données
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tdah_tasks (
                        task_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        title TEXT,
                        description TEXT,
                        priority INTEGER,
                        due_date TEXT,
                        completed BOOLEAN,
                        created_at TEXT,
                        category TEXT,
                        estimated_duration INTEGER,
                        emotional_state TEXT
                    )
                """)
                
                cursor.execute("""
                    INSERT INTO tdah_tasks 
                    (task_id, user_id, title, description, priority, due_date, completed, 
                     created_at, category, estimated_duration, emotional_state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id, user_id, title, description, priority,
                    due_date.isoformat() if due_date else None, False,
                    task.created_at.isoformat(), category, estimated_duration, emotional_state
                ))
                
                conn.commit()
                conn.close()
            
            # Ajouter en mémoire long terme
            if self.embedding_model:
                await self.store_memory(
                    content=f"Tâche TDAH créée: {title} - {description}",
                    user_id=user_id,
                    importance=0.7,
                    memory_type="tdah",
                    tags=["tâche", "tdah", category]
                )
            
            logger.info(f"Tâche TDAH créée pour {user_id}: {title}")
            return task_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la tâche TDAH: {e}")
            return ""
    
    async def get_tdah_tasks(
        self,
        user_id: str,
        completed: Optional[bool] = None,
        category: Optional[str] = None,
        priority_min: int = 1
    ) -> List[TDAHTask]:
        """Récupère les tâches TDAH d'un utilisateur"""
        try:
            # Charger depuis la base de données si pas en mémoire
            await self._load_tdah_tasks_from_db(user_id)
            
            tasks = []
            for task in self.tdah_tasks.values():
                # Filtrer par utilisateur
                task_user_id = getattr(task, 'user_id', None)
                if task_user_id != user_id:
                    continue
                    
                # Vérifier les filtres
                if completed is not None and task.completed != completed:
                    continue
                if category and task.category != category:
                    continue
                if task.priority < priority_min:
                    continue
                
                tasks.append(task)
            
            return sorted(tasks, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des tâches TDAH: {e}")
            return []
    
    async def _load_tdah_tasks_from_db(self, user_id: str):
        """Charge les tâches TDAH depuis la base de données"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT task_id, user_id, title, description, priority, due_date, 
                           completed, created_at, category, estimated_duration, emotional_state
                    FROM tdah_tasks WHERE user_id = ?
                """, (user_id,))
                
                for row in cursor.fetchall():
                    task_id, uid, title, description, priority, due_date, completed, created_at, category, estimated_duration, emotional_state = row
                    
                    if task_id not in self.tdah_tasks:
                        task = TDAHTask(
                            id=task_id,
                            title=title,
                            description=description or "",
                            priority=priority,
                            due_date=datetime.fromisoformat(due_date) if due_date else None,
                            completed=bool(completed),
                            created_at=datetime.fromisoformat(created_at),
                            category=category or "general",
                            estimated_duration=estimated_duration,
                            emotional_state=emotional_state
                        )
                        # Ajouter l'user_id comme attribut
                        task.user_id = uid
                        self.tdah_tasks[task_id] = task
                
                conn.close()
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des tâches TDAH: {e}")
    
    async def complete_tdah_task(self, user_id: str, task_id: str) -> bool:
        """Marque une tâche TDAH comme terminée"""
        try:
            if task_id not in self.tdah_tasks:
                await self._load_tdah_tasks_from_db(user_id)
            
            if task_id in self.tdah_tasks:
                task = self.tdah_tasks[task_id]
                task.completed = True
                
                # Mettre à jour en base
                with self.db_lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE tdah_tasks SET completed = 1 WHERE task_id = ?
                    """, (task_id,))
                    
                    conn.commit()
                    conn.close()
                
                logger.info(f"Tâche TDAH {task_id} marquée comme terminée")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la complétion de la tâche TDAH: {e}")
            return False
    
    async def get_tdah_suggestions(self, user_id: str) -> List[str]:
        """Génère des suggestions pour la gestion du TDAH"""
        try:
            # Récupérer les tâches de l'utilisateur
            tasks = await self.get_tdah_tasks(user_id)
            
            suggestions = []
            
            # Analyser les patterns
            active_tasks = [t for t in tasks if not t.completed]
            completed_tasks = [t for t in tasks if t.completed]
            
            if len(active_tasks) > 10:
                suggestions.append("Vous avez beaucoup de tâches en cours. Considérez en prioriser quelques-unes.")
            
            if len(active_tasks) == 0:
                suggestions.append("Excellent ! Vous n'avez aucune tâche en cours. C'est le moment d'en planifier de nouvelles.")
            
            # Analyser les priorités
            high_priority = [t for t in active_tasks if t.priority >= 4]
            if len(high_priority) > 3:
                suggestions.append("Vous avez plusieurs tâches haute priorité. Concentrez-vous sur une à la fois.")
            
            # Analyser les échéances
            overdue_tasks = []
            today = datetime.now()
            for task in active_tasks:
                if task.due_date and task.due_date < today:
                    overdue_tasks.append(task)
            
            if overdue_tasks:
                suggestions.append(f"Vous avez {len(overdue_tasks)} tâche(s) en retard. Considérez les reprioriser.")
            
            # Suggestions générales
            if len(completed_tasks) > len(active_tasks):
                suggestions.append("Bravo ! Vous terminez plus de tâches que vous n'en créez. Gardez cette dynamique !")
            
            if not suggestions:
                suggestions.append("Votre gestion des tâches semble équilibrée. Continuez comme ça !")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des suggestions TDAH: {e}")
            return ["Erreur lors de l'analyse de vos tâches."]