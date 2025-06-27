"""
Système de Mémoire Avancée pour EmoIA
Mémoire à court terme, long terme, apprentissage et gestion du TDAH
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
import uuid

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import faiss

from ..config import MemoryConfig
from ..emotional import EmotionalState, PersonalityProfile


logger = logging.getLogger(__name__)


@dataclass
class ShortTermMemory:
    """Mémoire à court terme (quelques minutes à quelques heures)"""
    
    content: str
    timestamp: datetime
    importance: float
    emotional_state: Optional[EmotionalState] = None
    context: str = ""
    memory_type: str = "episodic"
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    user_id: str = ""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)


@dataclass
class LongTermMemory:
    """Mémoire à long terme (consolidée et permanente)"""
    
    content: str
    timestamp: datetime
    importance: float
    emotional_state: Optional[EmotionalState] = None
    context: str = ""
    memory_type: str = "semantic"
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    consolidation_date: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    user_id: str = ""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    learned_concepts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)
        if isinstance(self.consolidation_date, str):
            self.consolidation_date = datetime.fromisoformat(self.consolidation_date)


@dataclass
class LearnedConcept:
    """Concept appris par l'utilisateur"""
    
    concept_name: str
    description: str
    examples: List[str]
    learned_date: datetime
    confidence_level: float  # 0.0 à 1.0
    user_id: str = ""
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if isinstance(self.learned_date, str):
            self.learned_date = datetime.fromisoformat(self.learned_date)


@dataclass
class TDAHManagement:
    """Gestion du TDAH - tâches, temps, émotions"""
    
    user_id: str
    current_tasks: List[Dict[str, Any]] = field(default_factory=list)
    time_blocks: List[Dict[str, Any]] = field(default_factory=list)
    emotional_regulation: Dict[str, Any] = field(default_factory=dict)
    focus_sessions: List[Dict[str, Any]] = field(default_factory=list)
    reminders: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UserProfile:
    """Profil utilisateur avec nom et préférences"""
    
    user_id: str
    name: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    learning_style: str = "visual"
    communication_preferences: Dict[str, Any] = field(default_factory=dict)
    tdah_management: Optional[TDAHManagement] = None


class AdvancedMemorySystem:
    """Système de mémoire avancé avec apprentissage et gestion TDAH"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        
        # Modèle d'embeddings
        self.embedding_model = None
        
        # Mémoires
        self.short_term_memory = {}  # user_id -> deque de ShortTermMemory
        self.long_term_memory = {}   # user_id -> dict de LongTermMemory
        self.learned_concepts = {}   # user_id -> dict de LearnedConcept
        
        # Profils utilisateurs
        self.user_profiles = {}  # user_id -> UserProfile
        
        # Index vectoriels
        self.vector_index = VectorIndex()
        
        # Base de données
        self.db_path = Path("emoia_advanced_memory.db")
        self.db_lock = threading.Lock()
        
        # Système de consolidation
        self.consolidation_thread = None
        self.should_consolidate = threading.Event()
        
        # Métriques
        self.stats = defaultdict(int)
        
    async def initialize(self):
        """Initialise le système de mémoire avancé"""
        try:
            logger.info("Initialisation du système de mémoire avancé...")
            
            # Charger le modèle d'embeddings
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Initialiser la base de données
            await self._init_database()
            
            # Charger les données existantes
            await self._load_existing_data()
            
            # Démarrer la consolidation
            self._start_consolidation_process()
            
            logger.info("Système de mémoire avancé initialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def _init_database(self):
        """Initialise la base de données"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table mémoire à court terme
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS short_term_memory (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    content TEXT,
                    timestamp TEXT,
                    importance REAL,
                    emotional_state TEXT,
                    context TEXT,
                    memory_type TEXT,
                    access_count INTEGER,
                    last_accessed TEXT
                )
            """)
            
            # Table mémoire à long terme
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    content TEXT,
                    timestamp TEXT,
                    importance REAL,
                    emotional_state TEXT,
                    context TEXT,
                    memory_type TEXT,
                    access_count INTEGER,
                    last_accessed TEXT,
                    consolidation_date TEXT,
                    embedding TEXT,
                    tags TEXT,
                    learned_concepts TEXT
                )
            """)
            
            # Table concepts appris
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_concepts (
                    concept_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    concept_name TEXT,
                    description TEXT,
                    examples TEXT,
                    learned_date TEXT,
                    confidence_level REAL
                )
            """)
            
            # Table profils utilisateurs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    preferences TEXT,
                    personality_traits TEXT,
                    learning_style TEXT,
                    communication_preferences TEXT
                )
            """)
            
            # Table gestion TDAH
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tdah_management (
                    user_id TEXT PRIMARY KEY,
                    current_tasks TEXT,
                    time_blocks TEXT,
                    emotional_regulation TEXT,
                    focus_sessions TEXT,
                    reminders TEXT
                )
            """)
            
            conn.commit()
            conn.close()
    
    async def _load_existing_data(self):
        """Charge les données existantes"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Charger les profils utilisateurs
            cursor.execute("SELECT * FROM user_profiles")
            for row in cursor.fetchall():
                user_id, name, prefs, traits, style, comm_prefs = row
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    name=name,
                    preferences=json.loads(prefs) if prefs else {},
                    personality_traits=json.loads(traits) if traits else {},
                    learning_style=style or "visual",
                    communication_preferences=json.loads(comm_prefs) if comm_prefs else {}
                )
            
            # Charger la gestion TDAH
            cursor.execute("SELECT * FROM tdah_management")
            for row in cursor.fetchall():
                user_id, tasks, blocks, regulation, sessions, reminders = row
                if user_id in self.user_profiles:
                    self.user_profiles[user_id].tdah_management = TDAHManagement(
                        user_id=user_id,
                        current_tasks=json.loads(tasks) if tasks else [],
                        time_blocks=json.loads(blocks) if blocks else [],
                        emotional_regulation=json.loads(regulation) if regulation else {},
                        focus_sessions=json.loads(sessions) if sessions else [],
                        reminders=json.loads(reminders) if reminders else []
                    )
            
            conn.close()
    
    async def create_user_profile(self, user_id: str, name: str, **kwargs) -> UserProfile:
        """Crée un nouveau profil utilisateur"""
        profile = UserProfile(
            user_id=user_id,
            name=name,
            preferences=kwargs.get('preferences', {}),
            personality_traits=kwargs.get('personality_traits', {}),
            learning_style=kwargs.get('learning_style', 'visual'),
            communication_preferences=kwargs.get('communication_preferences', {}),
            tdah_management=TDAHManagement(user_id=user_id)
        )
        
        self.user_profiles[user_id] = profile
        self.short_term_memory[user_id] = deque(maxlen=100)
        self.long_term_memory[user_id] = {}
        self.learned_concepts[user_id] = {}
        
        # Sauvegarder en base
        await self._save_user_profile(profile)
        
        return profile
    
    async def _save_user_profile(self, profile: UserProfile):
        """Sauvegarde un profil utilisateur"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, name, preferences, personality_traits, learning_style, communication_preferences)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.name,
                json.dumps(profile.preferences),
                json.dumps(profile.personality_traits),
                profile.learning_style,
                json.dumps(profile.communication_preferences)
            ))
            
            # Sauvegarder la gestion TDAH
            if profile.tdah_management:
                cursor.execute("""
                    INSERT OR REPLACE INTO tdah_management 
                    (user_id, current_tasks, time_blocks, emotional_regulation, focus_sessions, reminders)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    json.dumps(profile.tdah_management.current_tasks),
                    json.dumps(profile.tdah_management.time_blocks),
                    json.dumps(profile.tdah_management.emotional_regulation),
                    json.dumps(profile.tdah_management.focus_sessions),
                    json.dumps(profile.tdah_management.reminders)
                ))
            
            conn.commit()
            conn.close()
    
    async def store_short_term_memory(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        emotional_state: Optional[EmotionalState] = None,
        context: str = "",
        memory_type: str = "episodic"
    ) -> str:
        """Stocke une information en mémoire à court terme"""
        
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = deque(maxlen=100)
        
        memory = ShortTermMemory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            emotional_state=emotional_state,
            context=context,
            memory_type=memory_type,
            user_id=user_id
        )
        
        self.short_term_memory[user_id].append(memory)
        
        # Sauvegarder en base
        await self._save_short_term_memory(memory)
        
        return memory.memory_id
    
    async def _save_short_term_memory(self, memory: ShortTermMemory):
        """Sauvegarde une mémoire à court terme"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO short_term_memory 
                (memory_id, user_id, content, timestamp, importance, emotional_state, 
                 context, memory_type, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.memory_id,
                memory.user_id,
                memory.content,
                memory.timestamp.isoformat(),
                memory.importance,
                json.dumps(memory.emotional_state.__dict__) if memory.emotional_state else None,
                memory.context,
                memory.memory_type,
                memory.access_count,
                memory.last_accessed.isoformat()
            ))
            
            conn.commit()
            conn.close()
    
    async def store_long_term_memory(
        self,
        user_id: str,
        content: str,
        importance: float = 0.7,
        emotional_state: Optional[EmotionalState] = None,
        context: str = "",
        memory_type: str = "semantic",
        tags: List[str] = None,
        learned_concepts: List[str] = None
    ) -> str:
        """Stocke une information en mémoire à long terme"""
        
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = {}
        
        # Générer l'embedding
        embedding = self.embedding_model.encode([content])[0] if self.embedding_model else None
        
        memory = LongTermMemory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            emotional_state=emotional_state,
            context=context,
            memory_type=memory_type,
            tags=tags or [],
            learned_concepts=learned_concepts or [],
            user_id=user_id,
            embedding=embedding
        )
        
        self.long_term_memory[user_id][memory.memory_id] = memory
        
        # Sauvegarder en base
        await self._save_long_term_memory(memory)
        
        return memory.memory_id
    
    async def _save_long_term_memory(self, memory: LongTermMemory):
        """Sauvegarde une mémoire à long terme"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO long_term_memory 
                (memory_id, user_id, content, timestamp, importance, emotional_state, 
                 context, memory_type, access_count, last_accessed, consolidation_date,
                 embedding, tags, learned_concepts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.memory_id,
                memory.user_id,
                memory.content,
                memory.timestamp.isoformat(),
                memory.importance,
                json.dumps(memory.emotional_state.__dict__) if memory.emotional_state else None,
                memory.context,
                memory.memory_type,
                memory.access_count,
                memory.last_accessed.isoformat(),
                memory.consolidation_date.isoformat(),
                json.dumps(memory.embedding.tolist()) if memory.embedding is not None else None,
                json.dumps(memory.tags),
                json.dumps(memory.learned_concepts)
            ))
            
            conn.commit()
            conn.close()
    
    async def learn_concept(
        self,
        user_id: str,
        concept_name: str,
        description: str,
        examples: List[str],
        confidence_level: float = 0.8
    ) -> str:
        """Enregistre un concept appris par l'utilisateur"""
        
        if user_id not in self.learned_concepts:
            self.learned_concepts[user_id] = {}
        
        concept = LearnedConcept(
            concept_name=concept_name,
            description=description,
            examples=examples,
            learned_date=datetime.now(),
            confidence_level=confidence_level,
            user_id=user_id
        )
        
        self.learned_concepts[user_id][concept.concept_id] = concept
        
        # Sauvegarder en base
        await self._save_learned_concept(concept)
        
        return concept.concept_id
    
    async def _save_learned_concept(self, concept: LearnedConcept):
        """Sauvegarde un concept appris"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO learned_concepts 
                (concept_id, user_id, concept_name, description, examples, learned_date, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                concept.concept_id,
                concept.user_id,
                concept.concept_name,
                concept.description,
                json.dumps(concept.examples),
                concept.learned_date.isoformat(),
                concept.confidence_level
            ))
            
            conn.commit()
            conn.close()
    
    async def retrieve_memories(
        self,
        user_id: str,
        query: str = "",
        memory_type: str = "all",
        limit: int = 10
    ) -> List[Union[ShortTermMemory, LongTermMemory]]:
        """Récupère les mémoires pertinentes"""
        
        memories = []
        
        # Mémoire à court terme
        if memory_type in ["all", "short_term"] and user_id in self.short_term_memory:
            for memory in list(self.short_term_memory[user_id]):
                if query.lower() in memory.content.lower():
                    memories.append(memory)
        
        # Mémoire à long terme
        if memory_type in ["all", "long_term"] and user_id in self.long_term_memory:
            for memory in self.long_term_memory[user_id].values():
                if query.lower() in memory.content.lower():
                    memories.append(memory)
        
        # Trier par importance et date
        memories.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        
        return memories[:limit]
    
    async def get_user_name(self, user_id: str) -> Optional[str]:
        """Récupère le nom de l'utilisateur"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id].name
        return None
    
    async def get_learned_concepts(self, user_id: str) -> List[LearnedConcept]:
        """Récupère les concepts appris par l'utilisateur"""
        if user_id in self.learned_concepts:
            return list(self.learned_concepts[user_id].values())
        return []
    
    # Gestion TDAH
    async def add_task(self, user_id: str, task: Dict[str, Any]):
        """Ajoute une tâche pour la gestion TDAH"""
        if user_id not in self.user_profiles:
            await self.create_user_profile(user_id, "Utilisateur")
        
        if not self.user_profiles[user_id].tdah_management:
            self.user_profiles[user_id].tdah_management = TDAHManagement(user_id=user_id)
        
        self.user_profiles[user_id].tdah_management.current_tasks.append(task)
        await self._save_user_profile(self.user_profiles[user_id])
    
    async def get_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """Récupère les tâches de l'utilisateur"""
        if (user_id in self.user_profiles and 
            self.user_profiles[user_id].tdah_management):
            return self.user_profiles[user_id].tdah_management.current_tasks
        return []
    
    async def add_time_block(self, user_id: str, time_block: Dict[str, Any]):
        """Ajoute un bloc de temps pour la gestion TDAH"""
        if user_id not in self.user_profiles:
            await self.create_user_profile(user_id, "Utilisateur")
        
        if not self.user_profiles[user_id].tdah_management:
            self.user_profiles[user_id].tdah_management = TDAHManagement(user_id=user_id)
        
        self.user_profiles[user_id].tdah_management.time_blocks.append(time_block)
        await self._save_user_profile(self.user_profiles[user_id])
    
    async def get_time_blocks(self, user_id: str) -> List[Dict[str, Any]]:
        """Récupère les blocs de temps de l'utilisateur"""
        if (user_id in self.user_profiles and 
            self.user_profiles[user_id].tdah_management):
            return self.user_profiles[user_id].tdah_management.time_blocks
        return []
    
    def _start_consolidation_process(self):
        """Démarre le processus de consolidation"""
        self.consolidation_thread = threading.Thread(target=self._consolidation_worker, daemon=True)
        self.consolidation_thread.start()
    
    def _consolidation_worker(self):
        """Worker pour la consolidation des mémoires"""
        while True:
            try:
                asyncio.run(self._perform_consolidation())
                time.sleep(300)  # Toutes les 5 minutes
            except Exception as e:
                logger.error(f"Erreur dans la consolidation: {e}")
                time.sleep(60)
    
    async def _perform_consolidation(self):
        """Effectue la consolidation des mémoires"""
        for user_id in self.short_term_memory:
            await self._consolidate_user_memories(user_id)
    
    async def _consolidate_user_memories(self, user_id: str):
        """Consolide les mémoires d'un utilisateur"""
        if user_id not in self.short_term_memory:
            return
        
        current_time = datetime.now()
        memories_to_consolidate = []
        
        # Identifier les mémoires à consolider (plus de 1 heure)
        for memory in list(self.short_term_memory[user_id]):
            if (current_time - memory.timestamp).total_seconds() > 3600:  # 1 heure
                memories_to_consolidate.append(memory)
        
        # Consolider les mémoires importantes
        for memory in memories_to_consolidate:
            if memory.importance > 0.6:  # Seuil d'importance
                await self.store_long_term_memory(
                    user_id=user_id,
                    content=memory.content,
                    importance=memory.importance,
                    emotional_state=memory.emotional_state,
                    context=memory.context,
                    memory_type=memory.memory_type
                )
            
            # Retirer de la mémoire à court terme
            self.short_term_memory[user_id].remove(memory)
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Retourne les statistiques de mémoire pour un utilisateur"""
        short_term_count = len(self.short_term_memory.get(user_id, []))
        long_term_count = len(self.long_term_memory.get(user_id, {}))
        learned_concepts_count = len(self.learned_concepts.get(user_id, {}))
        
        return {
            "short_term_memory_count": short_term_count,
            "long_term_memory_count": long_term_count,
            "learned_concepts_count": learned_concepts_count,
            "total_memories": short_term_count + long_term_count
        }


class VectorIndex:
    """Index vectoriel pour recherche sémantique"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.memory_ids = []
        self._lock = threading.Lock()
    
    def add_vectors(self, vectors: np.ndarray, memory_ids: List[str]):
        """Ajoute des vecteurs à l'index"""
        with self._lock:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            self.memory_ids.extend(memory_ids)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Recherche les k vecteurs les plus similaires"""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.memory_ids):
                    results.append((self.memory_ids[idx], float(score)))
            
            return results 