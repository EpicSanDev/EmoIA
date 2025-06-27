"""
Système de Base de Connaissance sous forme de Graphe pour EmoIA
Gestion des connaissances avec visualisation et recherche sémantique
"""

import asyncio
import logging
import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import networkx as nx
from collections import defaultdict, deque

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """Noeud de connaissance dans le graphe"""
    id: str
    name: str
    type: str  # concept, fact, relation, person, event, etc.
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    created_at: datetime = None
    updated_at: datetime = None
    confidence: float = 1.0
    user_id: str = ""
    tags: List[str] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le noeud en dictionnaire"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "confidence": self.confidence,
            "user_id": self.user_id,
            "tags": self.tags,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        """Crée un noeud depuis un dictionnaire"""
        node = cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            user_id=data.get("user_id", ""),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0)
        )
        
        if data.get("embedding"):
            node.embedding = np.array(data["embedding"])
        if data.get("created_at"):
            node.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            node.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("last_accessed"):
            node.last_accessed = datetime.fromisoformat(data["last_accessed"])
        
        return node

@dataclass
class KnowledgeEdge:
    """Arête/Relation dans le graphe de connaissance"""
    id: str
    source_id: str
    target_id: str
    relationship: str  # "is_related_to", "is_a", "causes", "depends_on", etc.
    strength: float = 1.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    user_id: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'arête en dictionnaire"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "strength": self.strength,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_id": self.user_id,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEdge":
        """Crée une arête depuis un dictionnaire"""
        edge = cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship=data["relationship"],
            strength=data.get("strength", 1.0),
            metadata=data.get("metadata", {}),
            user_id=data.get("user_id", ""),
            confidence=data.get("confidence", 1.0)
        )
        
        if data.get("created_at"):
            edge.created_at = datetime.fromisoformat(data["created_at"])
        
        return edge

class KnowledgeGraphSystem:
    """Système de base de connaissance sous forme de graphe"""
    
    def __init__(self, database_path: str = "data/knowledge_graph.db"):
        self.database_path = Path(database_path)
        self.nodes = {}  # node_id -> KnowledgeNode
        self.edges = {}  # edge_id -> KnowledgeEdge
        self.user_graphs = defaultdict(lambda: {"nodes": set(), "edges": set()})  # user_id -> graph info
        
        # Modèle d'embeddings pour la recherche sémantique
        self.embedding_model = None
        
        # Index vectoriel pour recherche rapide
        self.vector_index = None
        self.node_id_to_index = {}
        self.index_to_node_id = {}
        
        # Graphe NetworkX pour analyses topologiques
        self.nx_graph = nx.MultiDiGraph()
        
        # Cache pour les recommandations
        self.recommendation_cache = {}
    
    async def initialize(self):
        """Initialise le système de graphe de connaissance"""
        try:
            logger.info("Initialisation du système de graphe de connaissance...")
            
            # Créer le répertoire si nécessaire
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Charger le modèle d'embeddings
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Initialiser la base de données
            await self._init_database()
            
            # Charger les données existantes
            await self._load_existing_data()
            
            # Initialiser l'index vectoriel
            await self._init_vector_index()
            
            logger.info(f"Système de graphe initialisé - {len(self.nodes)} noeuds, {len(self.edges)} arêtes")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du graphe: {e}")
            raise
    
    async def _init_database(self):
        """Initialise la base de données SQLite"""
        with sqlite3.connect(self.database_path) as conn:
            # Table des noeuds
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    created_at TEXT,
                    updated_at TEXT,
                    confidence REAL DEFAULT 1.0,
                    user_id TEXT,
                    tags TEXT,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            
            # Table des arêtes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TEXT,
                    user_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (source_id) REFERENCES knowledge_nodes (id),
                    FOREIGN KEY (target_id) REFERENCES knowledge_nodes (id)
                )
            """)
            
            # Index pour optimiser les requêtes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_user ON knowledge_nodes(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_user ON knowledge_edges(user_id)")
            
            conn.commit()
    
    async def _load_existing_data(self):
        """Charge les données existantes depuis la base"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Charger les noeuds
                cursor = conn.execute("SELECT * FROM knowledge_nodes")
                for row in cursor.fetchall():
                    node = self._row_to_node(row)
                    self.nodes[node.id] = node
                    self.user_graphs[node.user_id]["nodes"].add(node.id)
                    
                    # Ajouter au graphe NetworkX
                    self.nx_graph.add_node(node.id, **node.to_dict())
                
                # Charger les arêtes
                cursor = conn.execute("SELECT * FROM knowledge_edges")
                for row in cursor.fetchall():
                    edge = self._row_to_edge(row)
                    self.edges[edge.id] = edge
                    self.user_graphs[edge.user_id]["edges"].add(edge.id)
                    
                    # Ajouter au graphe NetworkX
                    self.nx_graph.add_edge(
                        edge.source_id, 
                        edge.target_id,
                        key=edge.id,
                        **edge.to_dict()
                    )
                
                logger.info(f"Chargé {len(self.nodes)} noeuds et {len(self.edges)} arêtes")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
    
    async def _init_vector_index(self):
        """Initialise l'index vectoriel pour la recherche sémantique"""
        try:
            if not self.nodes:
                return
            
            # Dimension des embeddings (384 pour all-MiniLM-L6-v2)
            dimension = 384
            self.vector_index = faiss.IndexFlatIP(dimension)
            
            # Préparer les vecteurs
            vectors = []
            node_ids = []
            
            for node_id, node in self.nodes.items():
                if node.embedding is not None:
                    vectors.append(node.embedding)
                    node_ids.append(node_id)
            
            if vectors:
                vectors_array = np.array(vectors).astype('float32')
                faiss.normalize_L2(vectors_array)  # Normaliser pour le produit scalaire
                
                self.vector_index.add(vectors_array)
                
                # Créer les mappings
                for i, node_id in enumerate(node_ids):
                    self.node_id_to_index[node_id] = i
                    self.index_to_node_id[i] = node_id
                
                logger.info(f"Index vectoriel initialisé avec {len(vectors)} vecteurs")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'index vectoriel: {e}")
    
    def _row_to_node(self, row) -> KnowledgeNode:
        """Convertit une ligne de base en KnowledgeNode"""
        (node_id, name, node_type, content, metadata_json, embedding_blob,
         created_at, updated_at, confidence, user_id, tags_json,
         importance, access_count, last_accessed) = row
        
        # Désérialiser les données JSON
        metadata = json.loads(metadata_json) if metadata_json else {}
        tags = json.loads(tags_json) if tags_json else []
        
        # Désérialiser l'embedding
        embedding = None
        if embedding_blob:
            try:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            except Exception:
                pass
        
        node = KnowledgeNode(
            id=node_id,
            name=name,
            type=node_type,
            content=content,
            metadata=metadata,
            embedding=embedding,
            confidence=confidence,
            user_id=user_id,
            tags=tags,
            importance=importance,
            access_count=access_count
        )
        
        if created_at:
            node.created_at = datetime.fromisoformat(created_at)
        if updated_at:
            node.updated_at = datetime.fromisoformat(updated_at)
        if last_accessed:
            node.last_accessed = datetime.fromisoformat(last_accessed)
        
        return node
    
    def _row_to_edge(self, row) -> KnowledgeEdge:
        """Convertit une ligne de base en KnowledgeEdge"""
        (edge_id, source_id, target_id, relationship, strength,
         metadata_json, created_at, user_id, confidence) = row
        
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        edge = KnowledgeEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            strength=strength,
            metadata=metadata,
            user_id=user_id,
            confidence=confidence
        )
        
        if created_at:
            edge.created_at = datetime.fromisoformat(created_at)
        
        return edge
    
    async def create_node(self, user_id: str, name: str, node_type: str, 
                         content: str, metadata: Dict[str, Any] = None,
                         tags: List[str] = None, importance: float = 0.5) -> str:
        """Crée un nouveau noeud de connaissance"""
        try:
            node_id = str(uuid.uuid4())
            
            # Générer l'embedding
            embedding = self.embedding_model.encode([content])[0].astype('float32')
            
            # Créer le noeud
            node = KnowledgeNode(
                id=node_id,
                name=name,
                type=node_type,
                content=content,
                metadata=metadata or {},
                embedding=embedding,
                user_id=user_id,
                tags=tags or [],
                importance=importance
            )
            
            # Sauvegarder en base
            await self._save_node(node)
            
            # Ajouter aux structures en mémoire
            self.nodes[node_id] = node
            self.user_graphs[user_id]["nodes"].add(node_id)
            
            # Ajouter au graphe NetworkX
            self.nx_graph.add_node(node_id, **node.to_dict())
            
            # Mettre à jour l'index vectoriel
            await self._update_vector_index()
            
            logger.info(f"Nouveau noeud créé: {node_id} - {name}")
            return node_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du noeud: {e}")
            raise
    
    async def create_edge(self, user_id: str, source_id: str, target_id: str,
                         relationship: str, strength: float = 1.0,
                         metadata: Dict[str, Any] = None) -> str:
        """Crée une nouvelle arête entre deux noeuds"""
        try:
            edge_id = str(uuid.uuid4())
            
            # Vérifier que les noeuds existent
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError("Les noeuds source et/ou target n'existent pas")
            
            # Créer l'arête
            edge = KnowledgeEdge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
                strength=strength,
                metadata=metadata or {},
                user_id=user_id
            )
            
            # Sauvegarder en base
            await self._save_edge(edge)
            
            # Ajouter aux structures en mémoire
            self.edges[edge_id] = edge
            self.user_graphs[user_id]["edges"].add(edge_id)
            
            # Ajouter au graphe NetworkX
            self.nx_graph.add_edge(source_id, target_id, key=edge_id, **edge.to_dict())
            
            logger.info(f"Nouvelle arête créée: {edge_id} - {source_id} -> {target_id}")
            return edge_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'arête: {e}")
            raise
    
    async def search_nodes(self, user_id: str, query: str, node_type: Optional[str] = None,
                          limit: int = 10) -> List[Tuple[KnowledgeNode, float]]:
        """Recherche sémantique dans les noeuds"""
        try:
            # Générer l'embedding de la requête
            query_embedding = self.embedding_model.encode([query])[0].astype('float32')
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Recherche vectorielle
            if self.vector_index and self.vector_index.ntotal > 0:
                scores, indices = self.vector_index.search(query_embedding, min(limit * 2, self.vector_index.ntotal))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and idx in self.index_to_node_id:
                        node_id = self.index_to_node_id[idx]
                        node = self.nodes[node_id]
                        
                        # Filtrer par utilisateur et type
                        if (node.user_id == user_id and 
                            (node_type is None or node.type == node_type)):
                            results.append((node, float(score)))
                
                # Trier par score et limiter
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]
            
            return []
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    async def get_node_neighbors(self, node_id: str, relationship: Optional[str] = None,
                                direction: str = "both") -> List[Tuple[KnowledgeNode, KnowledgeEdge]]:
        """Récupère les voisins d'un noeud"""
        try:
            neighbors = []
            
            for edge in self.edges.values():
                is_source = edge.source_id == node_id
                is_target = edge.target_id == node_id
                
                # Vérifier la direction
                if direction == "outgoing" and not is_source:
                    continue
                elif direction == "incoming" and not is_target:
                    continue
                elif direction == "both" and not (is_source or is_target):
                    continue
                
                # Vérifier le type de relation
                if relationship and edge.relationship != relationship:
                    continue
                
                # Trouver le noeud voisin
                neighbor_id = edge.target_id if is_source else edge.source_id
                if neighbor_id in self.nodes:
                    neighbors.append((self.nodes[neighbor_id], edge))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des voisins: {e}")
            return []
    
    async def get_user_graph(self, user_id: str) -> Dict[str, Any]:
        """Récupère le graphe complet d'un utilisateur"""
        try:
            user_nodes = []
            user_edges = []
            
            # Récupérer les noeuds de l'utilisateur
            for node_id in self.user_graphs[user_id]["nodes"]:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    # Mettre à jour le compteur d'accès
                    node.access_count += 1
                    node.last_accessed = datetime.now()
                    user_nodes.append(node.to_dict())
            
            # Récupérer les arêtes de l'utilisateur
            for edge_id in self.user_graphs[user_id]["edges"]:
                if edge_id in self.edges:
                    user_edges.append(self.edges[edge_id].to_dict())
            
            return {
                "nodes": user_nodes,
                "edges": user_edges,
                "stats": {
                    "node_count": len(user_nodes),
                    "edge_count": len(user_edges),
                    "node_types": self._get_node_type_distribution(user_id),
                    "relationship_types": self._get_relationship_type_distribution(user_id)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du graphe utilisateur: {e}")
            return {"nodes": [], "edges": [], "stats": {}}
    
    async def get_node_path(self, start_node_id: str, end_node_id: str,
                           max_depth: int = 5) -> List[List[str]]:
        """Trouve les chemins entre deux noeuds"""
        try:
            if start_node_id not in self.nx_graph.nodes or end_node_id not in self.nx_graph.nodes:
                return []
            
            # Utiliser NetworkX pour trouver les chemins
            try:
                paths = list(nx.all_simple_paths(
                    self.nx_graph.to_undirected(), 
                    start_node_id, 
                    end_node_id,
                    cutoff=max_depth
                ))
                return paths[:10]  # Limiter à 10 chemins
            except nx.NetworkXNoPath:
                return []
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de chemin: {e}")
            return []
    
    async def get_recommendations(self, user_id: str, node_id: str,
                                 limit: int = 5) -> List[Tuple[KnowledgeNode, float]]:
        """Recommande des noeuds similaires ou connectés"""
        try:
            if node_id not in self.nodes:
                return []
            
            node = self.nodes[node_id]
            recommendations = []
            
            # 1. Recommandations basées sur la similarité sémantique
            if node.embedding is not None:
                similar_results = await self.search_nodes(
                    user_id, node.content, limit=limit * 2
                )
                for similar_node, score in similar_results:
                    if similar_node.id != node_id:
                        recommendations.append((similar_node, score * 0.7))
            
            # 2. Recommandations basées sur les connexions
            neighbors = await self.get_node_neighbors(node_id)
            for neighbor_node, edge in neighbors:
                if neighbor_node.user_id == user_id:
                    score = edge.strength * 0.5
                    recommendations.append((neighbor_node, score))
            
            # 3. Recommandations basées sur les tags
            for other_node in self.nodes.values():
                if (other_node.user_id == user_id and 
                    other_node.id != node_id and 
                    set(node.tags) & set(other_node.tags)):
                    common_tags = len(set(node.tags) & set(other_node.tags))
                    score = common_tags / max(len(node.tags), len(other_node.tags)) * 0.3
                    recommendations.append((other_node, score))
            
            # Trier et dédupliquer
            seen = set()
            unique_recommendations = []
            for rec_node, score in sorted(recommendations, key=lambda x: x[1], reverse=True):
                if rec_node.id not in seen:
                    unique_recommendations.append((rec_node, score))
                    seen.add(rec_node.id)
            
            return unique_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Erreur lors des recommandations: {e}")
            return []
    
    def _get_node_type_distribution(self, user_id: str) -> Dict[str, int]:
        """Calcule la distribution des types de noeuds pour un utilisateur"""
        distribution = defaultdict(int)
        for node_id in self.user_graphs[user_id]["nodes"]:
            if node_id in self.nodes:
                distribution[self.nodes[node_id].type] += 1
        return dict(distribution)
    
    def _get_relationship_type_distribution(self, user_id: str) -> Dict[str, int]:
        """Calcule la distribution des types de relations pour un utilisateur"""
        distribution = defaultdict(int)
        for edge_id in self.user_graphs[user_id]["edges"]:
            if edge_id in self.edges:
                distribution[self.edges[edge_id].relationship] += 1
        return dict(distribution)
    
    async def _save_node(self, node: KnowledgeNode):
        """Sauvegarde un noeud en base de données"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_nodes (
                        id, name, type, content, metadata, embedding,
                        created_at, updated_at, confidence, user_id, tags,
                        importance, access_count, last_accessed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.id,
                    node.name,
                    node.type,
                    node.content,
                    json.dumps(node.metadata),
                    node.embedding.tobytes() if node.embedding is not None else None,
                    node.created_at.isoformat(),
                    node.updated_at.isoformat(),
                    node.confidence,
                    node.user_id,
                    json.dumps(node.tags),
                    node.importance,
                    node.access_count,
                    node.last_accessed.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du noeud: {e}")
            raise
    
    async def _save_edge(self, edge: KnowledgeEdge):
        """Sauvegarde une arête en base de données"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_edges (
                        id, source_id, target_id, relationship, strength,
                        metadata, created_at, user_id, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relationship,
                    edge.strength,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                    edge.user_id,
                    edge.confidence
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'arête: {e}")
            raise
    
    async def _update_vector_index(self):
        """Met à jour l'index vectoriel avec les nouveaux noeuds"""
        try:
            # Reconstruire l'index complet (plus simple pour cette implémentation)
            await self._init_vector_index()
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'index: {e}")
    
    async def delete_node(self, node_id: str) -> bool:
        """Supprime un noeud et ses arêtes associées"""
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # Supprimer toutes les arêtes connectées
            edges_to_delete = []
            for edge_id, edge in self.edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edges_to_delete.append(edge_id)
            
            for edge_id in edges_to_delete:
                await self.delete_edge(edge_id)
            
            # Supprimer le noeud de la base
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("DELETE FROM knowledge_nodes WHERE id = ?", (node_id,))
                conn.commit()
            
            # Supprimer des structures en mémoire
            del self.nodes[node_id]
            self.user_graphs[node.user_id]["nodes"].discard(node_id)
            self.nx_graph.remove_node(node_id)
            
            # Mettre à jour l'index vectoriel
            await self._update_vector_index()
            
            logger.info(f"Noeud supprimé: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du noeud: {e}")
            return False
    
    async def delete_edge(self, edge_id: str) -> bool:
        """Supprime une arête"""
        try:
            if edge_id not in self.edges:
                return False
            
            edge = self.edges[edge_id]
            
            # Supprimer de la base
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("DELETE FROM knowledge_edges WHERE id = ?", (edge_id,))
                conn.commit()
            
            # Supprimer des structures en mémoire
            del self.edges[edge_id]
            self.user_graphs[edge.user_id]["edges"].discard(edge_id)
            self.nx_graph.remove_edge(edge.source_id, edge.target_id, key=edge_id)
            
            logger.info(f"Arête supprimée: {edge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'arête: {e}")
            return False
    
    async def export_user_graph(self, user_id: str, format: str = "json") -> str:
        """Exporte le graphe d'un utilisateur"""
        try:
            graph_data = await self.get_user_graph(user_id)
            
            if format == "json":
                return json.dumps(graph_data, indent=2, ensure_ascii=False)
            elif format == "gexf":
                # Export au format GEXF pour Gephi
                user_graph = nx.MultiDiGraph()
                
                for node_data in graph_data["nodes"]:
                    user_graph.add_node(node_data["id"], **node_data)
                
                for edge_data in graph_data["edges"]:
                    user_graph.add_edge(
                        edge_data["source_id"],
                        edge_data["target_id"],
                        key=edge_data["id"],
                        **edge_data
                    )
                
                return "\n".join(nx.generate_gexf(user_graph))
            
            return ""
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export: {e}")
            return ""