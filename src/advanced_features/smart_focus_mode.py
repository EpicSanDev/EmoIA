"""
Smart Focus Mode - Mode de concentration intelligente pour EmoIA v3.0
Fonctionnalité #1 de la roadmap - Optimise la concentration et la productivité
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class FocusLevel(Enum):
    """Niveaux de concentration"""
    LIGHT = "light"          # Concentration légère (notifications réduites)
    MEDIUM = "medium"        # Concentration moyenne (musique focus, blocage partiel)
    DEEP = "deep"           # Concentration profonde (blocage total, environnement optimal)
    FLOW = "flow"           # État de flow (adaptation dynamique maximale)

class DistractionType(Enum):
    """Types de distractions"""
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    NOTIFICATIONS = "notifications"
    AUDIO = "audio"
    VISUAL = "visual"
    INTERRUPTIONS = "interruptions"

@dataclass
class FocusSession:
    """Session de concentration"""
    session_id: str
    user_id: str
    start_time: datetime
    planned_duration: int  # minutes
    focus_level: FocusLevel
    target_task: str
    blocked_distractions: List[DistractionType] = field(default_factory=list)
    background_music: Optional[str] = None
    break_intervals: List[int] = field(default_factory=lambda: [25, 50, 75])  # Pomodoro
    current_productivity_score: float = 0.0
    interruption_count: int = 0
    flow_state_detected: bool = False
    end_time: Optional[datetime] = None
    actual_duration: Optional[int] = None
    completion_rate: float = 0.0

@dataclass
class UserFocusProfile:
    """Profil de concentration personnalisé"""
    user_id: str
    optimal_focus_times: List[Tuple[int, int]] = field(default_factory=list)  # (hour, hour)
    preferred_session_duration: int = 45  # minutes
    preferred_break_duration: int = 10   # minutes
    most_effective_music: Optional[str] = None
    biggest_distractions: List[DistractionType] = field(default_factory=list)
    average_productivity_score: float = 0.0
    flow_state_triggers: List[str] = field(default_factory=list)
    energy_peaks: List[int] = field(default_factory=list)  # hours of day
    concentration_history: List[Dict[str, Any]] = field(default_factory=list)

class SmartFocusMode:
    """Mode de concentration intelligente avec IA adaptative"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Sessions actives
        self.active_sessions: Dict[str, FocusSession] = {}
        
        # Profils utilisateurs
        self.user_profiles: Dict[str, UserFocusProfile] = {}
        
        # Données en temps réel
        self.distraction_monitors: Dict[str, Any] = {}
        self.productivity_metrics: Dict[str, float] = {}
        
        # Intelligence adaptative
        self.ai_recommendations: Dict[str, List[str]] = {}
        self.learning_data: Dict[str, List[Any]] = {}
        
        # Configuration par défaut
        self.default_focus_music = [
            "binaural_beats_40hz",
            "classical_baroque",
            "ambient_nature",
            "white_noise",
            "brown_noise"
        ]
        
        self.distraction_keywords = {
            DistractionType.SOCIAL_MEDIA: ["facebook", "twitter", "instagram", "tiktok", "snapchat"],
            DistractionType.NEWS: ["news", "actualités", "breaking", "urgent"],
            DistractionType.NOTIFICATIONS: ["notification", "message", "email", "alert"]
        }
        
        logger.info("SmartFocusMode initialisé")
    
    async def start_focus_session(
        self,
        user_id: str,
        duration: int,
        task_description: str,
        focus_level: FocusLevel = FocusLevel.MEDIUM,
                 custom_settings: Dict[str, Any] = None
    ) -> str:
        """Démarre une session de concentration intelligente"""
        
        try:
            session_id = f"focus_{user_id}_{int(time.time())}"
            
            # Charger ou créer le profil utilisateur
            profile = await self._get_user_profile(user_id)
            
            # Optimiser les paramètres selon l'IA
            optimized_settings = await self._optimize_focus_settings(
                user_id, duration, task_description, focus_level, profile
            )
            
            # Créer la session
            session = FocusSession(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now(),
                planned_duration=duration,
                focus_level=focus_level,
                target_task=task_description,
                blocked_distractions=optimized_settings.get("blocked_distractions", []),
                background_music=optimized_settings.get("background_music"),
                break_intervals=optimized_settings.get("break_intervals", [25, 50, 75])
            )
            
            self.active_sessions[session_id] = session
            
            # Appliquer les optimisations
            await self._apply_focus_environment(session)
            
            # Démarrer le monitoring
            asyncio.create_task(self._monitor_focus_session(session_id))
            
            logger.info(f"Session focus démarrée: {session_id} pour {user_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Erreur démarrage session focus: {e}")
            raise
    
    async def _get_user_profile(self, user_id: str) -> UserFocusProfile:
        """Récupère ou crée le profil de concentration utilisateur"""
        
        if user_id not in self.user_profiles:
            # Créer un nouveau profil avec des données par défaut intelligentes
            profile = UserFocusProfile(user_id=user_id)
            
            # Analyser l'historique si disponible
            await self._analyze_user_patterns(profile)
            
            self.user_profiles[user_id] = profile
        
        return self.user_profiles[user_id]
    
    async def _optimize_focus_settings(
        self,
        user_id: str,
        duration: int,
        task: str,
        level: FocusLevel,
        profile: UserFocusProfile
    ) -> Dict[str, Any]:
        """Optimise les paramètres de concentration avec IA"""
        
        settings = {}
        
        # Sélection de la musique optimale
        if profile.most_effective_music:
            settings["background_music"] = profile.most_effective_music
        else:
            # IA: Recommandation basée sur le type de tâche
            music_type = await self._recommend_music_for_task(task, level)
            settings["background_music"] = music_type
        
        # Blocage des distractions
        blocked = []
        for distraction in DistractionType:
            if distraction in profile.biggest_distractions:
                blocked.append(distraction)
            elif level in [FocusLevel.DEEP, FocusLevel.FLOW]:
                blocked.append(distraction)
        
        settings["blocked_distractions"] = blocked
        
        # Intervalles de pause optimisés
        if duration <= 30:
            settings["break_intervals"] = []  # Pas de pause pour sessions courtes
        elif duration <= 60:
            settings["break_intervals"] = [duration // 2]  # Une pause au milieu
        else:
            # Pomodoro adaptatif basé sur l'historique
            optimal_interval = profile.preferred_session_duration
            intervals = []
            current = optimal_interval
            while current < duration - 10:
                intervals.append(current)
                current += optimal_interval + profile.preferred_break_duration
            settings["break_intervals"] = intervals
        
        return settings
    
    async def _recommend_music_for_task(self, task: str, level: FocusLevel) -> str:
        """Recommande le type de musique optimal pour la tâche"""
        
        task_lower = task.lower()
        
        # Analyse du type de tâche
        if any(word in task_lower for word in ["code", "programming", "développement", "debug"]):
            # Tâches de programmation
            if level == FocusLevel.FLOW:
                return "binaural_beats_40hz"
            else:
                return "ambient_electronic"
        
        elif any(word in task_lower for word in ["écriture", "rédaction", "write", "article"]):
            # Tâches d'écriture
            return "classical_baroque"
        
        elif any(word in task_lower for word in ["calcul", "math", "analyse", "data"]):
            # Tâches analytiques
            return "binaural_beats_40hz"
        
        elif any(word in task_lower for word in ["créatif", "design", "art", "brainstorm"]):
            # Tâches créatives
            return "ambient_nature"
        
        else:
            # Tâche générale
            return "white_noise"
    
    async def _apply_focus_environment(self, session: FocusSession):
        """Applique l'environnement de concentration optimal"""
        
        try:
            # Blocage des distractions
            for distraction in session.blocked_distractions:
                await self._block_distraction(distraction, session.user_id)
            
            # Lancement de la musique de fond
            if session.background_music:
                await self._start_background_music(session.background_music, session.user_id)
            
            # Configuration des notifications
            await self._configure_notifications(session)
            
            # Optimisation de l'affichage
            await self._optimize_visual_environment(session)
            
            logger.info(f"Environnement focus appliqué pour session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Erreur application environnement focus: {e}")
    
    async def _block_distraction(self, distraction: DistractionType, user_id: str):
        """Bloque un type de distraction spécifique"""
        
        # Ici, dans une implémentation complète, on interagirait avec:
        # - Le système d'exploitation pour bloquer des sites/apps
        # - Les APIs de notification pour les réduire
        # - Les services de messagerie pour les mettre en pause
        
        logger.info(f"Blocage de {distraction.value} pour utilisateur {user_id}")
        
        # Simulation du blocage
        if user_id not in self.distraction_monitors:
            self.distraction_monitors[user_id] = {}
        
        self.distraction_monitors[user_id][distraction.value] = {
            "blocked": True,
            "start_time": datetime.now(),
            "blocked_attempts": 0
        }
    
    async def _start_background_music(self, music_type: str, user_id: str):
        """Démarre la musique de fond optimale"""
        
        # Ici, dans une implémentation complète, on interagirait avec:
        # - Spotify API pour lancer des playlists spécifiques
        # - Services de streaming de musique focus
        # - Générateurs de bruits binauraux
        
        logger.info(f"Démarrage musique {music_type} pour utilisateur {user_id}")
        
        # Configuration audio simulée
        audio_config = {
            "type": music_type,
            "volume": 0.6,  # Volume optimal pour la concentration
            "user_id": user_id,
            "started_at": datetime.now()
        }
        
        return audio_config
    
    async def _configure_notifications(self, session: FocusSession):
        """Configure les notifications pendant la session"""
        
        if session.focus_level in [FocusLevel.DEEP, FocusLevel.FLOW]:
            # Mode silencieux total sauf urgences
            notification_config = {
                "mode": "emergency_only",
                "allowed_contacts": [],  # Seulement contacts d'urgence
                "keywords": ["urgence", "emergency", "urgent"]
            }
        elif session.focus_level == FocusLevel.MEDIUM:
            # Notifications importantes seulement
            notification_config = {
                "mode": "important_only",
                "delayed_notifications": True,
                "batch_delivery": True  # Grouper les notifications
            }
        else:
            # Mode léger - réduction simple
            notification_config = {
                "mode": "reduced",
                "delay_seconds": 300  # 5 minutes de délai
            }
        
        logger.info(f"Notifications configurées: {notification_config['mode']}")
        return notification_config
    
    async def _optimize_visual_environment(self, session: FocusSession):
        """Optimise l'environnement visuel pour la concentration"""
        
        visual_config = {}
        
        if session.focus_level == FocusLevel.FLOW:
            # Interface minimaliste maximale
            visual_config = {
                "ui_mode": "minimal",
                "hide_distracting_elements": True,
                "focus_overlay": True,
                "color_scheme": "high_contrast"
            }
        elif session.focus_level == FocusLevel.DEEP:
            # Mode sombre concentré
            visual_config = {
                "ui_mode": "dark_focus",
                "reduce_visual_clutter": True,
                "highlight_current_task": True
            }
        else:
            # Optimisations légères
            visual_config = {
                "ui_mode": "clean",
                "subtle_focus_indicators": True
            }
        
        logger.info(f"Environnement visuel optimisé: {visual_config.get('ui_mode')}")
        return visual_config
    
    async def _monitor_focus_session(self, session_id: str):
        """Monitoring continu de la session de concentration"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            while session_id in self.active_sessions and not session.end_time:
                # Calcul du score de productivité
                productivity_score = await self._calculate_productivity_score(session)
                session.current_productivity_score = productivity_score
                
                # Détection de l'état de flow
                flow_detected = await self._detect_flow_state(session)
                if flow_detected and not session.flow_state_detected:
                    session.flow_state_detected = True
                    await self._optimize_for_flow_state(session)
                
                # Vérification des pauses programmées
                await self._check_break_intervals(session)
                
                # Adaptation dynamique
                await self._dynamic_adaptation(session)
                
                # Attendre avant la prochaine vérification
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
        except Exception as e:
            logger.error(f"Erreur monitoring session {session_id}: {e}")
    
    async def _calculate_productivity_score(self, session: FocusSession) -> float:
        """Calcule le score de productivité en temps réel"""
        
        try:
            score = 0.0
            
            # Temps écoulé vs planifié
            elapsed = (datetime.now() - session.start_time).total_seconds() / 60
            time_factor = min(1.0, elapsed / session.planned_duration)
            
            # Réduction pour interruptions
            interruption_penalty = session.interruption_count * 0.1
            
            # Bonus pour état de flow
            flow_bonus = 0.2 if session.flow_state_detected else 0.0
            
            # Score basique basé sur la continuité
            base_score = 0.8 if elapsed > 10 else elapsed / 10  # Score de base
            
            score = max(0.0, min(1.0, base_score + flow_bonus - interruption_penalty))
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur calcul productivité: {e}")
            return 0.5
    
    async def _detect_flow_state(self, session: FocusSession) -> bool:
        """Détecte si l'utilisateur est en état de flow"""
        
        # Critères de détection du flow:
        # 1. Session longue sans interruption (>20 minutes)
        # 2. Score de productivité élevé et stable
        # 3. Pas de tentatives d'accès aux distractions bloquées
        
        elapsed = (datetime.now() - session.start_time).total_seconds() / 60
        
        if elapsed < 20:  # Minimum 20 minutes pour le flow
            return False
        
        if session.current_productivity_score < 0.8:
            return False
        
        if session.interruption_count > 2:  # Trop d'interruptions
            return False
        
        # Vérifier les tentatives de distraction
        user_distractions = self.distraction_monitors.get(session.user_id, {})
        total_blocked_attempts = sum(
            monitor.get("blocked_attempts", 0) 
            for monitor in user_distractions.values()
        )
        
        if total_blocked_attempts > 3:  # Trop de tentatives de distraction
            return False
        
        return True
    
    async def _optimize_for_flow_state(self, session: FocusSession):
        """Optimise l'environnement pour maintenir l'état de flow"""
        
        logger.info(f"État de flow détecté pour session {session.session_id}")
        
        # Adaptations pour maintenir le flow:
        # 1. Supprimer toutes les interruptions programmées
        # 2. Ajuster la musique si nécessaire
        # 3. Bloquer absolument toutes les distractions
        # 4. Prolonger automatiquement la session si proche de la fin
        
        # Suppression des pauses restantes
        current_time = (datetime.now() - session.start_time).total_seconds() / 60
        session.break_intervals = [
            interval for interval in session.break_intervals 
            if interval <= current_time
        ]
        
        # Blocage renforcé
        for distraction in DistractionType:
            if distraction not in session.blocked_distractions:
                session.blocked_distractions.append(distraction)
                await self._block_distraction(distraction, session.user_id)
        
        # Notification à l'utilisateur (non intrusive)
        await self._send_flow_state_notification(session)
    
    async def _send_flow_state_notification(self, session: FocusSession):
        """Envoie une notification discrète sur l'état de flow"""
        
        notification = {
            "type": "flow_state_detected",
            "message": "🌊 État de flow détecté ! Continuez votre excellent travail.",
            "session_id": session.session_id,
            "suggestions": [
                "Votre session sera automatiquement prolongée si nécessaire",
                "Toutes les distractions sont maintenant bloquées",
                "Profitez de cette concentration optimale"
            ]
        }
        
        logger.info(f"Notification flow envoyée pour {session.user_id}")
        return notification
    
    async def _check_break_intervals(self, session: FocusSession):
        """Vérifie et propose des pauses aux moments optimaux"""
        
        elapsed = (datetime.now() - session.start_time).total_seconds() / 60
        
        for break_time in session.break_intervals:
            if abs(elapsed - break_time) < 0.5:  # ±30 secondes
                # Il est temps pour une pause
                if not session.flow_state_detected:  # Pas de pause pendant le flow
                    await self._suggest_break(session, break_time)
                break
    
    async def _suggest_break(self, session: FocusSession, break_time: int):
        """Suggère une pause optimale"""
        
        profile = self.user_profiles.get(session.user_id)
        break_duration = profile.preferred_break_duration if profile else 10
        
        break_suggestion = {
            "type": "break_suggestion",
            "message": f"💡 Pause recommandée après {break_time} minutes de concentration",
            "duration": break_duration,
            "suggestions": [
                "Levez-vous et marchez",
                "Hydratez-vous",
                "Regardez au loin pour reposer vos yeux",
                "Faites quelques étirements"
            ],
            "auto_resume": True
        }
        
        logger.info(f"Pause suggérée pour session {session.session_id}")
        return break_suggestion
    
    async def _dynamic_adaptation(self, session: FocusSession):
        """Adaptation dynamique pendant la session"""
        
        # Adaptation basée sur les métriques en temps réel
        if session.current_productivity_score < 0.5:
            # Productivité faible - ajustements
            await self._adjust_for_low_productivity(session)
        
        elif session.current_productivity_score > 0.9:
            # Excellente productivité - maintenir
            await self._maintain_high_productivity(session)
    
    async def _adjust_for_low_productivity(self, session: FocusSession):
        """Ajustements pour améliorer la productivité"""
        
        adjustments = []
        
        # Changer la musique si elle joue depuis longtemps
        if session.background_music:
            adjustments.append("musique_adaptée")
        
        # Proposer une micro-pause si pas de pause récente
        elapsed = (datetime.now() - session.start_time).total_seconds() / 60
        if elapsed > 30:  # Plus de 30 minutes
            adjustments.append("micro_pause")
        
        # Réduire les distractions encore plus
        adjustments.append("blocage_renforcé")
        
        logger.info(f"Ajustements productivité pour {session.session_id}: {adjustments}")
        return adjustments
    
    async def _maintain_high_productivity(self, session: FocusSession):
        """Maintient une productivité élevée"""
        
        # Ne pas perturber - juste s'assurer que tout est optimal
        maintenance_actions = [
            "maintenir_environnement",
            "surveiller_fatigue",
            "préparer_fin_session"
        ]
        
        logger.debug(f"Maintenance productivité élevée: {session.session_id}")
        return maintenance_actions
    
    async def end_focus_session(self, session_id: str, completion_rate: float = 1.0) -> Dict[str, Any]:
        """Termine une session de concentration"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} non trouvée")
        
        try:
            # Finaliser la session
            session.end_time = datetime.now()
            session.actual_duration = int((session.end_time - session.start_time).total_seconds() / 60)
            session.completion_rate = completion_rate
            
            # Calculer les métriques finales
            final_metrics = await self._calculate_final_metrics(session)
            
            # Nettoyer l'environnement
            await self._cleanup_focus_environment(session)
            
            # Apprentissage pour l'IA
            await self._learn_from_session(session)
            
            # Sauvegarder dans l'historique
            await self._save_session_to_history(session)
            
            # Supprimer de la liste active
            del self.active_sessions[session_id]
            
            logger.info(f"Session focus terminée: {session_id}")
            
            return {
                "session_id": session_id,
                "duration": session.actual_duration,
                "completion_rate": completion_rate,
                "final_productivity_score": session.current_productivity_score,
                "flow_state_achieved": session.flow_state_detected,
                "metrics": final_metrics,
                "recommendations": await self._generate_recommendations(session)
            }
            
        except Exception as e:
            logger.error(f"Erreur fin session {session_id}: {e}")
            raise
    
    async def _calculate_final_metrics(self, session: FocusSession) -> Dict[str, Any]:
        """Calcule les métriques finales de la session"""
        
                     planned_ratio = session.actual_duration / session.planned_duration if session.planned_duration > 0 else 1.0
             return {
                 "planned_vs_actual_duration": planned_ratio,
            "average_productivity": session.current_productivity_score,
                             "interruption_rate": session.interruption_count / max(1, (session.actual_duration or 1) / 60),
            "flow_state_duration": 0,  # À calculer si on trackait le flow en continu
            "distraction_resistance": self._calculate_distraction_resistance(session)
        }
    
    def _calculate_distraction_resistance(self, session: FocusSession) -> float:
        """Calcule la résistance aux distractions"""
        
        user_distractions = self.distraction_monitors.get(session.user_id, {})
        total_attempts = sum(
            monitor.get("blocked_attempts", 0)
            for monitor in user_distractions.values()
        )
        
        # Score inversé : moins de tentatives = meilleure résistance
        if total_attempts == 0:
            return 1.0
        elif total_attempts <= 2:
            return 0.8
        elif total_attempts <= 5:
            return 0.6
        else:
            return 0.3
    
    async def _cleanup_focus_environment(self, session: FocusSession):
        """Nettoie l'environnement de concentration"""
        
        try:
            # Réactiver les notifications
            await self._restore_notifications(session.user_id)
            
            # Arrêter la musique de fond
            if session.background_music:
                await self._stop_background_music(session.user_id)
            
            # Débloquer les distractions
            for distraction in session.blocked_distractions:
                await self._unblock_distraction(distraction, session.user_id)
            
            # Restaurer l'interface normale
            await self._restore_visual_environment(session.user_id)
            
            logger.info(f"Environnement nettoyé pour session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage environnement: {e}")
    
    async def _restore_notifications(self, user_id: str):
        """Restaure les notifications normales"""
        logger.info(f"Notifications restaurées pour {user_id}")
    
    async def _stop_background_music(self, user_id: str):
        """Arrête la musique de fond"""
        logger.info(f"Musique arrêtée pour {user_id}")
    
    async def _unblock_distraction(self, distraction: DistractionType, user_id: str):
        """Débloque un type de distraction"""
        if user_id in self.distraction_monitors:
            if distraction.value in self.distraction_monitors[user_id]:
                del self.distraction_monitors[user_id][distraction.value]
        logger.info(f"Distraction {distraction.value} débloquée pour {user_id}")
    
    async def _restore_visual_environment(self, user_id: str):
        """Restaure l'environnement visuel normal"""
        logger.info(f"Interface normale restaurée pour {user_id}")
    
    async def _learn_from_session(self, session: FocusSession):
        """Apprentissage IA à partir de la session"""
        
        profile = self.user_profiles[session.user_id]
        
        # Mise à jour du profil basée sur les résultats
        if session.current_productivity_score > 0.8:
            # Session réussie - apprendre les bonnes pratiques
            if session.background_music and session.background_music not in profile.flow_state_triggers:
                profile.most_effective_music = session.background_music
            
            # Mémoriser l'heure comme optimal
            hour = session.start_time.hour
            if hour not in profile.energy_peaks:
                profile.energy_peaks.append(hour)
        
        # Ajuster la durée préférée
        if session.completion_rate > 0.9:
            # Session complétée avec succès
                         if session.actual_duration and session.actual_duration > profile.preferred_session_duration:
                # L'utilisateur peut gérer des sessions plus longues
                                 profile.preferred_session_duration = min(
                     profile.preferred_session_duration + 5,
                     session.actual_duration or profile.preferred_session_duration
                 )
        
        logger.info(f"Apprentissage mis à jour pour {session.user_id}")
    
    async def _save_session_to_history(self, session: FocusSession):
        """Sauvegarde la session dans l'historique"""
        
        profile = self.user_profiles[session.user_id]
        
        session_data = {
            "date": session.start_time.isoformat(),
            "duration": session.actual_duration,
            "planned_duration": session.planned_duration,
            "productivity_score": session.current_productivity_score,
            "completion_rate": session.completion_rate,
            "focus_level": session.focus_level.value,
            "task": session.target_task,
            "flow_achieved": session.flow_state_detected,
            "interruptions": session.interruption_count
        }
        
        profile.concentration_history.append(session_data)
        
        # Garder seulement les 100 dernières sessions
        if len(profile.concentration_history) > 100:
            profile.concentration_history = profile.concentration_history[-100:]
        
        # Mettre à jour la moyenne
        recent_scores = [s["productivity_score"] for s in profile.concentration_history[-10:]]
        profile.average_productivity_score = sum(recent_scores) / len(recent_scores)
    
    async def _generate_recommendations(self, session: FocusSession) -> List[str]:
        """Génère des recommandations personnalisées"""
        
        recommendations = []
        
        if session.current_productivity_score < 0.6:
            recommendations.extend([
                "Essayez des sessions plus courtes (25-30 minutes)",
                "Changez d'environnement de travail",
                "Testez différents types de musique de fond"
            ])
        
        if session.interruption_count > 3:
            recommendations.extend([
                "Activez le mode 'Ne pas déranger' sur vos appareils",
                "Informez votre entourage de vos créneaux de concentration",
                "Utilisez des créneaux de concentration plus courts mais plus intenses"
            ])
        
        if session.flow_state_detected:
            recommendations.extend([
                "Excellent ! Reproduisez ces conditions : même heure, même musique, même environnement",
                "Notez ce qui a déclenché cet état de flow pour le reproduire"
            ])
        
        return recommendations
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Retourne les analytics de concentration de l'utilisateur"""
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {"error": "Profil utilisateur non trouvé"}
        
        # Session active
        active_session = None
        for session in self.active_sessions.values():
            if session.user_id == user_id:
                active_session = {
                    "session_id": session.session_id,
                    "elapsed_minutes": int((datetime.now() - session.start_time).total_seconds() / 60),
                    "planned_duration": session.planned_duration,
                    "current_productivity": session.current_productivity_score,
                    "flow_state": session.flow_state_detected
                }
                break
        
        # Statistiques historiques
        history = profile.concentration_history
        if history:
            total_sessions = len(history)
            total_minutes = sum(s["duration"] for s in history)
            avg_productivity = sum(s["productivity_score"] for s in history) / total_sessions
            flow_sessions = sum(1 for s in history if s["flow_achieved"])
            
            # Tendance récente (7 dernières sessions)
            recent = history[-7:] if len(history) >= 7 else history
            recent_productivity = sum(s["productivity_score"] for s in recent) / len(recent)
            productivity_trend = "improving" if recent_productivity > avg_productivity else "stable"
        else:
            total_sessions = 0
            total_minutes = 0
            avg_productivity = 0
            flow_sessions = 0
            productivity_trend = "new_user"
        
        return {
            "active_session": active_session,
            "statistics": {
                "total_sessions": total_sessions,
                "total_focus_time_hours": round(total_minutes / 60, 1),
                "average_productivity_score": round(avg_productivity, 2),
                "flow_state_sessions": flow_sessions,
                "productivity_trend": productivity_trend
            },
            "profile": {
                "preferred_session_duration": profile.preferred_session_duration,
                "optimal_focus_times": profile.optimal_focus_times,
                "most_effective_music": profile.most_effective_music,
                "energy_peaks": profile.energy_peaks
            },
            "recommendations": await self._get_personalized_recommendations(profile)
        }
    
    async def _get_personalized_recommendations(self, profile: UserFocusProfile) -> List[str]:
        """Génère des recommandations personnalisées basées sur l'historique"""
        
        recommendations = []
        
        if profile.average_productivity_score < 0.7:
            recommendations.append("💡 Essayez le mode de concentration 'Deep' pour une meilleure productivité")
        
        if len(profile.energy_peaks) > 0:
            peak_hours = sorted(profile.energy_peaks)
            recommendations.append(f"⏰ Vos meilleures heures : {peak_hours[0]}h-{peak_hours[-1]}h")
        
        if not profile.most_effective_music:
            recommendations.append("🎵 Expérimentez différents types de musique pour trouver votre optimal")
        
        return recommendations
    
    async def _analyze_user_patterns(self, profile: UserFocusProfile):
        """Analyse les patterns de l'utilisateur pour personnaliser l'expérience"""
        
        # Dans une implémentation complète, ceci analyserait:
        # - L'historique d'utilisation du système
        # - Les patterns de productivité
        # - Les préférences comportementales
        # - Les données de wearables si disponibles
        
        logger.info(f"Analyse des patterns pour {profile.user_id}")
        
        # Valeurs par défaut intelligentes
        profile.optimal_focus_times = [(9, 11), (14, 16)]  # Matinée et après-midi
        profile.preferred_session_duration = 45
        profile.preferred_break_duration = 10