"""
Bot Telegram pour EmoIA
Permet à l'IA de contacter l'utilisateur via Telegram
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import os
import json

# Note: Pour utiliser ce module, installer python-telegram-bot:
# pip install python-telegram-bot

try:
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    Update = None
    Application = None
    CommandHandler = None
    MessageHandler = None
    filters = None
    ContextTypes = None

logger = logging.getLogger(__name__)


class TelegramBotManager:
    """Gestionnaire du bot Telegram pour EmoIA"""
    
    def __init__(self, token: str = None, emoia_instance=None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.emoia = emoia_instance
        self.application = None
        self.bot = None
        self.user_mappings = {}  # telegram_user_id -> emoia_user_id
        self.is_running = False
        
        if not TELEGRAM_AVAILABLE:
            logger.warning("Module python-telegram-bot non disponible. Installez-le avec: pip install python-telegram-bot")
        
        if not self.token:
            logger.warning("Token Telegram non configuré. Définissez TELEGRAM_BOT_TOKEN dans l'environnement.")
    
    async def initialize(self):
        """Initialise le bot Telegram"""
        if not TELEGRAM_AVAILABLE or not self.token:
            logger.info("Bot Telegram non disponible ou non configuré")
            return False
        
        try:
            # Créer l'application
            self.application = Application.builder().token(self.token).build()
            self.bot = self.application.bot
            
            # Ajouter les handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("register", self.register_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("tasks", self.tasks_command))
            self.application.add_handler(CommandHandler("learn", self.learn_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            logger.info("Bot Telegram initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du bot Telegram: {e}")
            return False
    
    async def start_bot(self):
        """Démarre le bot Telegram"""
        if not self.application:
            await self.initialize()
        
        if self.application:
            try:
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling()
                self.is_running = True
                logger.info("Bot Telegram démarré et en écoute")
            except Exception as e:
                logger.error(f"Erreur lors du démarrage du bot: {e}")
    
    async def stop_bot(self):
        """Arrête le bot Telegram"""
        if self.application and self.is_running:
            try:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                self.is_running = False
                logger.info("Bot Telegram arrêté")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du bot: {e}")
    
    async def send_message(self, telegram_user_id: str, message: str) -> bool:
        """Envoie un message à un utilisateur"""
        if not self.bot:
            logger.error("Bot Telegram non initialisé")
            return False
        
        try:
            await self.bot.send_message(chat_id=telegram_user_id, text=message)
            logger.info(f"Message envoyé à {telegram_user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi du message: {e}")
            return False
    
    async def send_proactive_message(self, emoia_user_id: str, message: str) -> bool:
        """Envoie un message proactif à un utilisateur EmoIA"""
        # Trouver l'ID Telegram correspondant
        telegram_id = None
        for tg_id, emo_id in self.user_mappings.items():
            if emo_id == emoia_user_id:
                telegram_id = tg_id
                break
        
        if telegram_id:
            return await self.send_message(telegram_id, f"🤖 EmoIA: {message}")
        else:
            logger.warning(f"Utilisateur {emoia_user_id} non trouvé sur Telegram")
            return False
    
    # Handlers des commandes
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour la commande /start"""
        welcome_message = """
🤖 Bonjour ! Je suis EmoIA, votre assistant IA émotionnel.

Voici ce que je peux faire pour vous :
• 🧠 Retenir et apprendre ce que vous m'enseignez
• 📝 Gérer vos tâches et votre organisation (TDAH)
• 💭 Vous aider avec la régulation émotionnelle
• 🎯 Vous contacter de manière proactive

Commandes disponibles :
/register - Vous enregistrer comme utilisateur
/help - Afficher l'aide
/status - Voir votre statut
/tasks - Gérer vos tâches
/learn - Apprendre un nouveau concept

Vous pouvez aussi simplement me parler !
        """
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour la commande /help"""
        help_message = """
🆘 Aide EmoIA

📋 Commandes disponibles :
/start - Message de bienvenue
/register <nom> - Vous enregistrer avec votre nom
/status - Voir vos statistiques
/tasks - Voir vos tâches en cours
/learn <concept> - Apprendre un nouveau concept

💬 Conversation :
Vous pouvez me parler naturellement ! Je comprends vos émotions et je peux :
• Retenir votre nom et nos conversations
• Apprendre de nouveaux concepts que vous m'enseignez
• Vous aider avec la gestion du temps et des tâches
• Vous soutenir émotionnellement

🔧 Gestion TDAH :
• Création et suivi de tâches
• Rappels et organisation
• Techniques de focus
• Régulation émotionnelle
        """
        await update.message.reply_text(help_message)
    
    async def register_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour la commande /register"""
        telegram_user_id = str(update.effective_user.id)
        telegram_username = update.effective_user.username or ""
        
        # Extraire le nom depuis les arguments
        if context.args:
            name = " ".join(context.args)
        else:
            name = update.effective_user.first_name or "Utilisateur"
        
        # Créer un ID EmoIA unique
        emoia_user_id = f"tg_{telegram_user_id}"
        
        # Enregistrer le mapping
        self.user_mappings[telegram_user_id] = emoia_user_id
        
        # Enregistrer dans EmoIA si disponible
        if self.emoia and hasattr(self.emoia, 'memory_system'):
            try:
                await self.emoia.memory_system.remember_user_name(emoia_user_id, name)
                success_message = f"✅ Enregistré avec succès !\n\nNom: {name}\nID EmoIA: {emoia_user_id}\n\nJe me souviendrai de vous ! 🧠"
            except Exception as e:
                success_message = f"✅ Enregistrement partiel\n\nNom: {name}\nErreur mémoire: {str(e)}"
        else:
            success_message = f"✅ Enregistrement local\n\nNom: {name}\nID: {emoia_user_id}\n\n⚠️ Connexion EmoIA non disponible"
        
        await update.message.reply_text(success_message)
        
        # Sauvegarder les mappings (simple fichier JSON pour persistance)
        try:
            with open("telegram_mappings.json", "w") as f:
                json.dump(self.user_mappings, f)
        except Exception as e:
            logger.error(f"Erreur sauvegarde mappings: {e}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour la commande /status"""
        telegram_user_id = str(update.effective_user.id)
        
        if telegram_user_id not in self.user_mappings:
            await update.message.reply_text("❌ Vous n'êtes pas encore enregistré. Utilisez /register <votre_nom>")
            return
        
        emoia_user_id = self.user_mappings[telegram_user_id]
        
        # Récupérer les stats depuis EmoIA
        if self.emoia and hasattr(self.emoia, 'memory_system'):
            try:
                user_name = await self.emoia.memory_system.get_user_name(emoia_user_id)
                concepts = await self.emoia.memory_system.get_learned_concepts(emoia_user_id)
                tasks = await self.emoia.memory_system.get_tdah_tasks(emoia_user_id)
                
                status_message = f"""
📊 Votre statut EmoIA

👤 Nom: {user_name or "Non défini"}
🆔 ID: {emoia_user_id}

📚 Apprentissage:
• Concepts appris: {len(concepts)}

📝 Tâches TDAH:
• Actives: {len([t for t in tasks if not t.completed])}
• Terminées: {len([t for t in tasks if t.completed])}

🔗 Connexion: ✅ Active
                """
            except Exception as e:
                status_message = f"📊 Statut de base\n\nID: {emoia_user_id}\n❌ Erreur connexion EmoIA: {str(e)}"
        else:
            status_message = f"📊 Statut local\n\nID: {emoia_user_id}\n⚠️ EmoIA non connecté"
        
        await update.message.reply_text(status_message)
    
    async def tasks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour la commande /tasks"""
        telegram_user_id = str(update.effective_user.id)
        
        if telegram_user_id not in self.user_mappings:
            await update.message.reply_text("❌ Utilisez d'abord /register <votre_nom>")
            return
        
        emoia_user_id = self.user_mappings[telegram_user_id]
        
        if self.emoia and hasattr(self.emoia, 'memory_system'):
            try:
                tasks = await self.emoia.memory_system.get_tdah_tasks(emoia_user_id, completed=False)
                
                if tasks:
                    task_list = "📝 Vos tâches actives:\n\n"
                    for i, task in enumerate(tasks[:10], 1):  # Limiter à 10 tâches
                        priority_emoji = "🔴" if task.priority >= 4 else "🟡" if task.priority >= 3 else "🟢"
                        task_list += f"{i}. {priority_emoji} {task.title}\n"
                        if task.due_date:
                            task_list += f"   📅 Échéance: {task.due_date.strftime('%d/%m/%Y')}\n"
                        task_list += "\n"
                else:
                    task_list = "✅ Aucune tâche active !\n\nVous pouvez créer une nouvelle tâche en me parlant."
                
                await update.message.reply_text(task_list)
                
            except Exception as e:
                await update.message.reply_text(f"❌ Erreur lors de la récupération des tâches: {str(e)}")
        else:
            await update.message.reply_text("⚠️ Service de tâches non disponible")
    
    async def learn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour la commande /learn"""
        telegram_user_id = str(update.effective_user.id)
        
        if telegram_user_id not in self.user_mappings:
            await update.message.reply_text("❌ Utilisez d'abord /register <votre_nom>")
            return
        
        if not context.args:
            await update.message.reply_text("💡 Usage: /learn <concept> <explication>\n\nExemple: /learn Python Un langage de programmation simple et puissant")
            return
        
        emoia_user_id = self.user_mappings[telegram_user_id]
        
        # Parser les arguments
        full_text = " ".join(context.args)
        parts = full_text.split(" ", 1)
        
        if len(parts) < 2:
            await update.message.reply_text("❌ Veuillez fournir un nom de concept et une explication")
            return
        
        concept_name = parts[0]
        explanation = parts[1]
        
        if self.emoia and hasattr(self.emoia, 'memory_system'):
            try:
                concept_id = await self.emoia.memory_system.learn_concept(
                    user_id=emoia_user_id,
                    concept_name=concept_name,
                    explanation=explanation,
                    category="telegram"
                )
                
                if concept_id:
                    await update.message.reply_text(f"🧠 Concept appris avec succès !\n\n📚 {concept_name}\n💭 {explanation}\n\nJe m'en souviendrai ! ✅")
                else:
                    await update.message.reply_text("❌ Erreur lors de l'apprentissage du concept")
                    
            except Exception as e:
                await update.message.reply_text(f"❌ Erreur: {str(e)}")
        else:
            await update.message.reply_text("⚠️ Service d'apprentissage non disponible")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler pour les messages texte normaux"""
        telegram_user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        if telegram_user_id not in self.user_mappings:
            await update.message.reply_text("👋 Bonjour ! Utilisez /register <votre_nom> pour commencer à discuter avec moi.")
            return
        
        emoia_user_id = self.user_mappings[telegram_user_id]
        
        # Traiter le message avec EmoIA si disponible
        if self.emoia and hasattr(self.emoia, 'process_message'):
            try:
                # Traiter le message avec EmoIA
                response_data = await self.emoia.process_message(
                    user_input=message_text,
                    user_id=emoia_user_id
                )
                
                # Envoyer la réponse
                response_text = response_data.get('response', 'Désolé, je n\'ai pas pu traiter votre message.')
                
                # Ajouter des informations émotionnelles si disponibles
                if response_data.get('emotional_analysis'):
                    emotion_info = response_data['emotional_analysis']
                    emotion_emoji = self._get_emotion_emoji(emotion_info.get('detected_emotion', ''))
                    if emotion_emoji:
                        response_text = f"{emotion_emoji} {response_text}"
                
                await update.message.reply_text(response_text)
                
            except Exception as e:
                logger.error(f"Erreur traitement message EmoIA: {e}")
                await update.message.reply_text("❌ Désolé, j'ai rencontré un problème. Pouvez-vous réessayer ?")
        else:
            # Réponse simple si EmoIA n'est pas disponible
            await update.message.reply_text(f"🤖 J'ai bien reçu votre message: \"{message_text}\"\n\n⚠️ Service EmoIA complet non disponible pour le moment.")
    
    def _get_emotion_emoji(self, emotion: str) -> str:
        """Retourne un emoji correspondant à l'émotion"""
        emotion_emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😰",
            "surprise": "😲",
            "love": "❤️",
            "excitement": "🎉",
            "anxiety": "😰",
            "contentment": "😌",
            "curiosity": "🤔"
        }
        return emotion_emojis.get(emotion.lower(), "")
    
    def load_mappings(self):
        """Charge les mappings depuis le fichier"""
        try:
            with open("telegram_mappings.json", "r") as f:
                self.user_mappings = json.load(f)
                logger.info(f"Mappings chargés: {len(self.user_mappings)} utilisateurs")
        except FileNotFoundError:
            logger.info("Aucun fichier de mappings trouvé, démarrage avec mappings vides")
        except Exception as e:
            logger.error(f"Erreur chargement mappings: {e}")


# Fonction pour démarrer le bot en standalone
async def main():
    """Fonction principale pour tester le bot"""
    bot_manager = TelegramBotManager()
    bot_manager.load_mappings()
    
    if await bot_manager.initialize():
        await bot_manager.start_bot()
        
        try:
            # Garder le bot en vie
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            await bot_manager.stop_bot()
    else:
        logger.error("Impossible d'initialiser le bot Telegram")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())


# Instance globale pour utilisation simple
telegram_bot = TelegramBotManager() 