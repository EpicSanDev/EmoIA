"""
Point d'entrée principal pour EmoIA
Démarrage de l'intelligence artificielle émotionnelle.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import EmoIA
from src.config import Config


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/emoia.log')
    ]
)

logger = logging.getLogger(__name__)


class EmoIADemo:
    """Interface de démonstration pour EmoIA"""
    
    def __init__(self):
        self.emoia = None
        self.user_id = "demo_user"
        
    async def initialize(self):
        """Initialise EmoIA"""
        try:
            # Charger la configuration
            config = Config()
            
            # Créer l'instance EmoIA
            self.emoia = EmoIA(config)
            
            # Initialiser tous les composants
            await self.emoia.initialize()
            
            logger.info("✨ EmoIA prêt pour les interactions !")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            raise
    
    async def run_interactive_mode(self):
        """Mode interactif en ligne de commande"""
        print("🤖 EmoIA - Intelligence Artificielle Émotionnelle")
        print("=" * 50)
        print("Tapez 'quit' pour quitter, 'help' pour l'aide")
        print("=" * 50)
        
        while True:
            try:
                # Saisie utilisateur
                user_input = input("\n💬 Vous: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir ! Prenez soin de vous.")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    await self.show_stats()
                    continue
                
                if user_input.lower() == 'insights':
                    await self.show_insights()
                    continue
                
                if not user_input:
                    continue
                
                # Traitement par EmoIA
                response_data = await self.emoia.process_message(
                    user_input=user_input,
                    user_id=self.user_id
                )
                
                # Affichage de la réponse
                print(f"\n🤖 EmoIA: {response_data['response']}")
                
                # Affichage des métadonnées (optionnel)
                if response_data.get('emotional_analysis'):
                    emotion_info = response_data['emotional_analysis']
                    print(f"   📊 Émotion détectée: {emotion_info['detected_emotion']} "
                          f"({emotion_info['emotion_intensity']:.2f})")
                
                # Vérifier la proactivité
                proactive_message = await self.emoia.check_proactivity(self.user_id)
                if proactive_message:
                    print(f"\n💭 EmoIA (proactif): {proactive_message}")
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir ! Prenez soin de vous.")
                break
            except Exception as e:
                logger.error(f"Erreur: {e}")
                print(f"❌ Une erreur s'est produite: {e}")
    
    def show_help(self):
        """Affiche l'aide"""
        help_text = """
🔧 Commandes disponibles:
- help: Affiche cette aide
- stats: Affiche les statistiques du système
- insights: Affiche vos insights émotionnels
- quit/exit/q: Quitte le programme

💡 Conseils d'utilisation:
- Parlez naturellement, EmoIA comprend vos émotions
- Plus vous interagissez, mieux EmoIA vous comprend
- EmoIA peut être proactif et vous contacter spontanément
- Vos conversations sont mémorisées pour un suivi personnalisé
        """
        print(help_text)
    
    async def show_stats(self):
        """Affiche les statistiques système"""
        try:
            stats = self.emoia.get_system_stats()
            
            print("\n📈 Statistiques EmoIA:")
            print(f"⏱️  Temps de fonctionnement: {stats['uptime']}")
            print(f"💬 Total interactions: {stats['total_interactions']}")
            print(f"👥 Utilisateurs actifs: {stats['active_users']}")
            print(f"🧠 Mémoires stockées: {stats['memory_stats']['long_term_memory_size']}")
            print(f"🎭 Modèle émotionnel: Activé (intensité {stats['config_summary']['emotional_intensity']})")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'affichage des stats: {e}")
    
    async def show_insights(self):
        """Affiche les insights émotionnels"""
        try:
            insights = await self.emoia.get_emotional_insights(self.user_id)
            
            if "error" in insights:
                print(f"❌ {insights['error']}")
                return
            
            if "message" in insights:
                print(f"ℹ️  {insights['message']}")
                return
            
            print(f"\n🧠 Insights Émotionnels ({insights['period_analyzed']}):")
            print(f"📊 Total interactions analysées: {insights['total_interactions']}")
            
            trends = insights['trends']
            print(f"😊 Émotion dominante: {trends['most_frequent_emotion']}")
            print(f"⚖️  Stabilité émotionnelle: {trends['emotional_stability']:.2f}/1.0")
            print(f"🌟 Ratio d'émotions positives: {trends['positive_ratio']:.2f}")
            
            print("\n💡 Recommandations:")
            for i, recommendation in enumerate(insights['recommendations'], 1):
                print(f"  {i}. {recommendation}")
                
        except Exception as e:
            print(f"❌ Erreur lors de l'affichage des insights: {e}")


async def main():
    """Fonction principale"""
    
    # Créer les répertoires nécessaires
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)
    
    # Créer et initialiser EmoIA
    demo = EmoIADemo()
    
    try:
        await demo.initialize()
        await demo.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        print(f"❌ Erreur fatale: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    """Point d'entrée du programme"""
    
    print("🚀 Démarrage d'EmoIA...")
    
    # Exécuter le programme principal
    exit_code = asyncio.run(main())
    sys.exit(exit_code)