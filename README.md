# EmoIA
Voici une description pour votre dépôt GitHub :

---

# AI Companion Bot

Ce dépôt contient le code source d'un assistant IA avancé, conçu pour interagir avec les utilisateurs via Telegram. Le bot utilise des modèles de traitement du langage naturel (NLP) et d'apprentissage automatique pour fournir des réponses intelligentes, émotionnellement conscientes et contextuellement pertinentes.

## Fonctionnalités

- **Analyse de Sentiment** : Utilisation de modèles comme DistilBERT et VADER pour analyser les émotions et les sentiments des utilisateurs.
- **Génération de Texte** : Intégration de modèles GPT-2 et T5 pour générer des réponses textuelles et des résumés.
- **Apprentissage Renforcé** : Utilisation d'un modèle de Q-learning pour améliorer les interactions basées sur les retours des utilisateurs.
- **Gestion de la Mémoire** : Système de mémoire à court et long terme pour stocker et récupérer des informations contextuelles.
- **Analyse d'Images** : Utilisation du modèle VGG16 pour extraire des caractéristiques d'images et générer des descriptions.
- **Synthèse et Reconnaissance Vocale** : Conversion de texte en parole et vice versa pour gérer les messages vocaux.
- **Base de Connaissances** : Système de gestion des connaissances avec catégorisation, recherche sémantique et évaluation de la pertinence.
- **Personnalisation** : Analyse du comportement et de la personnalité des utilisateurs pour adapter les réponses et les suggestions.

## Installation

1. Clonez le dépôt :
    ```sh
    git clone https://github.com/votre-utilisateur/ai-companion-bot.git
    cd ai-companion-bot
    ```

2. Installez les dépendances :
    ```sh
    pip install -r requirements.txt
    ```

3. Configurez les variables d'environnement en créant un fichier 

.env

 :
    ```env
    TELEGRAM_TOKEN=your_telegram_token
    OPENAI_API_KEY=your_openai_api_key
    YOUR_CHAT_ID=your_chat_id
    NEWSAPI_KEY=your_newsapi_key
    WOLFRAM_ALPHA_APP_ID=your_wolfram_alpha_app_id
    ENCRYPTION_KEY=your_encryption_key
    ```

4. Lancez le bot :
    ```sh
    python botv6.py
    ```

## Utilisation

- **Commandes Telegram** :
  - `/start` : Démarrer le bot.
  - `/feedback` : Donner un retour sur les réponses du bot.
  - `/analyze` : Obtenir une analyse de vos interactions récentes.
  - `/learn` : Apprendre de nouvelles informations.
  - `/knowledge` : Accéder à la base de connaissances.

## Contribution

Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir une issue pour discuter des changements que vous souhaitez apporter.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

N'hésitez pas à adapter cette description en fonction de vos besoins spécifiques.