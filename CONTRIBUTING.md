# Guide de Contribution à EmoIA

Merci de votre intérêt pour contribuer à EmoIA ! Ce document fournit les directives pour contribuer au projet.

## 🚀 Commencer

1. **Fork le repository**
   - Cliquez sur le bouton "Fork" en haut de la page GitHub

2. **Cloner votre fork**
   ```bash
   git clone https://github.com/VOTRE-USERNAME/emoia.git
   cd emoia
   ```

3. **Ajouter le repository upstream**
   ```bash
   git remote add upstream https://github.com/emoia/emoia.git
   ```

4. **Créer une branche**
   ```bash
   git checkout -b feature/ma-nouvelle-fonctionnalite
   ```

## 💻 Environnement de Développement

### Configuration
```bash
# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]  # Installe les dépendances de développement

# Frontend
cd frontend
npm install
```

### Standards de Code

- **Python** : Suivre PEP 8
  ```bash
  # Formatter le code
  black src/

  # Vérifier le style
  flake8 src/

  # Vérifier les types
  mypy src/
  ```

- **JavaScript/TypeScript** : Utiliser ESLint et Prettier
  ```bash
  cd frontend
  npm run lint
  npm run format
  ```

## 📝 Processus de Contribution

### 1. Types de Contributions

- 🐛 **Corrections de bugs**
- ✨ **Nouvelles fonctionnalités**
- 📚 **Documentation**
- 🧪 **Tests**
- 🎨 **Améliorations UI/UX**

### 2. Workflow

1. **Créer une issue** décrivant ce que vous voulez faire
2. **Attendre l'approbation** avant de commencer le travail majeur
3. **Développer** votre fonctionnalité/correction
4. **Tester** votre code
5. **Documenter** vos changements
6. **Soumettre** une Pull Request

### 3. Commit Messages

Utilisez des messages de commit descriptifs :
```
feat: Ajouter l'analyse des sentiments multilingue
fix: Corriger le bug de mémoire dans le chat
docs: Mettre à jour le README avec les exemples d'API
test: Ajouter des tests pour le module émotionnel
refactor: Restructurer le système de mémoire
```

## 🧪 Tests

### Écrire des Tests
```python
# tests/test_ma_fonctionnalite.py
import pytest
from src.mon_module import ma_fonction

def test_ma_fonction():
    result = ma_fonction("input")
    assert result == "expected_output"

@pytest.mark.asyncio
async def test_ma_fonction_async():
    result = await ma_fonction_async("input")
    assert result == "expected_output"
```

### Exécuter les Tests
```bash
# Tous les tests
pytest

# Tests spécifiques
pytest tests/test_emotional_core.py

# Avec couverture
pytest --cov=src --cov-report=html
```

## 📋 Checklist avant la PR

- [ ] Le code suit les standards du projet
- [ ] Les tests passent (`pytest`)
- [ ] La documentation est à jour
- [ ] Les commits sont propres et descriptifs
- [ ] La PR a une description claire
- [ ] Pas de conflits avec la branche main

## 🏗️ Structure des Pull Requests

### Titre
`[TYPE] Description courte`

Exemples :
- `[FEAT] Ajouter le support des émojis dans l'analyse`
- `[FIX] Corriger la fuite mémoire dans le WebSocket`
- `[DOCS] Améliorer la documentation API`

### Description
```markdown
## Description
Décrivez les changements effectués

## Motivation
Pourquoi ces changements sont nécessaires

## Changements
- Changement 1
- Changement 2

## Tests
Décrivez comment tester les changements

## Captures d'écran (si applicable)
Ajoutez des captures d'écran
```

## 🎯 Priorités du Projet

### Haute Priorité
- 🔒 Sécurité et confidentialité
- 🎯 Précision de l'analyse émotionnelle
- ⚡ Performance et scalabilité
- 🌍 Support multilingue

### Contributions Recherchées
- Amélioration des modèles d'IA
- Optimisation des performances
- Nouvelles visualisations analytics
- Support de nouvelles langues
- Tests d'intégration

## 📜 Code de Conduite

- Soyez respectueux et inclusif
- Acceptez les critiques constructives
- Focalisez-vous sur ce qui est meilleur pour la communauté
- Montrez de l'empathie envers les autres contributeurs

## ❓ Questions ?

- Ouvrez une issue avec le label `question`
- Rejoignez notre Discord : [discord.gg/emoia](https://discord.gg/emoia)
- Email : dev@emoia.ai

## 🙏 Merci !

Merci de prendre le temps de contribuer à EmoIA ! Chaque contribution, grande ou petite, est appréciée et aide à rendre l'IA plus émotionnellement intelligente.

---

Happy coding! 🚀