# Dockerfile pour EmoIA - Intelligence Artificielle Émotionnelle

# Image de base Python 3.11
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copier le code de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p logs cache models data

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["python", "-m", "uvicorn", "src.core.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]