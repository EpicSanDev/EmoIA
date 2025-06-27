# Dockerfile pour le frontend EmoIA

# Étape de build
FROM node:18-alpine AS builder

WORKDIR /app

# Copier package.json et package-lock.json
COPY package*.json ./

# Installer les dépendances
RUN npm ci

# Copier le code source
COPY . .

# Build de l'application
RUN npm run build

# Étape de production
FROM nginx:alpine

# Copier les fichiers buildés
COPY --from=builder /app/build /usr/share/nginx/html

# Configuration nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Exposition du port
EXPOSE 3000

# Démarrage de nginx
CMD ["nginx", "-g", "daemon off;"]