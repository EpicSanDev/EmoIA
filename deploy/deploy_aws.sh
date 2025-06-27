#!/bin/bash

# Script de déploiement d'EmoIA sur AWS

set -e

echo "🚀 Déploiement d'EmoIA sur AWS..."

# Variables de configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
ECR_REPOSITORY="emoia"
ECS_CLUSTER="emoia-cluster"
ECS_SERVICE="emoia-service"
TASK_DEFINITION="emoia-task"

# Vérifier les prérequis
check_requirements() {
    echo "📋 Vérification des prérequis..."
    
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI n'est pas installé"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker n'est pas installé"
        exit 1
    fi
    
    # Vérifier l'authentification AWS
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "❌ Non authentifié avec AWS. Exécutez 'aws configure'"
        exit 1
    fi
    
    echo "✅ Prérequis vérifiés"
}

# Construire et pousser l'image Docker
build_and_push() {
    echo "🔨 Construction de l'image Docker..."
    
    # Obtenir l'URL du registre ECR
    ECR_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text 2>/dev/null)
    
    if [ -z "$ECR_URI" ]; then
        echo "📦 Création du repository ECR..."
        aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
        ECR_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)
    fi
    
    # Se connecter à ECR
    echo "🔐 Connexion à ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Construire l'image
    echo "🏗️ Construction de l'image..."
    docker build -t $ECR_REPOSITORY:latest .
    
    # Taguer l'image
    docker tag $ECR_REPOSITORY:latest $ECR_URI:latest
    
    # Pousser l'image
    echo "📤 Push de l'image vers ECR..."
    docker push $ECR_URI:latest
    
    echo "✅ Image poussée avec succès"
}

# Déployer sur ECS
deploy_ecs() {
    echo "🚢 Déploiement sur ECS..."
    
    # Vérifier si le cluster existe
    if ! aws ecs describe-clusters --clusters $ECS_CLUSTER --region $AWS_REGION &> /dev/null; then
        echo "📋 Création du cluster ECS..."
        aws ecs create-cluster --cluster-name $ECS_CLUSTER --region $AWS_REGION
    fi
    
    # Mettre à jour la définition de tâche
    echo "📝 Mise à jour de la définition de tâche..."
    aws ecs register-task-definition \
        --family $TASK_DEFINITION \
        --network-mode awsvpc \
        --requires-compatibilities FARGATE \
        --cpu "1024" \
        --memory "2048" \
        --container-definitions "[
            {
                \"name\": \"emoia-api\",
                \"image\": \"$ECR_URI:latest\",
                \"portMappings\": [
                    {
                        \"containerPort\": 8000,
                        \"protocol\": \"tcp\"
                    }
                ],
                \"essential\": true,
                \"environment\": [
                    {
                        \"name\": \"ENV\",
                        \"value\": \"production\"
                    }
                ],
                \"logConfiguration\": {
                    \"logDriver\": \"awslogs\",
                    \"options\": {
                        \"awslogs-group\": \"/ecs/emoia\",
                        \"awslogs-region\": \"$AWS_REGION\",
                        \"awslogs-stream-prefix\": \"ecs\"
                    }
                }
            }
        ]" \
        --region $AWS_REGION
    
    # Vérifier si le service existe
    if aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION &> /dev/null; then
        echo "🔄 Mise à jour du service ECS..."
        aws ecs update-service \
            --cluster $ECS_CLUSTER \
            --service $ECS_SERVICE \
            --task-definition $TASK_DEFINITION \
            --force-new-deployment \
            --region $AWS_REGION
    else
        echo "➕ Création du service ECS..."
        # Note: Vous devez configurer les subnets et security groups appropriés
        echo "⚠️  Veuillez créer le service manuellement avec les bons paramètres réseau"
    fi
    
    echo "✅ Déploiement lancé"
}

# Afficher les informations de déploiement
show_deployment_info() {
    echo ""
    echo "📊 Informations de déploiement:"
    echo "================================"
    echo "Région: $AWS_REGION"
    echo "Cluster ECS: $ECS_CLUSTER"
    echo "Service: $ECS_SERVICE"
    echo ""
    echo "Pour voir les logs:"
    echo "aws logs tail /ecs/emoia --follow"
    echo ""
    echo "Pour voir l'état du service:"
    echo "aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE"
}

# Main
main() {
    check_requirements
    build_and_push
    deploy_ecs
    show_deployment_info
}

# Exécuter si appelé directement
if [ "$0" = "${BASH_SOURCE[0]}" ]; then
    main "$@"
fi