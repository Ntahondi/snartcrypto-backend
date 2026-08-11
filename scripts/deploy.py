#!/usr/bin/env python3
"""
SmartCrypto Deployment Script
Handles deployment to various environments (local, Docker, cloud)
"""

import os
import sys
import subprocess
import argparse
import shutil
import json
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self):
        self.settings = get_settings()
        self.project_root = Path(__file__).parent.parent
        self.env = os.getenv('DEPLOY_ENV', 'development')
        
    def deploy_local(self):
        """Deploy locally for testing"""
        logger.info("🚀 Deploying locally...")
        
        # Check if models exist
        model_path = self.project_root / self.settings.MODEL_PATH
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("📥 Run 'python scripts/train_model.py' first")
            return False
        
        # Install dependencies
        logger.info("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=False)
        
        # Create required directories
        directories = ['logs', 'storage', 'signal_history', 'positions']
        for d in directories:
            (self.project_root / d).mkdir(exist_ok=True)
        
        # Validate configuration
        self.validate_config()
        
        # Start the application
        logger.info("✅ Local deployment ready!")
        logger.info("▶️ Run: uvicorn main:app --host 0.0.0.0 --port 8000")
        return True
    
    def deploy_docker(self):
        """Deploy using Docker"""
        logger.info("🐳 Deploying with Docker...")
        
        # Build Docker image
        dockerfile = self.project_root / 'Dockerfile'
        if not dockerfile.exists():
            self.create_dockerfile()
        
        # Build and run
        subprocess.run(["docker-compose", "up", "--build", "-d"], cwd=self.project_root, check=True)
        
        logger.info("✅ Docker deployment complete!")
        logger.info("🌐 API available at: http://localhost:8000")
        return True
    
    def deploy_cloud(self, provider='aws'):
        """Deploy to cloud provider"""
        logger.info(f"☁️ Deploying to {provider.upper()}...")
        
        if provider == 'aws':
            return self.deploy_aws()
        elif provider == 'gcp':
            return self.deploy_gcp()
        elif provider == 'azure':
            return self.deploy_azure()
        else:
            logger.error(f"❌ Unsupported provider: {provider}")
            return False
    
    def deploy_aws(self):
        """Deploy to AWS ECS/EKS"""
        logger.info("☁️ Deploying to AWS...")
        
        # Build container
        subprocess.run(["docker", "build", "-t", "smartcrypto:latest", "."], check=True)
        
        # Tag and push to ECR (example)
        # subprocess.run(["docker", "tag", "smartcrypto:latest", "your-ecr-repo"], check=True)
        # subprocess.run(["docker", "push", "your-ecr-repo"], check=True)
        
        logger.info("✅ AWS deployment initiated!")
        logger.info("📝 Configure ECS/EKS to use the container image")
        return True
    
    def deploy_gcp(self):
        """Deploy to Google Cloud Run"""
        logger.info("☁️ Deploying to GCP...")
        subprocess.run(["gcloud", "run", "deploy", "smartcrypto", "--source", "."], check=True)
        return True
    
    def deploy_azure(self):
        """Deploy to Azure App Service"""
        logger.info("☁️ Deploying to Azure...")
        subprocess.run(["az", "webapp", "deploy", "--name", "smartcrypto", "--src-path", "."], check=True)
        return True
    
    def create_dockerfile(self):
        """Create Dockerfile if not exists"""
        dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs storage signal_history positions

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        with open(self.project_root / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        logger.info("✅ Dockerfile created")
    
    def validate_config(self):
        """Validate configuration"""
        errors = []
        
        # Check model paths
        if not (self.project_root / self.settings.MODEL_PATH).exists():
            errors.append(f"Model not found: {self.settings.MODEL_PATH}")
        
        if not (self.project_root / self.settings.SCALER_PATH).exists():
            errors.append(f"Scaler not found: {self.settings.SCALER_PATH}")
        
        if errors:
            logger.warning("⚠️ Configuration warnings:")
            for e in errors:
                logger.warning(f"   - {e}")
        else:
            logger.info("✅ Configuration validated")
        
        return len(errors) == 0
    
    def backup_models(self):
        """Backup current models before deployment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.project_root / f"storage/model_backups/{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = [
            self.settings.MODEL_PATH,
            self.settings.SCALER_PATH,
            self.settings.FEATURE_COLUMNS_PATH
        ]
        
        for f in model_files:
            src = self.project_root / f
            if src.exists():
                dst = backup_dir / src.name
                shutil.copy2(src, dst)
                logger.info(f"📁 Backed up: {src.name}")
        
        logger.info(f"✅ Models backed up to: {backup_dir}")
        return backup_dir

def main():
    parser = argparse.ArgumentParser(description='SmartCrypto Deployment')
    parser.add_argument('--env', choices=['local', 'docker', 'cloud'], default='local',
                       help='Deployment environment')
    parser.add_argument('--provider', choices=['aws', 'gcp', 'azure'], default='aws',
                       help='Cloud provider (if using cloud)')
    parser.add_argument('--backup', action='store_true',
                       help='Backup models before deployment')
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.backup:
        manager.backup_models()
    
    if args.env == 'local':
        success = manager.deploy_local()
    elif args.env == 'docker':
        success = manager.deploy_docker()
    elif args.env == 'cloud':
        success = manager.deploy_cloud(args.provider)
    else:
        logger.error(f"❌ Unknown environment: {args.env}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()