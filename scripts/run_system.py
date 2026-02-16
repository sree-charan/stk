#!/usr/bin/env python
"""Run the Stock Chat system."""
import sys
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

def train_models():
    """Train ML models if not already trained."""
    model_dir = ROOT / 'backend' / 'models' / 'saved'
    if not (model_dir / 'xgb_short.pkl').exists():
        print("Training models...")
        sys.path.insert(0, str(ROOT))
        from backend.models.train import ModelTrainer
        trainer = ModelTrainer('TSLA')
        trainer.train(days=200)
        trainer.save_models()
        print("Models trained and saved.")
    else:
        print("Models already trained.")

def build_frontend():
    """Build frontend if not already built."""
    frontend_dir = ROOT / 'frontend'
    build_dir = frontend_dir / 'build'
    
    if not build_dir.exists():
        print("Building frontend...")
        # Install dependencies
        subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
        # Build
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, check=True)
        print("Frontend built.")
    else:
        print("Frontend already built.")

def run_backend():
    """Start the FastAPI backend."""
    print("\n" + "="*50)
    print("Stock Chat")
    print("="*50)
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    os.chdir(ROOT)
    subprocess.run([
        sys.executable, '-m', 'uvicorn',
        'backend.api.server:app',
        '--host', '0.0.0.0',
        '--port', '8000'
    ])

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stock Chat')
    parser.add_argument('command', choices=['train', 'backend', 'frontend', 'all'], default='all', nargs='?')
    args = parser.parse_args()
    
    if args.command == 'train':
        train_models()
    elif args.command == 'backend':
        run_backend()
    elif args.command == 'frontend':
        build_frontend()
    else:
        train_models()
        build_frontend()
        run_backend()

if __name__ == '__main__':
    main()
