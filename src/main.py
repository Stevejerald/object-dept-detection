#!/usr/bin/env python3

import os
import sys
import logging
import yaml
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging(config):
    """Set up logging configuration."""
    log_dir = Path(config['logging']['file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration files."""
    config_dir = project_root / 'config'
    
    # Load model configuration
    with open(config_dir / 'model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Load application configuration
    with open(config_dir / 'app_config.yaml', 'r') as f:
        app_config = yaml.safe_load(f)
    
    return model_config, app_config

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/videos',
        'data/sample_outputs',
        'models/weights',
        'static/audio',
        'logs'
    ]
    
    for directory in directories:
        Path(project_root / directory).mkdir(parents=True, exist_ok=True)

def main():
    """Main application entry point."""
    # Create necessary directories
    create_directories()
    
    try:
        # Load configurations
        model_config, app_config = load_config()
        
        # Setup logging
        logger = setup_logging(app_config)
        logger.info("Starting Real-Time Monocular Depth Estimation System")
        
        # TODO: Initialize components
        # 1. Set up video input
        # 2. Initialize model
        # 3. Start web interface
        # 4. Begin processing loop
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 