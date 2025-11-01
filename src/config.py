"""Configuration management for the self-evolving agent."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
        self.load_env_vars()
    
    def load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}
    
    def load_env_vars(self):
        """Load sensitive API keys from environment variables."""
        self.config['api_keys'] = {
            'openai': os.getenv('OPENAI_API_KEY', ''),
            'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
            'firecrawl': os.getenv('FIRECRAWL_API_KEY', ''),
            'wandb': os.getenv('WANDB_API_KEY', ''),
            'composio': os.getenv('COMPOSIO_API_KEY', ''),
            'replicate': os.getenv('REPLICATE_API_TOKEN', ''),
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'models.llm.model')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a service."""
        return self.config.get('api_keys', {}).get(service, '')

# Global config instance
config = Config()

