"""Configuration management for MTG Judge Engine."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class Config:
    """Configuration manager for the MTG Judge Engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Try to find config file
            possible_paths = [
                "config/settings.yaml",
                "settings.yaml",
                "../config/settings.yaml"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError("Could not find settings.yaml file")
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Override with environment variables where applicable
            self._apply_env_overrides(config)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Apply environment variable overrides."""
        # LLM API keys
        if "llm" in config:
            if "openai" in config["llm"] and os.getenv("OPENAI_API_KEY"):
                config["llm"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
            
            if "anthropic" in config["llm"] and os.getenv("ANTHROPIC_API_KEY"):
                config["llm"]["anthropic"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        # Database URL override
        if os.getenv("DATABASE_URL"):
            config.setdefault("database", {})["url"] = os.getenv("DATABASE_URL")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "database": {
                "url": "sqlite:///mtg_judge.db",
                "echo": False
            },
            "llm": {
                "type": "local",
                "local": {
                    "model_name": "microsoft/DialoGPT-medium",
                    "use_quantization": True,
                    "max_length": 2048,
                    "temperature": 0.7
                }
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": True
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.get("llm", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get("database", {})
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web server configuration."""
        return self.get("web", {})


# Global config instance
_config_instance = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file."""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance