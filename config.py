"""Configuration settings for the video analysis pipeline."""

import os
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "keyframe_extraction": {
        "method": "content_based",
        "max_frames_per_scene": 10,
        "min_frames_per_scene": 5
    },
    "scene_detection": {
        "threshold": 0.65,
        "min_scene_length": 20.0  # in seconds
    },
    "entity_detection": {
        "min_confidence": 0.5,
        "max_detections": 20
    },
    "audio_processing": {
        "model": "large-v3",
        "min_speakers": 1,
        "max_speakers": 8
    },
    "gemini_vision": {
        "temperature": 0.2,
        "max_tokens": 1500,
        "top_p": 0.95
    }
}

def load_config(config_path=None):
    """
    Load configuration from file or return default config.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not config_path:
        logger.info("No config path provided. Using defaults.")
        return DEFAULT_CONFIG.copy()
        
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG.copy()
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get the config file's directory to resolve relative paths
        config_dir = config_path.parent.absolute()
        
        # Merge with defaults for any missing keys
        merged_config = DEFAULT_CONFIG.copy()
        for section, values in config.items():
            if section in merged_config and isinstance(values, dict):
                merged_config[section].update(values)
            else:
                merged_config[section] = values
        
        # Check and handle entity detection paths
        if 'entity_detection' in merged_config:
            entity_config = merged_config['entity_detection']
            required_paths = [
                "grounding_dino_config_path",
                "grounding_dino_checkpoint_path", 
                "sam_checkpoint_path"
            ]
            
            # Convert relative paths to absolute
            for path_key in required_paths:
                if path_key in entity_config:
                    path_str = entity_config[path_key]
                    path = Path(path_str)
                    
                    # If it's a relative path, make it absolute relative to config file location
                    if not path.is_absolute():
                        abs_path = (config_dir / path).resolve()
                        entity_config[path_key] = str(abs_path)
                        logger.info(f"Converted relative path '{path_str}' to absolute path '{abs_path}'")
            
            # Check for missing paths
            missing_paths = [path for path in required_paths if path not in entity_config]
            if missing_paths:
                logger.warning(f"Missing required entity detection paths: {', '.join(missing_paths)}")
            else:
                logger.info(f"Entity detection paths found in config")
                for path in required_paths:
                    logger.info(f"  {path}: {entity_config[path]}")
                    
                    # Check if paths exist
                    if not Path(entity_config[path]).exists():
                        logger.error(f"Path does not exist: {entity_config[path]}")
        
        return merged_config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}. Using defaults.")
        import traceback
        traceback.print_exc()
        return DEFAULT_CONFIG.copy() 