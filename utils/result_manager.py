"""
Result manager for caching and retrieving pipeline results.
Implements efficient caching mechanisms for intermediate results.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultManager:
    """
    Manages caching and retrieval of pipeline results.
    
    Features:
    - Efficient disk caching of intermediate results
    - Metadata tracking for cache invalidation
    - Fingerprinting of inputs for consistent cache keys
    """
    
    def __init__(self, cache_dir: str = None, use_cache: bool = True, cache_ttl: int = 86400):
        """
        Initialize the result manager.
        
        Args:
            cache_dir: Directory for caching results (defaults to ./.cache)
            use_cache: Whether to use caching
            cache_ttl: Time-to-live for cache entries in seconds (default 24 hours)
        """
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(".cache")
            
        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {self.cache_dir}")
            
    def cache_key(self, component: str, input_data: Any) -> str:
        """
        Generate a cache key for the given component and input data.
        
        Args:
            component: Component identifier (e.g., 'scene_detection')
            input_data: Input data to fingerprint (can be a file path, parameters, etc.)
            
        Returns:
            Cache key string
        """
        # Handle different input types
        if isinstance(input_data, str) and os.path.exists(input_data):
            # For file paths, use file stats and a sample of content
            stats = os.stat(input_data)
            file_size = stats.st_size
            file_mtime = stats.st_mtime
            
            # If it's a video file, include a content sample hash
            if Path(input_data).suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                # Read the first 1MB of the file for fingerprinting
                with open(input_data, 'rb') as f:
                    content_sample = f.read(1024 * 1024)
                content_hash = hashlib.md5(content_sample).hexdigest()
            else:
                content_hash = "file"
                
            key_data = f"{component}:{input_data}:{file_size}:{file_mtime}:{content_hash}"
        elif isinstance(input_data, (dict, list, tuple)):
            # For structured data, convert to JSON string
            try:
                key_data = f"{component}:{json.dumps(input_data, sort_keys=True)}"
            except (TypeError, ValueError):
                # If not JSON serializable, use pickle for fingerprinting
                key_data = f"{component}:{str(input_data)}"
        else:
            # For other types, use string representation
            key_data = f"{component}:{str(input_data)}"
            
        # Generate final key hash
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def save_result(self, component: str, input_data: Any, result: Any, metadata: Dict = None) -> bool:
        """
        Save a result to the cache.
        
        Args:
            component: Component identifier
            input_data: The input that generated this result (for key generation)
            result: The result to cache
            metadata: Optional metadata to store with the result
            
        Returns:
            True if save was successful, False otherwise
        """
        if not self.use_cache:
            return False
            
        try:
            # Generate cache key
            key = self.cache_key(component, input_data)
            
            # Create cache entry
            cache_entry = {
                "timestamp": time.time(),
                "component": component,
                "metadata": metadata or {},
                "result": result
            }
            
            # Save to disk using pickle for complex objects
            cache_path = self.cache_dir / f"{key}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_entry, f)
                
            # Save metadata separately for faster access
            meta_path = self.cache_dir / f"{key}.meta.json"
            meta_entry = {
                "timestamp": cache_entry["timestamp"],
                "component": component,
                "metadata": metadata or {}
            }
            with open(meta_path, 'w') as f:
                json.dump(meta_entry, f)
                
            logger.debug(f"Cached result for {component} with key {key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache result for {component}: {str(e)}")
            return False
            
    def load_result(self, component: str, input_data: Any) -> Tuple[bool, Any]:
        """
        Load a result from the cache.
        
        Args:
            component: Component identifier
            input_data: The input for key generation
            
        Returns:
            Tuple of (success, result)
        """
        if not self.use_cache:
            return False, None
            
        try:
            # Generate cache key
            key = self.cache_key(component, input_data)
            
            # Check if cache entry exists
            cache_path = self.cache_dir / f"{key}.pkl"
            if not cache_path.exists():
                return False, None
                
            # Check metadata for TTL
            meta_path = self.cache_dir / f"{key}.meta.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    try:
                        meta = json.load(f)
                        timestamp = meta.get("timestamp", 0)
                        if time.time() - timestamp > self.cache_ttl:
                            logger.debug(f"Cache entry for {component} expired")
                            return False, None
                    except json.JSONDecodeError:
                        # Corrupt metadata, ignore cache
                        return False, None
            
            # Load the actual result
            with open(cache_path, 'rb') as f:
                cache_entry = pickle.load(f)
                logger.debug(f"Loaded cached result for {component} with key {key}")
                return True, cache_entry.get("result")
                
        except Exception as e:
            logger.warning(f"Failed to load cached result for {component}: {str(e)}")
            return False, None
            
    def invalidate(self, component: str = None, older_than: int = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            component: Optional component to limit invalidation to
            older_than: Optional age in seconds to limit invalidation to
            
        Returns:
            Number of entries invalidated
        """
        if not self.use_cache:
            return 0
            
        count = 0
        now = time.time()
        
        try:
            # Walk through cache files
            for item in self.cache_dir.glob("*.meta.json"):
                try:
                    with open(item, 'r') as f:
                        meta = json.load(f)
                        
                    # Check component filter
                    if component and meta.get("component") != component:
                        continue
                        
                    # Check age filter
                    if older_than and now - meta.get("timestamp", 0) < older_than:
                        continue
                        
                    # Get the associated data file
                    key = item.stem.split('.')[0]
                    data_path = self.cache_dir / f"{key}.pkl"
                    
                    # Remove both files
                    if item.exists():
                        item.unlink()
                    if data_path.exists():
                        data_path.unlink()
                        
                    count += 1
                except Exception as e:
                    logger.warning(f"Error processing cache item {item}: {str(e)}")
                    
            logger.info(f"Invalidated {count} cache entries")
            return count
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return 0