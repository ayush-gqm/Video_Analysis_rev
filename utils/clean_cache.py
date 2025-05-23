#!/usr/bin/env python3
"""
Utility script to clean up cached results from the video analysis pipeline.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_cache(cache_dir: str = ".cache", older_than: int = None, component: str = None):
    """
    Clean up cache files.
    
    Args:
        cache_dir: Directory where cache files are stored
        older_than: Optional age in seconds to clean files older than this
        component: Optional component name to clean only files from this component
    
    Returns:
        Number of files cleaned
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.info(f"Cache directory {cache_dir} does not exist. Nothing to clean.")
        return 0
        
    count = 0
    
    # Try to import the ResultManager to use its invalidation logic
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.result_manager import ResultManager
        
        logger.info(f"Using ResultManager to clean cache in {cache_dir}")
        manager = ResultManager(cache_dir=cache_dir, use_cache=True)
        count = manager.invalidate(component=component, older_than=older_than)
        
    except ImportError:
        # Fallback to manual file cleaning
        logger.info(f"ResultManager not available. Cleaning cache manually.")
        
        # Clean based on file extension patterns
        for pattern in ["*.pkl", "*.meta.json"]:
            for file_path in cache_path.glob(pattern):
                try:
                    # Check component filter if specified
                    if component:
                        # Read meta file to check component
                        meta_path = file_path.with_suffix(".meta.json")
                        if meta_path.exists():
                            import json
                            with open(meta_path, 'r') as f:
                                try:
                                    meta = json.load(f)
                                    if meta.get("component") != component:
                                        continue
                                except json.JSONDecodeError:
                                    pass  # Clean it anyway if we can't read it
                    
                    # Check file age if specified
                    if older_than:
                        import time
                        file_age = time.time() - file_path.stat().st_mtime
                        if file_age < older_than:
                            continue
                    
                    # Delete the file
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Error cleaning {file_path}: {str(e)}")
    
    logger.info(f"Cleaned {count} cache files")
    return count

def main():
    parser = argparse.ArgumentParser(description="Clean pipeline cache files")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory to clean")
    parser.add_argument("--older-than", type=int, help="Clean files older than this many seconds")
    parser.add_argument("--component", help="Clean only files from this component")
    parser.add_argument("--all", action="store_true", help="Clean all cache files")
    
    args = parser.parse_args()
    
    if args.all:
        # Remove entire cache directory
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                logger.info(f"Completely removed cache directory: {args.cache_dir}")
            except Exception as e:
                logger.error(f"Error removing cache directory: {str(e)}")
        else:
            logger.info(f"Cache directory {args.cache_dir} does not exist. Nothing to clean.")
    else:
        # Selectively clean cache
        clean_cache(args.cache_dir, args.older_than, args.component)

if __name__ == "__main__":
    main()