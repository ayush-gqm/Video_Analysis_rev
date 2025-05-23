#!/usr/bin/env python3
"""
Utility script to download and prepare models needed by the video analysis pipeline.
This helps ensure dependencies are properly downloaded before running the main pipeline.
"""

import os
import sys
import logging
import argparse
import shutil
import tempfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_yolo():
    """Download and prepare YOLOv5 model."""
    logger.info("Setting up YOLOv5 model...")
    
    try:
        # Try importing torch first
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check if YOLOv5 is already downloaded in the torch hub cache
        cache_dir = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master")
        if os.path.exists(cache_dir):
            logger.info(f"YOLOv5 already exists at {cache_dir}")
            
            # Check if 'utils' directory exists
            utils_dir = os.path.join(cache_dir, 'utils')
            if not os.path.exists(utils_dir) or not os.path.exists(os.path.join(utils_dir, 'dataloaders.py')):
                logger.warning("YOLOv5 installation may be incomplete. Redownloading...")
                
                # Create a backup of the existing dir
                backup_dir = cache_dir + ".bak"
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.move(cache_dir, backup_dir)
                logger.info(f"Backed up existing installation to {backup_dir}")
                
                # Force redownload
                logger.info("Downloading YOLOv5 from scratch...")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)
                logger.info("YOLOv5 successfully downloaded and initialized")
            else:
                logger.info("YOLOv5 installation appears complete")
                
        else:
            # Download YOLOv5 model
            logger.info("Downloading YOLOv5 from scratch...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)
            logger.info("YOLOv5 successfully downloaded and initialized")
        
        # Test import of dataloaders to verify installation
        sys_path_orig = list(sys.path)
        
        try:
            # Temporarily modify path to check imports
            if os.path.exists(cache_dir):
                sys.path.insert(0, cache_dir)
                
            # Test imports
            from utils.dataloaders import exif_transpose, letterbox
            logger.info("Successfully verified YOLOv5 utils.dataloaders import")
        except ImportError as e:
            logger.error(f"Failed to import YOLOv5 modules: {e}")
            logger.info("Attempting alternative installation method...")
            
            # Clone the repository to a temporary location
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    subprocess.check_call([
                        "git", "clone", "https://github.com/ultralytics/yolov5.git", 
                        tmp_dir, "--depth", "1"
                    ])
                    
                    # Copy the repo to the torch hub cache location
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                    shutil.copytree(tmp_dir, cache_dir)
                    logger.info(f"Copied clean YOLOv5 repository to {cache_dir}")
                    
                    # Test import again
                    from utils.dataloaders import exif_transpose, letterbox
                    logger.info("Successfully verified YOLOv5 utils.dataloaders import after cleanup")
                    
                except Exception as clone_err:
                    logger.error(f"Failed to clone and install YOLOv5: {clone_err}")
                    raise
        finally:
            # Restore system path
            sys.path = sys_path_orig
            
    except Exception as e:
        logger.error(f"Error setting up YOLOv5: {e}")
        raise
    
    logger.info("YOLOv5 setup complete")
    return True

def setup_whisperx():
    """Ensure WhisperX is properly installed and available."""
    logger.info("Checking WhisperX installation...")
    
    try:
        import whisperx
        logger.info(f"WhisperX is installed")
        
        # Test key functionality
        if hasattr(whisperx, "load_model"):
            logger.info("WhisperX load_model function is available")
        else:
            logger.warning("WhisperX load_model function not found")
            
        # Check for VAD module
        try:
            from whisperx import vad
            logger.info("WhisperX VAD module is available")
        except ImportError:
            logger.warning("WhisperX VAD module not found")
            
    except ImportError:
        logger.warning("WhisperX is not installed")
        logger.info("To install WhisperX, run: pip install git+https://github.com/m-bain/whisperx.git")
        return False
        
    logger.info("WhisperX check complete")
    return True

def main():
    """Main function to prepare all necessary models."""
    parser = argparse.ArgumentParser(description="Download and prepare models for video analysis pipeline")
    parser.add_argument("--yolo", action="store_true", help="Prepare YOLOv5 model")
    parser.add_argument("--whisperx", action="store_true", help="Check WhisperX installation")
    parser.add_argument("--all", action="store_true", help="Prepare all models")
    
    args = parser.parse_args()
    
    # If no specific flags, prepare all
    if not (args.yolo or args.whisperx):
        args.all = True
        
    # Track failures
    failures = []
    
    # Prepare YOLOv5
    if args.all or args.yolo:
        try:
            setup_yolo()
        except Exception as e:
            logger.error(f"Failed to prepare YOLOv5: {e}")
            failures.append("YOLOv5")
    
    # Check WhisperX
    if args.all or args.whisperx:
        try:
            if not setup_whisperx():
                failures.append("WhisperX")
        except Exception as e:
            logger.error(f"Failed to check WhisperX: {e}")
            failures.append("WhisperX")
    
    # Report results
    if failures:
        logger.warning(f"The following components had issues: {', '.join(failures)}")
        logger.info("You may still be able to use the pipeline if you don't need these components.")
        return 1
    else:
        logger.info("All requested models have been prepared successfully")
        return 0

if __name__ == "__main__":
    sys.exit(main())