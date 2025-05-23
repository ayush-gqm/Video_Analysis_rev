"""
Pipeline enhancer module that integrates parallel processing into the existing pipeline.
This module dynamically enhances the VideoPipeline class without modifying the original code.

NOTE: This module must be imported before any other pipeline modules to ensure
proper path manipulation and module isolation for multiprocessing.
"""

import sys
import os
import logging
import importlib.util
import types
from pathlib import Path
from typing import Dict, Any, List, Optional

# Fix for potential module import conflicts
# This ensures that our utils module doesn't interfere with third-party library imports
original_sys_path = list(sys.path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_module_from_file(file_path: str, module_name: str) -> types.ModuleType:
    """
    Load a Python module from a file path.
    
    Args:
        file_path: Path to the Python file
        module_name: Name to assign to the module
        
    Returns:
        Loaded module object
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register module in sys.modules to prevent pickling issues
    spec.loader.exec_module(module)
    return module

def enhance_video_pipeline():
    """
    Enhance the VideoPipeline class with additional capabilities.
    
    This function:
    1. Locates the main pipeline module
    2. Imports the parallel processing enhancements
    3. Patches the VideoPipeline class to use parallel processing
    4. Enhances dialogue processing with full context support
    
    Returns:
        True if enhancement was successful, False otherwise
    """
    # Ensure sys.path is in a clean state
    sys.path = list(original_sys_path)
    # Add current directory to path to ensure relative imports work
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        # Load the main module that contains VideoPipeline
        main_module = sys.modules.get('main')
        if not main_module:
            # If not already imported, try to import it
            try:
                main_module = importlib.import_module('main')
            except ImportError:
                logger.error("Could not import main module")
                return False
        
        # Load the parallel enhancements
        parallel_module_path = Path(__file__).parent.parent / "main_parallel.py"
        if not parallel_module_path.exists():
            logger.error(f"Parallel module not found at {parallel_module_path}")
            return False
            
        # Load module and ensure it's registered in sys.modules
        # This is important for multiprocessing/pickling to work
        module_name = "main_parallel"
        parallel_module = load_module_from_file(str(parallel_module_path), module_name)
        
        # Get the original VideoPipeline class
        VideoPipeline = getattr(main_module, "VideoPipeline", None)
        if not VideoPipeline:
            # Try alternate class name (might be VideoAnalysisPipeline)
            VideoPipeline = getattr(main_module, "VideoAnalysisPipeline", None)
            if not VideoPipeline:
                logger.error("Pipeline class not found in main module")
                return False
            
        # Patch the process_video method to use parallel execution
        original_process_video = VideoPipeline.process_video
        
        def enhanced_process_video(self, video_path, output_dir, skip_steps=None):
            """Enhanced process_video method with parallel execution."""
            # Check if parallel processing is enabled in config
            parallel_config = self.config.get("parallel_processing", {})
            parallel_enabled = parallel_config.get("enabled", True)
            
            if parallel_enabled:
                logger.info("Using enhanced parallel processing")
                # Use the module's process_video_parallel function directly
                # This ensures we're using the exact same function object that was defined in main_parallel.py
                return parallel_module.process_video_parallel(self, video_path, output_dir, skip_steps)
            else:
                logger.info("Using original sequential processing")
                return original_process_video(self, video_path, output_dir, skip_steps)
        
        # Apply the process_video patch
        VideoPipeline.process_video = enhanced_process_video
        
        # Add utility components to the VideoPipeline class for convenience
        # Make sure to reference the function through the module to avoid pickling issues
        VideoPipeline.parallel_process_video = parallel_module.process_video_parallel
        
        # Apply the dialogue enhancement
        if enhance_dialogue_processor(VideoPipeline):
            logger.info("Successfully enhanced dialogue processing with full context support")
        
        logger.info("Successfully enhanced pipeline with all available enhancements")
        return True
    
    except Exception as e:
        logger.error(f"Error enhancing pipeline: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def enhance_dialogue_processor(pipeline_class):
    """
    Enhance the dialogue processing in the pipeline to use full context approach.
    
    Args:
        pipeline_class: The pipeline class to enhance
        
    Returns:
        True if enhancement was applied successfully, False otherwise
    """
    try:
        # Check if full_context_dialogue_enhancer.py exists
        root_dir = Path(__file__).parent.parent
        enhancer_path = root_dir / "full_context_dialogue_enhancer.py"
        
        if not enhancer_path.exists():
            logger.warning(f"Full context dialogue enhancer not found at {enhancer_path}")
            return False
            
        # Load the module
        module_name = "full_context_dialogue_enhancer"
        enhancer = load_module_from_file(str(enhancer_path), module_name)
        
        # Patch the _enhance_dialogue_with_gemini method
        def new_enhance_dialogue_with_gemini(self, structured_data_path):
            """Enhanced method that processes the entire structured analysis in a single API call."""
            logger.info("Using enhanced dialogue processing with full context...")
            return enhancer.enhance_dialogue_with_full_context(structured_data_path)
        
        # Apply the patch if the method exists
        if hasattr(pipeline_class, '_enhance_dialogue_with_gemini'):
            pipeline_class._enhance_dialogue_with_gemini = new_enhance_dialogue_with_gemini
            return True
        else:
            logger.warning("Could not find _enhance_dialogue_with_gemini method to patch")
            return False
            
    except Exception as e:
        logger.error(f"Error enhancing dialogue processor: {str(e)}")
        return False