#!/usr/bin/env python3
"""
Wrapper script to run the video analysis pipeline with the correct CUDA/NCCL settings.
This script sets necessary environment variables before importing any PyTorch-related modules
to avoid NCCL symbol errors while still using GPU acceleration.
"""

import os
import sys
import subprocess
import multiprocessing

# Set environment variables to fix NCCL issues while still using GPU
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer operations
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match device IDs to PCI bus order

# Optimize memory usage and GPU handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Limit memory splits
os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # Global symbol visibility

# Disable parallel processing to use the standard pipeline
os.environ["VIDEO_PIPELINE_PARALLEL"] = "0"  # Disable parallel component execution

# Set default number of parallel workers based on CPU cores
cpu_count = multiprocessing.cpu_count()
default_workers = max(1, min(cpu_count - 1, 4))  # Use N-1 cores up to 4 workers
os.environ["VIDEO_PIPELINE_WORKERS"] = str(default_workers)

def main():
    """Run the video analysis pipeline with the proper environment settings."""
    print("Starting video analysis pipeline with optimized CUDA settings...")
    
    # Import main only after environment variables are set
    from main import main as run_pipeline
    
    # Import and apply pipeline enhancements
    try:
        # Ensure current directory is in path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # First load utils.__init__ separately to ensure YOLOv5 compatibility
        try:
            import utils
        except ImportError as e:
            print(f"Warning: Could not import utils package: {e}")
        
        # Apply direct dialogue enhancement integration
        dialogue_integration_success = False
        try:
            direct_integration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_dialogue_integration.py")
            if os.path.exists(direct_integration_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("direct_dialogue_integration", direct_integration_path)
                direct_integration = importlib.util.module_from_spec(spec)
                sys.modules["direct_dialogue_integration"] = direct_integration
                spec.loader.exec_module(direct_integration)
                
                # Apply the direct integration
                dialogue_integration_success = direct_integration.integrate_full_context_dialogue_enhancement()
                if dialogue_integration_success:
                    print("Successfully integrated full context dialogue enhancement")
        except Exception as e:
            print(f"Error during direct dialogue integration: {e}")
            print("Using original pipeline without enhancements...")
        
        # Skip the main_parallel and pipeline enhancement
        print("Using original sequential pipeline from main.py")
            
    except Exception as e:
        print(f"Error setting up enhancements: {e}")
        print("Continuing with original pipeline")
    
    # Run the main function from main.py
    run_pipeline()

if __name__ == "__main__":
    # First check if we can import torch successfully
    try:
        # Import here to test if it works with our environment settings
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        main()
    except ImportError as e:
        print(f"Error importing PyTorch: {e}")
        print("Attempting to run in CPU-only mode...")
        
        # Set CPU-only mode and try again
        os.environ["FORCE_CPU"] = "1"
        try:
            from main import main as run_pipeline
            run_pipeline()
        except Exception as e:
            print(f"Failed to run in CPU-only mode: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1) 