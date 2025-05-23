"""Utility functions and modules for the video analysis pipeline.

This package contains various utility functions and modules used by the pipeline,
including parallel processing utilities, result caching, and pipeline enhancements.
"""

# YOLOv5 compatibility fix
# YOLOv5 looks for a 'TryExcept' function in 'utils', but we're using the same module name.
# This ensures that when YOLOv5 tries to import it, it will find our compatibility implementation.
class TryExcept:
    """
    Compatibility class for YOLOv5.
    Acts as a context manager to handle try-except blocks with a fallback value.
    """
    def __init__(self, *args, **kwargs):
        pass
        
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True  # Suppress exceptions