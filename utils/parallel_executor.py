"""
Parallel task execution utility for the video pipeline.
Allows independent pipeline components to execute concurrently.
"""

import os
import logging
import concurrent.futures
from typing import List, Dict, Callable, Any, Tuple, Optional
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineExecutor:
    """
    Executor for parallel processing of pipeline components.
    Manages task dependencies and optimizes execution flow.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_parallel: bool = True):
        """
        Initialize the executor with configuration parameters.
        
        Args:
            max_workers: Maximum number of worker processes/threads
            use_parallel: Whether to use parallel execution (can be disabled)
        """
        # Check environment variables for configuration
        parallel_env = os.environ.get("VIDEO_PIPELINE_PARALLEL", "1")
        self.use_parallel = use_parallel and parallel_env != "0"
        
        # Set default number of workers from environment or parameter
        if max_workers is None:
            workers_env = os.environ.get("VIDEO_PIPELINE_WORKERS")
            if workers_env and workers_env.isdigit():
                max_workers = int(workers_env)
            else:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                max_workers = max(1, min(cpu_count - 1, 4))
        
        self.max_workers = max_workers
        self.executor = None
        self.task_results = {}
        self.task_errors = {}
        
        logger.info(f"Initialized PipelineExecutor with parallel={self.use_parallel}, workers={self.max_workers}")
        
    def execute_tasks(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        Execute a set of tasks with their dependencies.
        
        Args:
            tasks: List of task definitions, each containing:
                  - name: Task identifier
                  - function: Function to execute
                  - args: Arguments for the function
                  - kwargs: Keyword arguments for the function
                  - dependencies: List of task names that must complete first
                  
        Returns:
            Dictionary mapping task names to their results
        """
        if not tasks:
            return {}
            
        # If parallel execution is disabled, run tasks serially
        if not self.use_parallel:
            return self._execute_serial(tasks)
            
        # Use parallel execution
        return self._execute_parallel(tasks)
        
    def _execute_serial(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Execute tasks serially in dependency order."""
        results = {}
        errors = {}
        
        # Convert task list to dictionary for easier lookup
        task_dict = {task["name"]: task for task in tasks}
        completed_tasks = set()
        
        # Track start time for performance logging
        start_time = time.time()
        
        # Process tasks in order, respecting dependencies
        while len(completed_tasks) < len(tasks):
            executed_any = False
            
            for task_name, task in task_dict.items():
                # Skip if already completed
                if task_name in completed_tasks:
                    continue
                    
                # Check if dependencies are satisfied
                dependencies = task.get("dependencies", [])
                if not all(dep in completed_tasks for dep in dependencies):
                    continue
                    
                # Execute task
                try:
                    logger.info(f"Executing task: {task_name}")
                    task_start = time.time()
                    
                    # Get function and arguments
                    func = task["function"]
                    args = task.get("args", [])
                    kwargs = task.get("kwargs", {})
                    
                    # Execute and store result
                    result = func(*args, **kwargs)
                    results[task_name] = result
                    
                    task_end = time.time()
                    logger.info(f"Task {task_name} completed in {task_end - task_start:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error in task {task_name}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    errors[task_name] = {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results[task_name] = None
                    
                completed_tasks.add(task_name)
                executed_any = True
                
            # If no tasks were executed in this iteration, there might be a dependency cycle
            if not executed_any and len(completed_tasks) < len(tasks):
                remaining = set(task_dict.keys()) - completed_tasks
                logger.error(f"Possible dependency cycle detected in tasks: {remaining}")
                break
                
        end_time = time.time()
        logger.info(f"All tasks completed in {end_time - start_time:.2f}s")
        
        self.task_results = results
        self.task_errors = errors
        return results
        
    def _execute_parallel(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Execute tasks in parallel with dependency management."""
        results = {}
        errors = {}
        
        # Convert task list to dictionary for easier lookup
        task_dict = {task["name"]: task for task in tasks}
        completed_tasks = set()
        
        # IMPORTANT: Always use ThreadPoolExecutor for compatibility
        # ProcessPoolExecutor causes pickling errors with imported functions
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Track start time for performance logging
            start_time = time.time()
            
            # Track futures and their corresponding tasks
            futures = {}
            
            # Process tasks in batches, submitting when dependencies are satisfied
            while len(completed_tasks) < len(tasks):
                # Submit ready tasks
                for task_name, task in task_dict.items():
                    # Skip if already submitted or completed
                    if task_name in completed_tasks or task_name in futures:
                        continue
                        
                    # Check if dependencies are satisfied
                    dependencies = task.get("dependencies", [])
                    if not all(dep in completed_tasks for dep in dependencies):
                        continue
                        
                    # Get function and arguments
                    func = task["function"]
                    args = task.get("args", [])
                    kwargs = task.get("kwargs", {})
                    
                    # Submit task to executor
                    logger.info(f"Submitting task: {task_name}")
                    future = executor.submit(func, *args, **kwargs)
                    futures[task_name] = future
                
                # Check for completed futures
                done, not_done = concurrent.futures.wait(
                    list(futures.values()),
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Process completed futures
                for task_name, future in list(futures.items()):
                    if future in done:
                        try:
                            result = future.result()
                            results[task_name] = result
                            logger.info(f"Task {task_name} completed successfully")
                        except Exception as e:
                            logger.error(f"Error in task {task_name}: {str(e)}")
                            logger.debug(traceback.format_exc())
                            errors[task_name] = {
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            }
                            results[task_name] = None
                            
                        completed_tasks.add(task_name)
                        del futures[task_name]
                
                # If all tasks are submitted but not completed, just wait
                if not futures and len(completed_tasks) < len(tasks):
                    remaining = set(task_dict.keys()) - completed_tasks
                    # Check if all remaining tasks have unmet dependencies due to errors
                    all_dependencies_failed = True
                    for task_name in remaining:
                        dependencies = task_dict[task_name].get("dependencies", [])
                        for dep in dependencies:
                            if dep not in completed_tasks and dep not in self.task_errors:
                                all_dependencies_failed = False
                                break
                    
                    if all_dependencies_failed:
                        logger.warning(f"Remaining tasks have dependencies that failed: {remaining}")
                        break
                    else:
                        logger.error(f"Possible dependency issue detected in tasks: {remaining}")
                        break
                
                # If no tasks were submitted or completed, there might be a dependency cycle
                if not futures:
                    remaining = set(task_dict.keys()) - completed_tasks
                    logger.error(f"No tasks can be executed due to dependency issues: {remaining}")
                    break
                    
        end_time = time.time()
        logger.info(f"All tasks completed in {end_time - start_time:.2f}s")
        
        self.task_results = results
        self.task_errors = errors
        return results
        
    def get_result(self, task_name: str) -> Any:
        """Get the result of a specific task."""
        return self.task_results.get(task_name)
        
    def get_error(self, task_name: str) -> Dict:
        """Get error information for a specific task."""
        return self.task_errors.get(task_name)
        
    def has_error(self, task_name: str) -> bool:
        """Check if a task had an error."""
        return task_name in self.task_errors