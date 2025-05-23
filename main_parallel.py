"""
Enhanced video analysis pipeline with parallel processing and improved resource management.
This patch adds parallel component execution capability to the original pipeline.
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import traceback
import time
# Import utility modules
from utils.parallel_executor import PipelineExecutor
from utils.result_manager import ResultManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_parallel_pipeline(config: Dict) -> Tuple[PipelineExecutor, ResultManager]:
    """
    Configure parallel processing components based on configuration.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Tuple of (PipelineExecutor, ResultManager)
    """
    # Get parallel processing configuration
    parallel_config = config.get("parallel_processing", {})
    parallel_enabled = parallel_config.get("enabled", True)
    max_workers = parallel_config.get("max_workers", 4)
    
    # Get result caching configuration
    cache_enabled = parallel_config.get("use_result_cache", True)
    cache_dir = parallel_config.get("cache_dir", ".cache")
    cache_ttl = parallel_config.get("cache_ttl", 86400)
    
    # Initialize components
    executor = PipelineExecutor(max_workers=max_workers, use_parallel=parallel_enabled)
    result_manager = ResultManager(cache_dir=cache_dir, use_cache=cache_enabled, cache_ttl=cache_ttl)
    
    logger.info(f"Configured parallel pipeline with parallel={parallel_enabled}, workers={max_workers}, cache={cache_enabled}")
    return executor, result_manager

# Component wrapper functions for parallel execution
# IMPORTANT: These functions must be defined in this module for pickling to work correctly

def run_scene_detection(scene_detector, video_path: str, output_dir: Optional[str] = None) -> List[Dict]:
    """Wrapper for scene detection to use in parallel execution."""
    logger.info(f"Running scene detection on {video_path}")
    start_time = time.time()
    try:
        scenes = scene_detector.detect_scenes(video_path, output_dir)
        end_time = time.time()
        logger.info(f"Scene detection completed in {end_time - start_time:.2f}s")
        return scenes
    except Exception as e:
        logger.error(f"Error in scene detection: {str(e)}")
        raise

def run_keyframe_extraction(keyframe_extractor, video_path: str, scenes: List[Dict], output_dir: Optional[str] = None) -> Dict:
    """Wrapper for keyframe extraction to use in parallel execution."""
    logger.info(f"Running keyframe extraction on {len(scenes)} scenes")
    start_time = time.time()
    try:
        # First, extract the keyframes
        keyframes = keyframe_extractor.extract_keyframes(video_path, scenes, output_dir)
        
        # Then, save the keyframes and get the metadata with paths
        if output_dir:
            keyframes_metadata = keyframe_extractor.save_keyframes(keyframes, output_dir)
            end_time = time.time()
            logger.info(f"Keyframe extraction completed in {end_time - start_time:.2f}s")
            return keyframes_metadata
        else:
            logger.warning("No output directory provided for keyframe extraction, paths will not be available")
            end_time = time.time()
            logger.info(f"Keyframe extraction completed in {end_time - start_time:.2f}s")
            return {"scenes": keyframes}  # Wrap in format compatible with metadata structure
    except Exception as e:
        logger.error(f"Error in keyframe extraction: {str(e)}")
        raise

def run_audio_processing(audio_processor, video_path: str, scenes: List[Dict], output_dir: str) -> Dict:
    """Wrapper for audio processing to use in parallel execution."""
    logger.info(f"Running audio processing on {video_path}")
    start_time = time.time()
    try:
        audio_results = audio_processor.process_video_audio(video_path, scenes, output_dir)
        end_time = time.time()
        logger.info(f"Audio processing completed in {end_time - start_time:.2f}s")
        return audio_results
    except Exception as e:
        logger.error(f"Error in audio processing: {str(e)}")
        logger.warning("Creating dummy audio results to continue pipeline")
        
        # Create a minimal audio_results to continue the pipeline
        audio_results = {"scenes": {}}
        if scenes:
            for i, scene in enumerate(scenes):
                audio_results["scenes"][str(i)] = {
                    "scene_info": {
                        "start_time": scene.get("start_time", 0),
                        "end_time": scene.get("end_time", 0),
                        "duration": scene.get("duration", 0) if "duration" in scene else scene.get("end_time", 0) - scene.get("start_time", 0)
                    },
                    "dialogue": [],
                    "dialogue_count": 0
                }
        
        # Save the dummy results
        try:
            import os
            import json
            from pathlib import Path
            
            audio_output_dir = Path(output_dir)
            audio_output_dir.mkdir(exist_ok=True)
            
            with open(audio_output_dir / "audio_results.json", 'w', encoding='utf-8') as f:
                json.dump(audio_results, f, indent=2, ensure_ascii=False)
            logger.info("Saved dummy audio results and continuing pipeline")
        except Exception as e2:
            logger.error(f"Error saving dummy audio results: {str(e2)}")
            
        return audio_results

def run_entity_detection(entity_detector, keyframes: Dict, keyframes_dir: str) -> Dict:
    """Wrapper for entity detection to use in parallel execution."""
    logger.info(f"Running entity detection")
    start_time = time.time()
    try:
        # Ensure the output directory for entities exists
        import os
        from pathlib import Path
        entities_dir = Path(os.path.dirname(keyframes_dir)) / "entities"
        entities_dir.mkdir(exist_ok=True)
        
        # Verify the keyframes directory exists
        keyframes_path = Path(keyframes_dir)
        if not keyframes_path.exists():
            raise FileNotFoundError(f"Keyframes directory not found: {keyframes_path}")
            
        # Run entity detection
        entity_results = entity_detector.detect_entities(keyframes, keyframes_dir)
        
        # Save the results to the entities directory
        import json
        with open(entities_dir / "entities.json", 'w', encoding='utf-8') as f:
            json.dump(entity_results, f, indent=2, ensure_ascii=False)
            
        end_time = time.time()
        logger.info(f"Entity detection completed in {end_time - start_time:.2f}s")
        return entity_results
    except Exception as e:
        logger.error(f"Error in entity detection: {str(e)}")
        raise

def run_gemini_analysis(gemini_analyzer, scenes_data, keyframes_metadata, entities_data, audio_results, output_dir: str, scene_videos=None) -> Dict:
    """Wrapper for Gemini Vision scene analysis."""
    logger.info("Running Gemini Vision analysis")
    start_time = time.time()
    
    try:
        import json
        from pathlib import Path
        
        # Create output directory
        gemini_output_dir = Path(output_dir)
        gemini_output_dir.mkdir(exist_ok=True)
        
        # Check if we have all required data
        if not keyframes_metadata or "scenes" not in keyframes_metadata:
            logger.error("Missing keyframes metadata for Gemini analysis")
            raise ValueError("Keyframes metadata required for Gemini analysis")
            
        # Prepare output
        gemini_output = {"scenes": {}}
        
        # Process each scene
        keyframes_dir = str(Path(output_dir).parent / "keyframes")
        
        # Ensure scene_videos is a dict
        if scene_videos is None:
            scene_videos = {}
            
        # Process each scene with available data
        for scene_idx, scene_info in keyframes_metadata["scenes"].items():
            logger.info(f"Analyzing scene {scene_idx} with Gemini")
            
            # Extract entity and audio data for this scene
            scene_entity_data = entities_data.get("scenes", {}).get(scene_idx, [])
            scene_audio_data = audio_results.get("scenes", {}).get(scene_idx, {})
            
            # Get video path for this scene if available
            scene_video_path = None
            if scene_videos and int(scene_idx) in scene_videos:
                scene_video_path = scene_videos[int(scene_idx)]
                
            # Use Gemini analyzer
            if gemini_analyzer is not None:
                try:
                    scene_result = gemini_analyzer.analyze_scene(
                        int(scene_idx),
                        scene_info,
                        keyframes_dir,
                        scene_video_path=scene_video_path,
                        entity_data=scene_entity_data,
                        audio_data=scene_audio_data
                    )
                    gemini_output["scenes"][scene_idx] = scene_result
                except Exception as scene_error:
                    logger.error(f"Error analyzing scene {scene_idx}: {str(scene_error)}")
                    # Create placeholder for failed scene
                    gemini_output["scenes"][scene_idx] = {
                        "scene_number": int(scene_idx),
                        "start_time": scene_info.get("start_time", 0),
                        "end_time": scene_info.get("end_time", 0),
                        "description": f"Scene analysis failed: {str(scene_error)}",
                        "entities": [],
                        "actions": [],
                        "settings": []
                    }
            else:
                # Create placeholder for scene when analyzer is not available
                gemini_output["scenes"][scene_idx] = {
                    "scene_number": int(scene_idx),
                    "start_time": scene_info.get("start_time", 0),
                    "end_time": scene_info.get("end_time", 0),
                    "description": "Scene analysis skipped - Gemini analyzer not available",
                    "entities": [],
                    "actions": [],
                    "settings": []
                }
                
        # Save results to JSON
        with open(gemini_output_dir / "scene_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(gemini_output, f, indent=2, ensure_ascii=False)
            
        end_time = time.time()
        logger.info(f"Gemini Vision analysis completed in {end_time - start_time:.2f}s")
        return gemini_output
        
    except Exception as e:
        logger.error(f"Error in Gemini Vision analysis: {str(e)}")
        # Return empty results to continue pipeline
        return {"scenes": {}}

def run_extract_scene_videos(video_path: str, scenes: List[Dict], output_dir: str) -> Dict:
    """Extract scene video segments for Gemini analysis."""
    logger.info(f"Extracting scene video segments from {video_path}")
    start_time = time.time()
    
    try:
        import os
        import subprocess
        from pathlib import Path
        
        # Create output directory
        scene_videos_dir = Path(output_dir)
        scene_videos_dir.mkdir(exist_ok=True)
        
        # Process each scene
        scene_videos = {}
        
        # Only process a subset of scenes to save time and space
        # Choose scenes evenly distributed throughout the video
        if len(scenes) > 10:
            # Process at most 10 representative scenes
            scene_indices = [int(i * len(scenes) / 10) for i in range(10)]
        else:
            # Process all scenes if less than 10
            scene_indices = range(len(scenes))
            
        # Extract scene videos
        for idx in scene_indices:
            scene = scenes[idx]
            scene_idx = idx
            
            # Get scene timestamps
            start_time_sec = scene.get("start_time", 0)
            end_time_sec = scene.get("end_time", 0)
            duration = end_time_sec - start_time_sec
            
            # Skip very short scenes
            if duration < 0.5:
                logger.info(f"Skipping scene {scene_idx} (too short: {duration:.2f}s)")
                continue
                
            # Limit scene duration to 10 seconds for efficiency
            if duration > 10:
                # Take the middle 10 seconds of longer scenes
                mid_point = (start_time_sec + end_time_sec) / 2
                start_time_sec = max(0, mid_point - 5)
                end_time_sec = start_time_sec + 10
                logger.info(f"Limiting scene {scene_idx} duration to 10s (was {duration:.2f}s)")
                
            # Format start time for ffmpeg
            h = int(start_time_sec / 3600)
            m = int((start_time_sec % 3600) / 60)
            s = start_time_sec % 60
            start_time_str = f"{h:02d}:{m:02d}:{s:06.3f}"
            
            # Output path
            output_path = scene_videos_dir / f"scene_{scene_idx:03d}.mp4"
            
            # Extract scene with ffmpeg
            try:
                cmd = [
                    "ffmpeg", "-y", "-ss", start_time_str, 
                    "-i", video_path, "-t", str(duration),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-an", str(output_path)
                ]
                
                process = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Add to scene videos dictionary
                if output_path.exists():
                    scene_videos[scene_idx] = str(output_path)
                    logger.info(f"Extracted scene {scene_idx} to {output_path}")
                else:
                    logger.warning(f"Failed to extract scene {scene_idx} (file not created)")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Error extracting scene {scene_idx}: {e.stderr.decode() if e.stderr else str(e)}")
            except Exception as e:
                logger.error(f"Error extracting scene {scene_idx}: {str(e)}")
                
        end_time = time.time()
        logger.info(f"Scene video extraction completed in {end_time - start_time:.2f}s")
        return scene_videos
        
    except Exception as e:
        logger.error(f"Error extracting scene videos: {str(e)}")
        return {}

def process_video_parallel(pipeline, video_path: Path, output_dir: Path, skip_steps: List[str] = None) -> Dict:
    """
    Enhanced parallel processing version of the process_video method.
    
    Args:
        pipeline: The VideoPipeline instance
        video_path: Path to the video file
        output_dir: Output directory for results
        skip_steps: List of steps to skip
    
    Returns:
        Dictionary of results
    """
    # Add start time for tracking processing duration
    start_time = time.time()
    
    # Initialize skip_steps if None
    if skip_steps is None:
        skip_steps = []
        
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure parallel execution components
    executor, result_manager = configure_parallel_pipeline(pipeline.config)
    
    # Define the task list with dependencies
    tasks = []
    
    # Check if we can use cached results
    video_path_str = str(video_path)
    cache_hit = False
    
    if "scene_detection" not in skip_steps and pipeline.scene_detector is not None:
        # Try to get scenes from cache
        cache_hit, scenes = result_manager.load_result("scene_detection", video_path_str)
        if not cache_hit:
            # Add scene detection task
            tasks.append({
                "name": "scene_detection",
                "function": run_scene_detection,
                "args": [pipeline.scene_detector, video_path_str, str(output_dir / "scenes")],
                "dependencies": []
            })
    else:
        # If scene detection is skipped or not available, we can't proceed
        logger.error("Scene detection is required but either skipped or unavailable")
        raise RuntimeError("Scene detection is required but either skipped or unavailable")
    
    if "keyframe_extraction" not in skip_steps and pipeline.keyframe_extractor is not None:
        # Add keyframe extraction task (depends on scene detection)
        tasks.append({
            "name": "keyframe_extraction",
            "function": run_keyframe_extraction,
            "args": [pipeline.keyframe_extractor, video_path_str, None, str(output_dir / "keyframes")],
            "kwargs": {},
            "dependencies": ["scene_detection"]
        })
    
    # Audio processing can run in parallel with keyframe extraction
    if "audio_processing" not in skip_steps and pipeline.audio_processor is not None:
        tasks.append({
            "name": "audio_processing",
            "function": run_audio_processing,
            "args": [pipeline.audio_processor, video_path_str, None, str(output_dir / "audio")],
            "dependencies": ["scene_detection"]
        })
    
    # Don't add entity detection task yet - it will be added after keyframe extraction completes
    # as it requires the keyframes_metadata from the keyframe extraction step
    
    # Execute tasks and collect results
    if cache_hit:
        # If scene detection was cached, we need to adjust the task dependencies
        for task in tasks:
            if "scene_detection" in task["dependencies"]:
                task["dependencies"].remove("scene_detection")
                
        # Add scenes to results directly
        results = {"scene_detection": scenes}
        
        # Execute remaining tasks
        if tasks:
            # Update task arguments with cached scenes
            for task in tasks:
                if task["name"] in ["keyframe_extraction", "audio_processing"]:
                    # Add scenes as the second argument
                    task["args"][2] = scenes
                    
            # Execute tasks in the correct dependency order
            task_results = executor.execute_tasks(tasks)
            results.update(task_results)
            
            # Now add entity detection task if needed
            if "entity_detection" not in skip_steps and pipeline.entity_detector is not None:
                # Check if we have keyframe_extraction results
                if "keyframe_extraction" in results and results["keyframe_extraction"]:
                    logger.info("Adding entity detection task after keyframe extraction completed")
                    entity_tasks = [{
                        "name": "entity_detection",
                        "function": run_entity_detection,
                        "args": [pipeline.entity_detector, results["keyframe_extraction"], str(output_dir / "keyframes")],
                        "dependencies": []
                    }]
                    
                    # Execute entity detection
                    entity_results = executor.execute_tasks(entity_tasks)
                    results.update(entity_results)
            
            # Continue with audio processing (if not done in parallel already) and scene video extraction
            scene_videos = {}
            if "gemini_vision" not in skip_steps and pipeline.scene_detector is not None and "scene_detection" in results:
                scenes = results["scene_detection"]
                # Extract scene videos if needed for Gemini analysis
                if pipeline.config.get("gemini_vision", {}).get("process_scene_videos", True):
                    logger.info("Extracting scene videos for Gemini Vision analysis")
                    scene_videos = run_extract_scene_videos(str(video_path), scenes, str(output_dir / "scene_videos"))
                    
            # Run Gemini Vision analysis as the final step
            if "gemini_vision" not in skip_steps and pipeline.gemini_analyzer is not None:
                # Make sure we have keyframes and at least entity or audio data
                if "keyframe_extraction" in results and "scene_detection" in results:
                    logger.info("Running Gemini Vision analysis")
                    
                    # Get entity results (if available)
                    entities_data = results.get("entity_detection", {})
                    
                    # Get audio results (if available)
                    audio_results = results.get("audio_processing", {})
                    
                    # Run Gemini analysis
                    gemini_results = run_gemini_analysis(
                        pipeline.gemini_analyzer,
                        results["scene_detection"],
                        results["keyframe_extraction"],
                        entities_data,
                        audio_results,
                        str(output_dir / "gemini"),
                        scene_videos
                    )
                    
                    # Add to results
                    results["gemini_vision"] = gemini_results
                    
            # Generate final report and combined results
            combined_results = {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat() if 'datetime' in globals() else time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pipeline_type": "parallel"
            }
            
            # Add all collected results
            combined_results.update(results)
            
            # Add scene videos if available
            if scene_videos:
                combined_results["scene_videos"] = scene_videos
                
            # Save combined results
            try:
                import json
                with open(output_dir / "full_structured_analysis.json", 'w', encoding='utf-8') as f:
                    json.dump(combined_results, f, indent=2, ensure_ascii=False)
                logger.info("Saved full structured analysis to full_structured_analysis.json")
            except Exception as e:
                logger.error(f"Error saving structured analysis: {str(e)}")
    else:
        # Execute initial tasks (scene detection, keyframe extraction, audio processing)
        results = executor.execute_tasks(tasks)
        
        # Cache the scene detection results for future use
        if "scene_detection" in results and results["scene_detection"]:
            result_manager.save_result(
                "scene_detection", 
                video_path_str, 
                results["scene_detection"]
            )
            
        # Now add entity detection task if needed
        if "entity_detection" not in skip_steps and pipeline.entity_detector is not None:
            # Check if we have keyframe_extraction results
            if "keyframe_extraction" in results and results["keyframe_extraction"]:
                logger.info("Adding entity detection task after keyframe extraction completed")
                entity_tasks = [{
                    "name": "entity_detection",
                    "function": run_entity_detection,
                    "args": [pipeline.entity_detector, results["keyframe_extraction"], str(output_dir / "keyframes")],
                    "dependencies": []
                }]
                
                # Execute entity detection
                entity_results = executor.execute_tasks(entity_tasks)
                results.update(entity_results)
                
        # Continue with audio processing (if not done in parallel already) and scene video extraction
        scene_videos = {}
        if "gemini_vision" not in skip_steps and pipeline.scene_detector is not None and "scene_detection" in results:
            scenes = results["scene_detection"]
            # Extract scene videos if needed for Gemini analysis
            if pipeline.config.get("gemini_vision", {}).get("process_scene_videos", True):
                logger.info("Extracting scene videos for Gemini Vision analysis")
                scene_videos = run_extract_scene_videos(str(video_path), scenes, str(output_dir / "scene_videos"))
                
        # Run Gemini Vision analysis as the final step
        if "gemini_vision" not in skip_steps and pipeline.gemini_analyzer is not None:
            # Make sure we have keyframes and at least entity or audio data
            if "keyframe_extraction" in results and "scene_detection" in results:
                logger.info("Running Gemini Vision analysis")
                
                # Get entity results (if available)
                entities_data = results.get("entity_detection", {})
                
                # Get audio results (if available)
                audio_results = results.get("audio_processing", {})
                
                # Run Gemini analysis
                gemini_results = run_gemini_analysis(
                    pipeline.gemini_analyzer,
                    results["scene_detection"],
                    results["keyframe_extraction"],
                    entities_data,
                    audio_results,
                    str(output_dir / "gemini"),
                    scene_videos
                )
                
                # Add to results
                results["gemini_vision"] = gemini_results
            
        # If we have scene detection results but other tasks failed due to dependency,
        # retry with explicit scene data
        if "scene_detection" in results and results["scene_detection"]:
            scenes = results["scene_detection"]
            
            # Get non-completed tasks
            completed_tasks = set(results.keys())
            pending_tasks = []
            
            for task in tasks:
                task_name = task["name"]
                if task_name not in completed_tasks and task_name != "scene_detection":
                    # Create a new task with updated dependencies
                    new_task = task.copy()
                    if "scene_detection" in new_task.get("dependencies", []):
                        new_task["dependencies"].remove("scene_detection")
                    
                    # Update args with scene data for tasks that need it
                    if task_name in ["keyframe_extraction", "audio_processing"]:
                        args = list(new_task["args"])
                        args[2] = scenes  # Set scenes as third argument
                        new_task["args"] = args
                        
                    pending_tasks.append(new_task)
            
            # If we have pending tasks, execute them
            if pending_tasks:
                logger.info(f"Retrying {len(pending_tasks)} tasks with explicit scene data")
                retry_results = executor.execute_tasks(pending_tasks)
                results.update(retry_results)
    
    # Generate final report and combined results
    combined_results = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "processing_time": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
        "pipeline_type": "parallel",
        "parallel_execution": executor.use_parallel,
        "workers": executor.max_workers,
        "cache_enabled": result_manager.use_cache,
        "cache_hit": cache_hit,
    }
    
    # Add all collected results
    combined_results.update(results)
    
    # Add scene videos if available
    if scene_videos:
        combined_results["scene_videos"] = scene_videos
        
    # Save combined results
    try:
        with open(output_dir / "full_structured_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved full structured analysis to full_structured_analysis.json")
    except Exception as e:
        logger.error(f"Error saving full analysis: {str(e)}")
        
    # Also save pipeline stats for debugging
    pipeline_stats = {
        "parallel_execution": executor.use_parallel,
        "workers": executor.max_workers,
        "cache_enabled": result_manager.use_cache,
        "cache_hit": cache_hit,
        "execution_times": {
            component: f"{time.time() - start_time:.2f}s" for component, result_value in results.items()
        }
    }
    
    # Save stats to output dir
    with open(output_dir / "pipeline_stats.json", "w") as f:
        json.dump(pipeline_stats, f, indent=2)
    
    # Optional: Generate a markdown report
    try:
        generate_markdown_report(combined_results, output_dir)
    except Exception as e:
        logger.error(f"Error generating markdown report: {str(e)}")
    
    return combined_results

def generate_markdown_report(results: Dict, output_dir: Path):
    """Generate a markdown report from the results."""
    report_path = output_dir / "analysis_report.md"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            # Write report header
            f.write(f"# Video Analysis Report\n\n")
            f.write(f"Analysis completed: {results.get('timestamp', datetime.now().isoformat())}\n\n")
            
            # Video information
            f.write(f"## Video Information\n\n")
            f.write(f"- File: `{results.get('video_path', 'Unknown')}`\n")
            
            # Scene summary
            scenes = results.get("scene_detection", [])
            if scenes:
                f.write(f"\n## Scene Summary\n\n")
                f.write(f"Total scenes detected: {len(scenes)}\n\n")
                
                # Add scene table
                f.write("| Scene | Start Time | End Time | Duration |\n")
                f.write("|-------|------------|----------|----------|\n")
                
                for i, scene in enumerate(scenes):
                    start = scene.get("start_time", 0)
                    end = scene.get("end_time", 0)
                    duration = end - start
                    
                    # Format times as MM:SS
                    start_fmt = f"{int(start // 60):02d}:{int(start % 60):02d}"
                    end_fmt = f"{int(end // 60):02d}:{int(end % 60):02d}"
                    duration_fmt = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
                    
                    f.write(f"| {i} | {start_fmt} | {end_fmt} | {duration_fmt} |\n")
                
                # Get Gemini scene descriptions
                gemini_results = results.get("gemini_vision", {})
                
                # If gemini_vision is not in main results, try loading from the scene_analysis.json file
                if not gemini_results or "scenes" not in gemini_results:
                    gemini_file = output_dir / "gemini" / "scene_analysis.json"
                    if gemini_file.exists():
                        try:
                            with open(gemini_file, 'r', encoding='utf-8') as gemini_f:
                                gemini_results = json.load(gemini_f)
                                logger.info(f"Loaded Gemini scene descriptions from {gemini_file}")
                        except Exception as e:
                            logger.error(f"Error loading Gemini scene descriptions: {str(e)}")
                
                # Add scene descriptions if available
                if gemini_results and "scenes" in gemini_results:
                    f.write(f"\n## Scene Descriptions\n\n")
                    
                    # Sort scenes by index to display in order
                    scene_indices = sorted([int(idx) for idx in gemini_results["scenes"].keys()])
                    
                    for idx in scene_indices:
                        scene_idx_str = str(idx)
                        scene_data = gemini_results["scenes"].get(scene_idx_str)
                        
                        if scene_data and "description" in scene_data:
                            # Add scene header
                            f.write(f"### Scene {idx}\n\n")
                            
                            # Get timestamps if available
                            start = 0
                            end = 0
                            if idx < len(scenes):
                                scene = scenes[idx]
                                start = scene.get("start_time", 0)
                                end = scene.get("end_time", 0)
                                    
                            start_fmt = f"{int(start // 60):02d}:{int(start % 60):02d}"
                            end_fmt = f"{int(end // 60):02d}:{int(end % 60):02d}"
                            
                            f.write(f"**Time:** {start_fmt} - {end_fmt}\n\n")
                            
                            # Add condensed description
                            description = scene_data["description"]
                            
                            # Split by paragraphs 
                            paragraphs = description.split("\n\n")
                            
                            # Take the first few paragraphs (up to 5) to get a more complete description
                            # but limit total length to 2000 characters
                            short_desc = "\n\n".join(paragraphs[:min(5, len(paragraphs))])
                            if len(short_desc) > 2000:
                                short_desc = short_desc[:2000] + "..."
                                
                            # Note: The scene numbers in descriptions appear to be off by one
                            f.write(f"{short_desc}\n\n")
            
            # Add information about entities if available
            entities = results.get("entity_detection", {})
            if entities:
                f.write(f"\n## Entity Detection Summary\n\n")
                
                # Count entity types
                entity_counts = {}
                for scene_idx, scene_entities in entities.get("scenes", {}).items():
                    for entity in scene_entities:
                        entity_type = entity.get("type", "unknown")
                        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                # Write entity type counts
                if entity_counts:
                    f.write("### Entity Types\n\n")
                    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- {entity_type}: {count}\n")
            
        logger.info(f"Generated markdown report at {report_path}")
    except Exception as e:
        logger.error(f"Error writing markdown report: {str(e)}")

def enhance_dialogue_processor(scene_processor_result):
    """
    Enhance the dialogue with speaker names based on scene context.
    
    Args:
        scene_processor_result: Result from scene processor
        
    Returns:
        Enhanced result with proper speaker names
    """
    try:
        # Get the output directory
        output_dir = scene_processor_result.get("output_dir")
        
        if not output_dir:
            logger.error("No output directory found for dialogue enhancement")
            return scene_processor_result
            
        # Load the structured data
        structured_file = os.path.join(output_dir, "structured_analysis.json")
        
        if not os.path.exists(structured_file):
            logger.error(f"Structured analysis file not found at {structured_file}")
            return scene_processor_result
        
        # Enhance the dialogue
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        
        from full_context_dialogue_enhancer import enhance_dialogue_with_full_context
        # Pass the API key to ensure we use the right one
        enhanced_file = enhance_dialogue_with_full_context(
            structured_file, 
            api_key=api_key,
            output_dir=output_dir
        )
        
        if enhanced_file:
            logger.info(f"Dialogue enhanced successfully, saved to {enhanced_file}")
            # Update the result with the enhanced file
            scene_processor_result["structured_analysis_json"] = enhanced_file
        else:
            logger.warning("Dialogue enhancement failed")
            
        return scene_processor_result
        
    except Exception as e:
        logger.error(f"Error in dialogue enhancement: {str(e)}")
        traceback.print_exc()
        return scene_processor_result