"""Main entry point for the video analysis pipeline."""

import argparse
import logging
import os
import sys
from pathlib import Path
import time
import json
from typing import Dict, Optional, List
import gc
from datetime import datetime
import traceback
import cv2
import re
import ffmpeg
import copy
import numpy as np

# Set environment variables to handle NCCL issues before importing PyTorch
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer which can cause symbol errors
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match device IDs to PCI bus order
# Add optimization settings for GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Limit memory splits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('video_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Check for CUDA availability
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available, using CPU. This will be much slower.")
except ImportError:
    logger.warning("PyTorch not installed, cannot check CUDA availability.")
    cuda_available = False

# Import pipeline components
try:
    from config import load_config
    from components.scene_detection import SceneDetector
    from components.keyframe_extraction import KeyframeExtractor
    from components.entity_detection import EntityDetector
    from components.audio_processing import AudioProcessor
    from components.gemini_vision import GeminiVisionAnalyzer
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure the package is installed or in your PYTHONPATH")
    sys.exit(1)

# Custom JSON Encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class VideoPipeline:
    """
    Main pipeline class that orchestrates the video analysis process.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Optional path to a configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Force CUDA if available and not explicitly disabled
        if cuda_available and self.config.get("device", "cuda") != "cpu":
            self.config["device"] = "cuda"
            logger.info("Using CUDA for processing")
        else:
            logger.info("Using CPU for processing")
            self.config["device"] = "cpu"
            
        logger.info("Initialized video analysis pipeline")
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        # Track failed components for skipping
        self.failed_components = []
        
        # Scene detection
        try:
            self.scene_detector = SceneDetector(self.config.get("scene_detection"))
        except Exception as e:
            logger.error(f"Error initializing Scene Detector: {str(e)}")
            self.failed_components.append("scene_detection")
            self.scene_detector = None
            
        # Keyframe extraction
        try:
            self.keyframe_extractor = KeyframeExtractor(self.config.get("keyframe_extraction"))
        except Exception as e:
            logger.error(f"Error initializing Keyframe Extractor: {str(e)}")
            self.failed_components.append("keyframe_extraction")
            self.keyframe_extractor = None
            
        # Entity detection
        try:
            entity_config = self.config.get("entity_detection", {})
            
            # Debug entity detection config
            logger.info(f"Entity detection config before initialization:")
            for key, value in entity_config.items():
                logger.info(f"  {key}: {value}")
                
            # Ensure paths exist and are valid
            if entity_config.get("detector", "").lower() == "grounding_sam":
                required_paths = [
                    "grounding_dino_config_path",
                    "grounding_dino_checkpoint_path", 
                    "sam_checkpoint_path"
                ]
                
                # Check for missing paths
                missing_paths = [path for path in required_paths if path not in entity_config]
                if missing_paths:
                    logger.error(f"Missing required paths for GroundingSAM: {', '.join(missing_paths)}")
                    raise ValueError(f"Missing required paths for GroundingSAM: {', '.join(missing_paths)}")
                
                # Check if paths exist
                for path_key in required_paths:
                    path = entity_config[path_key]
                    if not Path(path).exists():
                        logger.error(f"Path does not exist: {path}")
                        raise ValueError(f"Path does not exist: {path}")
            
            # Initialize detector
            self.entity_detector = EntityDetector(entity_config)
            
        except Exception as e:
            logger.error(f"Error initializing Entity Detector: {str(e)}")
            self.failed_components.append("entity_detection")
            self.entity_detector = None
            
        # Audio processing
        try:
            self.audio_processor = AudioProcessor(self.config.get("audio_processing"))
        except Exception as e:
            logger.error(f"Error initializing Audio Processor: {str(e)}")
            self.failed_components.append("audio_processing")
            self.audio_processor = None
            
        # Gemini Vision
        try:
            self.gemini_analyzer = GeminiVisionAnalyzer(self.config.get("gemini_vision"))
        except Exception as e:
            logger.error(f"Error initializing Gemini Vision Analyzer: {str(e)}")
            self.failed_components.append("gemini_vision")
            self.gemini_analyzer = None
            
        if not self.failed_components:
            logger.info("All pipeline components initialized successfully")
        else:
            logger.warning(f"Some components failed to initialize and will be skipped: {', '.join(self.failed_components)}")
    
    def process_video(self, video_path: Path, output_dir: Path, skip_steps: List[str] = None) -> Dict:
        """
        Process a video file through the entire pipeline.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files
            skip_steps: Optional list of steps to skip
            
        Returns:
            Dictionary with processing results
        """
        # Combine user-provided skip_steps with components that failed to initialize
        if skip_steps is None:
            skip_steps = []
        skip_steps = list(set(skip_steps + self.failed_components))
        
        # Log which steps will be skipped
        if skip_steps:
            logger.info(f"Skipping pipeline steps: {', '.join(skip_steps)}")
            
        # Initialize results dictionary
        results = {}
        
        # Initialize variables that might be referenced in error handling
        audio_results = None
        scenes = []
        keyframes_metadata = {}
        entities = {}
        gemini_output = {}
        scene_videos = {}
        
        # Store the video path for future use
        self.video_path = str(video_path)
        
        # Make sure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Step 1: Scene Detection
            if not skip_steps or "scene_detection" not in skip_steps:
                logger.info("Step 1: Scene Detection")
                scene_output_dir = output_dir / "scenes"
                scene_output_dir.mkdir(exist_ok=True)
                
                if self.scene_detector is not None:
                    scenes = self.scene_detector.detect_scenes(str(video_path), str(scene_output_dir))
                    logger.info(f"Detected {len(scenes)} scenes")
                else:
                    scenes = []
                    logger.error("Scene detection failed - no scene detector available")
                    raise RuntimeError("Scene detection is required but detector failed to initialize")
                
                # Save scenes to JSON
                with open(scene_output_dir / "scenes.json", 'w', encoding='utf-8') as f:
                    json.dump(scenes, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved scene detection results to {scene_output_dir}")
            else:
                logger.info("Skipping scene detection")
                # Try to load existing scenes from JSON
                try:
                    scene_output_dir = output_dir / "scenes"
                    with open(scene_output_dir / "scenes.json", 'r') as f:
                        scenes = json.load(f)
                        logger.info(f"Loaded {len(scenes)} scenes from existing data")
                except (FileNotFoundError, json.JSONDecodeError):
                    logger.warning("Could not load existing scenes, creating a dummy scene")
                    # Create a dummy scene spanning the entire video
                    try:
                        cap = cv2.VideoCapture(str(video_path))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = frame_count / fps
                            scenes = [{
                                "start_time": 0,
                                "end_time": duration,
                                "duration": duration
                            }]
                            cap.release()
                        else:
                            raise ValueError("Could not open video file")
                    except Exception as e:
                        logger.error(f"Error creating dummy scene: {str(e)}")
                        scenes = [{"start_time": 0, "end_time": 300, "duration": 300}]
                
            # Step 2: Keyframe Extraction
            if not skip_steps or "keyframe_extraction" not in skip_steps:
                logger.info("Step 2: Keyframe Extraction")
                keyframe_output_dir = output_dir / "keyframes"
                keyframe_output_dir.mkdir(exist_ok=True)
                
                try:
                    # Extract keyframes and save them directly to output directory
                    # This will return metadata in the format {"scenes": {scene_idx: [keyframe_metadata]}}
                    if self.keyframe_extractor is not None:
                        keyframes = self.keyframe_extractor.extract_keyframes(
                            str(video_path), 
                            scenes, 
                            str(keyframe_output_dir)
                        )
                    else:
                        logger.error("Keyframe extraction failed - no keyframe extractor available")
                        raise RuntimeError("Keyframe extraction is required but extractor failed to initialize")
                    
                    # Don't try to serialize the raw keyframes data which contains numpy arrays
                    # Instead, use the metadata object that's already saved by the extract_keyframes method
                    keyframes_metadata = {"scenes": {}}
                    for scene_idx, scene_keyframes in keyframes.items():
                        keyframes_metadata["scenes"][str(scene_idx)] = []
                        for kf in scene_keyframes:
                            # Only include serializable fields, not the image data
                            keyframes_metadata["scenes"][str(scene_idx)].append({
                                "frame_idx": kf["frame_idx"],
                                "timestamp": kf["timestamp"],
                                "path": f"scene_{scene_idx:04d}/keyframe_{len(keyframes_metadata['scenes'][str(scene_idx)]):04d}.jpg"
                            })
                    
                    # Save the metadata to JSON file
                    with open(keyframe_output_dir / "keyframes.json", 'w', encoding='utf-8') as f:
                        json.dump(keyframes_metadata, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Extracted and saved keyframes for {len(keyframes_metadata.get('scenes', {}))} scenes")
                except Exception as e:
                    logger.error(f"Error in keyframe extraction: {str(e)}")
                    # Create a minimal keyframes_metadata to continue the pipeline
                    keyframes_metadata = {"scenes": {}}
                    for i, scene in enumerate(scenes):
                        keyframes_metadata["scenes"][str(i)] = []
                    logger.warning("Created empty keyframes metadata to continue pipeline")
            else:
                logger.info("Skipping keyframe extraction")
                # Try to load existing keyframes from JSON
                try:
                    keyframe_output_dir = output_dir / "keyframes"
                    with open(keyframe_output_dir / "keyframes.json", 'r') as f:
                        keyframes_metadata = json.load(f)
                    logger.info(f"Loaded keyframes metadata from existing data")
                    # Ensure the format has a top-level "scenes" key
                    if "scenes" not in keyframes_metadata:
                        logger.warning("Loaded keyframes metadata doesn't have 'scenes' key, reformatting")
                        scenes_data = {}
                        for scene_idx, keyframes in keyframes_metadata.items():
                            scenes_data[scene_idx] = keyframes
                        keyframes_metadata = {"scenes": scenes_data}
                except (FileNotFoundError, json.JSONDecodeError):
                    logger.warning("Could not load existing keyframes metadata")
                    keyframes_metadata = {"scenes": {}}
                
            # Step 3: Entity Detection
            if not skip_steps or "entity_detection" not in skip_steps:
                logger.info("Step 3: Entity Detection")
                entity_output_dir = output_dir / "entities"
                entity_output_dir.mkdir(exist_ok=True)
                
                if self.entity_detector is not None:
                    entities = self.entity_detector.detect_entities(keyframes_metadata, str(keyframe_output_dir))
                    logger.info(f"Detected entities in all scenes")
                else:
                    entities = {}
                    logger.warning("Entity detection skipped due to initialization failure")
                
                # Save entities to JSON
                with open(entity_output_dir / "entities.json", 'w', encoding='utf-8') as f:
                    # Use custom NumpyEncoder for JSON serialization
                    json.dump(entities, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
                
                logger.info(f"Saved entity detection results to {entity_output_dir}")
            else:
                logger.info("Skipping entity detection")
                # Try to load existing entities from JSON
                try:
                    entity_output_dir = output_dir / "entities"
                    with open(entity_output_dir / "entities.json", 'r') as f:
                        entities = json.load(f)
                    logger.info(f"Loaded entity detection results from existing data")
                except (FileNotFoundError, json.JSONDecodeError):
                    logger.warning("Could not load existing entity detection results")
                    entities = {}
                
            # Step 4: Audio Processing
            if not skip_steps or "audio_processing" not in skip_steps:
                logger.info("Step 4: Audio Processing")
                audio_output_dir = output_dir / "audio"
                audio_output_dir.mkdir(exist_ok=True)
                
                try:
                    if self.audio_processor is not None:
                        audio_results = self.audio_processor.process_video_audio(str(video_path), scenes, str(audio_output_dir))
                        logger.info("Completed audio processing")
                        # --- ASR Ensemble Integration ---
                        if hasattr(self.audio_processor, 'config') and self.audio_processor.config.get('use_speaker_aware_vad', False):
                            auth_token = self.audio_processor.hf_token or os.environ.get("HF_TOKEN")
                            vad_segments = self.audio_processor._split_on_speech_gaps_with_diarization(str(audio_output_dir / "audio.wav"), auth_token)
                            chunk_paths = []
                            import soundfile as sf
                            for i, (start, end, _) in enumerate(vad_segments):
                                temp_path = str(audio_output_dir / f"chunk_{i}.wav")
                                with sf.SoundFile(str(audio_output_dir / "audio.wav")) as f:
                                    f.seek(int(start * f.samplerate))
                                    frames = int((end - start) * f.samplerate)
                                    data = f.read(frames)
                                    sf.write(temp_path, data, f.samplerate)
                                chunk_paths.append(temp_path)
                            asr_ensemble_result = self.audio_processor._asr_ensemble(chunk_paths)
                            audio_results['asr_ensemble'] = asr_ensemble_result
                        # --- end ASR Ensemble Integration ---
                    else:
                        logger.warning("Audio processing skipped due to initialization failure")
                        audio_results = {}
                except Exception as e:
                    logger.error(f"Error in audio processing: {str(e)}")
                    logger.warning("Creating dummy audio results to continue pipeline")
                    
                    # Create a minimal audio_results to continue the pipeline
                    audio_results = {"scenes": {}}
                    for i, scene in enumerate(scenes):
                        audio_results["scenes"][str(i)] = {
                            "scene_info": {
                                "start_time": scene.get("start_time", 0),
                                "end_time": scene.get("end_time", 0),
                                "duration": scene.get("duration", 0)
                            },
                            "dialogue": [],
                            "dialogue_count": 0
                        }
                    
                    # Save the dummy results
                    try:
                        with open(audio_output_dir / "audio_results.json", 'w', encoding='utf-8') as f:
                            json.dump(audio_results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
                        logger.info("Saved dummy audio results and continuing pipeline")
                    except Exception as e2:
                        logger.error(f"Error saving dummy audio results: {str(e2)}")
            else:
                logger.info("Skipping audio processing")
                # Try to load existing audio results from JSON
                try:
                    audio_output_dir = output_dir / "audio"
                    with open(audio_output_dir / "audio_results.json", 'r', encoding='utf-8') as f:
                        audio_results = json.load(f)
                    logger.info(f"Loaded audio processing results from existing data")
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Could not load existing audio results: {str(e)}")
                    # Create dummy audio results
                    audio_results = {"scenes": {}}
                    for i, scene in enumerate(scenes):
                        audio_results["scenes"][str(i)] = {
                            "scene_info": {
                                "start_time": scene.get("start_time", 0),
                                "end_time": scene.get("end_time", 0)
                            },
                            "dialogue": []
                        }
            
            # Extract scene videos for Gemini analysis
            if not skip_steps or "gemini_vision" not in skip_steps:
                scene_videos_dir = output_dir / "scene_videos"
                scene_videos_dir.mkdir(exist_ok=True)
                
                # Check if video processing is enabled in config
                if self.config.get("gemini_vision", {}).get("process_scene_videos", True):
                    logger.info("Extracting scene video segments for enhanced Gemini analysis")
                    try:
                        scene_videos = self._extract_scene_video_segments(str(video_path), scenes, str(scene_videos_dir))
                        logger.info(f"Extracted {len(scene_videos)} scene video segments")
                    except Exception as e:
                        logger.error(f"Error extracting scene videos: {str(e)}")
                        scene_videos = {}
                else:
                    logger.info("Scene video processing is disabled in config")
                    scene_videos = {}
                
            # Step 5: Scene Analysis with Gemini Vision
            if not skip_steps or "gemini_vision" not in skip_steps:
                logger.info("Step 5: Scene Analysis with Gemini Vision")
                gemini_output_dir = output_dir / "gemini"
                gemini_output_dir.mkdir(exist_ok=True)
                
                try:
                    # Process each scene with keyframes and video segments
                    gemini_output = {"scenes": {}}
                    
                    for scene_idx, scene_info in keyframes_metadata["scenes"].items():
                        logger.info(f"Analyzing scene {scene_idx}")
                        
                        # Extract entity and audio data for this scene
                        scene_entity_data = entities.get("scenes", {}).get(scene_idx, [])
                        scene_audio_data = audio_results.get("scenes", {}).get(scene_idx, {})
                        
                        # Get video path for this scene if available
                        scene_video_path = None
                        if int(scene_idx) in scene_videos:
                            scene_video_path = scene_videos[int(scene_idx)]
                            logger.info(f"Using video segment for scene {scene_idx}: {scene_video_path}")
                        
                        # Use enhanced scene analysis method
                        if self.gemini_analyzer is not None:
                            scene_result = self.gemini_analyzer.analyze_scene(
                                int(scene_idx),
                                scene_info,
                                str(keyframe_output_dir),
                                scene_video_path=scene_video_path,
                                entity_data=scene_entity_data,
                                audio_data=scene_audio_data
                            )
                            gemini_output["scenes"][scene_idx] = scene_result
                        else:
                            logger.warning(f"Skipping Gemini analysis for scene {scene_idx} due to missing analyzer")
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
                        json.dump(gemini_output, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
                    
                    logger.info(f"Completed scene analysis with Gemini Vision for {len(gemini_output['scenes'])} scenes")
                    
                except Exception as e:
                    logger.error(f"Error in Gemini Vision analysis: {str(e)}")
                    logger.warning("Skipping Gemini Vision analysis due to error")
                    gemini_output = {"scenes": {}}
            else:
                logger.info("Skipping Gemini Vision analysis")
                # Try to load existing Gemini results from JSON
                try:
                    gemini_output_dir = output_dir / "gemini"
                    with open(gemini_output_dir / "scene_analysis.json", 'r') as f:
                        gemini_output = json.load(f)
                    logger.info(f"Loaded Gemini Vision results from existing data")
                except (FileNotFoundError, json.JSONDecodeError):
                    logger.warning("Could not load existing Gemini Vision results")
                    gemini_output = {"scenes": {}}
            
            # Step 6: Generate Final Report
            logger.info("Step 6: Generating Final Report")
            self._generate_final_report({
                "scenes": scenes,
                "keyframes": keyframes_metadata,
                "entities": entities,
                "audio": audio_results,
                "gemini": gemini_output,
                "scene_videos": scene_videos
            }, output_dir)
            
            logger.info("Video analysis pipeline completed successfully")
            
            # Combine all results
            combined_results = {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "scenes": scenes,
                "keyframes": keyframes_metadata,
                "entities": entities,
                "audio": audio_results,
                "gemini": gemini_output,
                "scene_videos": scene_videos
            }
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
            
            # Collect partial data
            partial_data = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add any data that was successfully created
            if scenes:
                partial_data["scenes"] = scenes
            if keyframes_metadata:
                partial_data["keyframes"] = keyframes_metadata
            if entities:
                partial_data["entities"] = entities
            if audio_results:
                partial_data["audio"] = audio_results
            if gemini_output:
                partial_data["gemini"] = gemini_output
            if scene_videos:
                partial_data["scene_videos"] = scene_videos
                
            # Try to write partial results
            try:
                error_file = output_dir / "error_report.json"
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(partial_data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
                logger.info(f"Saved error report to {error_file}")
            except Exception as write_error:
                logger.error(f"Could not write error report: {str(write_error)}")
                
            return partial_data
        finally:
            # Clean up any temporary resources and force garbage collection
            gc.collect()
            
    def _generate_final_report(self, results: Dict, output_dir: Path):
        """Generate a final report summarizing all analysis results."""
        # Create a structured JSON with all combined results
        structured_data_path = output_dir / "structured_analysis.json"
        report_path = output_dir / "analysis_report.md"
        
        # Get video filename safely
        try:
            video_filename = str(Path(self.video_path).name)
        except (AttributeError, TypeError):
            # Fallback if video_path is not available
            video_filename = "unknown_video"
            logger.warning("Video filename not available, using placeholder in report")
        
        # Create a new structured JSON format - enhanced for downstream tasks
        structured_data = {
            "video_info": {
                "file_name": video_filename,
                "duration": 0,
                "num_scenes": 0,
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "fps": 0,  # Will be filled if available
                "resolution": {"width": 0, "height": 0}  # Will be filled if available
            },
            "scenes": {},
            "entities": {
                "total_objects": 0,
                "total_faces": 0,
                "total_text_snippets": 0,
                "object_types": {},
                "object_labels": {},
                "recurring_objects": []  # Objects appearing in multiple scenes
            },
            "audio": {
                "total_dialogue_segments": 0,
                "speakers": {},
                "language": "en"
            },
            "narrative_flow": {
                "scenes_with_dialogue": 0,
                "scenes_without_dialogue": 0,
                "average_scene_duration": 0
            },
            "narrative_structure": {
                "interweaving_storylines": False,
                "storyline_count": 0,
                "storylines": {},
                "transitions": [],
                "scene_to_storyline": {}
            },
            "characters": {}  # Will store character appearances across scenes
        }
        
        # Try to get video properties
        try:
            if isinstance(self.video_path, str) and os.path.exists(self.video_path):
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    structured_data["video_info"]["fps"] = cap.get(cv2.CAP_PROP_FPS)
                    structured_data["video_info"]["resolution"] = {
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    }
                    cap.release()
        except Exception as e:
            logger.warning(f"Could not extract video properties: {str(e)}")
        
        # Check for storyline analysis in Gemini output
        if "gemini" in results and "storyline_analysis" in results["gemini"]:
            storyline_analysis = results["gemini"]["storyline_analysis"]
            # Update the narrative structure with storyline info
            structured_data["narrative_structure"]["interweaving_storylines"] = storyline_analysis.get("interweaving_narrative", False)
            structured_data["narrative_structure"]["storyline_count"] = storyline_analysis.get("storyline_count", 0)
            
            # Copy storylines and transitions
            if "storylines" in storyline_analysis:
                structured_data["narrative_structure"]["storylines"] = storyline_analysis["storylines"]
            if "transitions" in storyline_analysis:
                structured_data["narrative_structure"]["transitions"] = storyline_analysis["transitions"]
                
            # Copy scene to storyline mapping if available
            if "scene_to_storyline" in storyline_analysis:
                structured_data["narrative_structure"]["scene_to_storyline"] = storyline_analysis["scene_to_storyline"]
                
            logger.info(f"Found storyline analysis with {storyline_analysis.get('storyline_count', 0)} storylines")
        
        # Character and object tracking across scenes
        character_tracker = {}  # Track character appearances across scenes
        object_tracker = {}     # Track object appearances across scenes
        
        # Process data and fill structured_data object
        if "scenes" in results and results["scenes"]:
            total_dialogue_segments = 0
            speakers_count = {}
            scenes_with_dialogue = 0
            scenes_without_dialogue = 0
            
            for i, scene in enumerate(results["scenes"]):
                start_time = scene["start_time"]
                end_time = scene["end_time"]
                duration = end_time - start_time
                
                scene_number = i + 1
                scene_key = str(i)
                
                # Initialize scene data structure with enhanced metadata
                scene_data = {
                    "scene_number": scene_number,
                    "scene_idx": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "timestamp": scene.get("timestamp", self._format_timestamp(start_time, end_time)),
                    "description": "",
                    "setting": "",  # Will be filled from scene analysis if available
                    "emotions": "",  # Emotional content of the scene
                    "entities": {
                        "objects": {},  # Will contain object_label -> count
                        "faces": 0,
                        "text": []
                    },
                    "dialogue": {
                        "transcript": "",
                        "segments": [],
                        "has_dialogue": False
                    },
                    "characters": [],  # Will contain characters detected in this scene
                    "semantic_keywords": []  # Key concepts extracted from the scene
                }
                
                # Add scene description and semantic content
                if ("gemini" in results and 
                    "scenes" in results["gemini"] and
                    scene_key in results["gemini"]["scenes"]):
                    
                    scene_analysis = results["gemini"]["scenes"][scene_key]
                    
                    # Extract structured fields if available
                    setting = ""
                    emotions = ""
                    characters = []
                    keywords = []
                    
                    # Try to extract structured fields from scene analysis
                    if "structured" in scene_analysis:
                        structured = scene_analysis["structured"]
                        setting = structured.get("setting", "")
                        emotions = structured.get("emotions", "")
                        
                        # Extract characters if available
                        if "characters" in structured:
                            char_text = structured["characters"]
                            # Extract character names using patterns like "Name:", "Character Name -", etc.
                            char_matches = re.findall(r'(?:^|\n)([A-Z][a-zA-Z\s]+(?::|-))', char_text)
                            characters = [match.rstrip(':- ').strip() for match in char_matches]
                            
                            # If no structured extraction, try simple name extraction
                            if not characters:
                                # Look for capitalized names (heuristic)
                                potential_names = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', char_text)
                                characters = [name for name in potential_names if len(name) > 3]
                        
                        # Track characters across scenes
                        for character in characters:
                            if character not in character_tracker:
                                character_tracker[character] = {
                                    "appearances": [],
                                    "total_screen_time": 0,
                                    "first_appearance": scene_data["timestamp"],
                                    "storylines": set()  # Track which storylines this character appears in
                                }
                            character_tracker[character]["appearances"].append(scene_number)
                            character_tracker[character]["total_screen_time"] += duration
                            
                            # Record storyline if this scene is part of one
                            scene_idx_str = str(i) if isinstance(i, int) else i
                            if scene_idx_str in structured_data["narrative_structure"]["scene_to_storyline"]:
                                storyline_id = structured_data["narrative_structure"]["scene_to_storyline"][scene_idx_str]
                                character_tracker[character]["storylines"].add(storyline_id)
                        
                        # Store characters in scene data
                        scene_data["characters"] = characters
                        
                        # Extract keywords for semantic search
                        if "significance" in structured:
                            significance = structured["significance"]
                            # Extract key terms and concepts
                            keywords = re.findall(r'\b([a-zA-Z]{4,})\b', significance.lower())
                            keywords = [word for word in keywords if word not in 
                                       ['this', 'that', 'with', 'from', 'have', 'there', 'they', 'their', 'scene']]
                            # Remove duplicates and limit
                            keywords = list(set(keywords))[:10]
                            scene_data["semantic_keywords"] = keywords
                    
                    # Check for description in scene_analysis (moved outside the 'structured' condition)
                    if "description" in scene_analysis and "error" not in scene_analysis:
                        scene_data["description"] = scene_analysis["description"]
                    # Also check for enhanced description which might be present
                    elif "enhanced_description" in scene_analysis and "error" not in scene_analysis:
                        scene_data["description"] = scene_analysis["enhanced_description"]
                    
                    # Add setting information if available
                    if setting:
                        scene_data["setting"] = setting
                        
                    # Add emotional content if available
                    if emotions:
                        scene_data["emotions"] = emotions
                    elif "error" in scene_analysis:
                        scene_data["error"] = scene_analysis['error']
                
                # Add entity information with detailed objects
                if ("entities" in results and 
                    "scenes" in results["entities"] and 
                    scene_key in results["entities"]["scenes"]):
                    
                    # Extract entity summary for this scene
                    if results["entities"]["scenes"][scene_key]:
                        # Count objects and store object types with counts
                        object_counts = {}
                        face_count = 0
                        text_snippets = set()
                        
                        for frame in results["entities"]["scenes"][scene_key]:
                            # Check if frame is a dictionary or if it has an "entities" field directly
                            if isinstance(frame, dict):
                                if "entities" in frame:
                                    entities_data = frame["entities"]
                                    
                                    # Process objects
                                    if isinstance(entities_data, dict):
                                        # Handle entities_data as a dictionary
                                        for obj in entities_data.get("objects", []):
                                            obj_type = obj.get("type", "unknown")
                                            obj_label = obj.get("label", "unknown")
                                            confidence = obj.get("confidence", 0)
                                            
                                            key = f"{obj_label} ({obj_type})"
                                            
                                            if key not in object_counts:
                                                object_counts[key] = {
                                                    "count": 1,
                                                    "confidence": confidence
                                                }
                                            else:
                                                object_counts[key]["count"] += 1
                                                # Update average confidence
                                                current = object_counts[key]
                                                current["confidence"] = (current["confidence"] * (current["count"] - 1) + confidence) / current["count"]
                                            
                                            # Track objects across scenes for recurring object detection
                                            if key not in object_tracker:
                                                object_tracker[key] = {
                                                    "scenes": set(),
                                                    "total_count": 0,
                                                    "storylines": set()  # Track which storylines this object appears in
                                                }
                                            object_tracker[key]["scenes"].add(scene_number)
                                            object_tracker[key]["total_count"] += 1
                                            
                                            # Record storyline if this scene is part of one
                                            scene_idx_str = str(i) if isinstance(i, int) else i
                                            if scene_idx_str in structured_data["narrative_structure"]["scene_to_storyline"]:
                                                storyline_id = structured_data["narrative_structure"]["scene_to_storyline"][scene_idx_str]
                                                object_tracker[key]["storylines"].add(storyline_id)
                                        
                                        # Count faces
                                        face_count += len(entities_data.get("faces", []))
                                        
                                        # Collect text snippets
                                        for text in entities_data.get("text", []):
                                            if text.get("text"):
                                                text_snippets.add(text.get("text"))
                                    elif isinstance(entities_data, list):
                                        # Handle entities_data as a list
                                        for item in entities_data:
                                            if isinstance(item, dict) and "type" in item and "label" in item:
                                                obj_type = item.get("type", "unknown")
                                                obj_label = item.get("label", "unknown")
                                                confidence = item.get("confidence", 0)
                                                
                                                key = f"{obj_label} ({obj_type})"
                                                
                                                if key not in object_counts:
                                                    object_counts[key] = {
                                                        "count": 1,
                                                        "confidence": confidence
                                                    }
                                                else:
                                                    object_counts[key]["count"] += 1
                                                    # Update average confidence
                                                    current = object_counts[key]
                                                    current["confidence"] = (current["confidence"] * (current["count"] - 1) + confidence) / current["count"]
                                                
                                                # Track objects across scenes for recurring object detection
                                                if key not in object_tracker:
                                                    object_tracker[key] = {
                                                        "scenes": set(),
                                                        "total_count": 0,
                                                        "storylines": set()  # Track which storylines this object appears in
                                                    }
                                                object_tracker[key]["scenes"].add(scene_number)
                                                object_tracker[key]["total_count"] += 1
                                                
                                                # Record storyline if this scene is part of one
                                                scene_idx_str = str(i) if isinstance(i, int) else i
                                                if scene_idx_str in structured_data["narrative_structure"]["scene_to_storyline"]:
                                                    storyline_id = structured_data["narrative_structure"]["scene_to_storyline"][scene_idx_str]
                                                    object_tracker[key]["storylines"].add(storyline_id)
                        
                        # Update scene data
                        scene_data["entities"]["objects"] = {k: v["count"] for k, v in object_counts.items()}
                        scene_data["entities"]["object_details"] = object_counts  # More detailed version with confidence
                        scene_data["entities"]["faces"] = face_count
                        scene_data["entities"]["text"] = list(text_snippets)
                
                # Add dialogue information with better timestamps and speaker clarity
                has_dialogue = False
                if ("audio" in results and 
                    "scenes" in results["audio"] and 
                    scene_key in results["audio"]["scenes"]):
                    
                    scene_audio = results["audio"]["scenes"][scene_key]
                    dialogue_sections = []
                    
                    # First check if we have a pre-formatted dialogue string
                    if "dialogue" in scene_audio and scene_audio["dialogue"]:
                        try:
                            if scene_audio["dialogue"] != "No dialogue in this scene" and scene_audio["dialogue"] != "No dialogue available":
                                dialogue_text = scene_audio["dialogue"]
                                if isinstance(dialogue_text, str):
                                    scene_data["dialogue"]["transcript"] = dialogue_text
                                    scene_data["dialogue"]["has_dialogue"] = True
                                    has_dialogue = True
                                    scenes_with_dialogue += 1
                                else:
                                    scene_data["dialogue"]["transcript"] = str(dialogue_text)
                                    scene_data["dialogue"]["has_dialogue"] = True
                                    has_dialogue = True
                                    scenes_with_dialogue += 1
                                # Structured version will still be empty list if we only have the formatted dialogue
                            else:
                                scene_data["dialogue"]["transcript"] = "No dialogue in this scene"
                                scene_data["dialogue"]["has_dialogue"] = False
                                scenes_without_dialogue += 1
                        except Exception as e:
                            logger.error(f"Error processing dialogue: {str(e)}")
                            scene_data["dialogue"]["transcript"] = "Error formatting dialogue"
                            scene_data["dialogue"]["has_dialogue"] = False
                            scenes_without_dialogue += 1
                    
                    # Process detailed segments if available
                    elif "segments" in scene_audio and scene_audio["segments"]:
                        try:
                            dialogue_text = ""
                            segments = []
                            for segment in scene_audio["segments"]:
                                speaker = segment.get("speaker", "Unknown")
                                text = segment.get("text", "").strip()
                                
                                # Count speakers for statistics
                                if speaker not in speakers_count:
                                    speakers_count[speaker] = 0
                                speakers_count[speaker] += 1
                                
                                # Get timestamp if available
                                timestamp = ""
                                start = segment.get("start", 0)
                                end = segment.get("end", 0)
                                
                                if start is not None and end is not None:
                                    timestamp = f"[{self._format_time(start)} - {self._format_time(end)}]"
                                
                                if text:
                                    # Add to dialogue text
                                    line = f"{speaker} {timestamp}: {text}\n"
                                    dialogue_text += line
                                    
                                    # Add to structured data
                                    segments.append({
                                        "speaker": speaker,
                                        "text": text,
                                        "start": start,
                                        "end": end
                                    })
                                    
                                    total_dialogue_segments += 1
                            
                            scene_data["dialogue"]["transcript"] = dialogue_text.strip()
                            scene_data["dialogue"]["segments"] = segments
                            
                            if segments:
                                scene_data["dialogue"]["has_dialogue"] = True
                                has_dialogue = True
                                scenes_with_dialogue += 1
                            else:
                                scene_data["dialogue"]["has_dialogue"] = False
                                scenes_without_dialogue += 1
                        except Exception as e:
                            logger.error(f"Error processing dialogue segments: {str(e)}")
                            scene_data["dialogue"]["transcript"] = "Error processing dialogue segments"
                            scene_data["dialogue"]["has_dialogue"] = False
                            scenes_without_dialogue += 1
                    else:
                        scene_data["dialogue"]["transcript"] = "No dialogue information available"
                        scene_data["dialogue"]["has_dialogue"] = False
                        scenes_without_dialogue += 1
                else:
                    scene_data["dialogue"]["transcript"] = "No dialogue information available"
                    scene_data["dialogue"]["has_dialogue"] = False
                    scenes_without_dialogue += 1
                
                # Add scene data to structured data
                structured_data["scenes"][scene_key] = scene_data
            
            # Update video info with calculated data
            if results["scenes"]:
                total_duration = sum(scene["duration"] for scene in results["scenes"])
                num_scenes = len(results["scenes"])
                structured_data["video_info"]["duration"] = total_duration
                structured_data["video_info"]["num_scenes"] = num_scenes
            
            # Update audio statistics in structured data
            structured_data["audio"]["total_dialogue_segments"] = total_dialogue_segments
            structured_data["audio"]["speakers"] = {
                speaker: count for speaker, count in speakers_count.items()
            }
            
            # Update narrative flow statistics
            structured_data["narrative_flow"]["scenes_with_dialogue"] = scenes_with_dialogue
            structured_data["narrative_flow"]["scenes_without_dialogue"] = scenes_without_dialogue
            if num_scenes > 0:
                structured_data["narrative_flow"]["average_scene_duration"] = total_duration / num_scenes
        
        # Add recurring objects to structured data
        recurring_objects = []
        for obj, data in object_tracker.items():
            if len(data["scenes"]) > 1:
                recurring_objects.append({
                    "name": obj,
                    "scenes": sorted(list(data["scenes"])),
                    "total_count": data["total_count"],
                    "storylines": sorted(list(data["storylines"]))
                })
        
        if recurring_objects:
            structured_data["entities"]["recurring_objects"] = recurring_objects
            
        # Store character data for downstream tasks
        for character, data in character_tracker.items():
            structured_data["characters"][character] = {
                "appearances": data["appearances"],
                "screen_time": data["total_screen_time"],
                "first_appearance": data["first_appearance"],
                "storylines": list(data.get("storylines", set()))
            }
        
        # Save structured data to JSON
        with open(structured_data_path, 'w', encoding='utf-8') as json_file:
            json.dump(structured_data, json_file, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved enhanced structured analysis data to {structured_data_path}")
        
        # Enhance dialogue with character names using Gemini
        self._enhance_dialogue_with_gemini(structured_data_path)
        
        # Reload the enhanced data from disk to get the updated character names
        try:
            with open(structured_data_path, 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
            logger.info("Successfully reloaded enhanced dialogue data for report generation")
        except Exception as e:
            logger.error(f"Error reloading enhanced data: {str(e)}. Using in-memory data for report.")
        
        # NOW generate the report AFTER dialogue enhancement and reloading
        self._generate_analysis_report(structured_data, report_path)
        
    def _generate_analysis_report(self, structured_data: Dict, report_path: Path):
        """Generate the analysis report from structured data."""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Video Analysis Report\n\n")
            
            # Video info
            f.write("## Video Information\n\n")
            video_info = structured_data["video_info"]
            f.write(f"- **File:** {video_info.get('file_name', 'Unknown')}\n")
            f.write(f"- **Duration:** {video_info.get('duration', 0):.2f} seconds ({self._format_timestamp(video_info.get('duration', 0))})\n")
            f.write(f"- **Number of Scenes:** {video_info.get('num_scenes', 0)}\n")
            f.write(f"- **Analysis Date:** {video_info.get('analysis_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
            
            # Add frame rate and resolution if available
            if video_info.get("fps", 0) > 0:
                f.write(f"- **Frame Rate:** {video_info.get('fps', 0):.2f} FPS\n")
            
            if video_info.get("resolution", {}).get("width", 0) > 0:
                res = video_info.get("resolution", {})
                f.write(f"- **Resolution:** {res.get('width', 0)}x{res.get('height', 0)}\n")
            
            f.write("\n")
            
            # Add Narrative Structure section for interweaving storylines
            if structured_data["narrative_structure"]["interweaving_storylines"]:
                f.write("## Narrative Structure\n\n")
                storyline_count = structured_data["narrative_structure"]["storyline_count"]
                
                f.write(f"This video features **{storyline_count} distinct storylines** that interweave throughout the narrative. ")
                f.write("The narrative alternates between different settings and character groups, creating a parallel storytelling structure.\n\n")
                
                # Add storyline descriptions
                f.write("### Storylines\n\n")
                
                for storyline_id, storyline in structured_data["narrative_structure"]["storylines"].items():
                    scenes_in_storyline = storyline.get("scenes", [])
                    scene_nums = [str(s + 1) for s in scenes_in_storyline]
                    setting = storyline.get("setting_summary", "Unknown setting")
                    
                    f.write(f"**Storyline {int(storyline_id)+1}**: {setting}\n")
                    f.write(f"- Appears in {len(scenes_in_storyline)} scenes: {', '.join(scene_nums)}\n")
                    f.write("\n")
                
                # Add narrative transitions
                if structured_data["narrative_structure"]["transitions"]:
                    f.write("### Narrative Transitions\n\n")
                    f.write("The story switches between storylines at these points:\n\n")
                    
                    for transition in structured_data["narrative_structure"]["transitions"]:
                        from_scene = transition.get("from_scene", 0)
                        to_scene = transition.get("to_scene", 0)
                        description = transition.get("description", "Setting change")
                        
                        f.write(f"- **Scene {from_scene+1} → Scene {to_scene+1}**: {description}\n")
                    
                    f.write("\n")
            
            # Scene breakdown
            f.write("## Scene Breakdown\n\n")
            
            for scene_idx, scene in structured_data.get("scenes", {}).items():
                scene_num = scene.get("scene_number", int(scene_idx) + 1)
                start_time = scene.get("start_time", 0)
                end_time = scene.get("end_time", 0)
                duration = scene.get("duration", 0)
                
                # Write scene header with clear temporal boundaries
                f.write(f"### Scene {scene_num}\n\n")
                
                # Include the formatted timestamp
                timestamp = scene.get("timestamp", self._format_timestamp(start_time, end_time))
                time_str = f"[{timestamp}]"
                f.write(f"- **Time Range:** {time_str}\n")
                f.write(f"- **Duration:** {duration:.2f}s\n")
                
                # Check if this scene is part of an identified storyline
                scene_idx_str = str(scene_idx) if isinstance(scene_idx, int) else scene_idx
                if scene_idx_str in structured_data["narrative_structure"]["scene_to_storyline"]:
                    storyline_id = structured_data["narrative_structure"]["scene_to_storyline"][scene_idx_str]
                    if str(storyline_id) in structured_data["narrative_structure"]["storylines"]:
                        storyline = structured_data["narrative_structure"]["storylines"][str(storyline_id)]
                        storyline_setting = storyline.get("setting_summary", "")
                        f.write(f"- **Storyline:** {int(storyline_id)+1} ({storyline_setting})\n")
                
                # Add scene description
                description = scene.get("description", "")
                if description:
                    f.write("\n**Description:**\n\n")
                    f.write(description + "\n\n")
                
                # Add setting information if available
                setting = scene.get("setting", "")
                if setting:
                    f.write(f"**Setting:** {setting}\n\n")
                    
                # Add emotional content if available
                emotions = scene.get("emotions", "")
                if emotions:
                    f.write(f"**Emotional Tone:** {emotions}\n\n")
                    
                # Add character list if available and not empty
                characters = scene.get("characters", [])
                if characters:
                    f.write(f"**Characters:** {', '.join(characters)}\n\n")
                
                # Add detected objects
                objects = scene.get("entities", {}).get("objects", {})
                if objects:
                    f.write("**Detected Objects:**\n\n")
                    object_details = scene.get("entities", {}).get("object_details", {})
                    for obj, count in sorted(objects.items(), key=lambda x: x[1], reverse=True):
                        if obj in object_details:
                            confidence = object_details[obj].get("confidence", 0)
                            f.write(f"- {obj}: {count} (Avg. Confidence: {confidence:.2f})\n")
                        else:
                            f.write(f"- {obj}: {count}\n")
                    f.write("\n")
                                
                # Add face count
                face_count = scene.get("entities", {}).get("faces", 0)
                if face_count > 0:
                    f.write(f"**Faces:** {face_count}\n\n")
                            
                # Add text in scene
                text_snippets = scene.get("entities", {}).get("text", [])
                if text_snippets:
                    f.write("**Text in Scene:**\n\n")
                    for text in sorted(text_snippets)[:10]:  # Show up to 10 text items
                        f.write(f"- {text}\n")
                            
                    if len(text_snippets) > 10:
                        f.write(f"- *...and {len(text_snippets) - 10} more text items*\n")
                    f.write("\n")
                    
                # Add dialogue with character names
                if scene.get("dialogue", {}).get("has_dialogue", False):
                    transcript = scene.get("dialogue", {}).get("transcript", "")
                    if transcript and transcript != "No dialogue in this scene" and transcript != "No dialogue information available":
                        f.write("**Dialogue:**\n\n")
                        f.write("```\n")
                        f.write(transcript)
                        f.write("\n```\n\n")
                
                # Add divider between scenes
                f.write("---\n\n")
            
            # Add recurring objects section
            recurring_objects = structured_data.get("entities", {}).get("recurring_objects", [])
            if recurring_objects:
                f.write("## Recurring Objects\n\n")
                
                # If we have interweaving storylines, show objects by storyline
                if structured_data["narrative_structure"]["interweaving_storylines"]:
                    f.write("### Objects by Storyline\n\n")
                    
                    # Group objects by storyline
                    storyline_objects = {}
                    for obj in recurring_objects:
                        if "storylines" in obj and obj["storylines"]:
                            for storyline_id in obj["storylines"]:
                                storyline_id_str = str(storyline_id)
                                if storyline_id_str not in storyline_objects:
                                    storyline_objects[storyline_id_str] = []
                                storyline_objects[storyline_id_str].append(obj)
                    
                    # Output objects by storyline
                    for storyline_id, objects in sorted(storyline_objects.items()):
                        if storyline_id in structured_data["narrative_structure"]["storylines"]:
                            storyline = structured_data["narrative_structure"]["storylines"][storyline_id]
                            setting = storyline.get("setting_summary", "Unknown setting")
                            f.write(f"**Storyline {int(storyline_id)+1}** ({setting}):\n\n")
                        else:
                            f.write(f"**Storyline {int(storyline_id)+1}**:\n\n")
                            
                        for obj in sorted(objects, key=lambda x: len(x["scenes"]), reverse=True)[:10]:
                            f.write(f"- {obj['name']}: appears in {len(obj['scenes'])} scenes\n")
                        
                        if len(objects) > 10:
                            f.write(f"- *...and {len(objects) - 10} more objects*\n")
                        
                        f.write("\n")
            
                f.write("### All Recurring Objects\n\n")
                f.write("Objects that appear across multiple scenes:\n\n")
                
                for obj in sorted(recurring_objects, key=lambda x: len(x["scenes"]), reverse=True)[:15]:
                    f.write(f"- **{obj['name']}**\n")
                    f.write(f"  - Appears in {len(obj['scenes'])} scenes: {', '.join(f'Scene {i}' for i in obj['scenes'])}\n")
                    f.write(f"  - Total appearances: {obj['total_count']}\n\n")
            
            # Add Character Summary section for recurring characters
            characters_data = structured_data.get("characters", {})
            if characters_data:
                f.write("\n## Character Summary\n\n")
                
                # If we have interweaving storylines, highlight characters appearing across storylines
                if structured_data["narrative_structure"]["interweaving_storylines"]:
                    f.write("### Characters by Storyline\n\n")
                    
                    # Group characters by storyline
                    storyline_characters = {}
                    crossover_characters = []
                    
                    for character, data in characters_data.items():
                        storylines = data.get("storylines", [])
                        if storylines:
                            # Character appears in multiple storylines
                            if len(storylines) > 1:
                                crossover_characters.append((character, data))
                            
                            # Add to each storyline
                            for storyline_id in storylines:
                                storyline_id_str = str(storyline_id)
                                if storyline_id_str not in storyline_characters:
                                    storyline_characters[storyline_id_str] = []
                                storyline_characters[storyline_id_str].append((character, data))
                    
                    # Output characters by storyline
                    for storyline_id, characters in sorted(storyline_characters.items()):
                        if storyline_id in structured_data["narrative_structure"]["storylines"]:
                            storyline = structured_data["narrative_structure"]["storylines"][storyline_id]
                            setting = storyline.get("setting_summary", "Unknown setting")
                            f.write(f"**Storyline {int(storyline_id)+1}** ({setting}):\n\n")
                        else:
                            f.write(f"**Storyline {int(storyline_id)+1}**:\n\n")
                            
                        for character, data in sorted(characters, key=lambda x: len(x[1].get("appearances", [])), reverse=True):
                            f.write(f"- {character}: appears in {len(data.get('appearances', []))} scenes\n")
                        
                        f.write("\n")
                    
                    # Output crossover characters
                    if crossover_characters:
                        f.write("### Narrative Crossover Characters\n\n")
                        f.write("Characters that appear in multiple storylines:\n\n")
                        
                        for character, data in sorted(crossover_characters, key=lambda x: len(x[1].get("storylines", [])), reverse=True):
                            f.write(f"- **{character}**\n")
                            storyline_nums = sorted([str(int(s)+1) for s in data.get("storylines", [])])
                            f.write(f"  - Appears in storylines: {', '.join(storyline_nums)}\n")
                            f.write(f"  - Total scenes: {len(data.get('appearances', []))}\n")
                            f.write(f"  - Screen time: {data.get('screen_time', 0):.2f}s\n\n")
                
                f.write("### Characters Across Multiple Scenes\n\n")
                f.write("Characters that appear across multiple scenes:\n\n")
                
                for character, data in sorted(characters_data.items(), 
                                            key=lambda x: len(x[1].get("appearances", [])), 
                                            reverse=True):
                    # Only include characters appearing in multiple scenes
                    appearances = data.get("appearances", [])
                    if appearances and len(appearances) > 1:
                        f.write(f"- **{character}**\n")
                        f.write(f"  - Appears in {len(appearances)} scenes: {', '.join(f'Scene {i}' for i in sorted(appearances))}\n")
                        f.write(f"  - First appearance: {data.get('first_appearance', '')}\n")
                        f.write(f"  - Total screen time: {data.get('screen_time', 0):.2f}s\n\n")
        
        logger.info(f"Generated enhanced analysis report at {report_path}")

    def _enhance_dialogue_with_gemini(self, structured_data_path: str) -> None:
        """
        Use the Gemini API to enhance dialogue in structured data by identifying characters
        and replacing "UNKNOWN" speaker tags.
        
        Args:
            structured_data_path: Path to the structured analysis JSON file
        """
        logger.info("Attempting to enhance dialogue with character names using Gemini API")
        
        # Load the structured data
        try:
            with open(structured_data_path, 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load structured data: {str(e)}")
            return
        
        # Create a deep copy to avoid modifying the original
        enhanced_data = copy.deepcopy(structured_data)
        
        # Setup caching to avoid redundant API calls
        output_dir = os.path.dirname(structured_data_path)
        cache_file = Path(output_dir) / "dialogue_api_cache.json"
        api_cache = {}
        
        # Load existing cache if available
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    api_cache = json.load(f)
                logger.info(f"Loaded {len(api_cache)} cached dialogue enhancement results")
            except Exception as e:
                logger.warning(f"Could not load API cache: {str(e)}")
        
        # Check for API keys - try dialogue-specific key first, fall back to regular key
        gemini_dialogue_key = os.environ.get("GEMINI_DIALOGUE_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        api_key = gemini_dialogue_key if gemini_dialogue_key else gemini_key
        
        if not api_key:
            logger.warning("No Gemini API key found. Skipping API-based character identification.")
            logger.info("Applying fallback character identification instead.")
            enhanced_data = self._apply_fallback_character_identification(enhanced_data)
            with open(structured_data_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            return
        
        try:
            import google.generativeai as genai
            from google.api_core.exceptions import InvalidArgument
            
            # Configure the Gemini API with error handling
            genai.configure(api_key=api_key)
            
            # Extract only the dialogue parts that need enhancement
            scenes_with_unknown = {}
            unknown_count = 0
            
            # Also collect scene summaries to provide context
            scene_summaries = {}
            
            for scene_key, scene in enhanced_data.get("scenes", {}).items():
                if isinstance(scene, dict) and "dialogue" in scene:
                    transcript = scene["dialogue"].get("transcript", "")
                    
                    # Include scene summary for context
                    if "summary" in scene:
                        scene_summaries[scene_key] = scene["summary"]
                    elif "description" in scene:
                        scene_summaries[scene_key] = scene["description"]
                    
                    if transcript and "UNKNOWN" in transcript:
                        count = transcript.count("UNKNOWN")
                        unknown_count += count
                        scenes_with_unknown[scene_key] = {
                            "dialogue": {"transcript": transcript},
                            "unknown_count": count
                        }
            
            logger.info(f"Found {unknown_count} 'UNKNOWN' speaker labels to replace in {len(scenes_with_unknown)} scenes")
            
            if unknown_count == 0:
                logger.info("No UNKNOWN speakers found. Skipping character identification.")
                return
            
            # Try multiple Gemini models in order of preference
            model = None
            models_to_try = [
                "gemini-1.5-flash",  # Lower quota impact
                "gemini-1.5-pro",    # Higher quota but better quality
                "gemini-2.0-flash-exp"  # Original model as last resort
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to use {model_name} for dialogue enhancement")
                    model = genai.GenerativeModel(model_name,
                                                 generation_config={
                                                     "temperature": 0.2,
                                                     "top_p": 0.95,
                                                     "max_output_tokens": 8192,
                                                 })
                    # Test with a tiny prompt to validate
                    test_response = model.generate_content("Hello")
                    if hasattr(test_response, 'text'):
                        logger.info(f"Successfully initialized {model_name} for dialogue enhancement")
                        break
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {str(e)}")
            
            if model is None:
                logger.error("Failed to initialize any Gemini model for dialogue enhancement")
                enhanced_data = self._apply_fallback_character_identification(enhanced_data)
                with open(structured_data_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
                return
            
            # Function to sanitize JSON (fix common JSON errors)
            def sanitize_json(json_str):
                """Fix common JSON errors and return a sanitized string."""
                # Replace unescaped quotes inside string values
                sanitized = ""
                in_string = False
                escape_next = False
                
                for i, char in enumerate(json_str):
                    if escape_next:
                        sanitized += char
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        sanitized += char
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    
                    # Handle newlines in strings
                    if in_string and char in ('\n', '\r'):
                        sanitized += '\\n'
                        continue
                    
                    sanitized += char
                
                # Fix unterminated strings by adding closing quotes if needed
                if in_string:
                    sanitized += '"'
                
                # Fix unterminated blocks
                open_braces = sanitized.count('{')
                close_braces = sanitized.count('}')
                for _ in range(open_braces - close_braces):
                    sanitized += "}"
                
                return sanitized
            
            # Try to parse JSON using a more resilient approach
            def parse_json_safely(json_str):
                # First, try the standard json parser
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Standard JSON parsing failed: {str(e)}")
                    
                    # Try sanitizing the JSON and parse again
                    try:
                        sanitized = sanitize_json(json_str)
                        logger.info("Attempting to parse sanitized JSON")
                        return json.loads(sanitized)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Sanitized JSON parsing failed: {str(e)}")
                
                    # Try a more selective approach - extract just what we need
                    try:
                        # This is a very restricted approach that might work in some cases
                        result = {"scenes": {}}
                        
                        # Extract scene keys and transcript data
                        scenes_pattern = re.compile(r'"(\d+)".*?dialogue.*?transcript.*?"(.*?)"', re.DOTALL)
                        matches = scenes_pattern.finditer(json_str)
                        
                        for match in matches:
                            scene_id, transcript = match.groups()
                            if "UNKNOWN" not in transcript:  # Only include if UNKNOWN was replaced
                                if scene_id not in result["scenes"]:
                                    result["scenes"][scene_id] = {"dialogue": {"transcript": transcript}}
                                else:
                                    result["scenes"][scene_id]["dialogue"] = {"transcript": transcript}
                        
                        if result["scenes"]:
                            logger.info(f"Extracted {len(result['scenes'])} scenes using pattern matching")
                            return result
                    except Exception as e:
                        logger.error(f"Pattern extraction failed: {str(e)}")
                
                return None
            
            # Process scenes in smaller batches if there are many
            max_scenes_per_batch = 3  # Reduced from original 10 to lower rate limit impact
            scene_keys = list(scenes_with_unknown.keys())
            total_batches = (len(scene_keys) + max_scenes_per_batch - 1) // max_scenes_per_batch
            
            # Track modified dialogues
            modified_dialogues = {}
            
            # Create output directory for debug files if it doesn't exist
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Process each batch of scenes
            for batch_idx in range(total_batches):
                start_idx = batch_idx * max_scenes_per_batch
                end_idx = min(start_idx + max_scenes_per_batch, len(scene_keys))
                batch_scene_keys = scene_keys[start_idx:end_idx]
                
                # Create the batch data
                batch_data = {
                    "video_info": structured_data.get("video_info", {}),
                    "scenes": {k: scenes_with_unknown[k] for k in batch_scene_keys},
                    "scene_summaries": {k: scene_summaries.get(k, "") for k in batch_scene_keys if k in scene_summaries}
                }
                
                # Create a cache key for this batch
                cache_key = str(hash(json.dumps(batch_data, sort_keys=True)))
                
                # Check if this exact batch is in the cache
                if cache_key in api_cache:
                    logger.info(f"Using cached result for batch {batch_idx+1}/{total_batches}")
                    dialogue_result = api_cache[cache_key]
                    
                    # Update modified dialogues from cache
                    if dialogue_result and "scenes" in dialogue_result:
                        for scene_key, scene_data in dialogue_result.get("scenes", {}).items():
                            if "dialogue" in scene_data and "transcript" in scene_data["dialogue"]:
                                modified_dialogues[scene_key] = scene_data["dialogue"]["transcript"]
                    
                    # Skip to next batch
                    continue
                
                logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_scene_keys)} scenes")
                
                # Create the prompt for Gemini API - focused only on the dialogue parts
                prompt = f"""
                You are tasked with replacing "UNKNOWN" speaker labels in dialogue transcripts with character names.
                
                INSTRUCTIONS:
                1. Extract character identities and names from the scene summaries and dialogue provided.
                2. Replace ALL instances of "UNKNOWN" speaker labels with the most appropriate character names.
                3. Be consistent with character names throughout the scenes.
                4. Return ONLY the scene IDs and their corresponding corrected dialogue transcripts.
                
                Here is the data with dialogue transcripts containing "UNKNOWN" speaker labels:
                ```json
                {json.dumps(batch_data, indent=2)}
                ```
                
                RESPONSE FORMAT REQUIREMENTS:
                - Return ONLY a JSON object with corrected dialogues in this exact format:
                {{
                  "scenes": {{
                    "scene_id": {{
                      "dialogue": {{
                        "transcript": "corrected dialogue with character names instead of UNKNOWN"
                      }}
                    }},
                    ...more scenes...
                  }}
                }}
                - Do not include explanations, comments, or any other text.
                - Focus ONLY on replacing "UNKNOWN" with proper character names.
                """
                
                # Implement robust API calling with retry logic
                max_retries = 3
                retry_count = 0
                api_success = False
                response_text = ""
                
                # Initialize dialogue_result outside the loop to fix scope issue
                dialogue_result = None
                
                while retry_count < max_retries and not api_success:
                    retry_count += 1
                    logger.info(f"Calling Gemini API for batch {batch_idx+1} (attempt {retry_count}/{max_retries})")
                    
                    try:
                        # Set a timeout to prevent hanging
                        import threading
                        import time
                        response = None
                        api_error = None
                        response_received = threading.Event()
                        
                        def make_api_call():
                            nonlocal response, api_error
                            try:
                                response = model.generate_content(prompt)
                                response_received.set()
                            except Exception as e:
                                api_error = str(e)
                                logger.error(f"API call error: {str(e)}")
                                response_received.set()
                        
                        # Start API call in a separate thread
                        api_thread = threading.Thread(target=make_api_call)
                        api_thread.daemon = True
                        api_thread.start()
                        
                        # Wait for the API call to complete with a reasonable timeout
                        api_success = response_received.wait(timeout=300)  # 5 minute timeout
                        
                        if not api_success:
                            logger.warning(f"API call timed out on attempt {retry_count}/{max_retries}")
                            time.sleep(10)  # Increased wait before retry
                            continue
                        
                        if api_error:
                            logger.warning(f"API error on attempt {retry_count}/{max_retries}: {api_error}")
                            
                            # If rate limit error, wait much longer
                            if "429" in api_error or "quota" in api_error.lower():
                                wait_time = 30 * (2 ** (retry_count - 1))  # Exponential backoff: 30, 60, 120 seconds
                                logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                            else:
                                time.sleep(10)  # Standard wait for other errors
                            
                            continue
                        
                        if not response:
                            logger.warning(f"Empty response on attempt {retry_count}/{max_retries}")
                            time.sleep(10)  # Increased wait before retry
                            continue
                        
                        # Process the response
                        response_text = response.text.strip()
                        original_text = response_text  # Save original for debugging
                        
                        # Save the raw response to a file for debugging
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        raw_response_file = os.path.join(debug_dir, f"gemini_raw_response_batch{batch_idx+1}_{timestamp}_attempt{retry_count}.txt")
                        with open(raw_response_file, 'w', encoding='utf-8') as f:
                            f.write(response_text)
                        logger.info(f"Saved raw Gemini API response to {raw_response_file}")
                        
                        # Validate we have some response
                        if not response_text:
                            logger.warning(f"Empty text response on attempt {retry_count}/{max_retries}")
                            time.sleep(10)  # Increased wait before retry
                            continue
                        
                        # Log first bit of response for debugging
                        logger.info(f"API Response first 100 chars: '{response_text[:100]}...'")
                        
                        # Extract JSON from code blocks if present
                        extracted_json = ""
                        
                        if "```json" in response_text:
                            # Find the start marker position
                            start_marker = "```json"
                            end_marker = "```"
                            
                            start_pos = response_text.find(start_marker)
                            if start_pos != -1:
                                # Skip the marker itself
                                json_start = start_pos + len(start_marker)
                                # Find the closing code block marker
                                json_end = response_text.find(end_marker, json_start)
                                
                                if json_end != -1:
                                    # Extract everything between the markers
                                    extracted_json = response_text[json_start:json_end].strip()
                                    logger.info(f"Extracted JSON from code block, length: {len(extracted_json)}")
                                else:
                                    logger.warning("Found starting JSON marker but no ending marker")
                        elif "```" in response_text:
                            # Try generic code block markers
                            start_marker = "```"
                            
                            start_pos = response_text.find(start_marker)
                            if start_pos != -1:
                                # Skip past the first marker
                                json_start = start_pos + len(start_marker)
                                # Find the next marker (ending the code block)
                                json_end = response_text.find(start_marker, json_start)
                                
                                if json_end != -1:
                                    # Extract everything between the markers
                                    extracted_json = response_text[json_start:json_end].strip()
                                    logger.info(f"Extracted JSON from generic code block, length: {len(extracted_json)}")
                                else:
                                    logger.warning("Found starting code block marker but no ending marker")
                        else:
                            # No code blocks, assume the entire response is JSON
                            extracted_json = response_text
                            logger.info("No code block markers found, using entire response")
                        
                        # Additional fallback: If response starts with ```json but extraction failed
                        if not extracted_json and response_text.startswith("```json"):
                            logger.warning("Response starts with code block but extraction failed. Trying alternative approach.")
                            lines = response_text.splitlines()
                            # Skip the first line with ```json
                            json_lines = []
                            capture = False
                            
                            for line in lines:
                                if line.strip() == "```json":
                                    capture = True
                                    continue
                                elif line.strip() == "```" and capture:
                                    break
                                elif capture:
                                    json_lines.append(line)
                            
                            if json_lines:
                                extracted_json = "\n".join(json_lines)
                                logger.info(f"Extracted JSON using line-by-line approach, length: {len(extracted_json)}")
                        
                        # Log the first part of extracted content for debugging
                        if extracted_json:
                            logger.info(f"First 100 chars of extracted content: '{extracted_json[:100]}...'")
                        else:
                            logger.warning("Failed to extract any content from response")
                            extracted_json = response_text  # Fall back to the full response
                        
                        # Save the extracted JSON to a file for debugging
                        extracted_json_file = os.path.join(debug_dir, f"gemini_extracted_json_batch{batch_idx+1}_{timestamp}_attempt{retry_count}.json")
                        with open(extracted_json_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_json)
                        logger.info(f"Saved extracted JSON to {extracted_json_file}")
                        
                        # One final cleanup: if it starts with whitespace and then {, trim the whitespace
                        extracted_json = extracted_json.lstrip()
                        
                        # Try the robust parsing approach
                        dialogue_result = parse_json_safely(extracted_json)
                        
                        if dialogue_result and "scenes" in dialogue_result:
                            api_success = True
                            logger.info(f"Successfully parsed the enhanced dialogue from Gemini API for batch {batch_idx+1}")
                            
                            # Save the successfully parsed JSON for reference
                            parsed_json_file = os.path.join(debug_dir, f"gemini_parsed_json_batch{batch_idx+1}_{timestamp}_attempt{retry_count}.json")
                            with open(parsed_json_file, 'w', encoding='utf-8') as f:
                                json.dump(dialogue_result, f, indent=2, ensure_ascii=False)
                            logger.info(f"Saved parsed JSON to {parsed_json_file}")
                            
                            # Store the result in cache
                            api_cache[cache_key] = dialogue_result
                            
                            # Update the cache file after each successful batch
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump(api_cache, f, indent=2, ensure_ascii=False)
                            logger.info(f"Updated API cache file with {len(api_cache)} entries")
                            
                            # Store the modified dialogues
                            for scene_key, scene_data in dialogue_result.get("scenes", {}).items():
                                if "dialogue" in scene_data and "transcript" in scene_data["dialogue"]:
                                    modified_dialogues[scene_key] = scene_data["dialogue"]["transcript"]
                        else:
                            logger.error(f"All JSON parsing approaches failed for batch {batch_idx+1}")
                            time.sleep(10)  # Increased wait before retry
                            continue
                    
                    except Exception as e:
                        logger.error(f"Unexpected error during API call attempt {retry_count} for batch {batch_idx+1}: {str(e)}")
                        time.sleep(10)  # Increased wait before retry
                
                # Enforce a cooldown period between batches to avoid rate limits
                if batch_idx < total_batches - 1:
                    cooldown_time = 15  # 15 second cooldown between batches
                    logger.info(f"Cooling down for {cooldown_time} seconds before next batch to avoid rate limits...")
                    time.sleep(cooldown_time)
            
            # After processing all batches, update the original JSON with the modified dialogues
            if not modified_dialogues:
                logger.warning("No modified dialogues were successfully obtained. Using fallback character identification.")
                enhanced_data = self._apply_fallback_character_identification(enhanced_data)
            else:
                # Apply the modifications to the original JSON
                logger.info(f"Applying {len(modified_dialogues)} modified dialogues to the original JSON")
                
                for scene_key, transcript in modified_dialogues.items():
                    if scene_key in enhanced_data.get("scenes", {}) and "dialogue" in enhanced_data["scenes"][scene_key]:
                        enhanced_data["scenes"][scene_key]["dialogue"]["transcript"] = transcript
                
                # Check if there are any remaining UNKNOWN labels
                remaining_unknown = 0
                for scene_key, scene in enhanced_data.get("scenes", {}).items():
                    if isinstance(scene, dict) and "dialogue" in scene:
                        transcript = scene["dialogue"].get("transcript", "")
                        if transcript and "UNKNOWN" in transcript:
                            remaining_unknown += transcript.count("UNKNOWN")
                
                if remaining_unknown > 0:
                    logger.warning(f"There are still {remaining_unknown} 'UNKNOWN' speaker labels remaining. Applying fallback character identification.")
                    enhanced_data = self._apply_fallback_character_identification(enhanced_data)
                else:
                    logger.info("All 'UNKNOWN' speaker labels have been successfully replaced")
            
            # Save the enhanced data
            with open(structured_data_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Dialogue enhancement completed")
        
        except Exception as e:
            logger.error(f"Error in _enhance_dialogue_with_gemini: {str(e)}")
            logger.info("Using fallback character identification due to unexpected error")
            enhanced_data = self._apply_fallback_character_identification(enhanced_data)
            
            # Save the enhanced data
            with open(structured_data_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    def _enhance_dialogue_in_batches(self, structured_data: Dict, structured_data_path: str, model) -> None:
        """
        Process scenes in batches to enhance dialogue when dealing with large videos.
        
        Args:
            structured_data: The full structured data
            structured_data_path: Path to save the enhanced data
            model: The Gemini API model to use
        """
        logger.info("Processing dialogue enhancement in batches")
        
        # Create a deep copy to avoid modifying the original
        enhanced_data = copy.deepcopy(structured_data)
        scenes = enhanced_data.get("scenes", {})
        
        # Extract potential character names from all scene descriptions
        all_character_names = []
        name_patterns = [
            r'\b([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\b',  # Capitalized words (single or double) that might be names
            r'(?:named|called|is|as) ([A-Z][a-z]+)',  # Words after "named" or "called" or "is" or "as"
            r'([A-Z][a-z]+)(?:\'s| is| was| looks| seems| appears)',  # Words before possessive or certain verbs
            r'(?:character|person|woman|man|girl|boy|child|lady|gentleman|actor|actress)\s+([A-Z][a-z]+)',  # Character descriptors
            r'"([A-Z][a-z]+)"',  # Names in quotes
            r'protagonist (?:is |named |called )?([A-Z][a-z]+)', # Protagonist references
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.) ([A-Z][a-z]+)' # Titles followed by names
        ]
        
        # First pass: extract potential character names from all scene descriptions
        for scene_key, scene in scenes.items():
            if isinstance(scene, dict):
                # Extract from description
                desc = scene.get("description", "") + " " + scene.get("setting", "")
                for pattern in name_patterns:
                    matches = re.findall(pattern, desc)
                    for match in matches:
                        if isinstance(match, str) and len(match) > 2 and match not in ["The", "She", "His", "Her", "They"]:
                            all_character_names.append(match)
                        elif isinstance(match, tuple):  # Handle tuple results from regex groups
                            for submatch in match:
                                if len(submatch) > 2 and submatch not in ["The", "She", "His", "Her", "They"]:
                                    all_character_names.append(submatch)
                
        # Remove duplicates while preserving order
        all_character_names = list(dict.fromkeys(all_character_names))
        logger.info(f"Extracted {len(all_character_names)} potential character names from all scenes")
        
        # Create batches of scenes (maximum 5 scenes per batch)
        batch_size = 5
        scene_keys = list(scenes.keys())
        total_batches = (len(scene_keys) + batch_size - 1) // batch_size
        
        # Process each batch
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(scene_keys))
            batch_keys = scene_keys[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} with scenes {batch_keys}")
            
            # Create a batch of scenes
            batch_data = {
                "video_info": structured_data.get("video_info", {}),
                "scenes": {k: scenes[k] for k in batch_keys}
            }
            
            # Add known character names to prompt context
            character_context = ""
            if all_character_names:
                character_context = f"Known character names from the video: {', '.join(all_character_names[:20])}"
            
            # Create prompt for this batch
            prompt = f"""
            You are a character identification expert for dialogue analysis. Your task is to review this JSON structure 
            and replace all "UNKNOWN" speaker labels in dialogue transcripts with appropriate character names.
            
            {character_context}
            
            Guidelines:
            - Assign consistent character names throughout all scenes
            - Use contextual clues from scene descriptions, dialogue content, and how characters address each other
            - Use realistic character names, not labels like "Character 1"
            - ONLY replace the "UNKNOWN" speaker labels in dialogue transcripts
            - DO NOT modify any other part of the JSON structure
            - Preserve all JSON structure, formatting, and non-speaker information
            
            Here is the partial structured data (in JSON format) for scenes {batch_keys}:
            ```json
            {json.dumps(batch_data, indent=2)}
            ```
            
            Return ONLY the complete modified JSON for these scenes with "UNKNOWN" speakers replaced by character names.
            Do not include any explanation or comments.
            """
            
            try:
                # Set a timeout to prevent hanging
                import threading
                response = None
                response_received = threading.Event()
                
                def make_api_call():
                    nonlocal response
                    try:
                        response = model.generate_content(prompt)
                        response_received.set()
                    except Exception as e:
                        logger.error(f"Batch {batch_idx+1} API call error: {str(e)}")
                        response_received.set()
                
                # Start API call in a separate thread
                api_thread = threading.Thread(target=make_api_call)
                api_thread.daemon = True
                api_thread.start()
                
                # Wait for the API call to complete with a timeout
                api_success = response_received.wait(timeout=30)  # 30 second timeout
                
                if not api_success or not response:
                    logger.warning(f"Batch {batch_idx+1} API call timed out or failed")
                    continue
                
                # Process the response
                response_text = response.text.strip()
                
                # Extract JSON from code blocks if present
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.rfind("```")
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.rfind("```")
                    response_text = response_text[json_start:json_end].strip()
                
                # Parse the response JSON
                batch_result = json.loads(response_text)
                
                # Update the scenes in our enhanced data
                for k in batch_keys:
                    if k in batch_result.get("scenes", {}):
                        # Only update dialogue fields to avoid overwriting other data
                        if "dialogue" in batch_result["scenes"][k] and "dialogue" in enhanced_data["scenes"][k]:
                            enhanced_data["scenes"][k]["dialogue"] = batch_result["scenes"][k]["dialogue"]
                            
                logger.info(f"Successfully processed batch {batch_idx+1}")
                
                # Save progress after each batch
                with open(structured_data_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                # Continue with next batch
        
        # Verify results and apply fallback for any remaining UNKNOWN labels
        has_unknown = False
        for scene_idx, scene in enhanced_data.get("scenes", {}).items():
            if isinstance(scene, dict) and "dialogue" in scene and "transcript" in scene["dialogue"]:
                if "UNKNOWN" in scene["dialogue"]["transcript"]:
                    has_unknown = True
                    logger.warning(f"Scene {scene_idx} still contains UNKNOWN speakers after batch processing")
        
        if has_unknown:
            logger.info("Applying fallback character identification for remaining UNKNOWN speakers")
            enhanced_data = self._apply_fallback_character_identification(enhanced_data)
            
            # Save the final enhanced data
            with open(structured_data_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    def _format_timestamp(self, start_time, end_time=None):
        """Format time as HH:MM:SS or time range if end_time is provided."""
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        if end_time is None:
            return format_time(start_time)
        else:
            return f"{format_time(start_time)} - {format_time(end_time)}"
            
    def _format_time(self, seconds):
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _extract_scene_video_segments(self, video_path: str, scenes: List[Dict], output_dir: str) -> Dict[int, str]:
        """
        Extract short video segments for each scene for more detailed analysis.
        
        Args:
            video_path: Path to the original video
            scenes: List of scene dictionaries with timestamps
            output_dir: Output directory to save video segments
            
        Returns:
            Dictionary mapping scene indices to video segment paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_videos = {}
        
        # For each scene, extract a video segment
        for i, scene in enumerate(scenes):
            start_time = scene.get("start_time", 0)
            end_time = scene.get("end_time", 0)
            duration = end_time - start_time
            
            # Skip very long scenes to avoid memory issues
            if duration > 60:  # Only process scenes up to 60 seconds
                start_time = max(0, end_time - 30)  # Take the last 30 seconds if scene is too long
                duration = end_time - start_time
                
            output_path = output_dir / f"scene_{i:04d}.mp4"
            
            try:
                # Use ffmpeg to extract the segment
                (
                    ffmpeg
                    .input(video_path, ss=start_time, t=duration)
                    .output(str(output_path), c="copy")
                    .global_args('-loglevel', 'error', '-y')
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                scene_videos[i] = str(output_path)
                logger.info(f"Extracted scene {i} video to {output_path}")
            except Exception as e:
                logger.error(f"Error extracting scene {i} video: {str(e)}")
        
        return scene_videos

    def _apply_fallback_character_identification(self, data: Dict) -> Dict:
        """
        Apply rule-based fallback character identification for any remaining UNKNOWN speakers.
        
        Args:
            data: The structured analysis data, potentially with some UNKNOWN speakers
            
        Returns:
            The data with remaining UNKNOWN speakers replaced with best-guess names
        """
        logger.info("Applying fallback character identification for any remaining UNKNOWN speakers")
        
        # Deep copy to avoid modifying the original
        enhanced_data = copy.deepcopy(data)
        
        # Extract character names from scene descriptions
        character_names = []
        name_patterns = [
            r'\b([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\b',  # Capitalized words (single or double) that might be names
            r'(?:named|called|is|as) ([A-Z][a-z]+)',  # Words after "named" or "called" or "is" or "as"
            r'([A-Z][a-z]+)(?:\'s| is| was| looks| seems| appears)',  # Words before possessive or certain verbs
            r'(?:character|person|woman|man|girl|boy|child|lady|gentleman|actor|actress)\s+([A-Z][a-z]+)',  # Character descriptors
            r'"([A-Z][a-z]+)"',  # Names in quotes
            r'protagonist (?:is |named |called )?([A-Z][a-z]+)', # Protagonist references
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.) ([A-Z][a-z]+)' # Titles followed by names
        ]

        # Words that shouldn't be treated as names even if they match patterns
        stopwords = {
            # Basic pronouns and articles
            "The", "She", "His", "Her", "They", "This", "That", "There", "These", "Those", 
            # Scene description terms
            "Scene", "Setting", "Action", "Plot", "Interior", "Exterior", "Day", "Night",
            # Common words in scene descriptions that are often capitalized
            "Okay", "Production", "Distribution", "Following", "Turkey", "Characterization", 
            "Emotions", "However", "Dialogue", "Communication", "Cinematography", "Visual", 
            "Composition", "Symbolism", "Themes", "Temporal", "Shifts", "Narrative", 
            "Structure", "Atmosphere", "Production", "Technical", "Aspects", "Continuity",
            # Analytical terms
            "Analysis", "Summary", "Description", "Overview", "Introduction", "Conclusion",
            # Time/place/setting terms
            "Morning", "Evening", "Afternoon", "Night", "Today", "Tomorrow", "Yesterday",
            "Inside", "Outside", "Location", "Setting", "Background", "Foreground",
            # Emotional states
            "Happy", "Sad", "Angry", "Frustrated", "Confused", "Surprised", "Shocked",
            "Worried", "Concerned", "Anxious", "Relieved", "Excited", "Bored", "Tired",
            # Direction words
            "North", "South", "East", "West", "Central", "Upper", "Lower", "Middle"
        }
        
        # Role-based names to extract from dialogue
        role_names = {
            "mom": ["Mom", "Mother", "Mommy"],
            "dad": ["Dad", "Father", "Daddy", "Papa"],
            "grandma": ["Grandmother", "Grandma", "Granny", "Nani", "Beeji"],
            "grandpa": ["Grandfather", "Grandpa", "Gramps", "Nana", "Dada"],
            "sister": ["Sister", "Sis"],
            "brother": ["Brother", "Bro"],
            "aunt": ["Aunt", "Auntie"],
            "uncle": ["Uncle"],
            "sir": ["Sir"],
            "madam": ["Madam", "Ma'am"],
            "teacher": ["Teacher", "Professor", "Instructor"],
            "doctor": ["Doctor", "Dr"],
            "boss": ["Boss", "Manager", "Chief"]
        }
        
        # Common character names to use as fallback
        common_names = [
            "James", "John", "Michael", "David", "Robert", "Sarah", "Maria", 
            "Anna", "Emily", "Lisa", "Thomas", "Daniel", "William", "Jennifer",
            "Elizabeth", "Karen", "Patricia", "Susan", "Jessica", "Richard",
            "Alex", "Chris", "Sam", "Taylor", "Jordan", "Morgan", "Jamie",
            "Avery", "Riley", "Casey", "Skyler", "Dakota", "Charlie", "Jules"
        ]
        
        # First pass: extract all known non-UNKNOWN speakers from dialogue
        known_speakers = set()
        
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict) and "dialogue" in scene:
                transcript = scene["dialogue"].get("transcript", "")
                if transcript:
                    # Find non-UNKNOWN speakers in the transcript
                    speaker_matches = re.findall(r'([A-Za-z\']+)(?: \[\d|\:)', transcript)
                    for match in speaker_matches:
                        if match != "UNKNOWN":
                            known_speakers.add(match)
                            character_names.append(match)
        
        logger.info(f"Found {len(known_speakers)} known speakers in dialogues: {', '.join(list(known_speakers)[:10])}{' and more' if len(known_speakers) > 10 else ''}")
        
        # Second pass: extract names from character lists in scenes
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict) and "characters" in scene:
                chars = scene.get("characters", [])
                if isinstance(chars, list):
                    for char in chars:
                        if isinstance(char, str) and char not in stopwords:
                            character_names.append(char)
        
        # Third pass: extract potential character names from scene descriptions
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict):
                # Extract from description and setting fields
                desc = scene.get("description", "") + " " + scene.get("setting", "")
                
                # Find all sentences mentioning characters
                character_sentences = []
                sentences = re.split(r'(?<=[.!?])\s+', desc)
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ["character", "protagonist", "person", "man", "woman", "boy", "girl", "named", "called"]):
                        character_sentences.append(sentence)
                
                # If we have character sentences, prioritize extracting names from them
                if character_sentences:
                    for sentence in character_sentences:
                        for pattern in name_patterns:
                            matches = re.findall(pattern, sentence)
                            for match in matches:
                                if isinstance(match, str):
                                    # Handle string matches
                                    if len(match) > 2 and match not in stopwords:
                                        character_names.append(match)
                                elif isinstance(match, tuple):
                                    # Handle tuple matches from regex groups
                                    for submatch in match:
                                        if len(submatch) > 2 and submatch not in stopwords:
                                            character_names.append(submatch)
                
                # Apply the patterns to the whole description as well
                for pattern in name_patterns:
                    matches = re.findall(pattern, desc)
                    for match in matches:
                        if isinstance(match, str):
                            # Handle string matches
                            if len(match) > 2 and match not in stopwords:
                                character_names.append(match)
                        elif isinstance(match, tuple):
                            # Handle tuple matches from regex groups
                            for submatch in match:
                                if len(submatch) > 2 and submatch not in stopwords:
                                    character_names.append(submatch)
                
                # Extract from dialogue content for role-based names
                if "dialogue" in scene:
                    transcript = scene["dialogue"].get("transcript", "")
                    
                    # Look for role indicators in dialogue
                    for role_key, role_variations in role_names.items():
                        for role in role_variations:
                            if re.search(r'\b' + re.escape(role) + r'\b', transcript, re.IGNORECASE):
                                character_names.append(role)
                                
                    # Look for direct address patterns in dialogue
                    if transcript:
                        # Direct address patterns like "Hey, [Name]" or "[Name], I..."
                        direct_address = re.findall(r'(?:Hey,?\s+|Hi,?\s+|Hello,?\s+|Okay,?\s+|Listen,?\s+)([A-Z][a-z]+)', transcript)
                        for name in direct_address:
                            if name not in stopwords and len(name) > 2:
                                character_names.append(name)
                                
                        # Look for speaking patterns - who's talking to whom
                        speaking_to = re.findall(r'(?:said to|tells|asked|speaking to|talking to|responded to)\s+([A-Z][a-z]+)', transcript)
                        for name in speaking_to:
                            if name not in stopwords and len(name) > 2:
                                character_names.append(name)
        
        # Remove duplicates while preserving order and standardize case (capitalize first letter)
        filtered_character_names = []
        seen = set()
        for name in character_names:
            standardized_name = name[0].upper() + name[1:] if name else ""
            if standardized_name and standardized_name not in seen and standardized_name not in stopwords:
                filtered_character_names.append(standardized_name)
                seen.add(standardized_name)
        
        character_names = filtered_character_names
        
        logger.info(f"Extracted {len(character_names)} potential character names: {', '.join(character_names[:10])}" + 
                   ("..." if len(character_names) > 10 else ""))
        
        # Check if we have actual character names or just descriptive words
        valid_names = [name for name in character_names if len(name) > 2 and name not in stopwords]
        
        # If no valid names found, use common names instead
        if not valid_names:
            logger.info("No valid character names found, using common names instead")
            character_names = common_names
            
        # If we have dialogue, analyze dialogue patterns for better character identification
        dialogue_patterns = {}
        speaker_assignments = {}
        
        # Track consistent speech patterns of characters
        speech_patterns = {}  # Maps character names to their speech patterns
        topic_interests = {}  # Maps character names to topics they discuss
        
        # First analyze speech patterns for known speakers
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict) and "dialogue" in scene:
                transcript = scene["dialogue"].get("transcript", "")
                if transcript:
                    lines = transcript.split('\n')
                    for line in lines:
                        speaker_match = re.match(r'([A-Za-z\']+) \[\d+:\d+ - \d+:\d+\]:\s*(.*)', line)
                        if speaker_match and speaker_match.group(1) != "UNKNOWN":
                            speaker = speaker_match.group(1)
                            text = speaker_match.group(2).lower()
                            
                            # Analyze speech pattern
                            if speaker not in speech_patterns:
                                speech_patterns[speaker] = {
                                    "avg_words": 0,
                                    "total_words": 0,
                                    "line_count": 0,
                                    "question_ratio": 0,
                                    "exclamation_ratio": 0,
                                    "common_words": {}
                                }
                                
                            # Count words
                            words = re.findall(r'\b\w+\b', text)
                            speech_patterns[speaker]["total_words"] += len(words)
                            speech_patterns[speaker]["line_count"] += 1
                            
                            # Track common words
                            for word in words:
                                if len(word) > 3:  # Skip short words
                                    if word not in speech_patterns[speaker]["common_words"]:
                                        speech_patterns[speaker]["common_words"][word] = 0
                                    speech_patterns[speaker]["common_words"][word] += 1
                                    
                            # Check for questions and exclamations
                            if "?" in text:
                                speech_patterns[speaker]["question_ratio"] += 1
                            if "!" in text:
                                speech_patterns[speaker]["exclamation_ratio"] += 1
                                
                            # Update average words
                            if speech_patterns[speaker]["line_count"] > 0:
                                speech_patterns[speaker]["avg_words"] = speech_patterns[speaker]["total_words"] / speech_patterns[speaker]["line_count"]
                                speech_patterns[speaker]["question_ratio"] = speech_patterns[speaker]["question_ratio"] / speech_patterns[speaker]["line_count"]
                                speech_patterns[speaker]["exclamation_ratio"] = speech_patterns[speaker]["exclamation_ratio"] / speech_patterns[speaker]["line_count"]
        
        # Group similar dialogue patterns to help identify the same speaker across scenes
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict) and "dialogue" in scene:
                transcript = scene["dialogue"].get("transcript", "")
                if "UNKNOWN" in transcript:
                    lines = transcript.split('\n')
                    for line in lines:
                        if "UNKNOWN [" in line:
                            # Extract dialogue content (after the colon)
                            match = re.match(r'UNKNOWN \[(.*?)\]:\s*(.*)', line)
                            if match:
                                timestamp = match.group(1)
                                text = match.group(2).lower()
                                
                                # Create a dialogue fingerprint based on characteristics
                                words = re.findall(r'\b\w+\b', text)
                                word_count = len(words)
                                has_question = "?" in text
                                has_exclamation = "!" in text
                                first_five_words = " ".join(words[:min(5, len(words))]) if words else ""
                                
                                # Build a fingerprint that combines multiple characteristics
                                fingerprint = f"{word_count}:{has_question}:{has_exclamation}:{first_five_words}"
                                
                                if fingerprint not in dialogue_patterns:
                                    dialogue_patterns[fingerprint] = []
                                
                                # Store scene_key, timestamp and the full line 
                                dialogue_patterns[fingerprint].append((scene_key, timestamp, line, text))
        
        # Now go through and replace UNKNOWN speakers with consistent names
        character_name_idx = 0
        scene_speaker_mapping = {}  # Maps scene -> unknown speaker position -> character name
        
        # First, match UNKNOWN speakers to known speakers based on speech patterns
        unknown_speech_patterns = {}
        
        # First extract speech patterns for all UNKNOWN speakers
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict) and "dialogue" in scene:
                transcript = scene["dialogue"].get("transcript", "")
                if "UNKNOWN" in transcript:
                    lines = transcript.split('\n')
                    
                    # Count positions of UNKNOWNs in this scene
                    unknown_positions = {}
                    unknown_pos = 0
                    
                    for line in lines:
                        match = re.match(r'UNKNOWN \[(.*?)\]:\s*(.*)', line)
                        if match:
                            timestamp = match.group(1)
                            text = match.group(2).lower()
                            
                            # Assign a position ID to this UNKNOWN
                            if scene_key not in unknown_positions:
                                unknown_positions[scene_key] = {}
                            
                            key = f"{scene_key}_{unknown_pos}"
                            unknown_positions[scene_key][unknown_pos] = (timestamp, text)
                            
                            # Create speech pattern
                            if key not in unknown_speech_patterns:
                                unknown_speech_patterns[key] = {
                                    "avg_words": 0,
                                    "total_words": 0,
                                    "line_count": 0,
                                    "question_ratio": 0,
                                    "exclamation_ratio": 0,
                                    "common_words": {}
                                }
                            
                            # Count words
                            words = re.findall(r'\b\w+\b', text)
                            unknown_speech_patterns[key]["total_words"] += len(words)
                            unknown_speech_patterns[key]["line_count"] += 1
                            
                            # Track common words
                            for word in words:
                                if len(word) > 3:  # Skip short words
                                    if word not in unknown_speech_patterns[key]["common_words"]:
                                        unknown_speech_patterns[key]["common_words"][word] = 0
                                    unknown_speech_patterns[key]["common_words"][word] += 1
                                    
                            # Check for questions and exclamations
                            if "?" in text:
                                unknown_speech_patterns[key]["question_ratio"] += 1
                            if "!" in text:
                                unknown_speech_patterns[key]["exclamation_ratio"] += 1
                                
                            unknown_pos += 1
                            
                    # Calculate average words for each UNKNOWN
                    for key in unknown_speech_patterns:
                        if unknown_speech_patterns[key]["line_count"] > 0:
                            unknown_speech_patterns[key]["avg_words"] = unknown_speech_patterns[key]["total_words"] / unknown_speech_patterns[key]["line_count"]
                            unknown_speech_patterns[key]["question_ratio"] = unknown_speech_patterns[key]["question_ratio"] / unknown_speech_patterns[key]["line_count"]
                            unknown_speech_patterns[key]["exclamation_ratio"] = unknown_speech_patterns[key]["exclamation_ratio"] / unknown_speech_patterns[key]["line_count"]
        
        # Try to match unknown speakers to known speakers by speech pattern similarity
        pattern_match_map = {}  # Maps unknown speaker key to character name
        
        for unknown_key, unknown_pattern in unknown_speech_patterns.items():
            best_match = None
            best_score = -1
            
            for speaker, known_pattern in speech_patterns.items():
                score = 0
                
                # Compare word count
                word_diff = abs(unknown_pattern["avg_words"] - known_pattern["avg_words"])
                word_score = 1 - min(1, word_diff / max(1, known_pattern["avg_words"]))
                score += word_score * 2  # Weight word count more heavily
                
                # Compare question ratio
                q_diff = abs(unknown_pattern["question_ratio"] - known_pattern["question_ratio"])
                q_score = 1 - min(1, q_diff * 5)  # Larger penalty
                score += q_score
                
                # Compare exclamation ratio
                e_diff = abs(unknown_pattern["exclamation_ratio"] - known_pattern["exclamation_ratio"])
                e_score = 1 - min(1, e_diff * 5)  # Larger penalty
                score += e_score
                
                # Compare common words
                word_match = 0
                for word, count in unknown_pattern["common_words"].items():
                    if word in known_pattern["common_words"]:
                        word_match += 1
                
                # Normalized word match score
                total_words = len(unknown_pattern["common_words"])
                if total_words > 0:
                    word_match_score = word_match / total_words
                    score += word_match_score * 3  # Weight common words heavily
                
                # Final score
                final_score = score / 7  # Normalize
                
                if final_score > best_score and final_score > 0.5:  # Threshold for match
                    best_score = final_score
                    best_match = speaker
            
            if best_match:
                pattern_match_map[unknown_key] = best_match
                logger.info(f"Matched unknown speaker {unknown_key} to character {best_match} with score {best_score:.2f}")
                
        # Now go through and apply all the identified mappings
        for scene_key, scene in enhanced_data["scenes"].items():
            if isinstance(scene, dict) and "dialogue" in scene:
                transcript = scene["dialogue"].get("transcript", "")
                if "UNKNOWN" in transcript:
                    lines = transcript.split('\n')
                    new_lines = []
                    
                    # Count UNKNOWNs in this scene
                    unknown_pos = 0
                    
                    for line in lines:
                        if "UNKNOWN [" in line:
                            # Get the timestamp and text
                            match = re.match(r'UNKNOWN \[(.*?)\]:\s*(.*)', line)
                            if match:
                                timestamp = match.group(1)
                                text = match.group(2)
                                
                                # Check if this UNKNOWN is matched by speech pattern
                                key = f"{scene_key}_{unknown_pos}"
                                if key in pattern_match_map:
                                    # Use the matched character name
                                    character_name = pattern_match_map[key]
                                    new_line = f"{character_name} [{timestamp}]: {text}"
                                    new_lines.append(new_line)
                                else:
                                    # No match by speech pattern, try dialogue patterns
                                    words = re.findall(r'\b\w+\b', text.lower())
                                    word_count = len(words)
                                    has_question = "?" in text
                                    has_exclamation = "!" in text
                                    first_five_words = " ".join(words[:min(5, len(words))]) if words else ""
                                    
                                    fingerprint = f"{word_count}:{has_question}:{has_exclamation}:{first_five_words}"
                                    
                                    if fingerprint in dialogue_patterns:
                                        # Look for this pattern in other scenes
                                        for other_scene, other_timestamp, other_line, other_text in dialogue_patterns[fingerprint]:
                                            # Skip if it's this scene/line
                                            if other_scene == scene_key and other_timestamp == timestamp:
                                                continue
                                                
                                            # If there's a match and we've assigned a character already
                                            if other_scene in scene_speaker_mapping and other_scene != scene_key:
                                                # Find what position this was in the other scene
                                                other_pos = 0
                                                found_match = False
                                                # Loop through all entries to find the right position
                                                for entry in dialogue_patterns[fingerprint]:
                                                    entry_scene, entry_ts, _, _ = entry
                                                    if entry_scene == other_scene and entry_ts == other_timestamp:
                                                        if other_pos in scene_speaker_mapping.get(other_scene, {}):
                                                            assigned_name = scene_speaker_mapping[other_scene][other_pos]
                                                            
                                                            # Use this name
                                                            new_line = f"{assigned_name} [{timestamp}]: {text}"
                                                            new_lines.append(new_line)
                                                            
                                                            # Remember this assignment for this scene
                                                            if scene_key not in scene_speaker_mapping:
                                                                scene_speaker_mapping[scene_key] = {}
                                                            scene_speaker_mapping[scene_key][unknown_pos] = assigned_name
                                                            found_match = True
                                                            break
                                                    other_pos += 1
                                                
                                                if found_match:
                                                    break
                                    
                                    # If no assignment was made, assign a new character
                                    if len(new_lines) == 0 or new_lines[-1] != line:
                                        # If this position isn't mapped yet
                                        if scene_key not in scene_speaker_mapping or unknown_pos not in scene_speaker_mapping[scene_key]:
                                            # Role-based assignment based on dialogue content
                                            role_assigned = False
                                            
                                            # Check for role names in the text
                                            for role_key, role_variations in role_names.items():
                                                for role in role_variations:
                                                    # Check exact match with word boundaries
                                                    if re.search(r'\b' + re.escape(role) + r'\b', text, re.IGNORECASE):
                                                        # If someone is talking about/to this role, make them the role
                                                        if scene_key not in scene_speaker_mapping:
                                                            scene_speaker_mapping[scene_key] = {}
                                                        scene_speaker_mapping[scene_key][unknown_pos] = role
                                                        new_line = f"{role} [{timestamp}]: {text}"
                                                        new_lines.append(new_line)
                                                        role_assigned = True
                                                        break
                                                if role_assigned:
                                                    break
                                            
                                            # If no role assigned, use the next character name
                                            if not role_assigned:
                                                if character_name_idx < len(character_names):
                                                    character_name = character_names[character_name_idx]
                                                    character_name_idx += 1
                                                else:
                                                    # Cycle back through names
                                                    character_name_idx = 0
                                                    character_name = character_names[character_name_idx]
                                                    character_name_idx += 1
                                                
                                                if scene_key not in scene_speaker_mapping:
                                                    scene_speaker_mapping[scene_key] = {}
                                                scene_speaker_mapping[scene_key][unknown_pos] = character_name
                                                new_line = f"{character_name} [{timestamp}]: {text}"
                                                new_lines.append(new_line)
                                    
                                unknown_pos += 1
                        else:
                            # Keep non-UNKNOWN lines as they are
                            new_lines.append(line)
                    
                    # Update the transcript
                    scene["dialogue"]["transcript"] = "\n".join(new_lines)
        
        logger.info("Fallback character identification completed with enhanced pattern matching")
        return enhanced_data

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Video Analysis Pipeline")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory for results")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--skip", nargs="+", choices=[
        "scene_detection", "keyframe_extraction", "entity_detection", 
        "audio_processing", "gemini_vision"
    ], help="Steps to skip (if already processed)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", 
                        help="Device to use for computation")
    
    args = parser.parse_args()
    
    # Set device environment variable
    if args.device == "cpu":
        os.environ["FORCE_CPU"] = "1"
        
    try:
        # Initialize pipeline
        pipeline = VideoPipeline(args.config)
        
        # Process video
        results = pipeline.process_video(Path(args.video_path), Path(args.output), args.skip)
        
        logger.info(f"Analysis results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 