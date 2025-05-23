"""Scene detection component for the video analysis pipeline."""

import os
# Set environment variables to handle NCCL issues before importing PyTorch
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer which can cause symbol errors
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import gc
from PIL import Image
import open_clip
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine
import copy
import random
import importlib.util
import sys
import tempfile
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Enhanced scene detection for video analysis.
    Uses CLIP-based frame similarity with semantic clustering to create coherent scenes.
    Optionally uses SceneSeg for more advanced scene detection.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the SceneDetector with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Set scene detection parameters
        self.threshold = self.config.get("threshold", 0.65)
        self.min_scene_length = self.config.get("min_scene_length", 0.0)  # No time constraint by default
        self.merge_threshold = self.config.get("merge_threshold", 0.8)  # Threshold for merging similar scenes
        
        # Add missing parameters needed by the scene detection algorithm
        self.similarity_window = self.config.get("similarity_window", 6)  # Window size for similarity smoothing
        self.detect_multiple_storylines = self.config.get("detect_multiple_storylines", False)
        self.storyline_similarity_threshold = self.config.get("storyline_similarity_threshold", 0.75)
        self.min_viable_scene_duration = self.config.get("min_viable_scene_duration", 15.0)
        self.max_temporal_gap = self.config.get("max_temporal_gap", 3.0)
        
        # Disable SceneSeg by default
        self.use_sceneseg = False
        self.sceneseg_path = None
        self.sceneseg_config = None
        
        # Set up device for processing
        self.device = self._initialize_device()
        
        # Log configuration
        logger.info(f"Initializing SceneDetector with content-based parameters: threshold={self.threshold}, merge_threshold={self.merge_threshold}, no time constraints applied")
        
        # Initialize scene detection model (CLIP for content-based detection)
        logger.info(f"Loading CLIP model ViT-L-14 on {self.device}...")
        try:
            model_and_transforms = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=self.device
            )
            self.model = model_and_transforms[0]
            self.preprocess = model_and_transforms[1]
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def _check_sceneseg_available(self):
        """Check if SceneSeg is available."""
        # Always return False to use built-in method instead
        logger.warning("SceneSeg disabled: using built-in scene detection method")
        return False
        
        # Original code below is commented out
        """
        try:
            if self.sceneseg_path:
                sys.path.append(self.sceneseg_path)
                
            from sceneseg.utils.config import Config
            from sceneseg.apis.inference import inference_model
            logger.info("SceneSeg is available")
            return True
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"SceneSeg not available: {str(e)}. Using built-in detection method.")
            return False
        """
    
    def _initialize_device(self) -> str:
        """Initialize the device (CUDA or CPU) based on configuration and availability."""
        config_device = self.config.get("device")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA is available")
            
            # Test CUDA to make sure it works
            try:
                test_tensor = torch.zeros(1).cuda()
                logger.info("CUDA test successful")
            except Exception as e:
                logger.warning(f"CUDA test failed: {str(e)}")
                device = "cpu"
                logger.info("Falling back to CPU")
        else:
            device = "cpu"
            logger.info("CUDA is not available, using CPU")
        
        # Override with config if specified
        if config_device:
            if config_device == "cuda" and device == "cpu":
                logger.warning("CUDA specified in config but not available or not working")
            else:
                device = config_device
                logger.info(f"Using device specified in config: {device}")
        
        return device
    
    def detect_scenes(self, video_path: str, output_dir: str = None) -> List[Dict]:
        """
        Detect scenes in a video.
        
        Args:
            video_path: Path to the video file
            output_dir: Optional directory to save scene detection results
            
        Returns:
            List of scene dictionaries
        """
        logger.info(f"Detecting scenes for video: {video_path}")
        
        # Directly use the built-in method without checking for SceneSeg
        logger.info("Using built-in scene detection method")
        return self._detect_scenes_builtin(video_path, output_dir)
    
    def _detect_scenes_with_sceneseg(self, video_path: str, output_dir: str = None) -> List[Dict]:
        """
        Detect scenes using SceneSeg.
        
        Args:
            video_path: Path to the video file
            output_dir: Optional directory to save scene detection results
            
        Returns:
            List of scene dictionaries
        """
        logger.info(f"Using SceneSeg for scene detection on {video_path}")
        
        # Import SceneSeg here to avoid dependency issues if not available
        try:
            if self.sceneseg_path:
                sys.path.append(self.sceneseg_path)
                logger.info(f"Added SceneSeg path to sys.path: {self.sceneseg_path}")
            
            # Try importing the required modules
            try:
                from sceneseg.utils.config import Config
                from sceneseg.apis.inference import inference_model
            except ImportError as e:
                logger.error(f"Failed to import SceneSeg modules: {e}")
                logger.error("SceneSeg is not installed or not properly configured.")
                logger.error("Falling back to built-in scene detection method.")
                return self._detect_scenes_builtin(video_path, output_dir)
            
            # Create a temp directory for SceneSeg output if output_dir is not provided
            temp_dir = None
            if output_dir is None:
                temp_dir = tempfile.mkdtemp()
                output_dir = temp_dir
            else:
                # Ensure output directory exists
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Use default config or load from provided path
            if self.sceneseg_config:
                cfg = Config.fromfile(self.sceneseg_config)
            else:
                # Create a simple default config
                # This assumes SceneSeg's default structure
                cfg_dict = {
                    'model': {
                        'type': 'TransNetV2',
                        'backbone': {
                            'pretrained': True
                        }
                    },
                    'data': {
                        'fps': 25,
                        'clip_len': 5,
                    },
                    'test_cfg': {
                        'batch_size': 1
                    }
                }
                
                # Create temporary config file
                config_file = os.path.join(output_dir, 'sceneseg_config.py')
                with open(config_file, 'w') as f:
                    f.write(f"model = {cfg_dict['model']}\n")
                    f.write(f"data = {cfg_dict['data']}\n")
                    f.write(f"test_cfg = {cfg_dict['test_cfg']}\n")
                
                cfg = Config.fromfile(config_file)
            
            # Get video properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            # SceneSeg has a limit on CPU RAM usage, so set environment variables
            os.environ["SCENESEG_MAX_FRAMES"] = str(min(10000, frame_count))
            
            # Run inference
            scenes = inference_model(
                cfg, 
                video_path,
                shot_threshold=self.threshold, 
                scene_threshold=self.merge_threshold
            )
            
            # Convert SceneSeg scenes to our format
            scene_dicts = []
            for i, scene in enumerate(scenes):
                start_frame = scene['start_frame']
                end_frame = scene['end_frame']
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                scene_dict = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "scene_idx": i,
                    "event_coherence": True,  # SceneSeg is designed for content coherence
                }
                
                if 'confidence' in scene:
                    scene_dict["similarity_score"] = scene['confidence']
                
                scene_dicts.append(scene_dict)
            
            # Save scenes if output directory is provided
            if output_dir:
                try:
                    output_path = Path(output_dir) / "scenes.json"
                    self.save_scenes(scene_dicts, str(output_path))
                    logger.info(f"Saved scene detection results to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving scene detection results: {str(e)}")
            
            # Clean up temporary directory if created
            if temp_dir:
                os.rmdir(temp_dir)
            
            return scene_dicts
            
        except Exception as e:
            logger.error(f"Error in SceneSeg scene detection: {str(e)}")
            logger.error("Falling back to built-in scene detection method.")
            return self._detect_scenes_builtin(video_path, output_dir)
    
    def _detect_scenes_with_sceneseg_cli(self, video_path: str, output_dir: str = None) -> List[Dict]:
        """
        Detect scenes using SceneSeg command-line interface.
        This is an alternative approach if the Python API is not working.
        
        Args:
        except Exception as e:\            logger.error(f"Error using SceneSeg for scene detection: {str(e)}")\            logger.error("Falling back to built-in scene detection method.")\            return self._detect_scenes_builtin(video_path, output_dir)
            video_path: Path to the video file
            output_dir: Optional directory to save scene detection results
            
        Returns:
            List of scene dictionaries
        """
        if self.sceneseg_path is None:
            logger.error("SceneSeg path not specified for CLI usage")
            return self._detect_scenes_builtin(video_path, output_dir)
            
        logger.info(f"Using SceneSeg CLI for scene detection on {video_path}")
        
        # Create temporary output directory if needed
        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        # Output JSON file
        output_json = os.path.join(output_dir, "sceneseg_output.json")
        
        # Build command
        cmd = [
            "python", f"{self.sceneseg_path}/tools/inference.py",
            video_path,
            "--out", output_json
        ]
        
        if self.sceneseg_config:
            cmd.extend(["--config", self.sceneseg_config])
        
        # Run SceneSeg CLI
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"SceneSeg CLI failed: {result.stderr}")
                return self._detect_scenes_builtin(video_path, output_dir)
                
            # Load results from JSON
            with open(output_json, 'r') as f:
                scenes_data = json.load(f)
            
        # Get video properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Convert to our format
            scene_dicts = []
            for i, scene in enumerate(scenes_data['scenes']):
                start_frame = scene['start_frame']
                end_frame = scene['end_frame']
                start_time = scene['start_time']
                end_time = scene['end_time']
                
                scene_dict = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "scene_idx": i,
                    "event_coherence": True
                }
                
                scene_dicts.append(scene_dict)
                
            # Save in our format
            if output_dir:
                try:
                    output_path = Path(output_dir) / "scenes.json"
                    self.save_scenes(scene_dicts, str(output_path))
                    logger.info(f"Saved scene detection results to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving scene detection results: {str(e)}")
                    
            # Clean up
            if temp_dir:
                os.rmdir(temp_dir)
                
            return scene_dicts
            
        except Exception as e:
            logger.error(f"Error using SceneSeg CLI: {str(e)}")
            return self._detect_scenes_builtin(video_path, output_dir)
    
    def _detect_scenes_builtin(self, video_path: str, output_dir: str = None) -> List[Dict]:
        """
        Built-in scene detection method with memory optimization.
        
        Args:
            video_path: Path to the video file
            output_dir: Optional directory to save scene detection results
            
        Returns:
            List of scene dictionaries
        """
        logger.info(f"Using built-in scene detection on {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Video properties: {fps:.2f} fps, {frame_count} frames, {duration:.2f} seconds")
        
        # Use a higher sampling rate to better capture content changes but balance with memory usage
        # Increase sampling rate for better scene detection accuracy
        sampling_rate = max(1, int(fps / 4))  # About 4 frames per second for better scene detection
        
        logger.info(f"Using sampling rate of {sampling_rate} ({fps/sampling_rate:.2f} frames per second)")
        
        # Initialize variables for frame processing
        all_embeddings = []
        all_timestamps = []
        all_frame_indices = []
        
        # Set up progress tracking
        progress = tqdm(total=frame_count, desc="Processing frames", unit="frames")
        
        try:
            # Process video in chunks to reduce memory usage
            batch_size = 100  # Process this many frames at a time
            
            for batch_start in range(0, frame_count, batch_size * sampling_rate):
                # Extract frames for this batch
                batch_frames = []
                batch_timestamps = []
                batch_frame_indices = []
                
                try:
                    for i in range(batch_start, min(batch_start + batch_size * sampling_rate, frame_count), sampling_rate):
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ret, frame = cap.read()
                            if not ret:
                                logger.warning(f"Failed to read frame at index {i}")
                                continue
                            
                            # Convert to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            batch_frames.append(frame_rgb)
                            batch_timestamps.append(i / fps)
                            batch_frame_indices.append(i)
                            progress.update(sampling_rate)
                        except Exception as e:
                            logger.error(f"Error processing frame {i}: {str(e)}")
                            continue
                    
                    if not batch_frames:
                        logger.warning(f"No frames extracted for batch starting at {batch_start}")
                        continue
                    
                    # Compute embeddings for this batch
                    batch_embeddings = self._compute_frame_embeddings(batch_frames)
                    
                    # Store results
                    all_embeddings.append(batch_embeddings)
                    all_timestamps.extend(batch_timestamps)
                    all_frame_indices.extend(batch_frame_indices)
                    
                    # Clear batch data to free memory
                    del batch_frames
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error processing batch starting at {batch_start}: {str(e)}")
                    # Continue to next batch
            
            progress.close()
            
            # Close video
            cap.release()
            
            # Concatenate all batch embeddings
            if all_embeddings:
                embeddings = torch.cat(all_embeddings, dim=0)
                logger.info(f"Processed {len(all_timestamps)} frames in total")
            else:
                logger.warning("No frames extracted, creating a single scene")
                return [{
                    "start_frame": 0,
                    "end_frame": frame_count - 1,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration,
                    "scene_idx": 0
                }]
            
            # Skip processing if too few frames
            if len(all_timestamps) < 10:
                logger.warning(f"Too few frames extracted ({len(all_timestamps)}), creating a single scene")
                scenes = [{
                    "start_frame": 0,
                    "end_frame": frame_count - 1,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration
                }]
                
                if output_dir:
                    self.save_scenes(scenes, output_dir)
                
                return scenes
            
            # Detect initial scene boundaries based on visual features
            logger.info("Detecting initial scene boundaries based on content changes...")
            initial_scenes = self._detect_initial_boundaries(embeddings, all_timestamps, all_frame_indices, fps, sampling_rate)
            
            logger.info(f"Detected {len(initial_scenes)} initial scenes")
            
            # Create fallback if no scenes detected
            if len(initial_scenes) < 2:
                logger.warning("No scene boundaries detected, creating fallback scenes")
                initial_scenes = self._create_fallback_scenes(all_timestamps, all_frame_indices, fps)
                logger.info(f"Created {len(initial_scenes)} fallback scenes")
            
            # Cluster scenes by similarity to create semantically coherent scenes
            logger.info(f"Clustering scenes by content similarity (merge_threshold={self.merge_threshold})")
            merged_scenes = self._cluster_scenes_by_similarity(initial_scenes, embeddings, all_timestamps, all_frame_indices)
            
            logger.info(f"Created {len(merged_scenes)} content-based scenes after clustering")
            
            # No duration-based adjustments
            scenes = merged_scenes
            
            # Format scenes as dictionaries
            scene_dicts = []
            for i, scene in enumerate(scenes):
                scene_dict = {
                    "start_frame": scene["start_frame"],
                    "end_frame": scene["end_frame"],
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "duration": scene["duration"],
                    "scene_idx": i
                }
                # Add similarity info if available
                if "similarity_score" in scene:
                    scene_dict["similarity_score"] = scene["similarity_score"]
                if "event_coherence" in scene:
                    scene_dict["event_coherence"] = scene["event_coherence"]
                scene_dicts.append(scene_dict)
            
            # Save scenes if output directory is provided
            if output_dir:
                try:
                    output_path = Path(output_dir) / "scenes.json"
                    self.save_scenes(scene_dicts, str(output_path))
                    logger.info(f"Saved scene detection results to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving scene detection results: {str(e)}")
            
            # Clean up to free memory
            del embeddings, all_embeddings
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return scene_dicts
            
        except Exception as e:
            logger.error(f"Error in scene detection: {str(e)}")
            # Try to release the capture if still open
            if cap is not None and cap.isOpened():
                cap.release()
            
            # Return a single scene as fallback
            logger.warning("Returning a single fallback scene for the entire video")
            return [{
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
                "scene_idx": 0
            }]
    
    def _compute_frame_embeddings(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Compute CLIP embeddings for a list of frames.
        Optimized implementation using mixed precision and batch processing.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Tensor of frame embeddings
        """
        if not frames:
            return torch.zeros((0, 512), device=self.device)  # Return empty tensor with CLIP dimensions
            
        # Process frames in batches
        batch_size = 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                
                # Preprocess frames
                processed_frames = []
                for frame in batch:
                    # Convert to PIL and apply preprocessing
                    pil_img = Image.fromarray(frame)
                    processed = self.preprocess(pil_img)
                    processed_frames.append(processed)
                
                # Stack frames into a batch tensor
                frame_tensor = torch.stack(processed_frames).to(self.device)
                
                # Use mixed precision for efficiency if using CUDA
                if self.device == "cuda":
                    with torch.amp.autocast('cuda', enabled=True):
                        embeddings = self.model.encode_image(frame_tensor)
                else:
                    embeddings = self.model.encode_image(frame_tensor)
                    
                # Normalize embeddings
                #embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  ## check by removing. maybe normalization can cause boundary detection issue
                
                # Store batch embeddings and clear GPU cache after each batch
                all_embeddings.append(embeddings)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
        # Concatenate all batch embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            return embeddings
        else:
            return torch.zeros((0, 512), device=self.device)
    
    def _detect_initial_boundaries(self, 
                          embeddings: torch.Tensor, 
                          timestamps: List[float], 
                          frame_indices: List[int],
                          fps: float, 
                          sampling_rate: int) -> List[Dict]:
        """
        Detect initial scene boundaries based on visual content changes.
        
        This enhanced method identifies meaningful content changes rather than just
        visual differences, focusing on narrative shifts and semantic changes that
        indicate a genuine scene transition. It uses adaptive thresholding based on
        content variation and temporal patterns.
        
        Args:
            embeddings: Tensor of frame embeddings
            timestamps: List of frame timestamps
            frame_indices: List of original frame indices
            fps: Frames per second
            sampling_rate: Frame sampling rate used
            
        Returns:
            List of scene dictionaries
        """
        logger.info("Detecting initial scene boundaries based on content changes...")
        
        if len(embeddings) < 2:
            logger.warning("Too few frames for boundary detection, creating single scene")
            if len(timestamps) > 0:
                return [{
                    "start_frame": frame_indices[0],
                    "end_frame": frame_indices[-1],
                    "start_time": timestamps[0],
                    "end_time": timestamps[-1],
                    "duration": timestamps[-1] - timestamps[0]
                }]
            return []

        # Check if this is a long video and adjust thresholds accordingly
        video_duration = timestamps[-1] - timestamps[0]
        is_long_video = video_duration > 900  # Consider videos over 15 minutes as long
        
        if is_long_video:
            logger.info(f"Long video detected ({video_duration:.2f} seconds). Using more sensitive scene detection.")
            # Use more sensitive threshold for long videos
            content_threshold_adjustment = 0.03  # Less aggressive adjustment for more logical boundaries
            window_adjustment = 1
        else:
            content_threshold_adjustment = 0
            window_adjustment = 0
            
        # Move embeddings to CPU for processing if they're not already there
        if embeddings.device.type != "cpu":
            embeddings = embeddings.cpu()
            
        # Calculate pairwise distances instead of just consecutive frames
        # This captures more meaningful content transitions
        similarities = []
        
        # Use both consecutive and window-based comparisons for better boundary detection
        window = min(self.similarity_window - window_adjustment, len(embeddings) // 4)  
        window = max(window, 3)  # Ensure window is at least 3 frames
        
        # First compute consecutive frame similarities
        for i in range(len(embeddings) - 1):
            # Get embeddings for current and next frame
            current_emb = embeddings[i]
            next_emb = embeddings[i + 1]
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                current_emb.unsqueeze(0), next_emb.unsqueeze(0)
            ).item()
            
            similarities.append(similarity)
        
        # Also compute similarities with a semantic window to capture gradual content shifts
        window_similarities = []
        window_step = 3  # Skip frames for efficiency
        for i in range(0, len(embeddings) - window - 1, window_step):
            # Compare beginning of potential scene with end
            begin_emb = embeddings[i:i+3].mean(dim=0)  # Average first few frames
            end_emb = embeddings[i+window:i+window+3].mean(dim=0)  # Average last few frames
            
            # Calculate similarity between beginning and end of window
            window_similarity = torch.nn.functional.cosine_similarity(
                begin_emb.unsqueeze(0), end_emb.unsqueeze(0)
            ).item()
            
            # Map this window similarity to its center frame
            center_idx = i + window // 2
            if center_idx < len(similarities):
                # Blend with consecutive similarity
                # This makes the algorithm sensitive to both immediate and semantic changes
                window_similarities.append((center_idx, window_similarity))
        
        # Incorporate window similarities to detect semantic changes
        for idx, win_sim in window_similarities:
            if idx < len(similarities):
                # Weight: 70% consecutive similarity, 30% window similarity
                similarities[idx] = 0.7 * similarities[idx] + 0.3 * win_sim
        
        # Smooth the similarities with a rolling window to reduce noise while preserving transitions
        if len(similarities) > window * 2:
            # Use exponential smoothing which preserves transitions better
            smoothed_similarities = []
            alpha = 0.3  # Smoothing factor - lower means more smoothing
            smoothed = similarities[0]
            
            for i in range(1, len(similarities)):
                # Exponential smoothing preserves sudden changes better
                smoothed = alpha * similarities[i] + (1 - alpha) * smoothed
                smoothed_similarities.append(smoothed)
                
            # Prepend first value that got lost in the smoothing
            smoothed_similarities.insert(0, similarities[0])
            
            # If we have fewer smoothed similarities, pad to match original
            while len(smoothed_similarities) < len(similarities):
                smoothed_similarities.append(smoothed_similarities[-1])
            
            similarities = smoothed_similarities
        
        # Calculate the mean and standard deviation of similarities
        mean_sim = sum(similarities) / len(similarities)
        std_sim = (sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)) ** 0.5
        
        # Adaptive thresholding: adjust based on content variance and video type
        # This prevents over-segmentation in high-variance videos and under-segmentation in stable ones
        
        # First, determine video content stability
        if std_sim > 0.15:  # High variation (action/fast-paced content)
            # More conservative for high-variance content to avoid over-segmentation
            content_factor = 1.0 
            logger.info(f"High visual variance detected ({std_sim:.4f}). Using conservative thresholding.")
        else:  # Low variation (stable camera, slow content changes)
            # More sensitive for stable content to catch subtle transitions
            content_factor = 1.5
            logger.info(f"Low visual variance detected ({std_sim:.4f}). Using sensitive thresholding.")
        
        # Adjust base threshold for long videos to catch more meaningful transitions
        adjusted_content_threshold = self.threshold - content_threshold_adjustment
        
        # Compute adaptive threshold using content factors
        adaptive_threshold = max(mean_sim - (std_sim * content_factor), adjusted_content_threshold)
        
        logger.info(f"Using adaptive similarity threshold: {adaptive_threshold:.4f} (mean: {mean_sim:.4f}, std: {std_sim:.4f})")
        
        # First pass: Find all potential boundaries where similarity drops significantly
        potential_boundaries = []
        for i in range(len(similarities)):
            if similarities[i] < adaptive_threshold:
                # This is a point of significant content change
                potential_boundaries.append(i + 1)  # +1 because similarity is between i and i+1
        
        # Second pass: Filter out boundaries that are too close together
        min_frames = max(3, int(self.min_scene_length * fps / sampling_rate))
        
        # For longer videos, use a more flexible approach to boundary filtering
        if is_long_video:
            # Slightly more flexible minimum scene length for long videos
            min_frames = max(2, int((self.min_scene_length * 0.8) * fps / sampling_rate))
        
        # Filter out boundaries that are too close together, keeping the stronger ones
        filtered_boundaries = []
        prev_boundary = -1
        
        for boundary in sorted(potential_boundaries):
            # Check if this boundary is far enough from the previous one
            if prev_boundary == -1 or boundary - prev_boundary >= min_frames:
                filtered_boundaries.append(boundary)
                prev_boundary = boundary
            else:
                # Boundaries are too close - keep the one with the lower similarity (stronger boundary)
                if prev_boundary > 0 and boundary < len(similarities) + 1:
                    # Get index within similarities list (account for +1 offset)
                    prev_idx = prev_boundary - 1
                    curr_idx = boundary - 1
                    
                    if curr_idx < len(similarities) and prev_idx < len(similarities):
                        # If current boundary is stronger, replace the previous one
                        if similarities[curr_idx] < similarities[prev_idx]:
                            filtered_boundaries.pop()  # Remove previous
                            filtered_boundaries.append(boundary)  # Add current
                            prev_boundary = boundary
        
        # Always include the start of the video
        if 0 not in filtered_boundaries:
            filtered_boundaries.insert(0, 0)
        
        # Always include the end of the video
        if len(embeddings) - 1 not in filtered_boundaries:
            filtered_boundaries.append(len(embeddings) - 1)
        
        # Sort boundaries
        filtered_boundaries.sort()
        
        # Create scene dictionaries
        scenes = []
        for i in range(len(filtered_boundaries) - 1):
            start_idx = filtered_boundaries[i]
            end_idx = filtered_boundaries[i + 1] - 1
            
            # Skip boundaries that are too close together
            if end_idx - start_idx < 2:
                continue
            
            # Convert to original frame indices and timestamps
            start_frame = frame_indices[start_idx]
            end_frame = frame_indices[min(end_idx, len(frame_indices) - 1)]
            start_time = timestamps[start_idx]
            end_time = timestamps[min(end_idx, len(timestamps) - 1)]
            
            # Calculate scene duration
            duration = end_time - start_time
            
            # Skip very short scenes (likely false positives)
            if duration < 2.0:  # Minimum 2 seconds
                continue
            
            # For each scene, add a confidence score based on how strong the boundary is
            boundary_strength = 1.0
            if i < len(filtered_boundaries) - 1 and i > 0:
                # Get similarity at the boundaries (lower similarity = stronger boundary)
                pre_boundary_idx = filtered_boundaries[i] - 1
                if 0 <= pre_boundary_idx < len(similarities):
                    boundary_strength = 1.0 - similarities[pre_boundary_idx]  # Invert so higher = stronger
            
            scenes.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "similarity_score": boundary_strength
            })
        
        # If no valid scenes were created, create a single scene for the entire video
        if not scenes:
            logger.warning("No valid scenes created, creating a single scene for the entire video")
            scenes = [{
                "start_frame": frame_indices[0],
                "end_frame": frame_indices[-1],
                "start_time": timestamps[0],
                "end_time": timestamps[-1],
                "duration": timestamps[-1] - timestamps[0]
            }]
        
        logger.info(f"Detected {len(scenes)} initial scene boundaries")
        return scenes
        
    def _create_fallback_scenes(self, timestamps: List[float], frame_indices: List[int], fps: float) -> List[Dict]:
        """
        Create fallback scenes when no regular scene boundaries are detected.
        
        Updated to create content-aware fallback scenes rather than just time-based splitting.
        
        Args:
            timestamps: List of frame timestamps
            frame_indices: List of original frame indices
            fps: Frames per second
            
        Returns:
            List of scene dictionaries
        """
        if not timestamps or not frame_indices:
            logger.error("No timestamps or frame indices available for fallback scene creation")
            return []
            
        logger.warning("No scene boundaries detected, creating content-based fallback scenes")
        
        total_duration = timestamps[-1] - timestamps[0]
        total_frames = frame_indices[-1] - frame_indices[0] + 1
        
        # For short videos, just create one scene
        if total_duration < self.min_scene_length * 1.5:
            logger.info(f"Video duration ({total_duration:.2f}s) too short, creating single scene")
            return [{
                "start_frame": frame_indices[0],
                "end_frame": frame_indices[-1],
                "start_time": timestamps[0],
                "end_time": timestamps[-1],
                "duration": total_duration
            }]
            
        # For longer videos, try to create content-aware scenes based on sampled frame analysis
        # This is a simplified version that focuses on major content shifts
        
        # First, estimate a reasonable number of scenes based on total duration
        # Aim for scenes of approximately target_duration length
        target_duration = 60.0  # Target 1 minute per scene as a starting point
        estimated_scenes = max(1, int(total_duration / target_duration))
        
        # Cap the number of scenes to avoid excessive fragmentation
        estimated_scenes = min(estimated_scenes, 10)
        
        logger.info(f"Creating {estimated_scenes} content-based fallback scenes for {total_duration:.2f}s video")
        
        # Create scenes with approximately equal durations but respecting content
        scenes = []
        target_frames_per_scene = total_frames / estimated_scenes
        
        for i in range(estimated_scenes):
            start_idx = i * len(timestamps) // estimated_scenes
            end_idx = (i + 1) * len(timestamps) // estimated_scenes - 1
            
            # Ensure end_idx is valid
            end_idx = min(end_idx, len(timestamps) - 1)
            if start_idx >= end_idx:
                continue
                
            start_frame = frame_indices[start_idx]
            end_frame = frame_indices[end_idx]
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]
            
            scenes.append({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            })
            
        logger.info(f"Created {len(scenes)} fallback scenes")
        return scenes
    
    def _cluster_scenes_by_similarity(self, 
                                scenes: List[Dict], 
                                embeddings: torch.Tensor,
                                timestamps: List[float],
                                frame_indices: List[int]) -> List[Dict]:
        """
        Cluster scenes based on visual similarity and dialogue coherence.
        
        This enhanced method:
        1. Uses dialogue information to prevent scene cuts during conversations
        2. Considers temporal proximity of dialogue segments
        3. Merges scenes that are part of the same conversation
        4. Ensures scenes don't cut in the middle of dialogue
        
        Args:
            scenes: List of scene dictionaries
            embeddings: Tensor of frame embeddings
            timestamps: List of frame timestamps
            frame_indices: List of original frame indices
            
        Returns:
            List of merged scene dictionaries
        """
        if not scenes:
            return []
            
        logger.info("Clustering scenes based on visual similarity and dialogue coherence...")
        
        # Calculate total video duration
        total_duration = timestamps[-1] - timestamps[0]
        logger.info(f"Total video duration: {total_duration:.2f} seconds")
        
        # Load dialogue information if available
        dialogue_info = None
        try:
            dialogue_path = Path("output/audio/dialogue.json")
            if dialogue_path.exists():
                with open(dialogue_path, 'r') as f:
                    dialogue_info = json.load(f)
                logger.info("Loaded dialogue information for scene clustering")
        except Exception as e:
            logger.warning(f"Could not load dialogue information: {e}")
        
        # First pass: Identify scenes with dialogue and calculate dialogue features
        scenes_with_dialogue = []
        for scene in scenes:
            scene_start = scene["start_time"]
            scene_end = scene["end_time"]
            
            # Find dialogue segments that overlap with this scene
            overlapping_dialogue = []
            if dialogue_info:
                for segment in dialogue_info.get("segments", []):
                    seg_start = segment.get("start", 0)
                    seg_end = segment.get("end", 0)
                    
                    # Check for overlap
                    if (seg_start <= scene_end and seg_end >= scene_start):
                        overlap_start = max(scene_start, seg_start)
                        overlap_end = min(scene_end, seg_end)
                        overlap_duration = overlap_end - overlap_start
                        
                        if overlap_duration > 0:
                            overlapping_dialogue.append({
                                "start": overlap_start,
                                "end": overlap_end,
                                "text": segment.get("text", ""),
                                "speaker": segment.get("speaker", "unknown")
                            })
            
            # Calculate dialogue features
            dialogue_duration = sum(d["end"] - d["start"] for d in overlapping_dialogue)
            dialogue_coverage = dialogue_duration / (scene_end - scene_start) if scene_end > scene_start else 0
            
            # Add dialogue information to scene
            scene["dialogue"] = overlapping_dialogue
            scene["dialogue_coverage"] = dialogue_coverage
            scene["dialogue_duration"] = dialogue_duration
            
            if overlapping_dialogue:
                scenes_with_dialogue.append(scene)
        
        # Calculate scene embeddings that better represent content
        scene_embeddings = []
        for scene in scenes:
            start_idx = frame_indices.index(scene["start_frame"])
            end_idx = frame_indices.index(scene["end_frame"])
            
            # Sample frames from beginning, middle, and end of scene
            mid_idx = (start_idx + end_idx) // 2
            quarter_idx = (start_idx + mid_idx) // 2
            three_quarter_idx = (mid_idx + end_idx) // 2
            
            # Get embeddings for key frames
            start_emb = embeddings[start_idx]
            quarter_emb = embeddings[quarter_idx]
            mid_emb = embeddings[mid_idx]
            three_quarter_emb = embeddings[three_quarter_idx]
            end_emb = embeddings[end_idx]
            
            # Weighted average: give more weight to middle frames
            scene_emb = (start_emb * 0.1 + quarter_emb * 0.2 + mid_emb * 0.4 + 
                        three_quarter_emb * 0.2 + end_emb * 0.1)
            
            scene_embeddings.append(scene_emb)
        
        # Convert to tensor for efficient computation
        scene_embeddings = torch.stack(scene_embeddings)
        
        # Calculate pairwise similarities between scenes
        similarities = torch.nn.functional.cosine_similarity(
            scene_embeddings.unsqueeze(1),
            scene_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Adjust merge threshold based on video length and dialogue presence
        base_merge_threshold = 0.75
        if total_duration > 900:  # Videos over 15 minutes
            merge_threshold = base_merge_threshold + 0.05
        else:
            merge_threshold = base_merge_threshold
        
        logger.info(f"Using merge threshold: {merge_threshold:.2f}")
        
        # First pass: Merge scenes that are part of the same conversation
        merged_scenes = []
        i = 0
        while i < len(scenes):
            current_scene = scenes[i]
            merged_scene = current_scene.copy()
            
            # Check if current scene has dialogue
            current_dialogue = current_scene.get("dialogue", [])
            current_coverage = current_scene.get("dialogue_coverage", 0)
            
            # Look ahead to find scenes that should be merged
            j = i + 1
            while j < len(scenes):
                next_scene = scenes[j]
                
                # Check temporal proximity
                time_gap = next_scene["start_time"] - current_scene["end_time"]
                if time_gap > self.max_temporal_gap:
                    break
                    
                # Check if next scene has dialogue
                next_dialogue = next_scene.get("dialogue", [])
                next_coverage = next_scene.get("dialogue_coverage", 0)
                
                # Calculate similarity between scenes
                similarity = similarities[i, j].item()
                
                # Determine if scenes should be merged based on dialogue and similarity
                should_merge = False
                
                # Case 1: Both scenes have significant dialogue
                if current_coverage > 0.3 and next_coverage > 0.3:
                    # Check if dialogue is continuous
                    last_current_dialogue = max(d["end"] for d in current_dialogue)
                    first_next_dialogue = min(d["start"] for d in next_dialogue)
                    
                    # If dialogue segments are close in time, merge the scenes
                    if first_next_dialogue - last_current_dialogue < 1.5:  # 1.5 second gap
                        should_merge = True
                        logger.info(f"Merging scenes {i} and {j} due to continuous dialogue")
                
                # Case 2: One scene has dialogue and the other is very short
                elif (current_coverage > 0.3 and next_scene["duration"] < 3.0) or \
                     (next_coverage > 0.3 and current_scene["duration"] < 3.0):
                    should_merge = True
                    logger.info(f"Merging short scene with dialogue scene")
                
                # Case 3: High visual similarity and temporal proximity
                elif similarity > merge_threshold and time_gap < 1.0:
                    should_merge = True
                    logger.info(f"Merging scenes {i} and {j} due to high similarity ({similarity:.2f})")
                
                if should_merge:
                    # Update merged scene
                    merged_scene["end_frame"] = next_scene["end_frame"]
                    merged_scene["end_time"] = next_scene["end_time"]
                    merged_scene["duration"] = merged_scene["end_time"] - merged_scene["start_time"]
                    
                    # Combine dialogue if present
                    if "dialogue" in merged_scene and next_dialogue:
                        merged_scene["dialogue"].extend(next_dialogue)
                    elif next_dialogue:
                        merged_scene["dialogue"] = next_dialogue
                    
                    # Update dialogue features
                    merged_scene["dialogue_duration"] = sum(d["end"] - d["start"] for d in merged_scene["dialogue"])
                    merged_scene["dialogue_coverage"] = merged_scene["dialogue_duration"] / merged_scene["duration"]
                    
                    j += 1
                else:
                    break
            
            # Add the merged scene
            merged_scenes.append(merged_scene)
            i = j
        
        # Second pass: Validate scene durations and check for overlaps
        final_scenes = []
        for i, scene in enumerate(merged_scenes):
            # Skip very short scenes (likely false positives)
            if scene["duration"] < self.min_viable_scene_duration:
                logger.info(f"Skipping short scene {i} (duration: {scene['duration']:.2f}s)")
                continue
            
            # Check for overlaps with previous scene
            if final_scenes:
                prev_scene = final_scenes[-1]
                if scene["start_time"] < prev_scene["end_time"]:
                    # Resolve overlap by adjusting boundaries
                    overlap = prev_scene["end_time"] - scene["start_time"]
                    if overlap > 0.5:  # Significant overlap
                        # Keep the scene with more dialogue if one has it
                        if scene.get("dialogue_coverage", 0) > prev_scene.get("dialogue_coverage", 0):
                            prev_scene["end_time"] = scene["start_time"]
                            prev_scene["duration"] = prev_scene["end_time"] - prev_scene["start_time"]
                        elif prev_scene.get("dialogue_coverage", 0) > scene.get("dialogue_coverage", 0):
                            scene["start_time"] = prev_scene["end_time"]
                            scene["duration"] = scene["end_time"] - scene["start_time"]
                        else:
                            # Split the overlap
                            mid_point = (scene["start_time"] + prev_scene["end_time"]) / 2
                            prev_scene["end_time"] = mid_point
                            scene["start_time"] = mid_point
                            prev_scene["duration"] = prev_scene["end_time"] - prev_scene["start_time"]
                            scene["duration"] = scene["end_time"] - scene["start_time"]
            
            final_scenes.append(scene)
        
        logger.info(f"Created {len(final_scenes)} content-based scenes after clustering")
        return final_scenes
    
    def _remove_scene_overlaps(self, scenes: List[Dict], 
                             embeddings: torch.Tensor,
                             timestamps: List[float],
                             frame_indices: List[int]) -> List[Dict]:
        """
        Memory-efficient implementation to remove overlaps between scenes.
        
        Args:
            scenes: List of scenes that may overlap
            embeddings: Frame embeddings
            timestamps: Frame timestamps
            frame_indices: Original frame indices
            
        Returns:
            List of scenes with overlaps removed
        """
        if len(scenes) <= 1:
            return scenes
            
        # Create a copy of scenes to avoid modifying the original
        adjusted_scenes = []
        for scene in scenes:
            adjusted_scenes.append({**scene})
            
        # Track how many overlaps we found and fixed
        overlap_count = 0
        max_overlap_duration = 0
            
        # Process overlaps in batches to reduce memory usage
        for i in range(1, len(adjusted_scenes)):
            current_scene = adjusted_scenes[i]
            prev_scene = adjusted_scenes[i-1]
            
            # Check if there's an overlap
            if current_scene["start_time"] < prev_scene["end_time"]:
                overlap_count += 1
                overlap_duration = prev_scene["end_time"] - current_scene["start_time"]
                max_overlap_duration = max(max_overlap_duration, overlap_duration)
                
                # Find all frames in the overlap region
                overlap_start = current_scene["start_time"]
                overlap_end = prev_scene["end_time"]
                
                # For very short overlaps, just use the midpoint
                if overlap_duration < 0.5:  # Less than half a second
                    mid_time = (overlap_start + overlap_end) / 2
                    
                    # Find closest frame to midpoint
                    closest_idx = None
                    closest_diff = float('inf')
                    for idx, t in enumerate(timestamps):
                        if closest_idx is None or abs(t - mid_time) < closest_diff:
                            closest_idx = idx
                            closest_diff = abs(t - mid_time)
                    
                    mid_frame = frame_indices[closest_idx] if closest_idx is not None else int((current_scene["start_frame"] + prev_scene["end_frame"]) / 2)
                    
                    # Update scene boundaries
                    prev_scene["end_time"] = mid_time
                    prev_scene["end_frame"] = mid_frame
                    prev_scene["duration"] = prev_scene["end_time"] - prev_scene["start_time"]
                    
                    current_scene["start_time"] = mid_time
                    current_scene["start_frame"] = mid_frame
                    current_scene["duration"] = current_scene["end_time"] - current_scene["start_time"]
        
        if overlap_count > 0:
            logger.info(f"Fixed {overlap_count} overlapping scene boundaries, max overlap was {max_overlap_duration:.2f}s")
        
        # Filter out any scenes that became too short
        valid_scenes = [s for s in adjusted_scenes if s["duration"] >= self.min_scene_length]
        
        if len(valid_scenes) < len(adjusted_scenes):
            logger.warning(f"Removed {len(adjusted_scenes) - len(valid_scenes)} scenes that became too short after overlap removal")
        
        return valid_scenes
    
    def _merge_scenes(self, scenes: List[Dict]) -> Dict:
        """
        Merge multiple scenes into a single scene.
        
        Args:
            scenes: List of scenes to merge
            
        Returns:
            Merged scene dictionary
        """
        if not scenes:
            return None
            
        if len(scenes) == 1:
            return scenes[0]
            
        # Get overall start and end
        start_time = min(scene["start_time"] for scene in scenes)
        end_time = max(scene["end_time"] for scene in scenes)
        start_frame = min(scene["start_frame"] for scene in scenes)
        end_frame = max(scene["end_frame"] for scene in scenes)
        
        # Combine frame indices
        all_frames = []
        for scene in scenes:
            if "frames_idx" in scene:
                all_frames.extend(scene["frames_idx"])
        
        # Create merged scene
        merged_scene = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "start_frame": start_frame,
            "end_frame": end_frame
        }
        
        # Add frames if we have them
        if all_frames:
            # Remove duplicates and sort
            all_frames = sorted(list(set(all_frames)))
            merged_scene["frames_idx"] = all_frames
            
        return merged_scene
        
    def save_scenes(self, scenes: List[Dict], output_path: str):
        """Save scene detection results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"scenes": scenes}, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(scenes)} scenes to {output_path}")

    def _format_timestamp(self, start_time: float, end_time: float) -> str:
        """
        Format time range as readable timestamp.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Formatted timestamp string
        """
        def format_time(seconds):
            """Format seconds as MM:SS."""
            minutes = int(seconds / 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
            
        return f"{format_time(start_time)} - {format_time(end_time)}"

    def _cluster_large_video_scenes(self, scenes: List[Dict], embeddings: torch.Tensor, 
                               timestamps: List[float], frame_indices: List[int]) -> List[Dict]:
        """
        Simplified clustering approach for extremely large videos with hundreds of scenes.
        Instead of trying to cluster all scenes at once, this method:
        1. Divides scenes into manageable windows (e.g., 30-minute segments)
        2. Clusters each window independently
        3. Joins the results

        Args:
            scenes: List of scene dictionaries
            embeddings: CLIP embeddings for frames
            timestamps: Timestamps for frames
            frame_indices: Frame indices
            
        Returns:
            List of merged scene dictionaries
        """
        logger.info(f"Using specialized large video clustering for {len(scenes)} scenes")
        
        if not scenes:
            return []
            
        # Sort scenes by start time to ensure temporal order
        scenes = sorted(scenes, key=lambda s: s["start_time"])
        
        # Determine window size - aim for 50-100 scenes per window
        scenes_per_window = 100
        total_scenes = len(scenes)
        
        # Calculate how many windows we need
        num_windows = max(1, total_scenes // scenes_per_window)
        
        # Alternatively, use time-based windows for longer videos
        video_duration = scenes[-1]["end_time"] - scenes[0]["start_time"]
        window_duration = 30 * 60  # 30 minutes per window
        
        # Choose the approach based on video length
        use_time_windows = video_duration > 60 * 60  # For videos longer than 1 hour
        
        # Process clusters in each window
        all_merged_scenes = []
        
        if use_time_windows:
            # Time-based windows
            logger.info(f"Using time-based windows for {video_duration/60:.1f} minute video")
            
            window_start = scenes[0]["start_time"]
            window_end = window_start + window_duration
            
            while window_start < scenes[-1]["end_time"]:
                # Get scenes in this time window
                window_scenes = [s for s in scenes if s["start_time"] < window_end and s["end_time"] > window_start]
                
                if window_scenes:
                    logger.info(f"Processing window {window_start/60:.1f}-{window_end/60:.1f} min with {len(window_scenes)} scenes")
                    
                    # Process this window with reduced memory settings
                    # Save memory by using a higher threshold for very large videos
                    temp_merge_threshold = min(0.85, self.merge_threshold + 0.1)
                    
                    try:
                        # Create a temporary instance with adjusted parameters
                        temp_detector = SceneDetector(config={
                            "threshold": self.threshold,
                            "merge_threshold": temp_merge_threshold,
                            "device": self.device
                        })
                        
                        # Use a specialized method for clustering that's more memory-efficient
                        window_merged = self._cluster_window(window_scenes, embeddings, timestamps, frame_indices, 
                                                          temp_merge_threshold)
                        
                        all_merged_scenes.extend(window_merged)
                    except Exception as e:
                        logger.error(f"Error clustering window {window_start}-{window_end}: {str(e)}")
                        # Fall back to original scenes in this window
                        all_merged_scenes.extend(window_scenes)
                
                # Advance to next window with a small overlap
                window_start = window_end - (window_duration * 0.1)  # 10% overlap
                window_end = window_start + window_duration
                
                # Clear memory between windows
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Count-based windows
            logger.info(f"Using count-based windows with {scenes_per_window} scenes per window")
            
            for i in range(0, total_scenes, scenes_per_window):
                window_scenes = scenes[i:i+scenes_per_window]
                
                if window_scenes:
                    logger.info(f"Processing window {i/total_scenes*100:.1f}% with {len(window_scenes)} scenes")
                    
                    try:
                        # Use a specialized method for clustering that's more memory-efficient
                        window_merged = self._cluster_window(window_scenes, embeddings, timestamps, frame_indices, 
                                                          self.merge_threshold)
                        
                        all_merged_scenes.extend(window_merged)
                    except Exception as e:
                        logger.error(f"Error clustering window {i}-{i+scenes_per_window}: {str(e)}")
                        # Fall back to original scenes in this window
                        all_merged_scenes.extend(window_scenes)
                
                # Clear memory between windows
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        # Sort final scenes by start time
        all_merged_scenes.sort(key=lambda s: s["start_time"])
        
        # Post-process to remove any overlaps resulting from the windowed approach
        return self._remove_scene_overlaps(all_merged_scenes, embeddings, timestamps, frame_indices)
        
    def _cluster_window(self, scenes: List[Dict], embeddings: torch.Tensor, 
                    timestamps: List[float], frame_indices: List[int],
                    merge_threshold: float) -> List[Dict]:
        """
        Specialized clustering for a window of scenes with minimal memory usage.
        
        Args:
            scenes: List of scene dictionaries in the window
            embeddings: CLIP embeddings for frames
            timestamps: Timestamps for frames
            frame_indices: Frame indices
            merge_threshold: Threshold for merging scenes
            
        Returns:
            List of merged scene dictionaries for this window
        """
        if not scenes:
            return []
            
        if len(scenes) == 1:
            return scenes.copy()
            
        # Create scene embeddings for this window only
        scene_embeddings = []
        
        for scene in scenes:
            # Find frames in the scene
            start_time = scene["start_time"]
            end_time = scene["end_time"]
            
            # Find frames in this scene
            scene_frame_indices = [i for i, t in enumerate(timestamps) 
                                  if start_time <= t <= end_time]
            
            if scene_frame_indices:
                # Use a maximum of 10 frames per scene to save memory
                if len(scene_frame_indices) > 10:
                    # Sample frames evenly throughout the scene
                    step = len(scene_frame_indices) / 10
                    scene_frame_indices = [scene_frame_indices[min(int(i*step), len(scene_frame_indices)-1)] 
                                         for i in range(10)]
                
                # Extract embeddings for these frames
                scene_frames_embeddings = embeddings[scene_frame_indices]
                
                # Average embeddings for the scene
                scene_embedding = torch.mean(scene_frames_embeddings, dim=0)
                
                # Normalize
                scene_embedding = scene_embedding / scene_embedding.norm()
            else:
                # Fallback if no frames found
                scene_embedding = torch.zeros(embeddings.shape[1], device=embeddings.device)
                
            scene_embeddings.append(scene_embedding)
        
        # Convert to tensor
        scene_embeddings = torch.stack(scene_embeddings)
        
        # Set threshold
        distance_threshold = 1.0 - merge_threshold
        
        # Initialize each scene as its own cluster
        num_scenes = len(scenes)
        clusters = [[i] for i in range(num_scenes)]
        
        # Simple greedy clustering - for each scene, merge with the most similar 
        # neighboring scene if similarity exceeds threshold
        merged = True
        iteration = 0
        max_iterations = min(10, num_scenes)  # Limit iterations for memory efficiency
        
        while merged and iteration < max_iterations and len(clusters) > 1:
            merged = False
            iteration += 1
            
            # Sort clusters by first scene index to maintain temporal order
            clusters = sorted(clusters, key=lambda c: c[0])
            
            i = 0
            while i < len(clusters) - 1:
                # Check only neighboring clusters
                j = i + 1
                
                # Compute average distance between clusters
                cluster_dists = []
                for idx1 in clusters[i]:
                    for idx2 in clusters[j]:
                        dist = 1.0 - torch.dot(scene_embeddings[idx1], scene_embeddings[idx2]).item()
                        cluster_dists.append(dist)
                
                if cluster_dists:
                    avg_dist = sum(cluster_dists) / len(cluster_dists)
                    
                    # Merge if similar enough
                    if avg_dist < distance_threshold:
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        merged = True
                        # Don't increment i since we removed j and need to check the new neighbor
                    else:
                        # Move to next pair
                        i += 1
                else:
                    i += 1
        
        # Create merged scenes
        merged_scenes = []
        
        for cluster in clusters:
            cluster.sort()  # Ensure temporal order
            cluster_scenes = [scenes[idx] for idx in cluster]
            
            # Merge scenes in this cluster
            if cluster_scenes:
                merged_scene = self._merge_scenes(cluster_scenes)
                merged_scene["event_coherence"] = True
                merged_scenes.append(merged_scene)
        
        # Clean up memory
        del scene_embeddings
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return merged_scenes

# Helper function
def get_scene_boundaries(video_path: str, config: Dict = None) -> List[Dict]:
    """
    Convenience function to detect scene boundaries in a video.
    
    Args:
        video_path: Path to the video file
        config: Optional configuration dictionary
        
    Returns:
        List of scene dictionaries
    """
    detector = SceneDetector(config)
    return detector.detect_scenes(video_path) 