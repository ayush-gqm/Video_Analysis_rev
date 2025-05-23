"""Keyframe extraction component for the video analysis pipeline."""

import os
# Set environment variables to handle NCCL issues before importing PyTorch
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer which can cause symbol errors
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
import gc
from PIL import Image
import open_clip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeyframeExtractor:
    """
    Extracts representative keyframes from video scenes using CLIP-based clustering.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the KeyframeExtractor with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.method = self.config.get("method", "content_based")
        
        # Base keyframe settings
        self.base_frames = self.config.get("frames_per_scene", 5)
        self.min_frames = self.config.get("min_frames_per_scene", 3)
        self.max_frames = self.config.get("max_frames_per_scene", 12)  # Increased from 8
        
        # Dynamic keyframe settings
        self.short_scene_threshold = 30.0  # seconds
        self.long_scene_threshold = 120.0  # seconds
        self.very_long_scene_threshold = 300.0  # seconds
        
        # Scale factors for dynamic frame count
        self.long_scene_scale = 1.5
        self.very_long_scene_scale = 2.0
        
        self.batch_size = self.config.get("batch_size", 32)
        
        # Always prioritize CUDA, only fall back to CPU if unavailable
        if torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU for keyframe extraction.")
            self.device = "cuda"
        else:
            logger.warning("CUDA is not available. Falling back to CPU for keyframe extraction.")
            self.device = "cpu"
        
        # Force device from config if explicitly specified
        config_device = self.config.get("device")
        if config_device:
            if config_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA specified in config but not available. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = config_device
                logger.info(f"Using device specified in config: {self.device}")
        
        # Load CLIP model for feature extraction
        if self.method == "content_based":
            logger.info(f"Loading CLIP model for content-based keyframe extraction on {self.device}...")
            try:
                model_and_transforms = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device=self.device
                )
                self.model = model_and_transforms[0]
                self.preprocess = model_and_transforms[1]
                self.model.eval()
                logger.info(f"CLIP model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {str(e)}")
                raise
        
        logger.info(f"Initialized KeyframeExtractor (method={self.method}, base_frames={self.base_frames}, device={self.device})")
        
    def extract_keyframes(self, video_path: str, scenes: List[Dict], output_dir: str = None) -> Dict[int, List[Dict]]:
        """
        Extract keyframes from each scene in the video.
        
        Args:
            video_path: Path to the video file
            scenes: List of scene dictionaries with start_frame and end_frame
            output_dir: Optional path to directory for saving keyframes
            
        Returns:
            Dictionary mapping scene indices to lists of keyframe information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        logger.info(f"Extracting keyframes from {video_path} using {self.method} method")
        
        # Validate scenes input
        if not isinstance(scenes, list):
            logger.error(f"Invalid scenes parameter: expected list, got {type(scenes)}")
            logger.debug(f"Scenes content: {scenes}")
            raise ValueError(f"Invalid scenes parameter: expected list, got {type(scenes)}")
            
        # Handle empty scenes list
        if not scenes:
            logger.warning("No scenes provided for keyframe extraction. Creating a dummy scene for the entire video.")
            
            # Open video to get properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                raise ValueError(f"Could not open video: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            # Create a dummy scene for the entire video
            scenes = [{
                "start_frame": 0,
                "end_frame": total_frames - 1,
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration
            }]
            
            logger.info(f"Created dummy scene for entire video: {duration:.2f} seconds")
        
        # Validate scene structure
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                logger.error(f"Invalid scene at index {i}: expected dict, got {type(scene)}")
                logger.debug(f"Scene content: {scene}")
                continue
                
            # Make sure start_frame and end_frame are available
            if "start_frame" not in scene:
                if "start_time" in scene and "fps" in scene:
                    # Calculate start_frame from start_time and fps
                    scene["start_frame"] = int(scene["start_time"] * scene["fps"])
                    logger.info(f"Calculated start_frame for scene {i}: {scene['start_frame']}")
                else:
                    logger.error(f"Scene {i} is missing start_frame and cannot be calculated")
                    logger.debug(f"Scene content: {scene}")
                    continue
                    
            if "end_frame" not in scene:
                if "end_time" in scene and "fps" in scene:
                    # Calculate end_frame from end_time and fps
                    scene["end_frame"] = int(scene["end_time"] * scene["fps"])
                    logger.info(f"Calculated end_frame for scene {i}: {scene['end_frame']}")
                else:
                    logger.error(f"Scene {i} is missing end_frame and cannot be calculated")
                    logger.debug(f"Scene content: {scene}")
                    continue
        
        # Log what scenes we're processing
        logger.info(f"Processing {len(scenes)} scenes for keyframe extraction")
        for i, scene in enumerate(scenes):
            if "start_frame" in scene and "end_frame" in scene:
                frames = scene["end_frame"] - scene["start_frame"] + 1
                logger.debug(f"Scene {i}: frames={frames}, start={scene.get('start_frame')}, end={scene.get('end_frame')}")
        
        # Open the video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties: {fps:.2f} fps, {total_frames} total frames")
        
        # Process each scene to extract keyframes
        all_keyframes = {}
        
        try:
            for scene_idx, scene in enumerate(scenes):
                logger.info(f"Processing scene {scene_idx+1}/{len(scenes)}")
                
                # Skip invalid scenes
                if "start_frame" not in scene or "end_frame" not in scene:
                    logger.warning(f"Skipping scene {scene_idx} due to missing frame information")
                    all_keyframes[scene_idx] = []
                    continue
                
                start_frame = scene["start_frame"]
                end_frame = scene["end_frame"]
                
                # Validate frame range
                if start_frame < 0:
                    logger.warning(f"Scene {scene_idx} has negative start_frame ({start_frame}), clamping to 0")
                    start_frame = 0
                    
                if end_frame >= total_frames:
                    logger.warning(f"Scene {scene_idx} has end_frame ({end_frame}) beyond video length ({total_frames}), clamping to {total_frames-1}")
                    end_frame = total_frames - 1
                    
                if start_frame > end_frame:
                    logger.error(f"Scene {scene_idx} has start_frame ({start_frame}) greater than end_frame ({end_frame})")
                    all_keyframes[scene_idx] = []
                    continue
                
                duration = end_frame - start_frame + 1
                logger.debug(f"Scene {scene_idx} has {duration} frames from {start_frame} to {end_frame}")
                
                # Skip very short scenes
                if duration < 10:
                    logger.warning(f"Scene {scene_idx} too short ({duration} frames), using uniform sampling")
                    keyframes = self._extract_uniform_keyframes(cap, start_frame, end_frame, fps)
                else:
                    if self.method == "content_based":
                        logger.debug(f"Using content-based keyframe extraction for scene {scene_idx}")
                        keyframes = self._extract_content_based_keyframes(cap, start_frame, end_frame, fps)
                    elif self.method == "uniform":
                        logger.debug(f"Using uniform keyframe extraction for scene {scene_idx}")
                        keyframes = self._extract_uniform_keyframes(cap, start_frame, end_frame, fps)
                    else:
                        logger.warning(f"Unknown method {self.method}, falling back to uniform sampling")
                        keyframes = self._extract_uniform_keyframes(cap, start_frame, end_frame, fps)
                
                all_keyframes[scene_idx] = keyframes
                gc.collect()  # Force garbage collection after each scene
                
                # Show progress
                logger.info(f"Extracted {len(keyframes)} keyframes for scene {scene_idx+1}")
                
        finally:
            cap.release()
            
        logger.info(f"Extracted keyframes for {len(all_keyframes)} scenes")
        
        # Count total keyframes
        total_keyframes = sum(len(kf) for kf in all_keyframes.values())
        logger.info(f"Total keyframes extracted: {total_keyframes}")
        
        # Save keyframes to output directory if provided
        keyframes_metadata = None
        if output_dir and total_keyframes > 0:
            try:
                keyframes_metadata = self.save_keyframes(all_keyframes, output_dir)
                logger.info(f"Saved keyframes to {output_dir}")
            except Exception as e:
                logger.error(f"Error saving keyframes to {output_dir}: {str(e)}")
                # Continue returning keyframes even if saving fails
        
        return all_keyframes
    
    def _calculate_target_frames(self, scene_duration: float, frame_count: int) -> int:
        """
        Calculate target number of keyframes based on scene duration and complexity.
        
        Args:
            scene_duration: Duration of scene in seconds
            frame_count: Total number of frames in scene
            
        Returns:
            Target number of keyframes
        """
        # Base number of frames from config
        target_frames = self.base_frames
        
        # Scale based on scene duration
        if scene_duration >= self.very_long_scene_threshold:
            target_frames = int(target_frames * self.very_long_scene_scale)
            logger.info(f"Very long scene ({scene_duration:.1f}s), scaling to {target_frames} target frames")
        elif scene_duration >= self.long_scene_threshold:
            target_frames = int(target_frames * self.long_scene_scale)
            logger.info(f"Long scene ({scene_duration:.1f}s), scaling to {target_frames} target frames")
        elif scene_duration < self.short_scene_threshold:
            # For very short scenes, reduce frames but ensure minimum
            target_frames = max(self.min_frames, int(target_frames * 0.7))
            logger.info(f"Short scene ({scene_duration:.1f}s), reducing to {target_frames} target frames")
        
        # Ensure within min/max bounds
        target_frames = min(self.max_frames, max(self.min_frames, target_frames))
        
        # Never extract more frames than we have
        target_frames = min(target_frames, frame_count)
        
        return target_frames

    def _extract_uniform_keyframes(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, fps: float) -> List[Dict]:
        """
        Extract keyframes using uniform sampling.
        """
        num_frames = end_frame - start_frame + 1
        scene_duration = num_frames / fps
        
        # Calculate target number of keyframes based on scene duration
        num_keyframes = self._calculate_target_frames(scene_duration, num_frames)
        
        if num_frames <= num_keyframes:
            # If the scene is shorter than requested keyframes, take all frames
            frame_indices = list(range(start_frame, end_frame + 1))
        else:
            # Otherwise, sample uniformly
            step = num_frames / num_keyframes
            frame_indices = [int(start_frame + i * step) for i in range(num_keyframes)]
        
        # Extract the frames
        keyframes = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue
                
            # Resize frame to reduce memory usage
            frame = cv2.resize(frame, (640, 360))
                
            # Convert to RGB for consistent color handling throughout the pipeline
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            keyframe = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "image": frame
            }
            keyframes.append(keyframe)
            
        return keyframes
        
    def _extract_content_based_keyframes(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, fps: float) -> List[Dict]:
        """
        Extract keyframes using CLIP-based clustering.
        Optimized version for large scenes with memory constraints.
        """
        # For very large scenes, limit the number of frames to process
        frame_count = end_frame - start_frame + 1
        scene_duration = frame_count / fps
        
        # Calculate target number of keyframes based on scene duration
        num_keyframes = self._calculate_target_frames(scene_duration, frame_count)
        
        # For extremely large scenes, use a more aggressive sampling strategy
        if frame_count > 10000:
            logger.warning(f"Very large scene detected ({frame_count} frames), using aggressive sampling")
            sampling_rate = max(1, frame_count // 1000)  # Sample at most 1000 frames
            logger.info(f"Using sampling rate of {sampling_rate} for large scene")
        elif frame_count > 3000:
            # For large scenes, sample more aggressively
            sampling_rate = max(1, frame_count // 500)
            logger.info(f"Using sampling rate of {sampling_rate} for scene with {frame_count} frames")
        else:
            # For normal scenes, use a reasonable sampling rate
            sampling_rate = max(1, frame_count // 200)
            
        # Calculate sampled frame indices
        sampled_indices = list(range(start_frame, end_frame + 1, sampling_rate))
        
        # Cap the number of samples to avoid memory issues
        if len(sampled_indices) > 500:
            step = len(sampled_indices) / 500
            sampled_indices = [sampled_indices[min(int(i*step), len(sampled_indices)-1)] 
                             for i in range(500)]
            
        logger.info(f"Sampling {len(sampled_indices)} frames from scene with {frame_count} frames")
        
        # Extract frames in batches to reduce memory usage
        frames = []
        frame_indices = []
        
        # Process in smaller batches for memory efficiency
        batch_size = min(self.batch_size, 32)  # Smaller batch size for large scenes
        
        for batch_start in range(0, len(sampled_indices), batch_size):
            batch_indices = sampled_indices[batch_start:batch_start+batch_size]
            
            batch_frames = []
            for frame_idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Resize to smaller dimensions for memory efficiency on large scenes
                if frame_count > 5000:
                    # Use even smaller dimensions for very large scenes
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, (320, 180))  # Smaller resolution for large scenes
                else:
                    # Standard resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, (640, 360))
                
                batch_frames.append(frame_rgb)
                frame_indices.append(frame_idx)
            
            frames.extend(batch_frames)
            
            # Force garbage collection between batches if scene is very large
            if frame_count > 10000:
                gc.collect()
        
        if not frames:
            logger.warning(f"No frames extracted from scene {start_frame}-{end_frame}")
            return []
            
        # Use the configured frames_per_scene value, constrained by min/max limits
        num_keyframes = min(self.max_frames, max(self.min_frames, self.base_frames))
        
        # For very large scenes, reduce number of keyframes further to save memory
        if frame_count > 10000:
            num_keyframes = min(5, num_keyframes)
        
        # If few frames, just use uniform sampling
        if len(frames) <= num_keyframes:
            keyframes = []
            for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                keyframe = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps,
                    "image": frame  # Already in RGB format
                }
                keyframes.append(keyframe)
            return keyframes
            
        try:
            # Extract features using CLIP in batches
            features = []
            
            # Move to CPU for preprocessing if GPU memory is limited
            device_for_large = "cpu" if frame_count > 10000 and self.device == "cuda" else self.device
            
            with torch.no_grad():
                for i in range(0, len(frames), self.batch_size):
                    batch = frames[i:i+self.batch_size]
                    try:
                        processed_batch = torch.stack([
                            # Use PIL Image from NumPy array (already in RGB format)
                            self.preprocess(Image.fromarray(frame)) for frame in batch
                        ]).to(self.device)
                        
                        # Use mixed precision for large batches if on GPU
                        if self.device == "cuda":
                            with torch.cuda.amp.autocast():
                                batch_features = self.model.encode_image(processed_batch)
                        else:
                            batch_features = self.model.encode_image(processed_batch)
                            
                        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                        
                        # Move to CPU immediately to free GPU memory
                        if self.device == "cuda":
                            batch_features = batch_features.cpu()
                            
                        features.append(batch_features)
                        
                        # Clear CUDA cache after each batch for large scenes
                        if frame_count > 5000 and self.device == "cuda":
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and self.device == "cuda":
                            # Handle OOM error by clearing cache and retrying with smaller batch
                            logger.warning("GPU OOM during feature extraction, retrying with CPU")
                            torch.cuda.empty_cache()
                            
                            # Move to CPU for this batch
                            processed_batch = torch.stack([
                                self.preprocess(Image.fromarray(frame)) for frame in batch
                            ]).to("cpu")
                            
                            # Use CPU for encoding this batch
                            self.model.to("cpu")
                            batch_features = self.model.encode_image(processed_batch)
                            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                            features.append(batch_features)
                            
                            # Move model back to GPU if available
                            if torch.cuda.is_available():
                                self.model.to("cuda")
                        else:
                            # Reraise other errors
                            raise
            
            # Free up memory
            del processed_batch
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Concatenate all features
            if features:
                features = torch.cat(features, dim=0).cpu().numpy()  # Move to CPU for clustering
            else:
                # Fallback if feature extraction fails
                logger.warning("Feature extraction failed, falling back to uniform sampling")
                return self._extract_uniform_keyframes(cap, start_frame, end_frame, fps)
            
            # For very large scenes, use a more efficient clustering approach
            if len(frames) > 1000:
                logger.info(f"Using mini-batch KMeans for large scene with {len(frames)} frames")
                try:
                    from sklearn.cluster import MiniBatchKMeans
                    kmeans = MiniBatchKMeans(n_clusters=num_keyframes, random_state=42, batch_size=100)
                except ImportError:
                    # Fall back to standard KMeans with fewer iterations
                    logger.info("MiniBatchKMeans not available, using standard KMeans with reduced iterations")
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=num_keyframes, random_state=42, n_init=3, max_iter=100)
            else:
                # Use standard KMeans for smaller scenes
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_keyframes, random_state=42, n_init=10)
            
            # Fit the clustering model
            kmeans.fit(features)
            
            # Find the frames closest to each cluster center
            keyframe_indices = []
            for center_idx in range(num_keyframes):
                center = kmeans.cluster_centers_[center_idx]
                distances = np.sqrt(np.sum((features - center) ** 2, axis=1))
                closest_frame_idx = np.argmin(distances)
                keyframe_indices.append(closest_frame_idx)
                
            # Sort by original frame order
            keyframe_indices.sort()
            
            # Create keyframe dictionaries
            keyframes = []
            for idx in keyframe_indices:
                frame_idx = frame_indices[idx]
                keyframe = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps,
                    "image": frames[idx]  # Already in RGB format
                }
                keyframes.append(keyframe)
                
            # Clear memory
            del features, kmeans
            gc.collect()
            
            return keyframes
            
        except Exception as e:
            logger.error(f"Error in content-based keyframe extraction: {str(e)}")
            logger.warning("Falling back to uniform sampling due to error")
            return self._extract_uniform_keyframes(cap, start_frame, end_frame, fps)
        
    def save_keyframes(self, keyframes: Dict[int, List[Dict]], output_dir: str):
        """
        Save keyframes to disk and return metadata.
        
        Args:
            keyframes: Dictionary mapping scene indices to lists of keyframe information
            output_dir: Directory where to save the keyframes
            
        Returns:
            Dictionary with keyframe metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving keyframes to {output_dir}")
        
        metadata = {"scenes": {}}
        
        # Check if keyframes dictionary is empty
        if not keyframes:
            logger.warning("No keyframes to save - keyframes dictionary is empty")
            return metadata
            
        # Debug info about keyframes being saved
        logger.debug(f"Keyframes to save: {len(keyframes)} scenes")
        for scene_idx, scene_keyframes in keyframes.items():
            logger.debug(f"Scene {scene_idx}: {len(scene_keyframes)} keyframes")
        
        for scene_idx, scene_keyframes in keyframes.items():
            # Skip if no keyframes for this scene
            if not scene_keyframes:
                logger.warning(f"No keyframes to save for scene {scene_idx}")
                metadata["scenes"][str(scene_idx)] = []
                continue
                
            # Create scene directory
            scene_dir = output_dir / f"scene_{scene_idx:04d}"
            try:
                scene_dir.mkdir(exist_ok=True)
                logger.debug(f"Created scene directory: {scene_dir}")
            except Exception as e:
                logger.error(f"Error creating scene directory {scene_dir}: {str(e)}")
                continue
            
            scene_metadata = []
            
            for i, keyframe in enumerate(scene_keyframes):
                try:
                    # Check if keyframe has image data
                    if "image" not in keyframe:
                        logger.warning(f"Keyframe {i} in scene {scene_idx} has no image data")
                        continue
                        
                    # Save the image
                    frame_path = scene_dir / f"keyframe_{i:04d}.jpg"
                    
                    # OpenCV's imwrite expects BGR format, so we need to convert from RGB to BGR
                    # Only convert if the image is in RGB format (it should be, but check to be safe)
                    img_to_save = keyframe["image"]
                    
                    # Make sure image data is valid
                    if img_to_save is None or not isinstance(img_to_save, np.ndarray):
                        logger.warning(f"Invalid image data for keyframe {i} in scene {scene_idx}")
                        continue
                        
                    # Check image shape
                    if len(img_to_save.shape) != 3 or img_to_save.shape[2] != 3:
                        logger.warning(f"Unexpected image shape for keyframe {i} in scene {scene_idx}: {img_to_save.shape}")
                        if len(img_to_save.shape) == 2:  # Grayscale image
                            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
                        else:
                            continue
                    else:
                        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                        
                    # Use higher quality JPEG encoding to preserve colors better
                    success = cv2.imwrite(str(frame_path), img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if not success:
                        logger.warning(f"Failed to save keyframe image to {frame_path}")
                        continue
                        
                    logger.debug(f"Saved keyframe image to {frame_path}")
                    
                    # Add metadata (without the image data)
                    frame_metadata = {
                        "frame_idx": keyframe["frame_idx"],
                        "timestamp": keyframe["timestamp"],
                        "path": str(frame_path.relative_to(output_dir))
                    }
                    scene_metadata.append(frame_metadata)
                    
                except Exception as e:
                    logger.error(f"Error saving keyframe {i} in scene {scene_idx}: {str(e)}")
                    continue
            
            metadata["scenes"][str(scene_idx)] = scene_metadata
            logger.info(f"Saved {len(scene_metadata)} keyframes for scene {scene_idx}")
            
        # Save metadata to JSON
        try:
            self._save_metadata(metadata, output_dir)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            
        total_keyframes = sum(len(scene) for scene in metadata["scenes"].values())
        logger.info(f"Saved {total_keyframes} keyframes to {output_dir}")
        
        return metadata

    def _save_metadata(self, metadata: Dict, output_dir: str):
        """Save keyframe metadata to JSON file."""
        metadata_path = Path(output_dir) / "keyframes.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved keyframe metadata to {metadata_path}")

# Helper function
def extract_scene_keyframes(video_path: str, scenes: List[Dict], output_dir: str, config: Dict = None) -> Dict:
    """
    Convenience function to extract and save keyframes from video scenes.
    
    Args:
        video_path: Path to the video file
        scenes: List of scene dictionaries
        output_dir: Directory where to save keyframes
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with keyframe metadata
    """
    extractor = KeyframeExtractor(config)
    keyframes = extractor.extract_keyframes(video_path, scenes)
    return extractor.save_keyframes(keyframes, output_dir) 