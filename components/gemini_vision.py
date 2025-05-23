"""Gemini Vision integration for scene description and visual analysis."""

import os
import logging
import json
import time
import re
import copy
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiVisionAnalyzer:
    """
    Uses Google's Gemini Vision models to analyze keyframes and generate scene descriptions.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the GeminiVisionAnalyzer with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 1500)
        self.top_p = self.config.get("top_p", 0.95)
        self.model_name = None
        self.dialogue_api_key = self.config.get("dialogue_api_key", None)
        
        # Structured output settings
        self.structured_output = self.config.get("structured_output", False)
        self.scene_analysis_fields = self.config.get("scene_analysis_fields", [
            "setting", "action", "characters", "dialogue", "cinematography", "significance", "technical_notes"
        ])
        
        # Initialize the model
        self._initialize_model()
        
        logger.info(f"Initialized GeminiVisionAnalyzer (temperature={self.temperature}, max_tokens={self.max_tokens}, structured_output={self.structured_output})")
        
    def _initialize_model(self):
        """Initialize and select the best available Gemini model."""
        # Get API key from environment variable if not already set
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # Default to not available
        self.model_available = False
        self.model = None  # Initialize self.model to None
        
        if not api_key:
            logger.warning("GEMINI_API_KEY not set in environment variables")
            return
            
        # Configure Gemini API
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # List available models
            models = genai.list_models()
            
            # Filter out deprecated models
            active_models = [m for m in models if not m.name.endswith("@deprecated")]
            
            # Define the preference order for models (latest first)
            preferred_models = [
                "gemini-2.0-flash-exp",
            ]
            
            # Find the best available model in our preference list
            best_model = None
            for preferred in preferred_models:
                for model in active_models:
                    if preferred in model.name:
                        best_model = model.name
                        break
                if best_model:
                    break
                    
            if best_model:
                logger.info(f"Selected Gemini model: {best_model}")
                self.model_name = best_model
                # Create the model object
                self.model = genai.GenerativeModel(
                    self.model_name,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                        "top_p": self.top_p,
                    }
                )
                logger.info(f"Created Gemini model instance")
                self.model_available = True
            else:
                available_models = ", ".join([m.name for m in active_models])
                logger.warning(f"No preferred Gemini models available. Available models: {available_models}")
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {str(e)}")
            # Make sure model is None in case of error
            self.model = None
            
    def get_dialogue_model(self):
        """Get a configured Gemini model for dialogue enhancement, using dialogue-specific API key if provided."""
        import google.generativeai as genai
        
        # Use dialogue-specific API key if provided, otherwise use the default key
        dialogue_key = self.dialogue_api_key or os.environ.get("GEMINI_DIALOGUE_API_KEY")
        api_key = dialogue_key if dialogue_key else os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning("No valid API key for dialogue enhancement")
            return None
            
        # Create a new GenAI configuration with the appropriate key
        genai.configure(api_key=api_key)
        
        # Preferred model for dialogue tasks
        preferred_model = "gemini-2.0-flash-exp" if "gemini-1.5-pro" in self.model_name else self.model_name
        if not preferred_model:
            preferred_model = "gemini-1.5-pro"  # Fallback if no model was initialized
            
        # Create and return the model
        return genai.GenerativeModel(
            preferred_model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": self.top_p,
            }
        )
        
    def analyze_scene_keyframes(self, scene_idx: int, scene_info: Dict, 
                               keyframes_dir: str, entity_data: Optional[Dict] = None,
                               audio_data: Optional[Dict] = None) -> Dict:
        """
        Analyze keyframes from a scene and generate a description.
        
        Args:
            scene_idx: Scene index
            scene_info: Scene information dictionary
            keyframes_dir: Directory containing keyframe images
            entity_data: Optional entity detection results
            audio_data: Optional audio processing results
            
        Returns:
            Dictionary with scene analysis results
        """
        if not self.model_name:
            logger.warning("Gemini Vision model not available")
            return {
                "scene_idx": scene_idx,
                "error": "Gemini Vision model not available",
                "description": "Scene analysis not available due to missing or deprecated Gemini Vision model. Please check logs for details."
            }
            
        keyframes_dir = Path(keyframes_dir)
        
        # Get keyframe paths for this scene
        keyframe_paths = []
        for keyframe in scene_info:
            path = keyframes_dir / keyframe["path"]
            if path.exists():
                keyframe_paths.append((keyframe["frame_idx"], keyframe["timestamp"], path))
                
        if not keyframe_paths:
            logger.warning(f"No keyframes found for scene {scene_idx}")
            return {
                "scene_idx": scene_idx,
                "error": "No keyframes found",
                "description": "Scene analysis not available due to missing keyframes."
            }
            
        logger.info(f"Analyzing scene {scene_idx} with {len(keyframe_paths)} keyframes")
        
        # Sort by frame index
        keyframe_paths.sort(key=lambda x: x[0])
        
        # Collect entity information for this scene if available
        entity_info = None
        if entity_data and "scenes" in entity_data and str(scene_idx) in entity_data["scenes"]:
            entity_info = self._extract_entity_summary(entity_data["scenes"][str(scene_idx)])
            
        # Collect audio/dialogue information if available
        dialogue_info = None
        if audio_data and "scenes" in audio_data and str(scene_idx) in audio_data["scenes"]:
            dialogue_info = self._extract_dialogue_summary(audio_data["scenes"][str(scene_idx)])
            
        # Prepare context for the model
        context = self._prepare_scene_context(scene_idx, scene_info, entity_info, dialogue_info)
        
        # Create the prompt
        prompt = self._create_scene_analysis_prompt(scene_idx, context)
        
        # Load images (limiting to maximum 8 to avoid token limits)
        # Gemini 2.0 has better multi-image handling, but we'll still keep a reasonable limit
        max_images = min(8, len(keyframe_paths))
        selected_indices = self._select_diverse_frames(keyframe_paths, max_images)
        images = []
        
        for idx in selected_indices:
            frame_idx, timestamp, path = keyframe_paths[idx]
            try:
                image = Image.open(path)
                images.append(image)
            except Exception as e:
                logger.warning(f"Error loading image {path}: {str(e)}")
                
        if not images:
            logger.warning(f"Could not load any images for scene {scene_idx}")
            return {
                "scene_idx": scene_idx,
                "error": "Failed to load images",
                "description": "Scene analysis not available due to image loading errors."
            }
            
        # Generate scene description with Gemini Vision
        try:
            logger.info(f"Calling Gemini API for scene {scene_idx} with {len(images)} images")
            
            # Create content parts: text prompt followed by images
            content_parts = [prompt]
            for image in images:
                content_parts.append(image)
            
            description = self._handle_gemini_api_call(content_parts)
            
            # Parse structured output if configured
            if self.structured_output:
                try:
                    structured_result = self._parse_structured_output(description)
            
            # Create and return the result
                    result = {
                        "scene_idx": scene_idx,
                        "num_keyframes_analyzed": len(images),
                        "keyframes_used": [keyframe_paths[idx][0] for idx in selected_indices],
                        "description": description,
                        "structured_analysis": structured_result,
                        "entity_summary": entity_info,
                        "dialogue_summary": dialogue_info
                    }
                except Exception as parse_err:
                    logger.warning(f"Failed to parse structured output: {str(parse_err)}")
                    result = {
                        "scene_idx": scene_idx,
                        "num_keyframes_analyzed": len(images),
                        "keyframes_used": [keyframe_paths[idx][0] for idx in selected_indices],
                        "description": description,
                        "structured_analysis": None,
                        "parse_error": str(parse_err),
                        "entity_summary": entity_info,
                        "dialogue_summary": dialogue_info
                    }
            else:
                # Create and return the result (original format)
                result = {
                "scene_idx": scene_idx,
                "num_keyframes_analyzed": len(images),
                "keyframes_used": [keyframe_paths[idx][0] for idx in selected_indices],
                "description": description,
                "entity_summary": entity_info,
                "dialogue_summary": dialogue_info
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating scene description: {str(e)}")
            # Create more useful error message for deprecated model errors
            error_msg = str(e)
            description = f"Scene analysis failed due to an error: {error_msg}"
            
            if "404" in error_msg and "deprecated" in error_msg.lower():
                description = ("Scene analysis failed because the Gemini model is deprecated. "
                              "Please update to a newer model version like gemini-1.5-flash or gemini-2.0. "
                              "You may need to reinstall the google-generativeai package or update your API key permissions.")
            
            return {
                "scene_idx": scene_idx,
                "error": error_msg,
                "description": description
            }
            
    def analyze_scene(self, scene_idx: int, scene_info: Dict, 
                     keyframes_dir: str, scene_video_path: Optional[str] = None,
                     entity_data: Optional[Dict] = None,
                     audio_data: Optional[Dict] = None) -> Dict:
        """
        Enhanced scene analysis with both keyframes and video segments.
        
        Args:
            scene_idx: Scene index
            scene_info: Scene information dictionary
            keyframes_dir: Directory containing keyframe images
            scene_video_path: Optional path to scene video segment 
            entity_data: Optional entity detection results
            audio_data: Optional audio processing results
            
        Returns:
            Dictionary with enhanced scene analysis
        """
        # Process keyframes as before
        result = self.analyze_scene_keyframes(
            scene_idx, 
            scene_info, 
            keyframes_dir, 
            entity_data=entity_data,
            audio_data=audio_data
        )
        
        # If no video segment to analyze, just return the keyframe analysis
        if not scene_video_path or not Path(scene_video_path).exists():
            return result
            
        # Enhanced analysis with video segment
        logger.info(f"Enhancing scene {scene_idx} analysis with video segment: {scene_video_path}")
        
        try:
            # Extract entity information if available
            entity_info = None
            if entity_data:
                # Different possible entity_data formats
                try:
                    # Handle entity_data as a dictionary with nested structure
                    if isinstance(entity_data, dict):
                        entity_info = self._extract_entity_summary(entity_data)
                    # Handle entity_data as a list
                    elif isinstance(entity_data, list):
                        entity_info = self._extract_entity_summary(entity_data)
                    else:
                        logger.warning(f"Unexpected entity_data format for scene {scene_idx}: {type(entity_data)}")
                        # Try with a best-effort approach
                        entity_info = self._extract_entity_summary([entity_data]) if entity_data else None
                except Exception as e:
                    logger.warning(f"Could not extract entity information: {str(e)}")
                    entity_info = None
                    
            # Collect audio/dialogue information if available
            dialogue_info = None
            if audio_data:
                try:
                    dialogue_info = self._extract_dialogue_summary(audio_data)
                except Exception as e:
                    logger.warning(f"Could not extract dialogue information: {str(e)}")
                    dialogue_info = None
            
            # Prepare context for the model
            context = self._prepare_scene_context(scene_idx, scene_info, entity_info, dialogue_info)
            
            # Process the video segment
            video_analysis = self._analyze_scene_video(scene_video_path, scene_idx, context)
            
            # Add video analysis to result
            result["video_analysis"] = video_analysis
            
            # Create enhanced description combining both analyses
            if "description" in result and "description" in video_analysis:
                result["enhanced_description"] = self._combine_analyses(
                    result.get("description", ""),
                    video_analysis.get("description", "")
                )
            
            logger.info(f"Successfully enhanced scene {scene_idx} analysis with video data")
        except Exception as e:
            logger.error(f"Error in video analysis for scene {scene_idx}: {str(e)}")
            result["video_analysis_error"] = str(e)
        
        return result
    
    def _analyze_scene_video(self, video_path: str, scene_idx: int, context: str) -> Dict:
        """
        Analyze a video segment using Gemini.
        
        Args:
            video_path: Path to the video segment
            scene_idx: Scene index
            context: Context information about the scene
            
        Returns:
            Dictionary with video analysis results
        """
        # For Gemini, we'll extract frames from the video at 1fps
        # and send them as a sequence to better capture motion
        import cv2
        from PIL import Image
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames at 1fps or less if video is very short
        sample_rate = max(1, int(fps))
        
        success = True
        count = 0
        
        while success and len(frames) < 32:  # Limit to 32 frames for Gemini
            success, frame = cap.read()
            if success and count % sample_rate == 0:
                # Convert to RGB and add to frames
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                frames.append(pil_img)
            count += 1
        
        cap.release()
        
        if not frames:
            logger.warning(f"No frames extracted from video segment: {video_path}")
            return {
                "error": "Failed to extract frames from video segment",
                "description": "Video analysis not available due to frame extraction failure."
            }
        
        logger.info(f"Extracted {len(frames)} frames from video segment for analysis")
        
        # Create a prompt specifically for video sequence analysis
        prompt = self._create_video_analysis_prompt(scene_idx, context)
        
        # Call Gemini with the sequence of frames
        content_parts = [prompt]
        for frame in frames:
            content_parts.append(frame)
        
        try:
            # Make the API call
            response_text = self._handle_gemini_api_call(content_parts)
            
            # Parse the response
            structured_analysis = self._parse_structured_output(response_text)
            
            return {
                "description": response_text,
                "structured": structured_analysis,
                "frames_analyzed": len(frames)
            }
        except Exception as e:
            logger.error(f"Error in Gemini API call for video analysis: {str(e)}")
            return {
                "error": str(e),
                "description": f"Video analysis failed: {str(e)}",
                "frames_analyzed": len(frames)
            }
    
    def _create_video_analysis_prompt(self, scene_idx: int, context: str) -> str:
        """Create a prompt specifically for video sequence analysis."""
        prompt = f"""# Video Sequence Analysis Task

## Overview
Analyze the video sequence from Scene {scene_idx + 1}, providing special attention to motion, transitions, and temporal narrative flow. Focus on creating a comprehensive analysis that captures both the visual elements and the dynamic aspects of the scene.

## Context Information
{context}

## Required Analysis Components

### Movement & Action
Describe the key movements, actions, and transitions in the sequence. How do characters or objects move through the frame? What actions occur and how do they develop? Pay special attention to the flow of motion and changing dynamics.

### Temporal Progression
Analyze how the scene evolves over time. Identify distinct phases or shifts in the narrative, mood, or visual composition as the sequence progresses. Note any significant changes in lighting, framing, or focus.

### Cinematography & Visual Flow
Examine the camera techniques and visual language. Does the camera move, cut, or employ special techniques? How do these choices enhance the storytelling? Comment on pacing, rhythm, and visual continuity.

### Dialogue & Sound Cues
Analyze how dialogue and action interrelate in the scene. Does the dialogue pace the scene? Are there meaningful pauses or interruptions? How do vocal intonations affect the scene's meaning?

### Character Development & Interactions
Observe how characters evolve within the scene. Do their emotions or positions change? How do interactions between characters develop over time? Note significant character moments.

## Output Format
Provide your analysis as a cohesive narrative that captures both the static visual elements and the dynamic flow of the scene. Integrate insights about motion, pacing, and temporal development that wouldn't be visible from isolated frames.
"""
        return prompt
    
    def _combine_analyses(self, keyframe_analysis: str, video_analysis: str) -> str:
        """
        Combine keyframe and video analyses into a unified enhanced description.
        
        Args:
            keyframe_analysis: Description from keyframe analysis
            video_analysis: Description from video sequence analysis
            
        Returns:
            Enhanced combined description
        """
        # Get the most important insights from each analysis
        keyframe_paragraphs = keyframe_analysis.split('\n\n')
        video_paragraphs = video_analysis.split('\n\n')
        
        # Start with a clear introduction
        combined = "# Enhanced Scene Analysis\n\n"
        
        # Add visual elements from keyframe analysis
        combined += "## Visual Content\n\n"
        # Take first 2-3 paragraphs from keyframe analysis, which usually describe the setting
        for i in range(min(2, len(keyframe_paragraphs))):
            combined += keyframe_paragraphs[i] + "\n\n"
        
        # Add dynamic elements from video analysis
        combined += "## Dynamic Elements\n\n"
        # Find paragraphs about movement and action
        movement_paragraphs = []
        for para in video_paragraphs:
            if any(term in para.lower() for term in ["movement", "motion", "action", "transition", "moves", "walking", "changes"]):
                movement_paragraphs.append(para)
        
        # Add all movement paragraphs or the first 2 video paragraphs if none found
        if movement_paragraphs:
            for para in movement_paragraphs[:2]:
                combined += para + "\n\n"
        else:
            for i in range(min(2, len(video_paragraphs))):
                combined += video_paragraphs[i] + "\n\n"
        
        # Add character interactions and dialogue
        combined += "## Characters & Dialogue\n\n"
        # Look for paragraphs about characters and dialogue in both analyses
        character_paragraphs = []
        for para in keyframe_paragraphs + video_paragraphs:
            if any(term in para.lower() for term in ["character", "speak", "dialogue", "conversation", "talks", "says"]):
                if para not in character_paragraphs:  # Avoid duplicates
                    character_paragraphs.append(para)
        
        # Add character paragraphs
        for para in character_paragraphs[:2]:
            combined += para + "\n\n"
        
        # Add cinematography insights
        combined += "## Cinematography & Significance\n\n"
        # Look for paragraphs about cinematography and significance
        cinematography_paragraphs = []
        for para in keyframe_paragraphs + video_paragraphs:
            if any(term in para.lower() for term in ["cinematography", "camera", "visual", "symbolism", "significance", "theme"]):
                if para not in cinematography_paragraphs:  # Avoid duplicates
                    cinematography_paragraphs.append(para)
        
        # Add cinematography paragraphs
        for para in cinematography_paragraphs[:2]:
            combined += para + "\n\n"
        
        return combined
        
    def _handle_gemini_api_call(self, content_parts: List) -> str:
        """Handle Gemini API call with error handling, retries and exponential backoff."""
        max_retries = 5
        base_delay = 1  # starting delay in seconds
        
        # Check if model is available and initialized
        if not self.model_available or self.model is None:
            logger.error("Gemini model not available or not initialized")
            raise RuntimeError("Gemini model not available or not initialized. Check API key and model availability.")
        
        for retry in range(max_retries):
            try:
                logger.info(f"Making Gemini API call (attempt {retry+1}/{max_retries})")
                
                # New updated safety settings format
                # Documentation: https://ai.google.dev/api/python/google/generativeai/GenerationConfig#safety_settings
                safety_settings = {
                    "harassment": "block_only_high",  # Changed from block_none to block_only_high
                    "hate": "block_only_high",        # Changed from block_none to block_only_high
                    "sexual": "block_only_high",      # Changed from block_none to block_only_high
                    "dangerous": "block_only_high"    # Changed from block_none to block_only_high
                }
                
                # Standard API call
                response = self.model.generate_content(
                    content_parts,
                    stream=False,
                    safety_settings=safety_settings
                )
                
                if hasattr(response, 'text'):
                    return response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    return response.candidates[0].content.parts[0].text.strip()
                else:
                    # Check if the prompt was blocked
                    if hasattr(response, 'prompt_feedback'):
                        block_reason = response.prompt_feedback.block_reason if hasattr(response.prompt_feedback, 'block_reason') else "Unknown"
                        logger.warning(f"Prompt was blocked. Reason: {block_reason}")
                        
                        # Try to moderate the prompt
                        if retry < max_retries - 1:  # Only try moderation if we have retries left
                            moderated_content = self._moderate_content_for_safety(content_parts)
                            if moderated_content != content_parts:
                                content_parts = moderated_content
                                logger.info("Using moderated content for next attempt")
                                continue  # Skip to next retry with moderated content
                    
                    logger.warning("Received empty response from Gemini API")
                    raise ValueError("Empty response from Gemini API")
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check for specific error types
                if "429" in error_msg:  # Rate limit error
                    logger.warning(f"Rate limit hit: {error_msg}")
                elif "404" in error_msg and "deprecated" in error_msg.lower():  # Deprecated model
                    logger.error(f"Model {self.model_name} is deprecated: {error_msg}")
                    logger.info("Attempting to reinitialize with a newer model")
                    self._initialize_model()
                    if not self.model_available:
                        raise RuntimeError("Failed to reinitialize with a non-deprecated model")
                elif "403" in error_msg:  # Authentication error
                    logger.error(f"Authentication error: {error_msg}")
                    # Check if GEMINI_API_KEY is set
                    if not os.environ.get("GEMINI_API_KEY"):
                        logger.error("GEMINI_API_KEY environment variable is not set")
                        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
                    # If API key exists but still getting 403, it might be invalid
                    raise RuntimeError("Invalid API key or insufficient permissions")
                elif "blocked" in error_msg.lower() or "prohibited" in error_msg.lower():
                    # Content was blocked - try with moderated content
                    logger.warning(f"Content blocked: {error_msg}")
                    if retry < max_retries - 1:  # Only try moderation if we have retries left
                        moderated_content = self._moderate_content_for_safety(content_parts)
                        if moderated_content != content_parts:
                            content_parts = moderated_content
                            logger.info("Using moderated content for next attempt")
                            continue  # Skip to next retry with moderated content
                else:
                    logger.error(f"Error in Gemini API call: {error_msg}")
                
                # Last retry - attempt with experimental safety settings as a last resort
                if retry == max_retries - 1:
                    try:
                        logger.info("Attempting final call with alternative safety settings")
                        
                        # Try with an alternative safety settings approach for the latest Gemini API
                        import google.generativeai as genai
                        generation_config = genai.GenerationConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                            top_p=self.top_p,
                        )
                        
                        # Create a fallback model instance with different safety settings
                        fallback_model = genai.GenerativeModel(
                            self.model_name,
                            generation_config=generation_config,
                            safety_settings={
                                "harassment": "block_only_high",
                                "hate": "block_only_high",
                                "sexual": "block_only_high",
                                "dangerous": "block_only_high"
                            }
                        )
                        
                        # Make the final fallback call 
                        response = fallback_model.generate_content(
                            self._create_fallback_prompt(content_parts)
                        )
                        
                        if hasattr(response, 'text'):
                            return response.text.strip()
                        elif hasattr(response, 'candidates') and response.candidates:
                            return response.candidates[0].content.parts[0].text.strip()
                        else:
                            # Return a graceful fallback message if all attempts fail
                            return self._generate_fallback_response(content_parts)
                    except Exception as inner_e:
                        # If all attempts fail, use fallback generation
                        logger.error(f"All Gemini API call attempts failed after {max_retries} retries. Last error: {str(inner_e)}")
                        return self._generate_fallback_response(content_parts)
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** retry) + random.uniform(0, 0.5)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # This should not be reached due to the exception in the last retry
        return self._generate_fallback_response(content_parts)
            
    def _moderate_content_for_safety(self, content_parts: List) -> List:
        """
        Attempt to modify content to pass safety filters by toning down potentially problematic language.
        
        Args:
            content_parts: The content parts to moderate
            
        Returns:
            Moderated content parts
        """
        # Create a copy of the content parts to modify
        moderated_parts = copy.deepcopy(content_parts)
        
        # Check if the first part is a string (prompt text)
        if moderated_parts and isinstance(moderated_parts[0], str):
            prompt_text = moderated_parts[0]
            
            # Replace potentially problematic terms with more neutral ones
            replacements = [
                (r'\b(explicit|adult|sexual|nude|nudity|naked)\b', 'appropriate'),
                (r'\b(violence|violent|gore|bloody|killing)\b', 'action'),
                (r'\b(hate|hateful|racial slur|slurs|offensive)\b', 'strong language'),
                (r'\b(extremist|terrorism|terrorist)\b', 'concerning'),
                # Add more patterns as needed
            ]
            
            # Apply replacements
            for pattern, replacement in replacements:
                prompt_text = re.sub(pattern, replacement, prompt_text, flags=re.IGNORECASE)
            
            # Add a disclaimer to ensure the prompt is treated as analytical
            disclaimer = "\nNote: This analysis is for educational and analytical purposes only, focusing on storytelling, cinematography, and narrative structure."
            
            # Add the disclaimer if it's not already there
            if disclaimer not in prompt_text:
                prompt_text += disclaimer
                
            # Update the first element with the moderated text
            moderated_parts[0] = prompt_text
            
        return moderated_parts
        
    def _create_fallback_prompt(self, content_parts: List) -> List:
        """
        Create a more neutral fallback prompt for when the original is blocked.
        
        Args:
            content_parts: The original content parts
            
        Returns:
            Modified content parts with a more neutral prompt
        """
        fallback_parts = copy.deepcopy(content_parts)
        
        # If there's text in the first part, replace it with a more neutral prompt
        if fallback_parts and isinstance(fallback_parts[0], str):
            neutral_prompt = """
            Please provide a simple, objective description of the visual content in the provided image.
            Focus only on:
            - General scene composition
            - Setting/location
            - Primary subjects visible
            - Basic actions occurring
            - Time of day/lighting conditions
            
            Keep the description brief, factual, and purely descriptive without interpretation.
            """
            
            fallback_parts[0] = neutral_prompt
            
        return fallback_parts
        
    def _generate_fallback_response(self, content_parts: List) -> str:
        """
        Generate a fallback response when all API calls fail due to safety filters.
        
        Args:
            content_parts: The original content parts
            
        Returns:
            A fallback response
        """
        # Create a generic scene description as fallback
        return """
        Scene Description:
        
        This scene contains visual content that could not be automatically analyzed due to content safety filters.
        
        The analysis system detected elements that may require human review. For a complete analysis, consider:
        
        1. Reviewing this scene manually
        2. Adjusting the content if needed
        3. Using a custom analysis approach for this specific section
        
        The automated analysis has continued with the remaining scenes.
        """
        
    def _select_diverse_frames(self, keyframe_paths: List[Tuple], max_frames: int) -> List[int]:
        """Select a diverse set of frames from the available keyframes."""
        if len(keyframe_paths) <= max_frames:
            return list(range(len(keyframe_paths)))
            
        # For simplicity, we'll use uniform sampling
        # In a more advanced implementation, we could use feature-based diversity measures
        step = len(keyframe_paths) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        
        # Always include first and last frame
        if 0 not in indices:
            indices[0] = 0
        if len(keyframe_paths) - 1 not in indices:
            indices[-1] = len(keyframe_paths) - 1
            
        return indices
        
    def _extract_entity_summary(self, scene_entities: List[Dict]) -> Dict:
        """Extract a summary of entities detected in the scene."""
        # Count objects
        objects = {}
        faces = 0
        text_snippets = []
        
        # Handle case where scene_entities is None or empty
        if not scene_entities:
            return {
                "objects": {},
                "avg_faces_per_frame": 0,
                "text_snippets": []
            }
        
        # Process each frame
        for frame in scene_entities:
            try:
                # Handle different possible data structures
                if isinstance(frame, dict):
                    # Case 1: Frame has an "entities" field
                    if "entities" in frame:
                        entities_data = frame["entities"]
                        
                        # Process entities_data as dictionary
                        if isinstance(entities_data, dict):
                            # Process objects
                            for obj in entities_data.get("objects", []):
                                obj_type = obj.get("type", "unknown")
                                if obj_type in objects:
                                    objects[obj_type] += 1
                                else:
                                    objects[obj_type] = 1
                                    
                            # Faces
                            faces += len(entities_data.get("faces", []))
                            
                            # Text
                            for text in entities_data.get("text", []):
                                if text.get("text") and text.get("confidence", 0) > 0.7:
                                    text_snippets.append(text.get("text"))
                        
                        # Process entities_data as list
                        elif isinstance(entities_data, list):
                            for item in entities_data:
                                if isinstance(item, dict):
                                    obj_type = item.get("type", "unknown")
                                    if obj_type in objects:
                                        objects[obj_type] += 1
                                    else:
                                        objects[obj_type] = 1
                    
                    # Case 2: Frame itself contains entity data directly
                    elif "label" in frame or "type" in frame:
                        obj_type = frame.get("type", frame.get("label", "unknown"))
                        if obj_type in objects:
                            objects[obj_type] += 1
                        else:
                            objects[obj_type] = 1
                
                # Handle case where frame is a list
                elif isinstance(frame, list):
                    for item in frame:
                        if isinstance(item, dict):
                            obj_type = item.get("type", item.get("label", "unknown"))
                            if obj_type in objects:
                                objects[obj_type] += 1
                            else:
                                objects[obj_type] = 1
            
            except Exception as e:
                logger.warning(f"Error processing entity frame: {e}")
                continue
        
        # Sort and limit
        sorted_objects = sorted(objects.items(), key=lambda x: x[1], reverse=True)
        top_objects = dict(sorted_objects[:10])
        
        # Deduplicate text snippets
        unique_text = list(set(text_snippets))
        
        # Calculate average faces per frame
        avg_faces = faces / max(1, len(scene_entities))
        
        return {
            "objects": top_objects,
            "avg_faces_per_frame": avg_faces,
            "text_snippets": unique_text[:5]  # Limit to 5 text snippets
        }
        
    def _extract_dialogue_summary(self, scene_audio: Dict) -> Dict:
        """Extract a summary of dialogue from the scene."""
        segments = scene_audio.get("segments", [])
        
        # Group by speaker
        speaker_lines = {}
        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            text = segment.get("text", "").strip()
            
            if speaker not in speaker_lines:
                speaker_lines[speaker] = []
                
            if text:
                speaker_lines[speaker].append(text)
                
        # Count speakers and lines
        num_speakers = len(speaker_lines)
        
        # Join speaker lines
        speaker_dialogue = {}
        for speaker, lines in speaker_lines.items():
            speaker_dialogue[speaker] = " ".join(lines)
            
        return {
            "num_speakers": num_speakers,
            "speaker_dialogue": speaker_dialogue,
            "total_segments": len(segments)
        }
        
    def _prepare_scene_context(self, scene_idx: int, scene_info: Dict, 
                             entity_info: Optional[Dict], dialogue_info: Optional[Dict]) -> str:
        """Prepare context information for the scene analysis."""
        context = []
        
        # Basic scene info
        if scene_info:
            try:
                # Handle scene_info as list (original approach)
                if isinstance(scene_info, list) and len(scene_info) > 0:
                    start_time = scene_info[0].get("timestamp", 0) if isinstance(scene_info[0], dict) else 0
                    end_time = scene_info[-1].get("timestamp", 0) if isinstance(scene_info[-1], dict) else 0
                    duration = end_time - start_time
                # Handle scene_info as dict
                elif isinstance(scene_info, dict):
                    start_time = scene_info.get("start_time", 0)
                    end_time = scene_info.get("end_time", 0)
                    duration = end_time - start_time
                else:
                    # Fallback
                    start_time = 0
                    end_time = 0
                    duration = 0
            except Exception as e:
                logger.warning(f"Error extracting time info from scene_info: {e}")
                start_time = 0
                end_time = 0
                duration = 0
                
            context.append(f"SCENE {scene_idx + 1}")
            context.append(f"Duration: {duration:.2f} seconds")
            
        # Entity information
        if entity_info:
            try:
                # Objects
                if entity_info.get("objects"):
                    try:
                        obj_text = ", ".join(
                            f"{obj} ({count})" for obj, count in list(entity_info["objects"].items())[:5]
                        )
                        context.append(f"Main objects detected: {obj_text}")
                    except Exception as e:
                        logger.warning(f"Error formatting object information: {e}")
                    
                # Faces
                if entity_info.get("avg_faces_per_frame", 0) > 0:
                    context.append(f"Average faces per frame: {entity_info['avg_faces_per_frame']:.1f}")
                    
                # Text
                if entity_info.get("text_snippets"):
                    try:
                        text_str = "; ".join(entity_info["text_snippets"])
                        context.append(f"Text visible in scene: {text_str}")
                    except Exception as e:
                        logger.warning(f"Error formatting text snippets: {e}")
            except Exception as e:
                logger.warning(f"Error processing entity information: {e}")
                # Provide a minimal context
                context.append("Entity detection information available but could not be formatted.")
                
        # Dialogue information
        if dialogue_info and dialogue_info.get("speaker_dialogue"):
            try:
                context.append("\nDIALOGUE:")
                for speaker, text in dialogue_info["speaker_dialogue"].items():
                    # Limit dialogue text to avoid making the prompt too long
                    if len(text) > 500:
                        text = text[:497] + "..."
                    context.append(f"{speaker}: {text}")
            except Exception as e:
                logger.warning(f"Error formatting dialogue information: {e}")
                context.append("Dialogue information available but could not be formatted.")
                
        return "\n".join(context)
        
    def _create_scene_analysis_prompt(self, scene_idx: int, context: str) -> str:
        """
        Create a prompt for scene analysis.
        
        Args:
            scene_idx: Index of the scene
            context: Context information about the scene
            
        Returns:
            Prompt for scene analysis
        """
        print(f"Creating scene analysis prompt for scene {scene_idx + 1}")
        prompt = f"""# Media Analysis Task

## Overview
Analyze Scene {scene_idx + 1} from the video sequence, providing a detailed description based on the provided images, video, audio and context information. Focus on creating a professional, objective analysis that describes the narrative elements, visual composition, and storytelling techniques.

## Context Information
{context}

## Required Analysis Components

### Setting
Describe the location, time of day, atmosphere, and key visual elements from the key frames. Include details about environmental features (e.g., architecture, landscapes) and visual impressions from the scene.

### Action & Plot
Summarize the main actions and events occurring in the scene. Identify the key narrative progression and how the events contribute to the story. Note any important shifts in the storyline.

### Characters & Expressions
Analyze the actions, gestures, and expressions of the people shown. Describe their apparent emotions and relationships. Note any significant changes in character states throughout the scene.

### Dialogue & Communication
Analyze any spoken or text elements in the scene. Focus on how the dialogue advances the narrative. For multiple speakers, differentiate between them and identify key exchanges.

### Visual Composition
Comment on the camera angles, framing, shot types, and composition of the images. Describe the role of lighting, color, and visual elements. Discuss how these technical elements contribute to the storytelling.

### Themes & Symbols
Identify recurring motifs or visual elements present in the scene. Discuss how these elements relate to the larger narrative themes.

### Narrative Structure
Examine how this scene fits into the overall story flow. If the scene presents different timelines or locations, provide insights into how these elements affect the narrative structure.

### Emotional Tone
Discuss the overall tone of the scene, focusing on the mood it conveys to the audience. Explain how the visual and auditory elements establish the atmosphere.

### Technical Aspects
Evaluate the production elements such as visual quality, set design, and other technical components that contribute to the scene's effectiveness.

## Output Format
Provide your analysis as a well-structured description with clear paragraphs. Focus on creating a cohesive narrative describing what's actually happening in the scene.

For videos with multiple storylines:
- Note when the scene shifts between different storylines or settings
- Describe how these narrative elements relate to each other
- Identify visual cues that signal a setting change

Note: This analysis is for educational and research purposes only, focusing on storytelling, cinematography, and narrative structure.
"""
        return prompt
        
    def _parse_structured_output(self, text: str) -> Dict[str, str]:
        """
        Parse structured output from the model response.
        
        Args:
            text: Text response from the model
            
        Returns:
            Dictionary with structured fields
        """
        result = {}
        
        # Remove any "Okay, let's analyze..." or similar introductory phrases
        text = re.sub(r'^(Okay|Let\'s|Now|Well),?\s+(let\'s|I will|I\'ll|we will|we\'ll)?\s*(analyze|examine|look at|study)\s*(Scene \d+.*?based on.*?context\.?)?', '', text, flags=re.IGNORECASE|re.MULTILINE)
        text = text.strip()
        
        # Split the text by markdown headers (###)
        sections = re.split(r'###\s+', text)
        
        # Process each section
        for section in sections:
            if not section.strip():
                continue
                
            # Extract the field name and content
            lines = section.split('\n', 1)
            if len(lines) < 2:
                continue
                
            field_name = lines[0].strip().lower()
            content = lines[1].strip()
            
            # Clean up field name
            field_name = re.sub(r'[^a-z0-9_]', '_', field_name)
            
            # Remove any remaining conversational elements from content
            content = re.sub(r'^(In this scene,?|Here,?|We can see that,?)\s+', '', content, flags=re.IGNORECASE)
            content = re.sub(r'(As we can see,?|Looking at the scene,?|It\'s worth noting that,?)\s+', '', content, flags=re.IGNORECASE)
            
            # Store in result
            result[field_name] = content.strip()
            
        # If no structured content was found, create a generic entry
        if not result and text.strip():
            # Clean up the general description
            description = text.strip()
            description = re.sub(r'^(In this scene,?|Here,?|We can see that,?)\s+', '', description, flags=re.IGNORECASE)
            description = re.sub(r'(As we can see,?|Looking at the scene,?|It\'s worth noting that,?)\s+', '', description, flags=re.IGNORECASE)
            result["general_description"] = description
            
        return result
            
    def analyze_scene_with_fallbacks(self, scene_idx: int, scene_info: Dict, 
                               keyframes_dir: str, entity_data: Optional[Dict] = None,
                               audio_data: Optional[Dict] = None) -> Dict:
        """
        Analyze a scene with multiple fallback approaches if content is blocked.
        
        Args:
            scene_idx: Scene index
            scene_info: Scene information
            keyframes_dir: Directory containing keyframes
            entity_data: Optional entity detection data
            audio_data: Optional audio processing data
            
        Returns:
            Scene analysis results
        """
        logger.info(f"Analyzing scene {scene_idx} with fallback approaches")
        
        # First attempt: Standard analysis
        try:
            result = self.analyze_scene(
                scene_idx, 
                scene_info, 
                keyframes_dir, 
                entity_data=entity_data,
                audio_data=audio_data
            )
            
            # If we got a successful result (no error), return it
            if "error" not in result or "content blocked" not in result.get("error", "").lower():
                return result
                
            logger.warning(f"Standard analysis failed for scene {scene_idx}, trying basic analysis")
        except Exception as e:
            logger.error(f"Error in standard analysis for scene {scene_idx}: {str(e)}")
            logger.warning(f"Falling back to basic analysis for scene {scene_idx}")
        
        # Second attempt: Basic analysis with minimal prompt
        try:
            # Get scene keyframes
            keyframe_paths = []
            for keyframe in scene_info:
                path = Path(keyframes_dir) / keyframe["path"]
                if path.exists():
                    keyframe_paths.append((keyframe["frame_idx"], keyframe["timestamp"], path))
                    
            if not keyframe_paths:
                return {
                    "scene_idx": scene_idx,
                    "error": "No keyframes found",
                    "description": "Scene analysis not available due to missing keyframes."
                }
                
            # Sort by frame index
            keyframe_paths.sort(key=lambda x: x[0])
            
            # Select a few representative frames
            selected_indices = [0]  # Always use first frame
            if len(keyframe_paths) > 1:
                selected_indices.append(len(keyframe_paths) - 1)  # Add last frame
            if len(keyframe_paths) > 2:
                selected_indices.append(len(keyframe_paths) // 2)  # Add middle frame
                
            selected_keyframes = [keyframe_paths[i] for i in selected_indices]
            
            # Load selected keyframes
            images = []
            for _, _, path in selected_keyframes:
                try:
                    img = Image.open(path)
                    images.append(img)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {str(e)}")
                    
            if not images:
                return {
                    "scene_idx": scene_idx,
                    "error": "Failed to load keyframe images",
                    "description": "Could not load any keyframe images for analysis."
                }
                
            # Create basic prompt
            basic_prompt = f"""
            Please provide a simple visual description of the images from Scene {scene_idx + 1}.
            Focus only on:
            - What is clearly visible in the images
            - The general setting/location
            - People or main subjects shown
            - Basic actions or events depicted
            
            Keep your description purely factual, objective, and brief.
            """
            
            # Create content parts
            content_parts = [basic_prompt]
            for img in images:
                content_parts.append(img)
                
            # Make API call with basic prompt
            try:
                description = self._handle_gemini_api_call(content_parts)
                
                # Create simple structured data
                structured_data = {
                    "setting": description,
                    "action": "",
                    "characters": "",
                    "significance": ""
                }
                
                return {
                    "scene_idx": scene_idx,
                    "description": description,
                    "structured_data": structured_data,
                    "num_keyframes_analyzed": len(images),
                    "analysis_method": "basic",
                    "keyframes_used": [str(path) for _, _, path in selected_keyframes]
                }
            except Exception as e:
                logger.error(f"Basic analysis failed for scene {scene_idx}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in basic analysis setup for scene {scene_idx}: {str(e)}")
            
        # Final fallback: Return a minimal result without API call
        return {
            "scene_idx": scene_idx,
            "error": "All analysis methods failed, possibly due to content restrictions",
            "description": "This scene could not be automatically analyzed due to content safety filters or technical errors.",
            "analysis_method": "fallback",
            "structured_data": {
                "setting": f"Scene {scene_idx + 1}",
                "action": "Scene content requires manual review",
                "characters": "",
                "significance": ""
            }
        }
        
    def analyze_all_scenes(self, keyframes_metadata: Dict, keyframes_dir: str,
                          entity_data: Optional[Dict] = None,
                          audio_data: Optional[Dict] = None,
                          output_path: Optional[str] = None) -> Dict:
        """
        Analyze all scenes in a video.
        
        Args:
            keyframes_metadata: Dictionary with keyframe metadata
            keyframes_dir: Directory containing keyframe images
            entity_data: Optional entity detection results
            audio_data: Optional audio processing results
            output_path: Optional path to save results
            
        Returns:
            Dictionary with scene analysis results
        """
        if not keyframes_metadata:
            logger.warning("Empty keyframes metadata provided")
            return {"scenes": {}, "error": "No keyframes metadata available"}
            
        if "scenes" not in keyframes_metadata:
            logger.warning("No 'scenes' key found in keyframes metadata, creating an empty structure")
            keyframes_metadata = {"scenes": {}}
            
        if not keyframes_metadata["scenes"]:
            logger.warning("No scenes found in keyframes metadata")
            return {"scenes": {}, "error": "No scenes in keyframes metadata"}
            
        results = {"scenes": {}}
        
        # Track potential storylines across scenes
        storyline_tracker = {
            "storylines": {},
            "scene_to_storyline": {}
        }
        
        # Track successful vs. blocked scene counts
        analysis_stats = {
            "total_scenes": len(keyframes_metadata["scenes"]),
            "successful_scenes": 0,
            "fallback_scenes": 0,
            "failed_scenes": 0
        }
        
        # Process each scene
        for scene_idx, scene_info in keyframes_metadata["scenes"].items():
            logger.info(f"Analyzing scene {scene_idx}")
            
            # Check if there are any keyframes for this scene
            if not scene_info:
                logger.warning(f"No keyframes found for scene {scene_idx}")
                # Create a placeholder result for the scene
                results["scenes"][scene_idx] = {
                    "scene_idx": int(scene_idx),
                    "error": "No keyframes available for analysis",
                    "description": "Cannot analyze this scene because no keyframes were extracted.",
                    "num_keyframes_analyzed": 0,
                    "keyframes_used": []
                }
                analysis_stats["failed_scenes"] += 1
                continue
            
            # Get entity data for this scene if available
            scene_entity_data = None
            if entity_data and "scenes" in entity_data and scene_idx in entity_data["scenes"]:
                scene_entity_data = entity_data["scenes"][scene_idx]
            
            # Get audio data for this scene if available
            scene_audio_data = None
            if audio_data and "scenes" in audio_data and scene_idx in audio_data["scenes"]:
                scene_audio_data = audio_data["scenes"][scene_idx]
            
            # Analyze the scene with fallbacks
            scene_result = self.analyze_scene_with_fallbacks(
                int(scene_idx), 
                scene_info, 
                keyframes_dir,
                entity_data=scene_entity_data,
                audio_data=scene_audio_data
            )
            
            # Track stats
            if "error" in scene_result:
                if "analysis_method" in scene_result and scene_result["analysis_method"] in ["basic", "fallback"]:
                    analysis_stats["fallback_scenes"] += 1
                else:
                    analysis_stats["failed_scenes"] += 1
            else:
                analysis_stats["successful_scenes"] += 1
                
            results["scenes"][scene_idx] = scene_result
        
        # Add analysis stats to results
        results["analysis_stats"] = analysis_stats
        logger.info(f"Scene analysis complete: {analysis_stats['successful_scenes']} successful, " 
                   f"{analysis_stats['fallback_scenes']} fallback, {analysis_stats['failed_scenes']} failed")
        
        # Only identify storylines if we have enough valid scene analyses
        if analysis_stats["successful_scenes"] + analysis_stats["fallback_scenes"] > 0:
            # Identify storylines and connections between scenes
            storyline_analysis = self._identify_storylines(results["scenes"])
            results["storyline_analysis"] = storyline_analysis
        else:
            # Create a placeholder storyline analysis
            results["storyline_analysis"] = {
                "storylines": [],
                "scene_to_storyline": {},
                "settings": [],
                "scene_to_setting": {}
            }
        
        # Save results if output path is provided
        if output_path:
            self._save_results(results, output_path)
            
        return results

    def _identify_storylines(self, scene_results: Dict) -> Dict:
        """
        Analyze scene results to identify interweaving storylines and connections.
        
        This method looks across all analyzed scenes to detect:
        1. Parallel storylines that may be interweaving
        2. Setting continuity and switches between settings
        3. Narrative connections between non-adjacent scenes
        
        Args:
            scene_results: Dictionary of scene analysis results
            
        Returns:
            Dictionary with storyline analysis
        """
        # Filter out non-scene entries
        scene_indices = sorted([int(idx) for idx in scene_results.keys() if idx.isdigit()])
        
        if not scene_indices:
            return {
                "storylines_detected": False,
                "interweaving_narrative": False,
                "storyline_count": 0
            }
            
        logger.info(f"Analyzing connections between {len(scene_indices)} scenes")
        
        # Extract settings from each scene to identify potential storylines
        settings = {}
        settings_by_scene = {}
        
        # Track scene transitions
        transitions = []
        
        # Identify potential storylines based on setting similarities
        for i, scene_idx in enumerate(scene_indices):
            scene_data = scene_results[str(scene_idx)]
            
            # Skip scenes with errors
            if "error" in scene_data:
                continue
                
            # Extract setting information if available
            setting_desc = ""
            if "setting" in scene_data:
                setting_desc = scene_data["setting"]
            elif "analysis" in scene_data and isinstance(scene_data["analysis"], dict) and "setting" in scene_data["analysis"]:
                setting_desc = scene_data["analysis"]["setting"]
            elif "structured" in scene_data and isinstance(scene_data["structured"], dict) and "setting" in scene_data["structured"]:
                setting_desc = scene_data["structured"]["setting"]
            elif "description" in scene_data:
                # Try to extract setting from description
                setting_desc = self._extract_setting_from_content(scene_data["description"])
                
            if setting_desc:
                settings_by_scene[scene_idx] = setting_desc
                
                # Check for storyline transitions
                if i > 0:
                    prev_idx = scene_indices[i-1]
                    prev_setting = settings_by_scene.get(prev_idx, "")
                    
                    # If settings are significantly different, it might indicate a storyline switch
                    if prev_setting and self._is_different_setting(prev_setting, setting_desc):
                        transitions.append({
                            "from_scene": prev_idx,
                            "to_scene": scene_idx,
                            "transition_type": "setting_change",
                            "description": f"Setting change from '{self._summarize_setting(prev_setting)}' to '{self._summarize_setting(setting_desc)}'"
                        })
        
        # If we have no settings, we can't identify storylines
        if not settings_by_scene:
            return {
                "storylines_detected": False,
                "interweaving_narrative": False,
                "storyline_count": 0,
                "error": "Could not extract setting information from any scenes"
            }
        
        # Group scenes into potential storylines based on setting similarity
        storylines = {}
        scene_to_storyline = {}
        
        # Initialize with the first scene
        if scene_indices:
            storylines[0] = {
                "scenes": [scene_indices[0]],
                "setting": settings_by_scene.get(scene_indices[0], "Unknown")
            }
            scene_to_storyline[scene_indices[0]] = 0
            
        current_storyline_id = 0
        
        # Go through scenes in order
        for i in range(1, len(scene_indices)):
            scene_idx = scene_indices[i]
            current_setting = settings_by_scene.get(scene_idx, "")
            
            # Check if this scene continues any existing storyline
            storyline_matched = False
            best_match = -1
            highest_similarity = -1
            
            for storyline_id, storyline in storylines.items():
                storyline_setting = storyline["setting"]
                similarity = self._setting_similarity(storyline_setting, current_setting)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = storyline_id
            
            # If it's similar enough to an existing storyline, add it there
            if highest_similarity > 0.7:
                storylines[best_match]["scenes"].append(scene_idx)
                scene_to_storyline[scene_idx] = best_match
                
                # If this isn't the immediately previous scene in the storyline, 
                # it's likely an interweaving structure
                last_scene_in_storyline = storylines[best_match]["scenes"][-2]
                if last_scene_in_storyline != scene_indices[i-1]:
                    transitions.append({
                        "from_scene": last_scene_in_storyline,
                        "to_scene": scene_idx,
                        "transition_type": "storyline_continuation",
                        "description": f"Returning to storyline {best_match} after interruption"
                    })
            else:
                # Create a new storyline
                current_storyline_id += 1
                storylines[current_storyline_id] = {
                    "scenes": [scene_idx],
                    "setting": current_setting
                }
                scene_to_storyline[scene_idx] = current_storyline_id
        
        # Compile final analysis
        interweaving = len(storylines) > 1 and any(len(s["scenes"]) > 1 for s in storylines.values())
        
        storyline_analysis = {
            "storylines_detected": len(storylines) > 1,
            "interweaving_narrative": interweaving,
            "storyline_count": len(storylines),
            "storylines": {},
            "transitions": transitions,
            "scene_to_storyline": scene_to_storyline
        }
        
        # Convert storylines to the final format
        for storyline_id, storyline in storylines.items():
            storyline_analysis["storylines"][str(storyline_id)] = {
                "scenes": storyline["scenes"],
                "setting_summary": self._summarize_setting(storyline["setting"]),
                "scene_count": len(storyline["scenes"])
            }
            
        return storyline_analysis
    
    def _extract_setting_from_content(self, content: str) -> str:
        """Extract setting information from scene content."""
        # Look for setting description paragraphs
        setting_indicators = ["setting:", "location:", "place:", "scene opens", "scene takes place"]
        
        for indicator in setting_indicators:
            if indicator in content.lower():
                # Find the sentence containing the indicator
                sentences = content.split('. ')
                for i, sentence in enumerate(sentences):
                    if indicator in sentence.lower():
                        # Return this sentence and the next one if available
                        if i < len(sentences) - 1:
                            return sentence + ". " + sentences[i+1]
                        return sentence
        
        # If no clear setting indicator, return the first paragraph
        paragraphs = content.split('\n\n')
        if paragraphs:
            return paragraphs[0]
        
        return ""
    
    def _is_different_setting(self, setting1: str, setting2: str) -> bool:
        """Determine if two setting descriptions represent different locations."""
        # This is a simplified version - a real implementation would use embedding similarity
        # or more sophisticated text comparison
        
        # Extract key location words
        location_terms1 = self._extract_location_terms(setting1)
        location_terms2 = self._extract_location_terms(setting2)
        
        # Check if there's significant overlap
        common_terms = set(location_terms1).intersection(set(location_terms2))
        if len(common_terms) >= 2:
            return False
            
        return True
    
    def _extract_location_terms(self, text: str) -> List[str]:
        """Extract location-related terms from text."""
        # Common location words
        location_words = ["room", "house", "building", "office", "kitchen", "bedroom", "living", 
                         "restaurant", "cafe", "hotel", "street", "road", "park", "garden", "forest",
                         "beach", "mountain", "inside", "outside", "indoor", "outdoor"]
                         
        words = text.lower().split()
        return [word for word in words if word in location_words]
    
    def _summarize_setting(self, setting: str) -> str:
        """Create a short summary of a setting description."""
        if not setting:
            return "Unknown setting"
            
        # If setting is short enough, return as is
        if len(setting) < 100:
            return setting
            
        # Otherwise find the first sentence describing a location
        sentences = setting.split('. ')
        for sentence in sentences:
            location_terms = self._extract_location_terms(sentence)
            if location_terms:
                return sentence
                
        # Fallback to first sentence
        if sentences:
            return sentences[0]
            
        return setting[:100] + "..."
        
    def _setting_similarity(self, setting1: str, setting2: str) -> float:
        """Calculate a simple similarity score between settings based on word overlap."""
        # Convert to lowercase and split into words
        words1 = set(setting1.lower().split())
        words2 = set(setting2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0
            
        return len(intersection) / len(union)
        
    def _save_results(self, results: Dict, output_path: str):
        """Save scene analysis results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved scene analysis results to {output_path}")
        
        # Also create a formatted text file for easy reading
        text_path = output_path.with_suffix('.txt')
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("SCENE DESCRIPTIONS\n")
            f.write("=================\n\n")
            
            for scene_idx, scene in sorted(results["scenes"].items(), key=lambda x: int(x[0])):
                f.write(f"SCENE {int(scene_idx) + 1}\n")
                f.write("=" * (len(f"SCENE {int(scene_idx) + 1}")) + "\n\n")
                
                if "error" in scene:
                    f.write(f"Error: {scene['error']}\n\n")
                else:
                    f.write(scene["description"] + "\n\n")
                    
        logger.info(f"Saved formatted scene descriptions to {text_path}")

    def analyze_scenes(self, scenes: List[Dict], keyframes_metadata: Dict, output_dir: str,
                     entity_data: Optional[Dict] = None, audio_data: Optional[Dict] = None) -> Dict:
        """
        Analyze scenes using keyframe metadata - compatibility wrapper for main.py.
        
        Args:
            scenes: List of scene dictionaries
            keyframes_metadata: Dictionary with keyframe metadata
            output_dir: Directory to save analysis results
            entity_data: Optional entity detection data
            audio_data: Optional audio transcription data
            
        Returns:
            Dictionary with scene analysis results
        """
        logger.info(f"Analyzing {len(scenes)} scenes with Gemini Vision")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir) / "scene_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find keyframes directory from metadata paths
        keyframes_dir = Path(output_dir).parent / "keyframes"
        if not keyframes_dir.exists():
            logger.warning(f"Keyframes directory not found at {keyframes_dir}. Creating directory.")
            keyframes_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using keyframes from {keyframes_dir}")
        
        # Prepare properly formatted keyframes metadata if needed
        formatted_keyframes = keyframes_metadata
        if "scenes" not in formatted_keyframes:
            logger.info("Converting keyframes_metadata to required format with 'scenes' key")
            formatted_keyframes = {"scenes": {}}
            for scene_idx, scene_keyframes in keyframes_metadata.items():
                formatted_keyframes["scenes"][str(scene_idx)] = scene_keyframes
            
        # Call the existing comprehensive method
        try:
            results = self.analyze_all_scenes(
                keyframes_metadata=formatted_keyframes,
                keyframes_dir=str(keyframes_dir),
                entity_data=entity_data,
                audio_data=audio_data,
                output_path=str(output_path)
            )
            return results
        except Exception as e:
            logger.error(f"Error in scene analysis: {str(e)}")
            # Return a valid structure even if analysis fails
            return {
                "scenes": {},
                "error": f"Scene analysis failed: {str(e)}",
                "storyline_analysis": {
                    "storylines_detected": False,
                    "interweaving_narrative": False,
                    "storyline_count": 0
                }
            }

# Helper function
def analyze_video_scenes(keyframes_metadata: Dict, keyframes_dir: str,
                        entity_data: Optional[Dict] = None,
                        audio_data: Optional[Dict] = None,
                        output_path: Optional[str] = None,
                        config: Dict = None) -> Dict:
    """
    Convenience function to analyze all scenes in a video.
    
    Args:
        keyframes_metadata: Dictionary with keyframe metadata
        keyframes_dir: Directory containing keyframe images
        entity_data: Optional entity detection results
        audio_data: Optional audio processing results
        output_path: Optional path to save results
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with scene analysis results
    """
    analyzer = GeminiVisionAnalyzer(config)
    return analyzer.analyze_all_scenes(
        keyframes_metadata, keyframes_dir, entity_data, audio_data, output_path
    ) 