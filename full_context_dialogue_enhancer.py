#!/usr/bin/env python3
"""
Full Context Dialogue Enhancer for Video Analysis Pipeline

This module enhances dialogue speaker labels in the structured analysis by processing 
the entire analysis JSON in a single API call to maintain full context.
"""

# Set environment variables to fix NCCL issues while still using GPU
import os
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer operations
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match device IDs to PCI bus order

import sys
import json
import logging
import time
import random
import copy
import re
from pathlib import Path
import threading
from datetime import datetime
import traceback
import argparse
import shutil

# Add Google's Generative AI import
try:
    import google.generativeai as genai
except ImportError:
    logging.error("google-generativeai package not installed. Install with: pip install google-generativeai")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the system prompt for dialogue enhancement
DIALOGUE_FULL_CONTEXT_SYSTEM_PROMPT = """
You are an expert dialogue analyst specialized in identifying speakers in video transcripts.
Your task is to analyze the provided video scenes with dialogue and enhance the transcripts by replacing "UNKNOWN" speaker labels with the actual character names based on context clues.

Input Format:
I will provide you with a JSON structure containing video scenes, each with dialogue transcripts where speakers are labeled as "UNKNOWN".

Response Format Requirements:
1. You MUST respond with a valid, properly formatted JSON object
2. The JSON structure MUST follow this exact format:
   {
     "scenes": {
       "scene_id": {
         "dialogue": {
           "transcript": "Speaker [00:01:23 - 00:01:25]: Text..."
         }
       },
       // Other scenes...
     }
   }
3. No explanations, markdown formatting, or text outside the JSON structure
4. Use double quotes for all keys and string values
5. Properly escape any special characters in the transcript text
6. Only include scenes that you have enhanced, not all scenes

Instructions:
1. Replace "UNKNOWN" with character names based on context clues from:
   - Scene descriptions
   - Character lists
   - Setting information
   - Dialogue content and conversational flow
2. Keep the original timestamp format intact: [HH:MM:SS - HH:MM:SS]
3. Maintain correct speaker attribution throughout conversations
4. Use consistent character naming across all scenes
5. If a speaker truly cannot be identified, keep it as "UNKNOWN"

Example Input Line:
UNKNOWN [00:01:23 - 00:01:25]: Hello there, Professor!

Example Output Line (enhanced):
Student [00:01:23 - 00:01:25]: Hello there, Professor!

Example Response Structure:
{"scenes":{"5":{"dialogue":{"transcript":"Student [00:01:23 - 00:01:25]: Hello there, Professor!"}}}}

IMPORTANT: Your ENTIRE response must be a single, valid JSON object that can be parsed directly.
"""

def mock_enhance_dialogue(scenes):
    """
    Create a mock enhanced dialogue response for testing.
    
    Args:
        scenes: Dictionary of scenes to enhance
        
    Returns:
        Dictionary with mock enhanced dialogue
    """
    enhanced_scenes = {}
    character_names = ["John", "Sarah", "Professor", "Student", "Doctor", "Officer", "Manager", "Employee", "Customer", "Clerk"]
    
    logger.info(f"Mock enhancing {len(scenes)} scenes")
    
    has_unknown = False
    # First check if there are any UNKNOWN speakers to process
    for scene_id, scene in scenes.items():
        if "dialogue" in scene and "transcript" in scene["dialogue"]:
            if "UNKNOWN" in scene["dialogue"]["transcript"]:
                has_unknown = True
                break
    
    if not has_unknown:
        logger.warning("No UNKNOWN speakers found in any of the scenes to enhance. Check data structure and dialogue formats.")
        # Print a sample scene for debugging
        if scenes:
            sample_id = next(iter(scenes))
            sample = scenes[sample_id]
            logger.debug(f"Sample scene {sample_id}: {json.dumps(sample, indent=2)}")
    
    # Now process each scene
    for scene_id, scene in scenes.items():
        if "dialogue" in scene and "transcript" in scene["dialogue"]:
            original_transcript = scene["dialogue"]["transcript"]
            
            # Skip if no UNKNOWN in transcript
            if "UNKNOWN" not in original_transcript:
                logger.info(f"Scene {scene_id} has no UNKNOWN speakers, keeping original transcript")
                enhanced_scenes[scene_id] = {
                    "dialogue": {
                        "transcript": original_transcript
                    }
                }
                continue
            
            # Create an enhanced transcript by replacing UNKNOWN with character names
            lines = original_transcript.split('\n')
            enhanced_lines = []
            
            characters_in_scene = scene.get("characters", [])
            # Use scene character list if available, otherwise use defaults
            if not characters_in_scene:
                characters_in_scene = character_names[:3]
            
            char_index = 0
            unknown_count = 0
            
            for line in lines:
                if "UNKNOWN" in line:
                    unknown_count += 1
                    # Replace UNKNOWN with a character name
                    char = characters_in_scene[char_index % len(characters_in_scene)]
                    enhanced_line = line.replace("UNKNOWN", char)
                    enhanced_lines.append(enhanced_line)
                    char_index += 1
                else:
                    enhanced_lines.append(line)
            
            enhanced_transcript = '\n'.join(enhanced_lines)
            
            logger.info(f"Mock enhanced scene {scene_id}: replaced {unknown_count} UNKNOWN labels")
            
            # Add to enhanced scenes
            enhanced_scenes[scene_id] = {
                "dialogue": {
                    "transcript": enhanced_transcript
                }
            }
    
    return {"scenes": enhanced_scenes}

def enhance_dialogue_with_full_context(analysis_json_path, api_key=None, model_name=None, output_dir=None, force_regenerate=False, chunk_size=10, demo_mode=False):
    """
    Enhance dialogue speaker labels by processing the entire structured analysis JSON in a single API call.
    
    Args:
        analysis_json_path: Path to the structured analysis JSON
        api_key: Optional gemini API key to use
        model_name: Optional model name to use
        output_dir: Optional output directory to save results
        force_regenerate: Force regeneration even if no UNKNOWN speakers
        chunk_size: Number of scenes to process in each API call (to avoid context length issues)
        demo_mode: Use mock responses instead of making real API calls (for testing)
    
    Returns:
        Path to the enhanced JSON file
    """
    try:
        # Load the structured analysis JSON
        logger.info(f"Enhancing dialogue with full context from {analysis_json_path}")
        
        # Define the output filename
        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            out_filename = os.path.join(output_dir, "structured_analysis_enhanced.json")
        else:
            # Save in the same directory
            output_dir = os.path.dirname(analysis_json_path)
            out_filename = analysis_json_path
            
        # Load the JSON data
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
            
        # Find scenes with UNKNOWN speakers to enhance
        scenes_to_process = {}
        
        # Handle different structured data formats
        if "scenes" in structured_data:
            # Format: {"scenes": {"0": {...}, "1": {...}}}
            scenes_dict = structured_data["scenes"]
            logger.info(f"Found structured data with {len(scenes_dict)} scenes under 'scenes' key")
            
            for scene_id, scene in scenes_dict.items():
                if "dialogue" in scene and "transcript" in scene["dialogue"]:
                    transcript = scene["dialogue"]["transcript"]
                    if "UNKNOWN" in transcript or force_regenerate:
                        scenes_to_process[scene_id] = scene
        else:
            # Format: {"scene_1": {...}, "scene_2": {...}}
            for scene_id, scene in structured_data.items():
                if not scene_id.startswith("scene_"):
                    continue  # Skip non-scene entries
                
                if "dialogue" in scene and "transcript" in scene["dialogue"]:
                    transcript = scene["dialogue"]["transcript"]
                    if "UNKNOWN" in transcript or force_regenerate:
                        scenes_to_process[scene_id] = scene
        
        if not scenes_to_process:
            logger.info("No scenes with UNKNOWN speakers found. Use --force to regenerate all scene dialogues.")
            return analysis_json_path
        
        logger.info(f"Found {len(scenes_to_process)} scenes with UNKNOWN speakers to enhance")
        
        # Set up Gemini client if not in demo mode
        if not demo_mode:
            api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("No API key provided and GEMINI_API_KEY environment variable not set")
                return analysis_json_path
            
            # Use provided model_name or default to 'gemini-1.5-flash' for higher quota
            selected_model = model_name or 'gemini-2.0-flash-lite'
            logger.info(f"Initialized Gemini model {selected_model} for dialogue enhancement")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                selected_model,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json"
                }
            )
        else:
            logger.info("Running in demo mode with mock responses")
        
        # Process scenes in chunks to avoid context length issues
        enhanced_data = copy.deepcopy(structured_data)
        total_scenes_processed = 0
        
        # Convert dictionary to list of items for chunking
        scenes_items = list(scenes_to_process.items())
        
        # Process scenes in chunks
        for chunk_start in range(0, len(scenes_items), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(scenes_items))
            current_chunk_items = scenes_items[chunk_start:chunk_end]
            current_chunk = dict(current_chunk_items)
            
            logger.info(f"Processing chunk of {len(current_chunk)} scenes (scenes {chunk_start}-{chunk_end-1} of {len(scenes_to_process)})")
            
            # Create a subset of structured data with only the current chunk
            chunk_data = {
                "video_info": structured_data.get("video_info", {}),
                "scenes": {}
            }
            
            # Add only the scenes from the current chunk
            for scene_id, scene in current_chunk.items():
                # Determine where to place the scene in the chunk data
                if "scenes" in structured_data:
                    chunk_data["scenes"][scene_id] = scene
                else:
                    chunk_data[scene_id] = scene
            
            # Load the system prompt
            system_prompt = DIALOGUE_FULL_CONTEXT_SYSTEM_PROMPT
            
            # Prepare the dialogue content from the chunk data
            dialogue_content = prepare_full_context_dialogue_input(chunk_data)
            
            # Set up the retry count
            max_retries = 5
            retry_count = 0
            success = False
            response_text = None
            dialogue_labels = None
            
            # In demo mode, use the mock_enhance_dialogue function instead of the API
            if demo_mode:
                try:
                    logger.info(f"Using mock dialogue enhancer for chunk {chunk_start//chunk_size + 1}")
                    dialogue_labels = mock_enhance_dialogue(current_chunk)
                    success = True
                except Exception as e:
                    logger.error(f"Error in mock dialogue enhancement: {e}")
                    continue
            else:
                # Try to get a response with retries for the real API
                start_time = time.time()
                max_repair_attempts = 3
                while retry_count < max_retries and not success:
                    retry_count += 1
                    try:
                        logger.info(f"Requesting dialogue labels for chunk {chunk_start//chunk_size + 1} (attempt {retry_count}/{max_retries})")
                        
                        # Check for timeout - don't get stuck in endless repair attempts
                        if time.time() - start_time > 300:  # 5 minutes timeout
                            logger.error(f"Timeout exceeded for chunk {chunk_start//chunk_size + 1}. Skipping after {retry_count} attempts.")
                            break
                        
                        # Make the API call
                        response = model.generate_content(
                            [
                                {"role": "user", "parts": [{"text": system_prompt}]},
                                {"role": "model", "parts": [{"text": "I understand. I'll follow these instructions to enhance the dialogue with speaker labels based on the full context and return only a valid JSON object."}]},
                                {"role": "user", "parts": [{"text": dialogue_content}]}
                            ],
                        generation_config={
                                "temperature": 0.2,  # Lower temperature for more deterministic output
                                "top_p": 0.95,
                                "top_k": 40,
                                "max_output_tokens": 8192,
                                "response_mime_type": "application/json"  # Encourage JSON output format
                            }
                        )
                        
                        response_text = response.text
                        
                        # Save the raw response for debugging
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        random_suffix = random.randint(1000, 9999)
                        debug_dir = "debug"
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        raw_response_path = os.path.join(debug_dir, f"gemini_raw_response_chunk{chunk_start//chunk_size + 1}_{timestamp}_{random_suffix}_attempt{retry_count}.txt")
                        with open(raw_response_path, 'w', encoding='utf-8') as f:
                            f.write(response_text)
                        logger.info(f"Saved raw Gemini API response to {raw_response_path}")
                        
                        # Try to extract and process JSON
                        success = False
                        
                        # Step 1: Extract JSON from the response text
                        json_str = extract_json_from_text(response_text)
                        
                        if json_str:
                            logger.info("Successfully extracted JSON-like structure from response")
                            
                            # Step 2: Try to repair the JSON if needed
                            repair_attempts = 0
                            while repair_attempts < max_repair_attempts and not success:
                                try:
                                    # First, attempt to parse it directly
                                    try:
                                        dialogue_labels = json.loads(json_str)
                                        logger.info("JSON is valid and parsed successfully")
                                        success = True
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Extracted JSON is not valid, attempting repair: {e}")
                                        
                                        # Try to repair the JSON
                                        repaired_json = repair_json(json_str)
                                        if repaired_json:
                                            dialogue_labels = json.loads(repaired_json)
                                            logger.info("Successfully repaired and parsed JSON")
                                            success = True
                                        else:
                                            logger.warning("JSON repair failed")
                                            # If we can't repair after multiple attempts, try a new API call
                                            if repair_attempts >= max_repair_attempts - 1:
                                                logger.warning(f"Failed to repair JSON after {repair_attempts + 1} attempts, requesting new response")
                                                break
                                            repair_attempts += 1
                                except Exception as e:
                                    logger.error(f"Error processing JSON: {e}")
                                    repair_attempts += 1
                                    if repair_attempts >= max_repair_attempts:
                                        logger.error(f"Failed to process JSON after {repair_attempts} attempts")
                                        break
                            
                            # If we have dialogue_labels, save them for debugging
                            if success and dialogue_labels:
                                json_path = os.path.join(debug_dir, f"gemini_extracted_json_chunk{chunk_start//chunk_size + 1}_{timestamp}_{random_suffix}_attempt{retry_count}.json")
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump(dialogue_labels, f, indent=2, ensure_ascii=False)
                                logger.info(f"Saved extracted JSON to {json_path}")
                                
                                # Validate the dialogue_labels structure
                                if "scenes" not in dialogue_labels:
                                    logger.warning("Extracted JSON is missing 'scenes' key")
                                    dialogue_labels = {"scenes": dialogue_labels}
                                    logger.info("Added 'scenes' wrapper to JSON")
                                
                                # Check if we need to do further processing
                                if not isinstance(dialogue_labels["scenes"], dict):
                                    logger.warning("'scenes' is not a dictionary")
                                    success = False
                                elif not dialogue_labels["scenes"]:
                                    logger.warning("'scenes' dictionary is empty")
                                    success = False
                        else:
                            logger.warning("No JSON structure found in response")
                    except Exception as e:
                        if "429" in str(e):
                            # Improved exponential backoff for rate limits
                            wait_time = min(60, 5 * (2 ** (retry_count - 1)))  # Exponential backoff (5, 10, 20, 40, 60)
                            logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry: {e}")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Error in enhancement for chunk {chunk_start//chunk_size + 1}: {e}")
                            logger.error(traceback.format_exc())
                            if retry_count >= max_retries:
                                logger.error(f"Skipping chunk {chunk_start//chunk_size + 1} after maximum retries")
                                break  # Skip this chunk and continue with the next one
                            # For non-rate-limit errors, wait a shorter time
                            time.sleep(2)
            
            # Process the successful response and update the structured data
            if success and dialogue_labels and "scenes" in dialogue_labels:
                logger.info(f"Successfully extracted dialogue for {len(dialogue_labels['scenes'])} scenes")
                
                # Update the structured data with the enhanced dialogue
                updated_scenes = 0
                for scene_id, scene_data in dialogue_labels["scenes"].items():
                    # Check if scene exists in structured_data or structured_data["scenes"]
                    scene_target = None
                    enhanced_scene_target = None
                    original_transcript = None
                    
                    # Determine where to find/update the scene data
                    if "scenes" in structured_data and scene_id in structured_data["scenes"]:
                        scene_target = structured_data["scenes"][scene_id]
                        enhanced_scene_target = enhanced_data["scenes"][scene_id]
                        if "dialogue" in scene_target and "transcript" in scene_target["dialogue"]:
                            original_transcript = scene_target["dialogue"]["transcript"]
                    elif scene_id in structured_data and "dialogue" in structured_data[scene_id]:
                        scene_target = structured_data[scene_id]
                        enhanced_scene_target = enhanced_data[scene_id]
                        if "dialogue" in scene_target and "transcript" in scene_target["dialogue"]:
                            original_transcript = scene_target["dialogue"]["transcript"]
                    
                    # Skip if we couldn't find the scene or it doesn't have transcript
                    if not scene_target or not original_transcript or not enhanced_scene_target:
                        logger.warning(f"Scene {scene_id} not found in structured data or missing dialogue/transcript")
                        continue
                        
                    # Check if the scene has dialogue data with transcript
                    if "dialogue" in scene_data and "transcript" in scene_data["dialogue"]:
                        enhanced_transcript = scene_data["dialogue"]["transcript"]
                        
                        # Check if the enhanced transcript is different from the original
                        if enhanced_transcript != original_transcript:
                            # Update the transcript in both structured_data and enhanced_data
                            scene_target["dialogue"]["transcript"] = enhanced_transcript
                            enhanced_scene_target["dialogue"]["transcript"] = enhanced_transcript
                            
                            # Also update the audio info if applicable
                            if "audio" in scene_target["dialogue"]:
                                audio_data = scene_target["dialogue"]["audio"]
                                if "transcript" in audio_data:
                                    audio_data["transcript"] = enhanced_transcript
                                    
                            if "audio" in enhanced_scene_target["dialogue"]:
                                enhanced_audio_data = enhanced_scene_target["dialogue"]["audio"]
                                if "transcript" in enhanced_audio_data:
                                    enhanced_audio_data["transcript"] = enhanced_transcript
                            
                            # Update the scene's character list based on enhanced transcript
                            character_names = apply_dialogue_labels(structured_data, scene_id, enhanced_transcript)
                            # Also apply dialogue labels to enhanced_data
                            apply_dialogue_labels(enhanced_data, scene_id, enhanced_transcript)
                            
                            if character_names:
                                logger.info(f"Added characters for scene {scene_id}: {character_names}")
                            updated_scenes += 1
                            total_scenes_processed += 1  # Increment the total scenes counter
                        else:
                            logger.debug(f"No changes in transcript for {scene_id}")
                    else:
                        logger.warning(f"Scene {scene_id} in response is missing dialogue/transcript")
                
                logger.info(f"Updated dialogue for {updated_scenes} scenes in this chunk")
            
            # If we've successfully processed the chunk, save the intermediate results
            if total_scenes_processed > 0 and (chunk_end % (chunk_size * 2) == 0 or chunk_end == len(scenes_items)):
                # Save intermediate results
                intermediate_path = os.path.join(output_dir, f"structured_analysis_enhanced_intermediate_{chunk_end}.json")
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved intermediate results after processing {total_scenes_processed} scenes to {intermediate_path}")
        
        # Save the final enhanced data
        with open(out_filename, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully enhanced dialogue for {total_scenes_processed} scenes and saved to {out_filename}")
        
        # Regenerate the analysis report with enhanced dialogues
        report_path = os.path.join(output_dir, "analysis_report.md")
        regenerate_analysis_report(enhanced_data, report_path)
        
        return out_filename
        
    except Exception as e:
        logger.error(f"Error in dialogue enhancement: {e}")
        logger.error(traceback.format_exc())
        return analysis_json_path
        
def create_prompt_data(structured_data: dict) -> dict:
    """
    Create a simplified version of the structured data for the prompt.
    
    Args:
        structured_data: The full structured analysis JSON
        
    Returns:
        Simplified data with only the necessary information for the prompt
    """
    # Create a deep copy to avoid modifying the original
    simplified_data = {
        "video_info": structured_data.get("video_info", {}),
        "scenes": {}
    }
    
    # Include only necessary scene information
    for scene_key, scene in structured_data.get("scenes", {}).items():
        if isinstance(scene, dict):
            simplified_scene = {}
            
            # Include dialogue information
            if "dialogue" in scene:
                simplified_scene["dialogue"] = scene["dialogue"]
                
            # Include visual context information
            for context_key in ["description", "setting", "summary", "characters", "action"]:
                if context_key in scene:
                    simplified_scene[context_key] = scene[context_key]
                    
            # Include the simplified scene
            simplified_data["scenes"][scene_key] = simplified_scene
    #print(simplified_data)
    
    return simplified_data
    
def create_full_context_prompt(simplified_data: dict) -> str:
    """
    Create a prompt for Gemini API that focuses on dialogue speaker enhancement.
    
    Args:
        simplified_data: Simplified structured data
        
    Returns:
        Prompt string for the API call
    """
    # Count the number of UNKNOWN speakers to be replaced
    unknown_count = 0
    scene_count = 0
    
    for scene_key, scene in simplified_data.get("scenes", {}).items():
        if "dialogue" in scene and "transcript" in scene.get("dialogue", {}):
            transcript = scene["dialogue"].get("transcript", "")
            if "UNKNOWN" in transcript:
                scene_count += 1
                unknown_count += transcript.count("UNKNOWN")
    
    prompt = f"""# Dialogue Speaker Enhancement Task

## Context
You are analyzing a video with {scene_count} scenes containing dialogue where {unknown_count} speaker labels are marked as "UNKNOWN". Your task is to identify and replace these generic "UNKNOWN" speaker labels with appropriate character names based on the context.

## Instructions
1. Analyze all dialogue and scene descriptions to understand the characters present in each scene
2. Replace all "UNKNOWN" speaker labels with specific character names based on the following rules:
    - Extract character identities from actions and references in the scene descriptions
    - Identify names referred to in 1st person, 2nd person, or 3rd person dialogues
    - Label dialogues scene by scene with the appropriate character names
    - Ensure consistent character naming throughout the video
    - Use realistic character names, not placeholder labels like "SPEAKER_1"
    - Pay attention to character roles, relationships, and interactions
3. Be consistent with character naming throughout the video
4. Pay special attention to character roles, relationships, and how they interact

## Important
- Return ONLY a JSON object with the complete transcripts and character names
- DO NOT modify or include any other aspects of the original JSON structure
- Keep the same timestamp format in your response
- Maintain the same JSON structure as shown in the output format example

## Video Analysis Data
```json
{json.dumps(simplified_data, indent=2)}
```

## Output Format
Return ONLY a JSON type object with proper formating that contains the following information:
- the scene id
- the transcript with the changed character names

IMPORTANT: Only return the JSON with this exact structure. Do not include any explanation text or markdown formatting.
"""
    return prompt
    
def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text response that might have additional formatting.
    
    Args:
        text: The raw text response from the API
        
    Returns:
        Extracted JSON string
    """
    logger.debug("Extracting JSON from response...")
    text = text.strip()
    
    # Check if response is formatted as a code block
    if "```json" in text and "```" in text[text.find("```json")+7:]:
        start_marker = "```json"
        end_marker = "```"
        
        start_pos = text.find(start_marker)
        if start_pos != -1:
            # Skip the marker itself
            json_start = start_pos + len(start_marker)
            # Find the closing code block marker
            json_end = text.find(end_marker, json_start)
            
            if json_end != -1:
                # Extract everything between the markers
                extracted = text[json_start:json_end].strip()
                logger.debug("Extracted JSON from code block")
                return extracted
    
    # If text starts with { and ends with }, it's likely JSON already
    elif text.startswith("{") and text.endswith("}"):
        logger.debug("Response is already JSON format")
        return text
        
    # Try to extract JSON part from text using regex
    else:
        # First, look for a complete JSON object with the expected structure
        scenes_json_pattern = r'({[\s\S]*"scenes"[\s\S]*})'
        match = re.search(scenes_json_pattern, text)
        if match:
            logger.debug("Extracted JSON using scenes pattern")
            return match.group(1)
        
        # If no complete JSON found, try to extract the important sections and reconstruct
        logger.debug("Trying to reconstruct JSON from fragments...")
        
        # Extract all scene sections with proper regex capturing groups
        scenes_sections = []
        for scene_match in re.finditer(r'"(\d+)"\s*:\s*{[^{]*"dialogue"\s*:\s*{[^{]*"transcript"\s*:\s*"[^"]*"[^}]*}[^}]*?)}', text):
            scene_id = scene_match.group(1)
            scene_content = scene_match.group(2)
            scenes_sections.append((scene_id, scene_content))
        
        if scenes_sections:
            # Reconstruct a valid JSON structure
            reconstructed = '{"scenes":{'
            for i, (scene_id, content) in enumerate(scenes_sections):
                if i > 0:
                    reconstructed += ','
                reconstructed += f'"{scene_id}":{{{content}}}'
            reconstructed += '}}'
            
            logger.debug(f"Reconstructed JSON with {len(scenes_sections)} scene sections")
            return reconstructed
        
        # Try simpler approach for finding the first JSON-like block
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, text)
        if match:
            logger.debug("Extracted potential JSON using general pattern")
            return match.group(1)
            
        # If regex didn't work, try simpler approach
        json_start = text.find("{")
        json_end = text.rfind("}")
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            logger.debug("Extracted JSON using start/end braces approach")
            return text[json_start:json_end+1]
    
    logger.warning("No JSON structure found in response")
    
    # Last resort: construct a minimal JSON with scene data from response
    scene_blocks = re.findall(r'Scene (\d+).*?transcript.*?([^\n]+)', text, re.DOTALL | re.IGNORECASE)
    if scene_blocks:
        minimal_json = '{"scenes":{'
        for i, (scene_id, transcript) in enumerate(scene_blocks):
            if i > 0:
                minimal_json += ','
            # Clean up transcript
            transcript = transcript.strip().strip('"').strip(':').strip()
            if len(transcript) < 10:
                continue
            minimal_json += f'"{scene_id}": {{"dialogue": {{"transcript": "{transcript}"}}}}'
        minimal_json += '}}'
        
        logger.debug(f"Created minimal JSON with {len(scene_blocks)} scene blocks")
        return minimal_json
            
    return ""

def repair_json(json_str: str) -> str:
    """
    Try to repair malformed JSON that might be returned by the API.
    This is a best-effort function and may not work in all cases.
    
    Args:
        json_str: The potentially malformed JSON string
        
    Returns:
        A repaired JSON string, or None if repair failed
    """
    try:
        # First, log the specific issue to help diagnose
        try:
            json.loads(json_str)
            return json_str  # Already valid
        except json.JSONDecodeError as e:
            logger.warning(f"JSON error details: {str(e)}")
            logger.debug(f"Error at position {e.pos}, context: '{json_str[max(0, e.pos-30):min(len(json_str), e.pos+30)]}'")
        
        # Handle duplicate scene keys by keeping the first occurrence
        cleaned_str = json_str
        
        # Look for typical pattern of duplicated scene keys
        pattern = r'("scenes"\s*:\s*{[^}]*"(\d+)"\s*:\s*{[^}]*})[^}]*"\2"\s*:\s*{'
        match = re.search(pattern, cleaned_str)
        while match:
            # Find the end of the first scene entry
            start_pos = match.start(2)
            section_to_scan = cleaned_str[start_pos:]
            
            # Extract just the first occurrence of the duplicate key section
            logger.debug(f"Found duplicate key: {match.group(2)}")
            
            # Remove the second occurrence
            second_key_pos = cleaned_str.find(f'"{match.group(2)}":', match.end(1))
            if second_key_pos != -1:
                # Find the end of the duplicated section (next scene key or end of scenes)
                next_key_match = re.search(r'"(\d+)"\s*:', cleaned_str[second_key_pos + 10:])
                if next_key_match:
                    end_pos = second_key_pos + 10 + next_key_match.start()
                    cleaned_str = cleaned_str[:second_key_pos] + cleaned_str[end_pos:]
                else:
                    # If no next key, find the end of scenes object
                    end_of_scenes = cleaned_str.find('}}', second_key_pos)
                    if end_of_scenes != -1:
                        cleaned_str = cleaned_str[:second_key_pos] + cleaned_str[end_of_scenes:]
            
            # Look for more duplicates
            match = re.search(pattern, cleaned_str)
        
        # Common JSON errors and their fixes
        
        # 1. Missing quotes around keys
        fixed = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', cleaned_str)
        
        # 2. Single quotes instead of double quotes
        fixed = fixed.replace("'", '"')
        
        # 3. Trailing commas in arrays or objects
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # 4. Missing commas between array elements or object properties
        # This is a common issue and may be causing your error at column 1847
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r'"\s*"', '","', fixed)
        fixed = re.sub(r'"}\s*"', '"},"', fixed)
        
        # 5. Fix specific pattern with missing commas between scene entries
        fixed = re.sub(r'(\}\s*)\}\s*"(\d+)"\s*:', r'\1},"$2":', fixed)
        
        # 6. Fix missing comma between transcript and other fields
        fixed = re.sub(r'(transcript":\s*"[^"]*")\s*"', r'\1,"', fixed)
        
        # 7. Unescaped newlines or tabs in strings
        fixed = fixed.replace('\n', '\\n').replace('\t', '\\t')
        
        # 8. Fix escaped quotes within strings
        fixed = re.sub(r'\\+"', '"', fixed)
        
        # 9. Fix mismatched quotes in transcript strings
        pattern = r'"transcript"\s*:\s*"([^"]*)"([^"]*)"'
        match = re.search(pattern, fixed)
        while match:
            replacement = f'"transcript":"\\"{match.group(1)}\\"{match.group(2)}\\"'
            fixed = fixed[:match.start()] + replacement + fixed[match.end():]
            match = re.search(pattern, fixed)
        
        # 10. Fix truncated JSON with missing closing brackets
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        
        # 11. Remove description fields that tend to cause problems
        fixed = re.sub(r'"description"\s*:\s*"[^"]*(?<!\\)"', '"description":""', fixed)
        
        # Validate if it's parseable now
        try:
            json.loads(fixed)
            logger.info("Successfully repaired JSON")
            return fixed
        except json.JSONDecodeError as e:
            logger.warning(f"Initial repair attempt failed: {e}")
            logger.debug(f"Error at position {e.pos}, context: '{fixed[max(0, e.pos-30):min(len(fixed), e.pos+30)]}'")
            
            # If it's still not valid, try a more aggressive approach
            # Extract just the scenes structure which is what we need
            scenes_match = re.search(r'"scenes"\s*:\s*{(.+?)}(?=\s*}\s*$)', fixed, re.DOTALL)
            if scenes_match:
                scenes_str = scenes_match.group(1)
                try:
                    scene_json = "{\"scenes\":{" + scenes_str + "}}"
                    json.loads(scene_json)
                    logger.info("Successfully extracted and repaired scenes JSON")
                    return scene_json
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to repair scenes JSON: {e}")
            
            # Most aggressive approach: fix specific position-based errors
            # This is useful if the error is consistently at a specific position like char 1846
            if len(fixed) > 1800:  # Only try this for longer JSON strings
                try:
                    # The error mentioned is at char 1846, try inserting a comma there
                    position_to_fix = 1846  # Adjust based on the error message
                    fixed_with_comma = fixed[:position_to_fix] + ',' + fixed[position_to_fix:]
                    json.loads(fixed_with_comma)
                    logger.info("Successfully repaired JSON with position-based fix")
                    return fixed_with_comma
                except:
                    logger.warning("Position-based repair failed")
            
            # Last resort: force the structure with a regex-based approach
            try:
                scenes_dict = {}
                scene_pattern = r'"(\d+)"\s*:\s*{\s*"dialogue"\s*:\s*{\s*"transcript"\s*:\s*"([^"]*)"'
                for match in re.finditer(scene_pattern, fixed):
                    scene_id = match.group(1)
                    transcript = match.group(2)
                    scenes_dict[scene_id] = {"dialogue": {"transcript": transcript}}
                
                if scenes_dict:
                    logger.info(f"Created minimal JSON with {len(scenes_dict)} scenes using regex")
                    return json.dumps({"scenes": scenes_dict})
            except Exception as e:
                logger.warning(f"Regex-based repair failed: {e}")
            
            return None
    except Exception as e:
        logger.error(f"Error in repair_json: {e}")
        # If repair failed, return None
        return None

def extract_scenes_from_malformed_json(json_str: str) -> dict:
    """
    Extract scenes data from potentially malformed JSON by parsing it line by line.
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        Dictionary of scenes data
    """
    scenes = {}
    current_scene = None
    in_transcript = False
    transcript_content = ""
    
    # Process line by line
    lines = json_str.split('\n')
    for line in lines:
        line = line.strip()
        
        # Detect scene start
        scene_match = re.search(r'"([0-9]+)"\s*:\s*{', line)
        if scene_match:
            current_scene = scene_match.group(1)
            scenes[current_scene] = {"dialogue": {}}
            
        # Detect transcript start
        if current_scene and '"transcript"' in line:
            in_transcript = True
            transcript_start = line.find(':', line.find('"transcript"')) + 1
            # Extract content after the colon
            if transcript_start > 0:
                content = line[transcript_start:].strip()
                # Remove starting quote if present
                if content.startswith('"'):
                    content = content[1:]
                transcript_content = content
                
        # Collect transcript content
        elif in_transcript and current_scene:
            # Check for end of transcript
            if line.endswith('",') or line.endswith('"'):
                # Remove ending quote
                if line.endswith('",'):
                    line = line[:-2]
                elif line.endswith('"'):
                    line = line[:-1]
                transcript_content += line
                
                # Save transcript and reset
                scenes[current_scene]["dialogue"]["transcript"] = transcript_content
                in_transcript = False
                transcript_content = ""
            else:
                # Continue collecting transcript content
                transcript_content += line
                
    return scenes
    
def get_dialogue_labels_with_retries(model, prompt: str) -> dict:
    """
    Get speaker label mappings from Gemini API with robust retry logic for rate limits.
    
    Args:
        model: Gemini API model instance
        prompt: The prompt to send to the API
        
    Returns:
        Dictionary mapping scene IDs to speaker label mappings
    """
    max_retries = 5
    base_delay = 2  # Start with a longer base delay
    
    # Create debug directory if it doesn't exist
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    for retry in range(max_retries):
        try:
            logger.info(f"Requesting dialogue labels (attempt {retry+1}/{max_retries})")
            
            # Set a timeout to prevent hanging
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
            
            # Wait for the API call to complete with a longer timeout
            api_success = response_received.wait(timeout=600)  # 10 minute timeout
            
            if not api_success:
                logger.warning(f"API call timed out on attempt {retry+1}/{max_retries}")
                delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            
            if api_error:
                error_msg = api_error
                
                # Handle specific error types
                if "429" in error_msg:
                    logger.warning(f"Rate limit hit: {error_msg}")
                    
                    # Extract retry delay from the error message if available
                    retry_match = re.search(r"retry_delay.*?seconds: (\d+)", error_msg)
                    if retry_match:
                        retry_delay = int(retry_match.group(1))
                        logger.info(f"Using suggested retry delay of {retry_delay} seconds")
                        time.sleep(retry_delay + random.uniform(1, 5))  # Add jitter
                    else:
                        delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    continue
                elif "404" in error_msg and "deprecated" in error_msg.lower():
                    logger.error(f"Model deprecated: {error_msg}")
                elif "403" in error_msg:
                    logger.error(f"Authentication error: {error_msg}")
                else:
                    logger.error(f"Error in Gemini API call: {error_msg}")
                
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All attempts failed after {max_retries} retries")
                    return {}
                continue
            
            if not response:
                logger.warning(f"Empty response on attempt {retry+1}/{max_retries}")
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                return {}
            
            # Extract response text
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                response_text = response.candidates[0].content.parts[0].text.strip()
            else:
                logger.warning(f"Empty response from Gemini API")
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                return {}
            
            # Save the raw response to a file for debugging
            timestamp = time.strftime("%Y%m%d-%H%M%S") + f"_{random.randint(1000, 9999)}"
            raw_response_file = os.path.join(debug_dir, f"gemini_raw_response_full_context_{timestamp}_attempt{retry+1}.txt")
            with open(raw_response_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.info(f"Saved raw Gemini API response to {raw_response_file}")
            
            # Extract JSON from response text
            json_text = extract_json_from_text(response_text)
            if not json_text:
                logger.warning(f"Could not extract JSON from response")
                timestamp = time.strftime("%Y%m%d-%H%M%S") + f"_{random.randint(1000, 9999)}"
                extracted_json_file = os.path.join(debug_dir, f"gemini_extracted_json_full_context_{timestamp}_attempt{retry+1}.txt")
                with open(extracted_json_file, 'w', encoding='utf-8') as f:
                    f.write("JSON EXTRACTION FAILED")
                
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                return {}
                
            # Try to repair the JSON if needed
            repaired_json = repair_json(json_text)
                
            # Save the extracted and repaired JSON to a file for debugging
            timestamp = time.strftime("%Y%m%d-%H%M%S") + f"_{random.randint(1000, 9999)}"
            extracted_json_file = os.path.join(debug_dir, f"gemini_extracted_json_full_context_{timestamp}_attempt{retry+1}.json")
            with open(extracted_json_file, 'w', encoding='utf-8') as f:
                f.write(repaired_json)
            logger.info(f"Saved extracted JSON to {extracted_json_file}")
            
            # Parse the JSON
            try:
                labels = json.loads(repaired_json)
                return labels
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {str(e)}")
                
                # For the last attempt, try to handle partial results 
                if retry == max_retries - 1:
                    try:
                        # Try the structural extraction approach as a last resort
                        scenes_data = extract_scenes_from_malformed_json(repaired_json)
                        if scenes_data:
                            logger.info("Extracted partial results using structural approach")
                            return {"scenes": scenes_data}
                    except Exception as ex:
                        logger.error(f"Failed to extract partial results: {str(ex)}")
                
                # Retry for earlier attempts
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                return {}
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error on attempt {retry+1}: {error_msg}")
            
            if retry < max_retries - 1:
                delay = base_delay * (2 ** retry) + random.uniform(0, 1.0)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All attempts failed after {max_retries} retries")
                return {}
                
    return {}
    
def apply_dialogue_labels(structured_data: dict, scene_id: str, enhanced_transcript: str) -> set:
    """
    Apply dialogue labels to a scene and extract character names.
    
    Args:
        structured_data: The structured data dictionary (either old format with scenes key or new format)
        scene_id: The scene ID to update
        enhanced_transcript: The enhanced transcript with speaker labels
        
    Returns:
        Set of character names extracted from the transcript
    """
    character_names = set()
    
    # Extract character names from the transcript
    for line in enhanced_transcript.split('\n'):
        if line and "[" in line and not line.startswith("UNKNOWN"):
            match = re.match(r'^([^\[]+)', line)
            if match:
                char_name = match.group(1).strip()
                if char_name:
                    character_names.add(char_name)
    
    # Check if the scene exists directly in structured_data
    if scene_id in structured_data:
        # Update the character list
        scene_data = structured_data[scene_id]
        if character_names:
            if "characters" in scene_data:
                # Combine existing characters with newly detected ones
                characters_list = list(set(scene_data.get("characters", [])) | character_names)
                scene_data["characters"] = characters_list
            else:
                # Add new character list
                scene_data["characters"] = list(character_names)
    
    # Check if the scene exists in structured_data["scenes"]
    elif "scenes" in structured_data and scene_id in structured_data["scenes"]:
        # Update the character list
        scene_data = structured_data["scenes"][scene_id]
        if character_names:
            if "characters" in scene_data:
                # Combine existing characters with newly detected ones
                characters_list = list(set(scene_data.get("characters", [])) | character_names)
                scene_data["characters"] = characters_list
    else:
                # Add new character list
                scene_data["characters"] = list(character_names)
    
    # Also check audio section
    if "audio" in structured_data and "scenes" in structured_data["audio"] and scene_id in structured_data["audio"]["scenes"]:
        audio_scene = structured_data["audio"]["scenes"][scene_id]
        dialogue_segments = []
        
        # Parse the transcript to extract individual dialogue segments
        for line in enhanced_transcript.split('\n'):
            if not line:
                continue
            
            # Match pattern: "Speaker [00:00:00 - 00:00:00]: Text"
            match = re.match(r'([^\[]+)\s*\[([^\]]+)\]:\s*(.*)', line)
            if match:
                speaker = match.group(1).strip()
                timestamp = match.group(2).strip()
                text = match.group(3).strip()
                
                # Skip if the speaker is still UNKNOWN
                if speaker == "UNKNOWN":
                    continue
                    
                # Parse timestamps
                times = timestamp.split(" - ")
                if len(times) == 2:
                    # Convert to seconds
                    def time_to_seconds(time_str):
                        parts = time_str.split(":")
                        if len(parts) == 3:
                            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                        elif len(parts) == 2:
                            return int(parts[0])*60 + float(parts[1])
                        return float(time_str)
                    
                    start_time = time_to_seconds(times[0])
                    end_time = time_to_seconds(times[1])
                    
                    # Find matching segment in audio dialogues or create new one
                    found_segment = False
                    for segment in audio_scene.get("dialogue", []):
                        seg_start = segment.get("start_time", 0)
                        seg_end = segment.get("end_time", 0)
                        
                        # If timestamps approximately match
                        if (abs(seg_start - start_time) < 1.0 and
                            abs(seg_end - end_time) < 1.0):
                            # Update the segment
                            segment["original_speaker"] = segment.get("speaker", "UNKNOWN")
                            segment["speaker"] = speaker
                            segment["text"] = text
                            found_segment = True
                            break
                    
                    # If no matching segment found, create a new one
                    if not found_segment:
                        dialogue_segments.append({
                            "speaker": speaker,
                            "text": text,
                            "start_time": start_time,
                            "end_time": end_time
                        })
        
        # Append any new segments that didn't match existing ones
        if dialogue_segments:
            if "dialogue" not in audio_scene:
                audio_scene["dialogue"] = []
            audio_scene["dialogue"].extend(dialogue_segments)
        
        # Mark audio scene as enhanced
        audio_scene["dialogue_enhanced"] = True
    
    return character_names

def integrate_with_pipeline():
    """
    Modify VideoAnalysisPipeline._enhance_dialogue_with_gemini to use the full context approach.
    This function should be called from pipeline_enhancer.py.
    """
    from main import VideoAnalysisPipeline
    import types
    
    # Define the new _enhance_dialogue_with_gemini method
    def new_enhance_dialogue_with_gemini(self, structured_data_path):
        """
        Enhanced method that processes the entire structured analysis in a single API call.
        
        Args:
            structured_data_path: Path to the structured analysis JSON file
        """
        logger.info("Using enhanced dialogue processing with full context...")
        enhance_dialogue_with_full_context(structured_data_path)
    
    # Replace the method in the VideoAnalysisPipeline class
    VideoAnalysisPipeline._enhance_dialogue_with_gemini = types.MethodType(
        new_enhance_dialogue_with_gemini, 
        VideoAnalysisPipeline
    )
    
    logger.info("Successfully integrated full context dialogue enhancer with pipeline")

# Register this module as a direct replacement for _enhance_dialogue_with_gemini in the pipeline
# This makes the enhancement available even if the pipeline_enhancer fails
try:
    import inspect
    import importlib.util
    
    # Try to find main module
    if 'main' in sys.modules:
        main_module = sys.modules['main']
    else:
        # Try to import it
        try:
            spec = importlib.util.spec_from_file_location(
                "main", 
                str(Path(__file__).parent / "main.py")
            )
            main_module = importlib.util.module_from_spec(spec)
            sys.modules["main"] = main_module
            spec.loader.exec_module(main_module)
        except Exception as e:
            logger.warning(f"Could not import main module: {e}")
            main_module = None
            
    # If main module is found, try to patch the VideoPipeline class
    if main_module:
        # Find the pipeline class
        VideoPipeline = getattr(main_module, "VideoPipeline", None)
        if not VideoPipeline:
            # Try alternate name
            VideoPipeline = getattr(main_module, "VideoAnalysisPipeline", None)
        
        if VideoPipeline and hasattr(VideoPipeline, '_enhance_dialogue_with_gemini'):
            # Store the original method for fallback
            original_method = VideoPipeline._enhance_dialogue_with_gemini
            
            # Create a replacement method that uses our full context approach
            def enhanced_dialogue_method(self, structured_data_path):
                logger.info("Using full context dialogue enhancement (direct integration)")
                try:
                    return enhance_dialogue_with_full_context(structured_data_path)
                except Exception as e:
                    logger.error(f"Full context enhancement failed: {e}")
                    logger.info("Falling back to original method")
                    return original_method(self, structured_data_path)
            
            # Replace the method
            VideoPipeline._enhance_dialogue_with_gemini = enhanced_dialogue_method
            logger.info("Successfully integrated full context dialogue enhancement with pipeline (direct method)")
    
except Exception as e:
    logger.warning(f"Failed to directly integrate with pipeline: {e}")
    logger.info("The dialogue enhancer will still be available as a standalone tool")

def regenerate_analysis_report(structured_data, output_path=None):
    """
    Regenerate the analysis report to reflect enhanced dialogues.
    
    Args:
        structured_data: Either a file path to the structured data JSON or the data dictionary
        output_path: Path to save the report (optional, if None uses the same directory as input)
        
    Returns:
        Path to the generated report
    """
    try:
        # Handle either file path or data dictionary
        if isinstance(structured_data, str):
            # It's a file path
            with open(structured_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # If output_path is not specified, use the same directory
            if output_path is None:
                output_dir = os.path.dirname(structured_data)
                output_path = os.path.join(output_dir, "analysis_report.md")
        else:
            # It's already a data dictionary
            data = structured_data
            
            # If output_path is not specified, raise an error
            if output_path is None:
                raise ValueError("Output path must be specified when providing data dictionary")
        
        logger.info(f"Regenerating analysis report to {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write report header
            f.write("# Video Analysis Report\n\n")
            
            # Write video information
            f.write("## Video Information\n\n")
            video_info = data.get("video_info", {})
            f.write(f"- **File:** {video_info.get('filename', 'Unknown')}\n")
            f.write(f"- **Duration:** {format_duration(video_info.get('duration', 0))}\n")
            f.write(f"- **Number of Scenes:** {len(data.get('scenes', {}))}\n")
            f.write(f"- **Analysis Date:** {video_info.get('analysis_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n\n")
            
            # Write scene breakdown
            f.write("## Scene Breakdown\n\n")
            
            # Sort scenes by their numerical index, converting to integer where possible
            scene_indices = []
            for idx in data.get("scenes", {}).keys():
                try:
                    scene_indices.append(int(idx))
                except ValueError:
                    # If conversion fails, use the original string
                    scene_indices.append(idx)
            
            # Sort the indices, this will sort integers numerically and strings lexicographically
            scene_indices.sort()
            
            for idx in scene_indices:
                # Convert back to string for dictionary lookup
                scene_idx = str(idx)
                scene = data["scenes"][scene_idx]
                
                # Get scene time range
                start_time = format_timestamp(scene.get("start_time", 0))
                end_time = format_timestamp(scene.get("end_time", 0))
                duration = format_duration(scene.get("end_time", 0) - scene.get("start_time", 0))
                
                # Write scene header
                f.write(f"### Scene {scene_idx}: {start_time} - {end_time} ({duration})\n\n")
                
                # Write scene description
                if scene.get("description"):
                    f.write(f"**Description:** {scene['description']}\n\n")
                
                # Write scene setting
                if scene.get("setting"):
                    f.write(f"**Setting:** {scene['setting']}\n\n")
                
                # Write emotional tone
                if scene.get("emotional_tone"):
                    f.write(f"**Emotional Tone:** {scene['emotional_tone']}\n\n")
                
                # Write characters
                if scene.get("characters"):
                    f.write("**Characters:** ")
                    f.write(", ".join(scene["characters"]))
                    f.write("\n\n")
                
                # Write detected objects
                if scene.get("detected_objects"):
                    f.write("**Detected Objects:** ")
                    
                    # If objects is a list
                    if isinstance(scene["detected_objects"], list):
                        f.write(", ".join(scene["detected_objects"]))
                    # If objects is a dict with counts
                    elif isinstance(scene["detected_objects"], dict):
                        obj_strs = []
                        for obj, count in scene["detected_objects"].items():
                            obj_strs.append(f"{obj} ({count})")
                        f.write(", ".join(obj_strs))
                    
                    f.write("\n\n")
                
                # Write face count
                if "face_count" in scene:
                    f.write(f"**Face Count:** {scene['face_count']}\n\n")
                
                # Write text in scene
                if scene.get("text_in_scene"):
                    f.write("**Text in Scene:**\n\n```\n")
                    if isinstance(scene["text_in_scene"], list):
                        for text in scene["text_in_scene"]:
                            f.write(f"{text}\n")
                    else:
                        f.write(f"{scene['text_in_scene']}\n")
                    f.write("```\n\n")
                
                # Write dialogue
                if "dialogue" in scene and "transcript" in scene["dialogue"] and scene["dialogue"]["transcript"]:
                    f.write("**Dialogue:**\n\n```\n")
                    
                    # Get and clean transcript
                    transcript = scene["dialogue"]["transcript"]
                    
                    # Check if transcript still has UNKNOWN speakers (only if there are speakers at all)
                    if "[" in transcript:  # Only check if the transcript has speaker markers
                        unknown_count = transcript.count("UNKNOWN [")
                        total_speaker_lines = len([line for line in transcript.split('\n') if "[" in line])
                        
                        unknown_ratio = unknown_count / total_speaker_lines if total_speaker_lines > 0 else 0
                        if unknown_ratio > 0.8:  # If more than 80% of speakers are UNKNOWN
                            # If most lines are still UNKNOWN, mark it
                            f.write("Note: Speaker identification failed for this scene\n\n")
                    
                    f.write(f"{transcript}\n")
                    f.write("```\n\n")
            
            # Add a note about dialogue enhancement
            f.write("\n\n---\n\n")
            f.write("*Report generated with dialogue enhancement by Advanced Video Analysis Tool*\n")
        
        logger.info(f"Successfully regenerated analysis report at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error regenerating analysis report: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def format_timestamp(seconds, end_time=None):
    """Format seconds into a human-readable timestamp."""
    if end_time is not None:
        return f"{format_time(seconds)} - {format_time(end_time)}"
    return format_time(seconds)

def format_time(seconds):
    """Format seconds into HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def format_duration(seconds):
    """Format seconds into a human-readable duration string (HH:MM:SS or MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def prepare_full_context_dialogue_input(structured_data: dict) -> str:
    """
    Prepare the input for the dialogue enhancement API.
    This creates a simplified version of the structured data that focuses on dialogue.
    
    Args:
        structured_data: The structured analysis data
        
    Returns:
        A string containing the formatted input for the API
    """
    # Create a simplified version of the data with only relevant information
    dialogue_data = {
        "video_info": structured_data.get("video_info", {}),
        "scenes": {}
    }
    
    # Add scenes that have dialogue
    for scene_idx, scene in structured_data.get("scenes", {}).items():
        if "dialogue" in scene and "transcript" in scene["dialogue"] and scene["dialogue"].get("transcript"):
            # Create a simplified scene with only relevant information
            dialogue_data["scenes"][scene_idx] = {
                "start_time": scene.get("start_time", 0),
                "end_time": scene.get("end_time", 0),
                "description": scene.get("description", ""),
                "setting": scene.get("setting", ""),
                "characters": scene.get("characters", []),
                "dialogue": {
                    "transcript": scene["dialogue"]["transcript"]
                }
            }
    
    # Convert to pretty JSON string
    json_str = json.dumps(dialogue_data, indent=2)
    
    # Create the complete prompt with the JSON data
    complete_prompt = f"""
I need you to analyze the following video scenes and enhance the dialogue by replacing "UNKNOWN" speaker labels with character names based on contextual clues.

TASK SUMMARY:
- Replace all "UNKNOWN" speakers with character names
- Keep timestamp formats intact
- Maintain exact dialogue content
- Use character names consistently

DATA STRUCTURE:
The video contains {len(dialogue_data["scenes"])} scenes with dialogue. Each scene includes description, setting, character list, and dialogue transcript.

INPUT DATA:
```json
{json_str}
```

OUTPUT REQUIREMENTS:
1. You MUST return ONLY a valid JSON object with the following structure:
{{{{
  "scenes": {{{{
    "scene_id": {{{{
      "dialogue": {{{{
        "transcript": "CHARACTER_NAME [timestamp]: Dialogue text..."
      }}}}
    }}}},
    // other scenes...
  }}}}
}}}}

2. Include ONLY scenes where you've replaced "UNKNOWN" with character names
3. Maintain exact timestamp formats and dialogue text
4. Use ONLY double quotes for JSON keys and string values 
5. Properly escape all special characters in transcript text
6. Output NOTHING but the JSON object - no explanations, no markdown

EXAMPLE OUTPUT FORMAT:
{{"scenes":{{"5":{{"dialogue":{{"transcript":"Student [00:01:23 - 00:01:25]: Hello there, Professor!"}}}}}}}}
"""
    
    return complete_prompt

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Enhance dialogue with speaker identification')
    parser.add_argument('--analysis', '-a', required=True, help='Path to the structured analysis JSON file')
    parser.add_argument('--output', '-o', help='Output directory for enhanced results')
    parser.add_argument('--api-key', '-k', help='Gemini API key (optional, will use environment variable if not provided)')
    parser.add_argument('--model', '-m', default='gemini-2.0-flash-lite', help='Gemini model to use (default: gemini-1.5-pro)')
    parser.add_argument('--force', '-f', action='store_true', help='Force regeneration even if no UNKNOWN speakers')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')
    parser.add_argument('--chunk-size', '-c', type=int, default=10, help='Number of scenes to process in each API call (default: 10)')
    parser.add_argument('--extract-method', '-e', choices=['auto', 'strict', 'regex'], default='auto', 
                        help='Method to extract JSON from response (auto: try all methods, strict: only use JSON code blocks, regex: use regex patterns)')
    parser.add_argument('--demo', '-D', action='store_true', help='Run in demo mode with mock responses')
    
    args = parser.parse_args()
    
    # Set up debugging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Main script execution
    try:
        # Check if CUDA is available
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0) if cuda_available else 'No'}")
        except ImportError:
            pass
        
        # Enhance dialogue with full context
        enhanced_file = enhance_dialogue_with_full_context(
            args.analysis,
            api_key=args.api_key,
            model_name=args.model,
            output_dir=args.output,
            force_regenerate=args.force,
            chunk_size=args.chunk_size,
            demo_mode=args.demo
        )
        
        if enhanced_file:
            logger.info(f"Successfully enhanced dialogues and saved to {enhanced_file}")
            sys.exit(0)
        else:
            logger.error("Failed to enhance dialogues")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)