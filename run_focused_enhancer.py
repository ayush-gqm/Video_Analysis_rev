#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool for enhancing dialogue of specific scene(s) in a structured data file with
focused context. This allows for improved dialogue enhancement by providing
extra context about a scene.
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found.")
    print("Please install it with: pip install google-generativeai")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("dialogue_enhancer")

def enhance_dialogue_with_focused_input(
    structured_data_path: str,
    scene_numbers: List[int],
    context: Optional[str] = None,
    character_list: Optional[str] = None,
    api_key: Optional[str] = None,
    save_backup: bool = True
) -> bool:
    """
    Enhance dialogue for specific scenes with additional context.
    
    Args:
        structured_data_path: Path to the JSON file with structured data
        scene_numbers: List of scene numbers to enhance
        context: Additional context about the scene (optional)
        character_list: Comma-separated list of characters in the scene (optional)
        api_key: Gemini API key (optional, will use env var if not provided)
        save_backup: Whether to save a backup of the original file
        
    Returns:
        True if enhancement was successful, False otherwise
    """
    logger.info(f"Enhancing dialogue for scenes {scene_numbers} with focused input")
    
    try:
        # Load the structured data
        with open(structured_data_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        # Create a backup of the original file
        if save_backup:
            backup_path = structured_data_path + f".{int(time.time())}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created backup at {backup_path}")
        
        # Find the scenes to enhance
        scenes_to_enhance = {}
        scene_key_to_idx = {}  # Mapping from scene key to numeric index
        
        for scene_key, scene in structured_data.get("scenes", {}).items():
            scene_num = scene.get("scene_number")
            if scene_num in scene_numbers:
                scenes_to_enhance[scene_key] = scene
                scene_key_to_idx[scene_key] = scene_num
                
                # Check if there's dialogue to enhance
                if not scene.get("dialogue") or not scene.get("dialogue", {}).get("transcript"):
                    logger.warning(f"Scene {scene_num} has no dialogue transcript to enhance")
        
        if not scenes_to_enhance:
            logger.error(f"No scenes found with numbers {scene_numbers}")
            return False
            
        logger.info(f"Found {len(scenes_to_enhance)} scenes to enhance")
        
        # Configure Gemini
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("No API key provided. Set GEMINI_API_KEY environment variable or pass --api-key")
            return False
            
        genai.configure(api_key=api_key)
        
        # Select model
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
        )
        
        # Prepare the prompt for each scene
        for scene_key, scene in scenes_to_enhance.items():
            scene_idx = scene_key_to_idx[scene_key]
            logger.info(f"Enhancing dialogue for scene {scene_idx}")
            
            # Extract dialogue transcript
            dialogue = scene.get("dialogue", {})
            transcript = dialogue.get("transcript", "")
            
            if not transcript or transcript == "No dialogue in this scene":
                logger.warning(f"Scene {scene_idx} has no dialogue to enhance. Skipping.")
                continue
                
            # Create the prompt
            prompt = create_focused_prompt(
                scene=scene,
                context=context,
                character_list=character_list
            )
            
            # Call the model with retries
            success = False
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Calling Gemini API (attempt {attempt+1}/{max_retries})")
                    response = model.generate_content(prompt)
                    
                    try:
                        result = json.loads(response.text)
                        enhanced_transcript = result.get("enhanced_transcript", "")
                        
                        if enhanced_transcript:
                            # Update the transcript in the structured data
                            structured_data["scenes"][scene_key]["dialogue"]["transcript"] = enhanced_transcript
                            logger.info(f"Successfully enhanced dialogue for scene {scene_idx}")
                            success = True
                            break
                        else:
                            logger.warning(f"API response didn't contain enhanced_transcript (attempt {attempt+1})")
                    except json.JSONDecodeError:
                        logger.warning(f"API response was not valid JSON (attempt {attempt+1})")
                        logger.debug(f"API raw response: {response.text}")
                        
                except Exception as e:
                    logger.warning(f"API call failed (attempt {attempt+1}): {str(e)}")
        
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            if not success:
                logger.error(f"Failed to enhance dialogue for scene {scene_idx} after {max_retries} attempts")
                
        # Save the enhanced data
        with open(structured_data_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved enhanced dialogue to {structured_data_path}")
        
        # Regenerate the analysis report
        regenerate_analysis_report(structured_data_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing dialogue: {str(e)}")
        return False

def regenerate_analysis_report(structured_data_path: str) -> bool:
    """
    Regenerate the analysis report to reflect the enhanced dialogues.
    
    Args:
        structured_data_path: Path to the structured analysis JSON file
        
    Returns:
        True if report regeneration was successful, False otherwise
    """
    try:
        # Determine the output directory from the structured data path
        output_dir = Path(structured_data_path).parent
        report_path = output_dir / "analysis_report.md"
        
        logger.info(f"Regenerating analysis report at {report_path}")
        
        # Load the enhanced structured data
        with open(structured_data_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        # Generate the report
        with open(report_path, 'w', encoding='utf-8') as f:
            # Write report header
            f.write("# Video Analysis Report\n\n")
            
            # Video info
            f.write("## Video Information\n\n")
            video_info = structured_data.get("video_info", {})
            f.write(f"- **File:** {video_info.get('file_name', 'Unknown')}\n")
            f.write(f"- **Duration:** {video_info.get('duration', 0):.2f} seconds ({format_timestamp(video_info.get('duration', 0))})\n")
            f.write(f"- **Number of Scenes:** {video_info.get('num_scenes', 0)}\n")
            f.write(f"- **Analysis Date:** {video_info.get('analysis_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
            
            # Add frame rate and resolution if available
            if video_info.get("fps", 0) > 0:
                f.write(f"- **Frame Rate:** {video_info.get('fps', 0):.2f} FPS\n")
            
            if video_info.get("resolution", {}).get("width", 0) > 0:
                res = video_info.get("resolution", {})
                f.write(f"- **Resolution:** {res.get('width', 0)}x{res.get('height', 0)}\n")
            
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
                timestamp = scene.get("timestamp", format_timestamp(start_time, end_time))
                time_str = f"[{timestamp}]"
                f.write(f"- **Time Range:** {time_str}\n")
                f.write(f"- **Duration:** {duration:.2f}s\n")
                
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
                    
                # Add dialogue with character names - This is the critical part we need to fix
                if "dialogue" in scene and "transcript" in scene["dialogue"]:
                    transcript = scene["dialogue"].get("transcript", "")
                    if transcript and transcript != "No dialogue in this scene" and transcript != "No dialogue information available":
                        f.write("**Dialogue:**\n\n")
                        f.write("```\n")
                        f.write(transcript)
                        f.write("\n```\n\n")
                
                # Add divider between scenes
                f.write("---\n\n")
        
        logger.info(f"Successfully regenerated analysis report with enhanced dialogues at {report_path}")
            return True
        
    except Exception as e:
        logger.error(f"Error regenerating analysis report: {str(e)}")
        return False

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

def create_focused_prompt(scene, context=None, character_list=None):
    """
    Create a prompt for enhancing dialogue with focused input.
    
    Args:
        scene: The scene data
        context: Additional context about the scene
        character_list: Comma-separated list of characters
        
    Returns:
        The prompt text
    """
    transcript = scene.get("dialogue", {}).get("transcript", "")
    scene_description = scene.get("description", "")
    
    # If character list is not provided, try to get it from the scene
    if not character_list and scene.get("characters"):
        character_list = ", ".join(scene.get("characters", []))
    
    # Create a base prompt
    prompt = """
You are a dialogue enhancer for a video analysis system. Your task is to take a dialogue transcript from a video scene and enhance it by adding proper speaker labels (names) to each line of dialogue. 

The transcript is currently not properly attributed to specific characters. Each line might start with "UNKNOWN: " or have no speaker label at all.

I need you to:
1. Analyze the dialogue and the context provided.
2. Determine which character is likely speaking each line.
3. Replace "UNKNOWN" with the specific character's name.
4. Format the dialogue consistently with each speaker clearly labeled.
5. Return the enhanced transcript maintaining the exact dialogue content, only changing the speaker labels.

Output a JSON object with a single key "enhanced_transcript" containing the improved dialogue text.

"""
    
    # Add the context if provided
    if context:
        prompt += f"\nAdditional context about this scene: {context}\n"
    
    # Add the scene description
    prompt += f"\nScene description: {scene_description}\n"
    
    # Add the character list if provided
    if character_list:
        prompt += f"\nCharacters in this scene: {character_list}\n"
    
    # Add the original transcript
    prompt += f"\nOriginal dialogue transcript:\n{transcript}\n\n"
    
    # Add output instructions
    prompt += """
Analyze this dialogue and respond with properly labeled dialogue. Output ONLY a JSON object like this:
{"enhanced_transcript": "PERSON1: Hello there.\nPERSON2: Hi, how are you?"}

Replace PERSON1 and PERSON2 with the appropriate character names based on the context.
If you're unsure about a specific speaker, use a name like "Man", "Woman", "Child", etc., based on context clues.
Maintain the exact wording, pauses, and formatting of the original dialogue - only add or replace speaker labels.
"""

    return prompt

def main():
    """Main function for CLI interface."""
    parser = argparse.ArgumentParser(description="Enhance dialogue for specific scenes with additional context")
    parser.add_argument("file_path", help="Path to the structured data JSON file")
    parser.add_argument("--scenes", "-s", type=int, nargs="+", required=True, help="Scene number(s) to enhance")
    parser.add_argument("--context", "-c", help="Additional context about the scene")
    parser.add_argument("--characters", "-ch", help="Comma-separated list of characters in the scene")
    parser.add_argument("--api-key", "-k", help="Gemini API key (will use env var if not provided)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    
    args = parser.parse_args()
    
    result = enhance_dialogue_with_focused_input(
        args.file_path,
        args.scenes,
        args.context,
        args.characters,
        args.api_key,
        not args.no_backup
    )
    
    if result:
        print("Dialogue enhancement completed successfully")
        sys.exit(0)
    else:
        print("Dialogue enhancement failed")
        sys.exit(1)

if __name__ == "__main__":
    main()