#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that enhances the VideoAnalysisPipeline by providing improved
dialogue enhancement capabilities.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("pipeline_enhancer")

def enhance_dialogue_processor(scene_processor_result):
    """
    Enhance dialogue with speaker labels based on scene context.
    
    Args:
        scene_processor_result: The result from the scene processor
        
    Returns:
        Updated result with enhanced dialogue
    """
    try:
        import os
        import traceback
        import logging
        logger = logging.getLogger(__name__)
        
        # Get the output directory
        output_dir = scene_processor_result.get("output_dir")
        
        if not output_dir:
            logger.error("No output directory found in scene processor result")
            return scene_processor_result
            
        # Check for structured analysis file
        structured_file = os.path.join(output_dir, "structured_analysis.json")
        
        if not os.path.exists(structured_file):
            logger.error(f"Structured analysis file not found: {structured_file}")
            return scene_processor_result
            
        # Import dialogue enhancer
        try:
            from full_context_dialogue_enhancer import enhance_dialogue_with_full_context
            logger.info("Imported dialogue enhancer")
        except ImportError:
            logger.error("Failed to import dialogue enhancer - ensure full_context_dialogue_enhancer.py is in the path")
            return scene_processor_result
            
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("DIALOGUE_API_KEY")
        
        if not api_key:
            logger.warning("No API key found for dialogue enhancement - using default if available")
            
        # Call the dialogue enhancer
        logger.info(f"Enhancing dialogues with full context from {structured_file}")
        enhanced_file = enhance_dialogue_with_full_context(
            structured_file,
            api_key=api_key,
            output_dir=output_dir
        )
        
        if enhanced_file:
            logger.info(f"Successfully enhanced dialogues and saved to {enhanced_file}")
            scene_processor_result["structured_analysis_json"] = enhanced_file
        else:
            logger.warning("Failed to enhance dialogues")
            
        return scene_processor_result
        
    except Exception as e:
        logger.error(f"Error in dialogue enhancement: {str(e)}")
        logger.error(traceback.format_exc())
        return scene_processor_result

def regenerate_analysis_report(structured_data_path, pipeline_instance=None):
    """
    Regenerate the analysis report to ensure it reflects the enhanced dialogues.
    
    Args:
        structured_data_path: Path to the structured data file
        pipeline_instance: Optional reference to the pipeline instance
    """
    try:
        # Check if the pipeline has a report generation method we can call
        if pipeline_instance and hasattr(pipeline_instance, '_generate_analysis_report'):
            logger.info("Regenerating analysis report using pipeline's method")
            pipeline_instance._generate_analysis_report(structured_data_path)
            return
        
        # If not, use our standalone method to regenerate the report
        from pathlib import Path
        from datetime import datetime
        import json
        
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
        
    except Exception as e:
        logger.error(f"Error regenerating analysis report: {str(e)}")

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