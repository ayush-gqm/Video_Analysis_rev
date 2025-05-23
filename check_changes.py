#!/usr/bin/env python3
"""
Script to check if dialogue enhancements were applied correctly in the structured_analysis.json file.
"""

import json
import sys
import os
import re
from pathlib import Path

def check_enhancements(json_file):
    """Check if dialogue enhancements were applied correctly."""
    print(f"Checking dialogue enhancements in {json_file}")
    
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if the file has the expected structure
    if "scenes" in data:
        scenes = data["scenes"]
    else:
        scenes = {k: v for k, v in data.items() if k.isdigit() or (isinstance(k, str) and k.startswith("scene_"))}
    
    print(f"Found {len(scenes)} scenes in the file")
    
    # Count UNKNOWN speakers and check character matching
    total_unknowns = 0
    total_speakers = 0
    character_mismatches = 0
    
    for scene_id, scene in scenes.items():
        if "dialogue" not in scene or "transcript" not in scene["dialogue"]:
            continue
            
        transcript = scene["dialogue"]["transcript"]
        
        # Count UNKNOWN speakers
        unknown_count = transcript.count("UNKNOWN")
        
        # Extract all speaker names from the transcript
        speaker_pattern = r'^([^[]+)\s*\['
        speakers = re.findall(speaker_pattern, transcript, re.MULTILINE)
        
        # Check if speakers are in the characters list
        characters = scene.get("characters", [])
        speakers_not_in_characters = set()
        
        for speaker in speakers:
            if speaker.strip() != "UNKNOWN" and speaker.strip() not in characters:
                speakers_not_in_characters.add(speaker.strip())
        
        # Print the results for this scene
        if unknown_count > 0 or speakers_not_in_characters:
            print(f"\nScene {scene_id}:")
            if unknown_count > 0:
                print(f"  - {unknown_count} UNKNOWN speakers")
            if speakers_not_in_characters:
                print(f"  - Speakers not in characters list: {', '.join(speakers_not_in_characters)}")
            
            if "characters" in scene:
                print(f"  - Characters: {', '.join(characters)}")
            else:
                print("  - No characters list in scene")
            
            # Print part of the transcript for verification
            transcript_lines = transcript.split("\n")
            preview = "\n    ".join(transcript_lines[:3]) + (
                f"\n    ... ({len(transcript_lines) - 3} more lines)" if len(transcript_lines) > 3 else ""
            )
            print(f"  - Transcript preview:\n    {preview}")
        
        total_unknowns += unknown_count
        total_speakers += len(speakers)
        character_mismatches += len(speakers_not_in_characters)
    
    # Print overall statistics
    print("\nOverall statistics:")
    print(f"Total speakers: {total_speakers}")
    print(f"UNKNOWN speakers: {total_unknowns} ({total_unknowns / total_speakers * 100:.2f}% of all speakers)")
    print(f"Character mismatches: {character_mismatches} (speakers not in characters list)")
    
    if total_unknowns == 0 and character_mismatches == 0:
        print("\nAll dialogue has been properly enhanced!")
    else:
        print("\nSome dialogue still needs enhancement.")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Use default file
        json_file = "Kuma_test/structured_analysis.json"
    
    check_enhancements(json_file) 