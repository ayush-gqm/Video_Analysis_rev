import json
import sys
import logging
from pathlib import Path
from main import VideoPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default output directory
output_dir = Path("Kuma_ep5")

# Load existing data
try:
    with open(output_dir / "full_structured_analysis.json", 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Check if gemini_vision data is in the results
    if "gemini_vision" in results:
        print(f"Found gemini_vision in results with {len(results['gemini_vision'].get('scenes', {}))} scenes")
    else:
        print("No gemini_vision data found in results")
        
        # Try to load it directly
        gemini_file = output_dir / "gemini" / "scene_analysis.json"
        if gemini_file.exists():
            print(f"Found Gemini file at {gemini_file}")
            with open(gemini_file, 'r', encoding='utf-8') as f:
                gemini_data = json.load(f)
                print(f"Loaded Gemini data with {len(gemini_data.get('scenes', {}))} scenes")
                
                # Add it to results
                results["gemini_vision"] = gemini_data
                print("Added Gemini data to results")
    
    # Generate structured data for report
    structured_data_path = output_dir / "structured_analysis.json"
    
    # Load or create structured data
    if structured_data_path.exists():
        print(f"Using existing structured analysis data from {structured_data_path}")
        with open(structured_data_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
    else:
        print("Creating new structured analysis data")
        # Create a VideoPipeline instance without initializing components (just for report generation)
        pipeline = VideoPipeline()
        
        # Build structured data from combined results
        structured_data = {
            "video_info": {
                "file_name": Path(results.get("video_path", "unknown")).name,
                "duration": sum(scene.get("duration", 0) for scene in results.get("scenes", [])),
                "num_scenes": len(results.get("scenes", [])),
                "analysis_timestamp": results.get("timestamp", "")
            },
            "scenes": {},
            "entities": {},
            "audio": {},
            "narrative_flow": {},
            "narrative_structure": {},
            "characters": {}
        }
        
        # Add scenes to structured data
        for i, scene in enumerate(results.get("scenes", [])):
            scene_key = str(i)
            structured_data["scenes"][scene_key] = {
                "scene_number": i + 1,
                "scene_idx": i,
                "start_time": scene.get("start_time", 0),
                "end_time": scene.get("end_time", 0),
                "duration": scene.get("duration", 0),
                "timestamp": pipeline._format_timestamp(
                    scene.get("start_time", 0), 
                    scene.get("end_time", 0)
                )
            }
            
            # Add Gemini description if available
            if "gemini_vision" in results and "scenes" in results["gemini_vision"]:
                if scene_key in results["gemini_vision"]["scenes"]:
                    scene_data = results["gemini_vision"]["scenes"][scene_key]
                    if "description" in scene_data:
                        structured_data["scenes"][scene_key]["description"] = scene_data["description"]
        
        # Save the structured data
        with open(structured_data_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        print(f"Saved structured analysis data to {structured_data_path}")
    
    # Generate the final report
    report_path = output_dir / "analysis_report.md"
    pipeline = VideoPipeline()
    pipeline._generate_analysis_report(structured_data, report_path)
    print(f"Report generated successfully at {report_path}")
    
except Exception as e:
    print(f"Error generating report: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 