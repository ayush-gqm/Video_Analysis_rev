# Video Analysis Pipeline

A comprehensive pipeline for analyzing video content with scene detection, keyframe extraction, entity detection, audio processing, and AI-powered scene analysis.

## Features

- **Scene Detection**: Identifies meaningful scenes based on visual content and dialogue
- **Keyframe Extraction**: Extracts representative frames from each scene
- **Entity Detection**: Identifies objects, people, and text in video frames using YOLOv5 and GroundingSAM
- **Audio Processing**: 
  - Speech transcription with ASR ensemble (Whisper-Large, Faster-Whisper, Google STT)
  - Speaker diarization with improved VAD (WebRTC + Silero)
  - Overlap detection using EEND-EDA
  - Forced alignment with subtitles via Gentle
- **Speaker Bank**: Dynamic per-video speaker tracking with x-vector embeddings
- **Scene Analysis**: Provides detailed scene analysis using Gemini Vision API
- **Dialogue Enhancement**: Identifies characters in dialogue using context-aware AI
- **Parallel Processing**: Executes independent pipeline components concurrently
- **Result Caching**: Caches intermediate results for faster repeated runs

## Repository Structure

This repository contains all the Python code and configuration files needed to run the video analysis pipeline:

- **Main Pipeline**:
  - `main.py`: Main pipeline orchestrator
  - `run_pipeline.py`: Wrapper script with CUDA optimization

- **Components**:
  - `components/scene_detection.py`: Scene detection module
  - `components/keyframe_extraction.py`: Keyframe extraction module
  - `components/entity_detection.py`: Entity detection with YOLOv5 and GroundingSAM
  - `components/audio_processing.py`: Audio processing and transcription
  - `components/gemini_vision.py`: Scene analysis with Gemini Vision API
  - `components/speaker_bank.py`: Speaker tracking and identification
  - `components/character_graph.py`: Character relationship tracking
  - `components/grounding_sam_wrapper.py`: GroundingSAM integration

- **Configuration**:
  - `config.py`: Configuration loading utilities
  - `config.yaml`: Default configuration
  - `enhanced_config.yaml`: Enhanced configuration for improved results

- **Utilities**:
  - `memory_tools/`: Memory optimization utilities
  - `utils/`: Various utility functions
    - `parallel_executor.py`: Parallel task execution utilities
    - `result_manager.py`: Result caching and management
    - `pipeline_enhancer.py`: Dynamic pipeline enhancements
    - `clean_cache.py`: Cache cleanup utility
  - `cleanup.py`: Repository maintenance script
  
- **Models**:
  - `models/`: Directory for all model files
    - `grounding_sam/`: GroundingSAM model files
      - `GroundingDINO_SwinT_OGC.py`: GroundingDINO configuration
      - `groundingdino_swint_ogc.pth`: GroundingDINO weights
      - `sam_vit_h_4b8939.pth`: SAM model weights
  
- **Helpers**:
  - `fetch_models.py`: Downloads and sets up all required models
  - `check_pipeline.py`: Verifies pipeline configuration and model setup
  - `check_models.py`: Tests loading of specific model components

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- FFmpeg installed on your system
- Tesseract OCR installed on your system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nelumbo08/Video_Analysis.git
cd Video_Analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install GroundingSAM dependencies:
```bash
pip install groundingdino-py segment-anything
```

5. Prepare required models:
```bash
# Download and initialize all models (YOLOv5, GroundingSAM, etc.)
python fetch_models.py --all

# For GroundingSAM models only
python fetch_models.py --grounding-sam

# To force redownload models
python fetch_models.py --grounding-sam --force
```

6. Set up environment variables:
```bash
export GEMINI_API_KEY="your_api_key_here"  # Required for scene analysis
```

### Troubleshooting Installation

If you encounter any issues with model setup, you can verify your configuration:
```bash
# Verify the entire pipeline configuration
python check_pipeline.py

# Test loading of GroundingSAM models only
python check_models.py
```

If you encounter any issues with YOLOv5 model imports, you can specifically fix them with:
```bash
# Repair YOLOv5 model installation
python fetch_models.py --yolo
```

This will ensure proper model setup and fix common import errors with YOLOv5.

## Usage

Basic usage:
```bash
python run_pipeline.py path/to/video.mp4 --output results
```

With enhanced configuration (recommended for better performance):
```bash
python run_pipeline.py path/to/video.mp4 --output results --config enhanced_config.yaml
```

Directly using main.py (without the wrapper):
```bash
python main.py path/to/video.mp4 --output results --config enhanced_config.yaml
```

### Parallel Processing

The pipeline automatically uses parallel processing for independent components, which significantly improves performance on multi-core systems. This behavior is controlled in the configuration:

```yaml
# In config.yaml or enhanced_config.yaml
parallel_processing:
  enabled: true  # Set to false to disable parallel processing
  max_workers: 4  # Adjust based on your CPU cores
```

You can also set environment variables to control parallel processing:
```bash
# Disable parallel processing
VIDEO_PIPELINE_PARALLEL=0 python run_pipeline.py path/to/video.mp4

# Specify number of workers
VIDEO_PIPELINE_WORKERS=6 python run_pipeline.py path/to/video.mp4
```

### Result Caching

The pipeline can cache intermediate results to speed up repeated runs on the same video:

```yaml
# In config.yaml or enhanced_config.yaml
parallel_processing:
  use_result_cache: true  # Enable/disable result caching
  cache_dir: ".cache"     # Directory to store cached results
  cache_ttl: 86400        # Cache time-to-live in seconds (24 hours)
```

To clean the cache:
```bash
# Clean all cache
python -m utils.clean_cache --all

# Clean cache older than 1 week (604800 seconds)
python -m utils.clean_cache --older-than 604800
```

## Configuration

The pipeline behavior can be customized by editing `config.yaml` or creating your own config file. Key configuration areas:

- **Device Selection**: CUDA/CPU processing options
- **Scene Detection**: Thresholds, scene length parameters
- **Keyframe Extraction**: Method, number of frames
- **Entity Detection**: 
  - Detection model (GroundingSAM or YOLOv5)
  - Model paths for GroundingSAM
  - Confidence thresholds
  - Detection prompts
- **Audio Processing**: Transcription model, language
- **Gemini Vision**: Temperature, token limits, structured output fields

## Output

The pipeline generates a comprehensive analysis in the specified output directory:

- Scene detection results with timestamps
- Representative keyframes for each scene
- Entity detection with bounding boxes and labels
- Transcribed dialogue with speaker identification
- Detailed scene analysis with Gemini Vision
- Final analysis report in Markdown format

## Documentation

For detailed documentation on the pipeline architecture, components, and workflow, see [Video_Analysis_Pipeline.md](Video_Analysis_Pipeline.md).

## Contributing

Contributions to improve the pipeline are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 