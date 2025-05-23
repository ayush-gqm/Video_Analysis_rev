# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python fetch_models.py --all

# Set up Gemini API key
export GEMINI_API_KEY="your_api_key_here"
```

### Running the Pipeline

```bash
# Basic usage
python run_pipeline.py path/to/video.mp4 --output results

# With enhanced configuration (recommended)
python run_pipeline.py path/to/video.mp4 --output results --config enhanced_config.yaml

# Direct pipeline execution
python main.py path/to/video.mp4 --output results --config enhanced_config.yaml

# Run with parallel processing disabled
VIDEO_PIPELINE_PARALLEL=0 python run_pipeline.py path/to/video.mp4

# Run with specific number of workers
VIDEO_PIPELINE_WORKERS=6 python run_pipeline.py path/to/video.mp4
```

### Debugging & Verification

```bash
# Check dialogue enhancements
python check_changes.py results/structured_analysis.json

# Clean cache
python -m utils.clean_cache --all
python -m utils.clean_cache --older-than 604800  # Clean cache older than 1 week
```

## Code Architecture

The Video Analysis Pipeline is a sophisticated system for analyzing video content through multiple dimensions: visual, auditory, and narrative. The codebase is structured around core components that work together to process videos.

### Core Components

1. **Pipeline Orchestration**
   - `main.py`: Central orchestrator that manages the pipeline execution
   - `run_pipeline.py`: Wrapper script that sets up CUDA environment before running the pipeline
   - `main_parallel.py`: Enhanced version with parallel processing capability

2. **Scene Detection** (`components/scene_detection.py`)
   - Divides videos into meaningful scenes based on visual content and semantic meaning
   - Uses CLIP embeddings to measure visual/semantic similarity between frames
   - Implements hierarchical clustering for scene boundary detection

3. **Keyframe Extraction** (`components/keyframe_extraction.py`)
   - Extracts representative frames from each scene
   - Supports uniform sampling and content-based extraction methods
   - Uses clustering to find diverse representative frames

4. **Entity Detection** (`components/entity_detection.py`)
   - Identifies objects, people, and text within keyframes using YOLOv5 and GroundingSAM
   - Classifies entities into semantic categories (Person, Setting, Object, Action)
   - Tracks objects across scenes to identify recurring elements

5. **Audio Processing** (`components/audio_processing.py`)
   - Transcribes speech using an ASR ensemble (Whisper, Faster-Whisper, Google STT)
   - Performs speaker diarization with PyAnnote
   - Uses advanced VAD (WebRTC + Silero) for speech detection
   - Maps dialogue to corresponding scenes

6. **Speaker Bank** (`components/speaker_bank.py`)
   - Maintains a database of speaker identities and their voice embeddings
   - Uses x-vector embeddings for speaker identification
   - Maps character names to speakers for consistent dialogue attribution

7. **Gemini Vision Analysis** (`components/gemini_vision.py`)
   - Generates rich descriptive analysis of scenes using Google's Gemini Vision models
   - Incorporates entity and dialogue information as context
   - Provides structured scene descriptions (setting, characters, actions, etc.)

8. **Dialogue Enhancement** (`full_context_dialogue_enhancer.py`)
   - Identifies speakers in dialogue and replaces "UNKNOWN" labels with character names
   - Uses Gemini Pro to analyze dialogue in context
   - Implements robust JSON parsing to handle API responses

9. **Parallel Processing** (`utils/parallel_executor.py`)
   - Enables concurrent execution of independent pipeline components
   - Manages worker pools and task dependencies
   - Implements result caching to speed up repeated runs

### Pipeline Workflow

1. **Scene Detection**: Video is divided into meaningful scenes
2. **Keyframe Extraction**: Representative frames are extracted from each scene
3. **Entity Detection**: Objects and people are identified in keyframes
4. **Audio Processing**: Speech is transcribed and speakers are identified
5. **Scene Analysis**: Gemini Vision API analyzes each scene's content
6. **Dialogue Enhancement**: Character names are added to dialogue
7. **Report Generation**: All outputs are aggregated into structured analysis

### Configuration System

The pipeline uses a flexible configuration system:
- `config.yaml`: Default configuration for all components
- `enhanced_config.yaml`: Optimized settings for improved results
- Environment variables for API keys and processing options

## Memory Management

The codebase includes specialized memory optimization for handling large videos:
- Windowed processing for large videos
- Batch processing for frames and operations
- Progressive cleanup to release memory
- Dynamic batch sizing based on available memory
- Mixed precision (FP16) to reduce memory usage

## Parallel Processing Architecture

The parallel implementation (`main_parallel.py`) allows independent pipeline components to run concurrently, which significantly improves performance on multi-core systems:

- **PipelineExecutor**: Manages worker pools and executes tasks based on dependency graph
- **ResultManager**: Handles caching of intermediate results to avoid redundant processing
- **Task Definitions**: Each pipeline component is wrapped in a task definition with dependencies
- **Adaptive Execution**: Falls back to sequential processing when dependencies aren't met