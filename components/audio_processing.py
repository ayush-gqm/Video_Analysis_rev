"""Audio processing component for extracting speech and identifying speakers in video."""

import os
# Set environment variables to handle NCCL issues before importing PyTorch
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL peer-to-peer which can cause symbol errors
os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking NCCL operations

import logging
import json
import os
import tempfile
import subprocess
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import datetime
import torch
import cv2
import ffmpeg
import gc
import soundfile as sf
import webrtcvad
from pydub import AudioSegment
import wave
import struct
import collections
import difflib
import requests
from components.speaker_bank import SpeakerBank
from components.character_graph import CharacterGraph

# Try to import specialized modules
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    
try:
    import openai.whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    
try:
    import pyannote.audio
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Check if CUDA is available and working properly."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        return False
        
    try:
        # Test CUDA with a small tensor
        test_tensor = torch.zeros(1).cuda()
        logger.info("CUDA is available and working")
        return True
    except Exception as e:
        logger.warning(f"CUDA is available but encountered an error: {str(e)}")
        logger.info("Falling back to CPU")
        return False

class AudioProcessor:
    """
    Processes audio from video files for speech recognition and speaker diarization.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the AudioProcessor with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        # Set use_speaker_aware_vad to True by default
        if "use_speaker_aware_vad" not in self.config:
            self.config["use_speaker_aware_vad"] = True
        # Set use_eend_eda to True by default
        if "use_eend_eda" not in self.config:
            self.config["use_eend_eda"] = True
        
        # Set device
        if os.environ.get("FORCE_CPU") == "1":
            self.device = "cpu"
        else:
            self.device = self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
            
        self.model = self.config.get("model", "large-v3")
        
        # Transcription options
        self.use_whisperx = self.config.get("use_whisperx", True)
        self.use_whisper = self.config.get("use_whisper", False)
        self.use_faster_whisper = self.config.get("use_faster_whisper", False)
        
        # Speaker diarization options
        self.use_diarization = self.config.get("use_diarization", True)
        self.hf_token = self.config.get("hf_token") or os.environ.get("HF_TOKEN")
        
        # WhisperX specific settings
        self.whisperx_config = self.config.get("whisperx", {})
        self.language = self.whisperx_config.get("language", self.config.get("language", "auto"))
        self.batch_size = self.whisperx_config.get("batch_size", 16)
        self.compute_type = self.whisperx_config.get("compute_type", "float16" if self.device == "cuda" else "float32")
        
        # Diarization settings
        self.diarization_config = self.config.get("diarization", {})
        self.min_speakers = self.diarization_config.get("min_speakers", 1)
        self.max_speakers = self.diarization_config.get("max_speakers", 10)
        
        # Check for required dependencies and tools
        self._check_dependencies()
        
        logger.info(f"Initialized AudioProcessor (model={self.model}, device={self.device}, whisperx={self.use_whisperx})")
        if self.use_diarization and not self.hf_token:
            logger.warning("Speaker diarization enabled but no HuggingFace token provided. Diarization will be skipped.")
        
        # Initialize a dynamic SpeakerBank for this video
        self.speaker_bank = SpeakerBank(backend=self.config.get("speaker_bank_backend", "faiss"), config=self.config.get("speaker_bank_config", {}))
        
    def _check_dependencies(self):
        """
        Check if required dependencies are available and set flags accordingly.
        This includes checking for FFmpeg, WhisperX, Whisper, and faster-whisper.
        """
        # Reset flags
        self.ffmpeg_available = False
        self.whisperx_available = WHISPERX_AVAILABLE
        self.whisper_available = WHISPER_AVAILABLE
        self.faster_whisper_available = FASTER_WHISPER_AVAILABLE
        self.pyannote_available = PYANNOTE_AVAILABLE
        
        # Check FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.ffmpeg_available = True
            logger.info("FFmpeg is available")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available, audio extraction may fail")
            self.ffmpeg_available = False
            
        # Additional verification of WhisperX capabilities if it's available
        if self.use_whisperx and WHISPERX_AVAILABLE:
            try:
                # Verify WhisperX by testing key functionality
                audio_load_fn = getattr(whisperx, "load_audio", None)
                if audio_load_fn is None:
                    logger.warning("WhisperX load_audio function not found")
                    self.whisperx_available = False
                
                model_load_fn = getattr(whisperx, "load_model", None)
                if model_load_fn is None:
                    logger.warning("WhisperX load_model function not found")
                    self.whisperx_available = False
                
                # Check for VAD module
                try:
                    from whisperx import vad
                    # Try different known module paths for SileroVAD across WhisperX versions
                    try:
                        # Newer versions might use this structure
                        from whisperx.vad import SileroVAD
                        logger.info("WhisperX with VAD module is available (vad.SileroVAD)")
                    except ImportError:
                        # Just check if the basic VAD module exists, don't require specific SileroVAD import
                        logger.info("WhisperX with basic VAD module is available")
                except ImportError:
                    logger.warning("WhisperX VAD module not found. Some functionality may be limited")
                    # We'll still keep WhisperX as available, but note the VAD limitation
                    logger.info("WhisperX is available (without VAD)")
            except Exception as e:
                logger.warning(f"WhisperX verification failed: {str(e)}")
                self.whisperx_available = False
        elif self.use_whisperx and not WHISPERX_AVAILABLE:
            logger.warning("WhisperX is requested but not available. Please install it using: pip install git+https://github.com/m-bain/whisperx.git")
        
        # Additional verification of faster-whisper if it's available
        if self.use_faster_whisper and FASTER_WHISPER_AVAILABLE:
            try:
                # Test loading a model (but don't actually load it to save memory)
                model_path = faster_whisper.utils.download_model("tiny", output_dir=None)
                if not model_path:
                    logger.warning("faster-whisper model check failed")
                    self.faster_whisper_available = False
                else:
                    logger.info("faster-whisper is available")
            except Exception as e:
                logger.warning(f"faster-whisper verification failed: {str(e)}")
                self.faster_whisper_available = False
        elif self.use_faster_whisper and not FASTER_WHISPER_AVAILABLE:
            logger.warning("faster-whisper is requested but not available")
        
        # Log overall availability status
        available_engines = []
        if self.whisperx_available:
            available_engines.append("WhisperX")
        if self.faster_whisper_available:
            available_engines.append("faster-whisper")
        if self.whisper_available:
            available_engines.append("Whisper")
        
        if available_engines:
            logger.info(f"Available transcription engines: {', '.join(available_engines)}")
        else:
            logger.warning("No transcription engines available")
            
        # If none of the requested engines are available, try to use any available engine
        if self.use_whisperx and not self.whisperx_available and self.use_faster_whisper and not self.faster_whisper_available and self.use_whisper and not self.whisper_available:
            if self.whisperx_available or self.faster_whisper_available or self.whisper_available:
                logger.info("Requested engines unavailable, will use any available engine as fallback")
                self.use_whisperx = self.whisperx_available
                self.use_faster_whisper = self.faster_whisper_available 
                self.use_whisper = self.whisper_available
            
    def process_video_audio(self, video_path: str, scenes: List[Dict], output_dir: str, subtitle_path: str = None) -> Dict:
        """
        Process audio from a video file to extract speech and identify speakers. Optionally perform forced alignment with subtitles.

        Args:
            video_path: Path to the video file
            scenes: List of scene dictionaries with start_time and end_time
            output_dir: Directory to save intermediate and final results
            subtitle_path: Optional path to subtitle file for forced alignment

        Returns:
            Dictionary with audio processing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing audio from {video_path}")
        if not scenes:
            logger.warning("No scenes provided for audio processing. Creating a dummy scene for the entire video.")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            scenes = [{
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration
            }]
        audio_path = output_dir / "audio.wav"
        self._extract_audio(str(video_path), str(audio_path))
        use_speaker_aware_vad = self.config.get("use_speaker_aware_vad", False)
        auth_token = self.hf_token or os.environ.get("HF_TOKEN")
        if use_speaker_aware_vad and auth_token:
            logger.info("Using speaker-aware VAD with diarization for speech segmentation.")
            vad_segments = self._split_on_speech_gaps_with_diarization(str(audio_path), auth_token)
            logger.info(f"Speaker-aware VAD produced {len(vad_segments)} segments.")
        transcription = self._transcribe_audio(str(audio_path))
        if self.pyannote_available:
            diarization = self._perform_diarization(str(audio_path))
            results = self._combine_transcription_with_speakers(transcription, diarization, scenes)
        else:
            results = self._assign_transcription_to_scenes(transcription, scenes)
        # Forced alignment with Gentle if subtitle_path is provided
        if subtitle_path:
            logger.info(f"Performing forced alignment with Gentle using subtitles: {subtitle_path}")
            try:
                alignment = self._forced_align_with_gentle(str(audio_path), subtitle_path)
                results["forced_alignment"] = alignment
            except Exception as e:
                logger.error(f"Gentle forced alignment failed: {e}")
        # --- CharacterGraph integration ---
        try:
            char_graph = CharacterGraph()
            for idx, scene_dict in results.get("scenes", {}).items():
                # Get the scene data correctly - it might be nested under 'scene_info'
                scene_data = scene_dict.get("scene_info", scene_dict)
                dialogue = scene_dict.get("dialogue", [])
                if isinstance(dialogue, str):
                    # If dialogue is a string (transcript), wrap it in a list
                    dialogue = [{"text": dialogue}]
                # Extract speakers from dialogue segments
                speakers = list({seg.get("speaker", "UNKNOWN") for seg in dialogue if isinstance(seg, dict) and seg.get("speaker")})
                char_graph.add_scene(int(idx), speakers, dialogue)
            char_graph.resolve_characters(self.speaker_bank)
            results["character_graph"] = char_graph.export_json()
        except Exception as e:
            logger.error(f"CharacterGraph integration failed: {e}")
        # --- end CharacterGraph integration ---
        self._save_results(results, output_dir)
        return results

    def _extract_audio(self, video_path: str, output_path: str):
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Path where to save the extracted audio
        """
        try:
            logger.info(f"Extracting audio from {video_path} to {output_path}")
            
            if not self.ffmpeg_available:
                logger.error("FFmpeg is not available, cannot extract audio")
                raise RuntimeError("FFmpeg is not available for audio extraction")
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
            # Extract audio using FFmpeg
            command = [
                'ffmpeg', '-y', '-i', video_path, 
                '-ac', '1',  # Mono audio
                '-ar', '16000',  # 16kHz sample rate
                '-vn',  # No video
            output_path
        ]
        
            # Run the command
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Audio extracted successfully: {output_path}")
                return True
            else:
                logger.error("Audio extraction failed: output file is empty or missing")
                raise RuntimeError("Audio extraction failed")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error during audio extraction: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"FFmpeg error: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
            
    def _transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio using available engines with fallback.
        
        Args:
            audio_path: Path to the audio file to transcribe
        
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"Transcribing audio from {audio_path}")
        
        # Check if any of the engines are available
        if not self.whisperx_available and not self.faster_whisper_available and not self.whisper_available:
            logger.warning("No transcription engines available. Using dummy transcription.")
            return self._create_dummy_transcription(audio_path)
            
        # Try WhisperX first (if available and enabled)
        if self.use_whisperx and self.whisperx_available:
            try:
                logger.info("Attempting transcription with WhisperX...")
                result = self._transcribe_with_whisperx(audio_path)
                if result and "segments" in result and result["segments"]:
                    logger.info("WhisperX transcription successful")
                    return result
            except Exception as e:
                logger.warning(f"WhisperX transcription failed: {str(e)}")
                logger.info("Falling back to alternative transcription engine")
        
        # Try faster-whisper next (if available and enabled)
        if self.use_faster_whisper and self.faster_whisper_available:
            try:
                logger.info("Attempting transcription with faster-whisper...")
                result = self._transcribe_with_faster_whisper(audio_path)
                if result and "segments" in result and result["segments"]:
                    logger.info("faster-whisper transcription successful")
                    return result
            except Exception as e:
                    logger.warning("faster-whisper returned empty or invalid results, falling back to next engine")
            except Exception as e:
                logger.warning(f"faster-whisper transcription failed: {str(e)}")
                logger.info("Falling back to next transcription engine")
        
        # Try regular Whisper last (if available and enabled)
        if self.use_whisper and self.whisper_available:
            try:
                logger.info("Attempting transcription with Whisper...")
                result = self._transcribe_with_whisper(audio_path)
                if result and "segments" in result and result["segments"]:
                    logger.info("Whisper transcription successful")
                    return result
                else:
                    logger.warning("Whisper returned empty or invalid results, falling back to dummy transcription")
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {str(e)}")
                logger.info("All transcription attempts failed, using dummy transcription")
        
        # If all engines failed or none were available, return dummy results
        logger.warning("All transcription attempts failed or no engines available. Using dummy transcription.")
        return self._create_dummy_transcription(audio_path)
        
    def _transcribe_with_whisperx(self, audio_path: str) -> Dict:
        """
        Transcribe audio using WhisperX with improved VAD and diarization.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results
        """
        # First check if whisperx is properly imported
        if not WHISPERX_AVAILABLE:
            logger.error("WhisperX is not available or properly installed")
            return self._create_dummy_transcription(audio_path)
            
        try:
            # Import whisperx here as well to ensure it's available in this scope
            import whisperx
            
            logger.info("Starting WhisperX transcription process with enhanced VAD")
            
            # Step 1: Load audio
            logger.info("Step 1: Loading audio from file")
            audio = whisperx.load_audio(audio_path)
            
            # Step 2: Load ASR model
            logger.info(f"Step 2: Loading WhisperX model {self.model} on device {self.device}")
            
            # Don't pass language parameter if set to 'auto'
            if self.language == "auto":
                logger.info("Using automatic language detection (not specifying language for model loading)")
                model = whisperx.load_model(
                    self.model,
                    device=self.device,
                    compute_type=self.compute_type
                )
            else:
                logger.info(f"Using specified language: {self.language}")
                model = whisperx.load_model(
                    self.model,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language
                )
            
            # Step 3: Use more robust VAD approach - try multiple methods
            logger.info("Step 3: Setting up VAD for speech detection")
            vad_segments = None
            
            # First try the integrated WhisperX VAD
            try:
                logger.info("Attempting to use WhisperX integrated VAD")
                
                # Try different module paths for VAD based on WhisperX version
                vad_model = None
                
                # Method 1: Try newer versions with dedicated vad module
                if hasattr(whisperx, 'vad'):
                    logger.info("Found whisperx.vad module")
                    try:
                        # Try direct import of SileroVAD from the vad module
                        from whisperx.vad import SileroVAD
                        logger.info("Using SileroVAD from whisperx.vad")
                        vad_model = SileroVAD(self.device)
                    except ImportError:
                        # If SileroVAD isn't available, try VAD class
                        try:
                            from whisperx.vad import VAD
                            logger.info("Using VAD from whisperx.vad")
                            vad_model = VAD(self.device)
                        except ImportError:
                            logger.warning("Could not import VAD classes from whisperx.vad")
                
                # Method 2: Try older versions where VAD might be in different locations
                if vad_model is None and hasattr(whisperx, 'audio'):
                    logger.info("Checking whisperx.audio module for VAD")
                    try:
                        from whisperx.audio import VAD
                        logger.info("Using VAD from whisperx.audio")
                        vad_model = VAD(self.device)
                    except ImportError:
                        logger.warning("Could not import VAD from whisperx.audio")
                
                # Method 3: Try main module
                if vad_model is None:
                    logger.info("Checking whisperx main module for VAD")
                    if hasattr(whisperx, 'VAD'):
                        logger.info("Using VAD from whisperx main module")
                        vad_model = whisperx.VAD(self.device)
                    elif hasattr(whisperx, 'SileroVAD'):
                        logger.info("Using SileroVAD from whisperx main module")
                        vad_model = whisperx.SileroVAD(self.device)
                
                # If we found a VAD model, use it
                if vad_model is not None:
                    # Use more sensitive parameters for better dialogue detection
                    vad_segments = vad_model(
                        audio,
                        threshold=0.2,  # More sensitive threshold (was 0.3)
                        min_speech_duration_ms=250,  # Shorter min speech duration (was 300)
                        window_size_samples=512
                    )
                    logger.info(f"VAD detected {len(vad_segments)} potential speech segments")
                else:
                    logger.warning("Could not initialize any WhisperX VAD model")
            except Exception as e:
                logger.error(f"Error with WhisperX VAD: {str(e)}. Trying alternative VAD method.")
            
            # If WhisperX VAD failed, try using Silero VAD directly
            if vad_segments is None:
                try:
                    logger.info("Attempting to use Silero VAD directly")
                    import torch
                    
                    # Initialize Silero VAD model directly
                    repo_location = None
                    
                    # First try pip installed version
                    try:
                        # First try if the package exists 
                        from importlib.util import find_spec
                        if find_spec("silero") is not None:
                            import silero
                            logger.info("Using pip installed silero-vad package")
                            repo_location = "silero-vad"
                        else:
                            raise ImportError("silero not found")
                    except ImportError:
                        # Try direct repository access
                        logger.info("Using direct repository access for silero-vad")
                        repo_location = "snakers4/silero-vad"
                        
                        # If direct import fails, download manually using torch.hub.download_url_to_file
                        try:
                            import os
                            import torch.hub
                            cache_dir = torch.hub.get_dir()
                            os.makedirs(os.path.join(cache_dir, 'checkpoints'), exist_ok=True)
                            
                            logger.info(f"Downloading silero-vad model to {cache_dir}")
                            torch.hub.download_url_to_file(
                                'https://github.com/snakers4/silero-vad/raw/master/silero_vad.jit',
                                os.path.join(cache_dir, 'checkpoints', 'silero_vad.jit'),
                                progress=False
                            )
                            logger.info("Downloaded silero-vad model successfully")
                        except Exception as download_error:
                            logger.warning(f"Manual download attempt failed: {str(download_error)}")
                    
                    # Load the model, trying different methods
                    # Try multiple loading methods in sequence
                    vad_model = None
                    utils = None
                    
                    # Method 1: Standard torch.hub.load
                    try:
                        logger.info("Trying standard torch.hub.load method")
                        vad_model, utils = torch.hub.load(
                            repo_or_dir=repo_location,
                            model="silero_vad",
                            force_reload=False,
                            onnx=False,
                            verbose=False
                        )
                        logger.info("Standard loading successful")
                    except Exception as load_error:
                        logger.warning(f"Standard loading failed: {str(load_error)}")
                    
                    # Method 2: Explicit GitHub source
                    if vad_model is None:
                        try:
                            logger.info("Trying loading with explicit GitHub source")
                            vad_model, utils = torch.hub.load(
                                repo_or_dir="snakers4/silero-vad:main",
                                model="silero_vad",
                                source="github",
                                force_reload=True,
                                onnx=False,
                                verbose=False
                            )
                            logger.info("GitHub source loading successful")
                        except Exception as github_error:
                            logger.warning(f"GitHub source loading failed: {str(github_error)}")
                    
                    # Method 3: Direct path to JIT model
                    if vad_model is None:
                        try:
                            logger.info("Trying direct loading of JIT model")
                            cache_dir = torch.hub.get_dir()
                            model_path = os.path.join(cache_dir, 'checkpoints', 'silero_vad.jit')
                            
                            if os.path.exists(model_path):
                                # Load the model directly from file
                                vad_model = torch.jit.load(model_path)
                                vad_model.eval()
                                
                                # Create basic utils functions
                                def get_speech_timestamps(audio, model, threshold=0.5, min_speech_duration_ms=250, 
                                                         min_silence_duration_ms=100, window_size_samples=512, 
                                                         speech_pad_ms=30, return_seconds=False):
                                    """Basic implementation of VAD function"""
                                    sample_rate = 16000  # Fixed for this model
                                    
                                    if return_seconds:
                                        time_scale = sample_rate
                                    else:
                                        time_scale = 1
                                        
                                    min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
                                    min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
                                    speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
                                    
                                    # Process in chunks
                                    result = []
                                    speeches = []
                                    current_speech = {}
                                    
                                    stride = window_size_samples
                                    for i in range(0, len(audio), stride):
                                        chunk = audio[i:i + window_size_samples]
                                        if len(chunk) < window_size_samples:
                                            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
                                            
                                        speech_prob = model(chunk)[0]
                                        
                                        if speech_prob >= threshold:
                                            if not current_speech:
                                                current_speech = {'start': i / time_scale}
                                        elif current_speech:
                                            current_speech['end'] = (i + stride) / time_scale
                                            speeches.append(current_speech)
                                            current_speech = {}
                                    
                                    # Handle last speech segment
                                    if current_speech:
                                        current_speech['end'] = len(audio) / time_scale
                                        speeches.append(current_speech)
                                    
                                    # Merge short silence intervals
                                    result = []
                                    for i, speech in enumerate(speeches):
                                        if i == 0:
                                            result.append(speech)
                                            continue
                                        
                                        gap = speech['start'] - result[-1]['end']
                                        if gap < min_silence_samples / time_scale:
                                            result[-1]['end'] = speech['end']
                                        else:
                                            result.append(speech)
                                    
                                    return result
                                
                                # Create a simple utils tuple
                                utils = (get_speech_timestamps, None, None, None, None)
                                logger.info("Direct JIT model loading successful")
                            else:
                                logger.warning(f"JIT model file not found at {model_path}")
                        except Exception as jit_error:
                            logger.warning(f"Direct JIT loading failed: {str(jit_error)}")
                    
                    # Check if we have a model
                    if vad_model is None:
                        raise RuntimeError("All loading methods failed for Silero VAD")
                    
                    # Move model to appropriate device
                    vad_model.to(self.device)
                    
                    # Get utils
                    (get_speech_timestamps, _, read_audio, _, _) = utils
                    
                    # Convert audio to format expected by Silero VAD if needed
                    if isinstance(audio, np.ndarray):
                        if audio.dtype != np.float32:
                            audio_tensor = torch.tensor(audio.astype(np.float32))
                        else:
                            audio_tensor = torch.tensor(audio)
                    else:
                        audio_tensor = audio
                        
                    # Ensure audio is on the correct device
                    audio_tensor = audio_tensor.to(self.device)
                    
                    # Get speech timestamps
                    speech_timestamps = get_speech_timestamps(
                        audio_tensor, 
                        vad_model,
                        threshold=0.2,  # More sensitive threshold
                        min_speech_duration_ms=250,  # Short enough for brief utterances
                        min_silence_duration_ms=300,  # Allow reasonable pauses
                        window_size_samples=512,
                        return_seconds=True
                    )
                    
                    # Convert to format expected by WhisperX
                    if speech_timestamps:
                        vad_segments = {"segments": []}
                        for segment in speech_timestamps:
                            vad_segments["segments"].append({
                                "start": segment["start"],
                                "end": segment["end"]
                            })
                        logger.info(f"Silero VAD detected {len(vad_segments['segments'])} speech segments")
                except Exception as e:
                    logger.error(f"Error using direct Silero VAD: {str(e)}. Continuing without VAD.")
                    vad_segments = None
            
            # Step 4: Transcribe audio with appropriate parameters and VAD if available
            logger.info("Step 4: Transcribing audio with WhisperX")
            transcribe_language = None if self.language == "auto" else self.language
            
            # Check if various methods/parameters are supported in this WhisperX version
            try:
                # Check if this is a FasterWhisperPipeline
                is_faster_whisper = False
                if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
                    is_faster_whisper = 'FasterWhisperPipeline' in model.__class__.__name__
                    logger.info(f"Detected model type: {'FasterWhisperPipeline' if is_faster_whisper else 'Standard WhisperX'}")
                
                # Try transcription with VAD segments if available and NOT using FasterWhisperPipeline
                # (which doesn't support vad parameters)
                if vad_segments is not None and len(vad_segments.get("segments", [])) > 0 and not is_faster_whisper:
                    try:
                        logger.info("Using VAD segments for transcription with standard WhisperX")
                        result = model.transcribe(
                            audio,
                            batch_size=self.batch_size,
                            language=transcribe_language,
                            vad_filter=True,
                            vad_parameters=vad_segments
                        )
                    except TypeError as e:
                        logger.warning(f"VAD parameter error: {str(e)}. Trying alternative format.")
                        # If vad_parameters is not supported, try vad_segments
                        try:
                            result = model.transcribe(
                                audio,
                                batch_size=self.batch_size,
                                language=transcribe_language,
                                vad_filter=True,
                                vad_segments=vad_segments
                            )
                        except TypeError as e2:
                            logger.warning(f"VAD segments parameter error: {str(e2)}. Using only vad_filter.")
                            # If neither works, use only vad_filter
                            result = model.transcribe(
                                audio,
                                batch_size=self.batch_size,
                                language=transcribe_language,
                                vad_filter=True
                            )
                else:
                    # For FasterWhisperPipeline or when VAD segments aren't available
                    logger.info(f"Using {'FasterWhisperPipeline' if is_faster_whisper else 'standard'} transcription without VAD parameters")
                    # FasterWhisperPipeline doesn't support vad_filter
                    if is_faster_whisper:
                        # Basic parameters for FasterWhisperPipeline
                        result = model.transcribe(
                            audio,
                            batch_size=self.batch_size,
                            language=transcribe_language
                        )
                    else:
                        # Standard WhisperX with vad_filter
                        result = model.transcribe(
                            audio,
                            batch_size=self.batch_size,
                            language=transcribe_language,
                            vad_filter=True  # Still use VAD filtering but without custom segments
                        )
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}. Trying with basic settings.")
                # Most basic approach if the standard one fails
                try:
                    # Try without language parameter
                    logger.info("Falling back to minimal parameters")
                    result = model.transcribe(audio, batch_size=self.batch_size)
                except Exception as e2:
                    logger.error(f"Error during basic transcription: {str(e2)}.")
                    return self._create_dummy_transcription(audio_path)
            
            # Step 5: Align timestamps
            logger.info("Step 5: Aligning timestamps with phoneme-level precision")
            try:
                detected_language = result.get("language", "en")
                logger.info(f"Detected language for alignment: {detected_language}")
                
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device
                )
                
                # Basic alignment parameters that should work with most versions
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    self.device
                )
                logger.info(f"Aligned {len(result.get('segments', []))} segments")
                
            except Exception as e:
                logger.error(f"Error during alignment: {str(e)}. Continuing with unaligned segments.")
                # Continue with unaligned segments
            
            # Step 6: Perform diarization if enabled
            if self.use_diarization and self.hf_token:
                logger.info("Step 6: Performing speaker diarization")
                try:
                    # Initialize diarization pipeline with basic parameters
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    
                    # Basic diarization parameters
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=self.min_speakers,
                        max_speakers=self.max_speakers
                    )
                    
                    # Use standard function to assign speakers
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Fix speaker assignments if needed
                    self._fix_speaker_assignments(result)
                    
                    logger.info(f"Speaker diarization complete with {len(diarize_segments.get('segments', []))} speaker segments")
                    
                    # After diarization, add speakers to the speaker bank if x-vectors are available
                    # (Assume diarization result includes 'embedding' for each speaker if available)
                    for entry in result.get("speakers", []):
                        speaker_id = entry.get("speaker")
                        embedding = entry.get("embedding")
                        if speaker_id and embedding is not None:
                            self.speaker_bank.add_speaker(speaker_id, np.array(embedding))
                    
                except Exception as e:
                    logger.error(f"Error during diarization: {str(e)}")
                    logger.warning("Continuing without speaker assignment")
            else:
                logger.warning("Speaker diarization skipped (token not provided or diarization disabled)")
            
            # Format and clean up the results
            transcription = self._format_whisperx_results(result)
            logger.info(f"WhisperX transcription complete: {len(transcription['segments'])} segments")
            
            # Merge speaker segments for continuity
            if "segments" in transcription and transcription["segments"]:
                transcription["segments"] = self._merge_speaker_segments(transcription["segments"])
                logger.info(f"Merged into {len(transcription['segments'])} speaker segments")
            
            return transcription
            
        except ImportError as e:
            logger.error(f"WhisperX import error: {str(e)}. Make sure WhisperX is properly installed.")
            return self._create_dummy_transcription(audio_path)
        except Exception as e:
            logger.error(f"Error in WhisperX transcription: {str(e)}")
            logger.warning("Falling back to dummy transcription")
            return self._create_dummy_transcription(audio_path)
            
    def _fix_speaker_assignments(self, result: Dict) -> None:
        """
        Apply post-processing fixes to speaker assignments to improve coherence.
        
        Args:
            result: WhisperX result dictionary with speaker assignments (modified in-place)
        """
        segments = result.get("segments", [])
        if not segments or len(segments) < 2:
            return
            
        # Create mapping of segment to dominant speaker for each segment
        segment_speakers = {}
        for idx, segment in enumerate(segments):
            words = segment.get("words", [])
            if not words:
                continue
                
            # Count speaker occurrences in words
            speaker_counts = {}
            for word in words:
                speaker = word.get("speaker", "UNKNOWN")
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
            # Find dominant speaker
            dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0] if speaker_counts else "UNKNOWN"
            segment_speakers[idx] = dominant_speaker
        
        # Apply temporal continuity fixes (avoid rapid speaker switching)
        for i in range(1, len(segments) - 1):
            curr_speaker = segment_speakers.get(i)
            prev_speaker = segment_speakers.get(i-1)
            next_speaker = segment_speakers.get(i+1)
            
            # If current segment is sandwiched between two segments with the same speaker,
            # and is very short, it's likely the same speaker continuing
            curr_duration = segments[i]["end"] - segments[i]["start"]
            if prev_speaker == next_speaker and prev_speaker != curr_speaker and curr_duration < 2.0:
                # This is likely a misclassification - fix it by copying the surrounding speaker
                segment_speakers[i] = prev_speaker
                
                # Update all words in this segment to the corrected speaker
                for word in segments[i].get("words", []):
                    word["speaker"] = prev_speaker
        
        # Fix speaker labels on segments based on the dominant corrected speaker
        for idx, segment in enumerate(segments):
            segment["speaker"] = segment_speakers.get(idx, "UNKNOWN")
        
    def _format_whisperx_results(self, result: Dict) -> Dict:
        """
        Format WhisperX results into a standardized structure.
        
        Args:
            result: Raw WhisperX result dictionary
            
        Returns:
            Standardized transcription dictionary
        """
        segments = []
        
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if not text:
                continue
                
            segment_dict = {
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
                "words": segment.get("words", [])
            }
            segments.append(segment_dict)
            
        return {
            "segments": segments,
            "language": result.get("language", "en"),
            "text": " ".join(s.get("text", "") for s in segments)
        }
        
    def _create_dummy_transcription(self, audio_path: str) -> Dict:
        """
        Create a dummy transcription result when transcription fails.
        
        Args:
            audio_path: Path to the audio file (used to get duration)
            
        Returns:
            Dictionary with dummy transcription
        """
        # Try to get audio duration if possible
        duration = 0
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['format']['duration'])
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {str(e)}")
            duration = 300.0  # Assume 5 minutes if we can't determine duration
            
        logger.warning(f"Creating dummy transcription for {audio_path} (duration: {duration:.2f}s)")
        
        # Create a single segment spanning the entire audio
        return {
            "text": "Transcription unavailable",
            "segments": [
                {
                    "id": 0,
                    "start": 0,
                    "end": duration,
                    "text": "Transcription unavailable",
                    "speaker": "unknown"
                }
            ],
            "language": "en"
        }
        
    def _transcribe_with_faster_whisper(self, audio_path: str) -> Dict:
        """Transcribe audio using faster-whisper."""
        # Check if faster-whisper is available
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper is not available or properly installed")
            return self._create_dummy_transcription(audio_path)
        
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Transcribing audio with faster-whisper ({self.model} model)")
        
        # Load the model
            device = "cuda" if self.device == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            model = WhisperModel(self.model, device=device, compute_type=compute_type)
        
        # Transcribe
            segments, info = model.transcribe(
            audio_path,
                language=self.language if self.language != "auto" else None,
                word_timestamps=True
            )
        
            # Convert to our standard format
            formatted_result = {
            "segments": []
            }
        
            # Process segments
            for i, segment in enumerate(segments):
                segment_dict = {
                    "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": []
            }
            
                # Add word-level timestamps if available
                if hasattr(segment, "words") and segment.words:
                    for word in segment.words:
                        word_dict = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": getattr(word, "probability", 1.0)
                        }
                segment_dict["words"].append(word_dict)
                
                formatted_result["segments"].append(segment_dict)
            
            logger.info(f"faster-whisper transcription completed: {len(formatted_result['segments'])} segments")
            return formatted_result
        
        except ImportError as e:
            logger.error(f"faster-whisper import error: {str(e)}. Make sure faster-whisper is properly installed.")
            return self._create_dummy_transcription(audio_path)
        except Exception as e:
            logger.error(f"Error in faster-whisper transcription: {str(e)}")
            logger.warning("Falling back to dummy transcription")
            return self._create_dummy_transcription(audio_path)
            
    def _transcribe_with_whisper(self, audio_path: str) -> Dict:
        """Transcribe audio using OpenAI's Whisper."""
        # Check if whisper is available
        if not WHISPER_AVAILABLE:
            logger.error("Whisper is not available or properly installed")
            return self._create_dummy_transcription(audio_path)
            
        try:
            import openai.whisper
            
            logger.info(f"Transcribing audio with Whisper ({self.model} model)")
            
            # Load the model
            model = openai.whisper.load_model(self.model, device=self.device)
            
            # Transcribe
            result = model.transcribe(
                audio_path,
                verbose=True,
                word_timestamps=True
            )
                
            logger.info(f"Whisper transcription completed: {len(result.get('segments', []))} segments")
            return result
            
        except ImportError as e:
            logger.error(f"Whisper import error: {str(e)}. Make sure Whisper is properly installed.")
            return self._create_dummy_transcription(audio_path)
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}")
            logger.warning("Falling back to dummy transcription")
            return self._create_dummy_transcription(audio_path)
        
    def _perform_diarization(self, audio_path: str) -> Dict:
        """
        Perform enhanced speaker diarization using pyannote.audio or EEND-EDA if enabled.

        Returns:
            Dictionary with diarization results
        """
        logger.info("Performing enhanced speaker diarization")
        
        # Check if pyannote.audio is available
        if not PYANNOTE_AVAILABLE:
            logger.error("pyannote.audio is not available or properly installed")
            return self._create_dummy_diarization()
            
        # Check if HF token is available
        if not self.hf_token:
            logger.warning("HuggingFace token not provided, cannot use pyannote.audio diarization")
            return self._create_dummy_diarization()
        
        try:
            import torch
            from pyannote.audio import Pipeline
            
            # Initialize the pipeline
            try:
                # Use the latest v3.1 model for better speaker separation
                pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
            
            # Move to GPU if available
                if torch.cuda.is_available() and self.device == "cuda":
                    pipeline = pipeline.to(torch.device("cuda"))
                
                # Apply diarization with basic settings for compatibility
                diarization = pipeline(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
                )
            
            # Convert to a dictionary format
                result = {"speakers": []}
            
            # Extract speaker turns
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    entry = {
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start
                    }
                    result["speakers"].append(entry)
                    
                # Merge segments if the method is available
                if hasattr(self, '_merge_speaker_segments'):
                    result["speakers"] = self._merge_speaker_segments(result["speakers"])
                
                logger.info(f"Speaker diarization completed: {len(set(s['speaker'] for s in result['speakers']))} speakers")
                
                # After diarization, add speakers to the speaker bank if x-vectors are available
                # (Assume diarization result includes 'embedding' for each speaker if available)
                for entry in result.get("speakers", []):
                    speaker_id = entry.get("speaker")
                    embedding = entry.get("embedding")
                    if speaker_id and embedding is not None:
                        self.speaker_bank.add_speaker(speaker_id, np.array(embedding))
                
                return result
            
            except Exception as e:
                logger.error(f"Error initializing diarization pipeline: {str(e)}")
                return self._create_dummy_diarization()
                
        except ImportError as e:
            logger.error(f"pyannote.audio import error: {str(e)}. Make sure pyannote.audio is properly installed.")
            return self._create_dummy_diarization()
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            return self._create_dummy_diarization()
            
    def _merge_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge adjacent segments from the same speaker to improve continuity.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            List of merged speaker segments
        """
        if not segments:
            return []
            
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        # Initialize with the first segment
        merged = [sorted_segments[0]]
        
        # Merge adjacent segments from the same speaker if they're close
        max_gap = 0.3  # Maximum silence gap to merge (in seconds)
        
        for segment in sorted_segments[1:]:
            prev = merged[-1]
            
            # If same speaker and small gap, merge them
            if (segment["speaker"] == prev["speaker"] and 
                segment["start"] - prev["end"] <= max_gap):
                # Extend previous segment
                prev["end"] = segment["end"]
                prev["duration"] = prev["end"] - prev["start"]
            else:
                # Add as new segment
                merged.append(segment)
                
        return merged
            
    def _combine_transcription_with_speakers(self, transcription: Dict, diarization: Dict, scenes: List[Dict]) -> Dict:
        """
        Combine transcription with speaker diarization and assign to scenes.
        
        Args:
            transcription: Dictionary with transcription data
            diarization: Dictionary with speaker diarization data
            scenes: List of scene dictionaries
        
        Returns:
            Dictionary with combined results
        """
        if not transcription or not diarization:
            logger.warning("Missing transcription or diarization data")
            return {"scenes": {}}
            
        # Initialize results
        results = {
            "scenes": {},
            "speakers": {},
            "speaker_count": 0
        }
        
        # Track speaker statistics
        speaker_tracking = {}
        
        # Process each scene
        for scene_idx, scene in enumerate(scenes):
            scene_start = scene.get("start_time", 0)
            scene_end = scene.get("end_time", 0)
            
            # Find segments that belong to this scene
            scene_segments = []
            
            for segment in transcription.get("segments", []):
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)
                
                # Check if segment overlaps with scene
                if (scene_start <= segment_start <= scene_end or
                    scene_start <= segment_end <= scene_end or
                    (segment_start <= scene_start and segment_end >= scene_start) or
                    (segment_start <= scene_end and segment_end >= scene_end)):
                    
                    # Find speaker for this segment
                    speaker = self._find_speaker_for_segment(
                        segment_start, segment_end, diarization.get("segments", [])
                    )
                    
                    # Track speaker statistics
                    if speaker not in speaker_tracking:
                        speaker_tracking[speaker] = {
                            "speech_time": 0,
                            "segments": 0,
                            "first_seen_at": segment_start
                        }
                    
                    speaker_tracking[speaker]["speech_time"] += segment_end - segment_start
                    speaker_tracking[speaker]["segments"] += 1
                    speaker_tracking[speaker]["first_seen_at"] = min(
                        speaker_tracking[speaker]["first_seen_at"], 
                        segment_start
                    )
                    
                    # Add segment with speaker
                    segment_copy = dict(segment)
                    segment_copy["speaker"] = speaker
                    scene_segments.append(segment_copy)
            
            # Group consecutive segments by the same speaker with improved logic
            grouped_segments = self._group_segments_by_speaker(scene_segments)
            
            # Generate a human-readable dialogue for the scene
            dialogue = ""
            for segment in grouped_segments:
                speaker = segment.get("speaker", "Unknown")
                text = segment.get("text", "").strip()
                start_time = segment.get("start", 0)  # This is already video-relative
                end_time = segment.get("end", 0)      # This is already video-relative
                
                timestamp = f"[{self._format_time(start_time)} - {self._format_time(end_time)}]"
                
                if text:
                    dialogue += f"{speaker} {timestamp}: {text}\n"
            
            # Add to results
            results["scenes"][str(scene_idx)] = {
                "start_time": scene_start,
                "end_time": scene_end,
                "duration": scene_end - scene_start,
                "timestamp": self._format_timestamp(scene_start, scene_end),
                "segments": grouped_segments,
                "dialogue": dialogue.strip() if dialogue.strip() else "No dialogue in this scene"
            }
            
        # Add speaker analysis to the overall results
        results["speakers"] = {}
        
        # Convert speaker tracking to a useful summary
        for speaker, data in sorted(speaker_tracking.items(), key=lambda x: x[1]["speech_time"], reverse=True):
            results["speakers"][speaker] = {
                "total_speech_time": data["speech_time"],
                "total_segments": data["segments"],
                "first_appearance": self._format_time(data["first_seen_at"])
            }
        
        # Add global speaker statistics
        results["speaker_count"] = len(results["speakers"])
            
        return results
        
    def _assign_transcription_to_scenes(self, transcription: Dict, scenes: List[Dict]) -> Dict:
        """
        Assign transcription segments to corresponding scenes based on timestamps.
        Implements v2 logic: If a sentence spans a scene cut, assign to the scene with ≥ 60% duration; else split text.

        Args:
            transcription: Dictionary with transcription data
            scenes: List of scene dictionaries with timestamps

        Returns:
            Dictionary with transcription segments assigned to scenes
        """
        if not transcription or "segments" not in transcription or not transcription["segments"]:
            logger.warning("No transcription segments to assign to scenes")
            return {"scenes": {}}
        if not scenes:
            logger.warning("No scenes to assign transcription to")
            return {"scenes": {}}
        logger.info(f"Assigning {len(transcription['segments'])} transcription segments to {len(scenes)} scenes (v2 logic)")
        scenes_with_dialogue = []
        for scene in scenes:
            scene_copy = dict(scene)
            scene_copy["dialogue"] = []
            scenes_with_dialogue.append(scene_copy)
        result = {
            "scenes": {},
            "total_segments": len(transcription["segments"]),
            "segments_assigned": 0,
            "segments_unassigned": 0
        }
        assigned_segments = []
        unassigned_segments = []
        for segment in transcription["segments"]:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            segment_duration = segment_end - segment_start
            # Find all scenes this segment overlaps
            overlaps = []
            for scene_idx, scene in enumerate(scenes_with_dialogue):
                scene_start = scene.get("start_time", 0)
                scene_end = scene.get("end_time", 0)
                overlap_start = max(segment_start, scene_start)
                overlap_end = min(segment_end, scene_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                if overlap_duration > 0:
                    overlaps.append((scene_idx, overlap_duration))
            if not overlaps:
                unassigned_segments.append(segment)
                continue
            # If only one scene overlaps, assign directly
            if len(overlaps) == 1:
                scene_idx = overlaps[0][0]
                scenes_with_dialogue[scene_idx]["dialogue"].append(segment)
                assigned_segments.append(segment)
                continue
            # If multiple scenes overlap, check for ≥60% rule
            max_scene_idx, max_overlap = max(overlaps, key=lambda x: x[1])
            if max_overlap / segment_duration >= 0.6:
                scenes_with_dialogue[max_scene_idx]["dialogue"].append(segment)
                assigned_segments.append(segment)
            else:
                # Split the segment text proportionally and assign to each scene
                text = segment.get("text", "")
                if not text or segment_duration <= 0:
                    unassigned_segments.append(segment)
                    continue
                # Split text by overlap proportion
                total_overlap = sum([d for _, d in overlaps])
                char_idx = 0
                for scene_idx, overlap in overlaps:
                    prop = overlap / total_overlap if total_overlap > 0 else 0
                    n_chars = int(round(len(text) * prop))
                    if n_chars <= 0:
                        continue
                    part_text = text[char_idx:char_idx + n_chars]
                    char_idx += n_chars
                    if part_text.strip():
                        part_segment = dict(segment)
                        part_segment["text"] = part_text
                        part_segment["start"] = max(segment_start, scenes_with_dialogue[scene_idx]["start_time"])
                        part_segment["end"] = min(segment_end, scenes_with_dialogue[scene_idx]["end_time"])
                        scenes_with_dialogue[scene_idx]["dialogue"].append(part_segment)
                        assigned_segments.append(part_segment)
        for scene_idx, scene in enumerate(scenes_with_dialogue):
            scene["dialogue"].sort(key=lambda x: x.get("start", 0))
            result["scenes"][str(scene_idx)] = {
                "scene_info": {
                    "start_time": scene.get("start_time", 0),
                    "end_time": scene.get("end_time", 0),
                    "duration": scene.get("duration", 0)
                },
                "dialogue": scene["dialogue"],
                "dialogue_count": len(scene["dialogue"])
            }
        result["segments_assigned"] = len(assigned_segments)
        result["segments_unassigned"] = len(unassigned_segments)
        result["unassigned_segments"] = unassigned_segments if unassigned_segments else []
        logger.info(f"Assigned {result['segments_assigned']} segments to scenes, {result['segments_unassigned']} unassigned (v2 logic)")
        return result
        
    def _group_segments_by_speaker(self, segments: List[Dict]) -> List[Dict]:
        """
        Group consecutive segments by the same speaker with improved temporal coherence.
        Handles overlapping speech and ensures proper ordering.
        """
        if not segments:
            return []
            
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        # Normalize speaker names for consistency
        for segment in sorted_segments:
            speaker = segment.get("speaker", "Unknown").strip()
            
            # Clean up speaker labels for consistency
            if speaker.lower().startswith("speaker_"):
                # Keep standard format like SPEAKER_1
                segment["speaker"] = speaker.upper()
            elif speaker.lower() == "unknown":
                segment["speaker"] = "UNKNOWN"
                
            # Add segment duration for context
            segment["duration"] = segment["end"] - segment["start"]
        
        # Group segments by speaker with improved merging logic
        grouped = []
        
        if not sorted_segments:
            return grouped
            
        current_group = {
            "speaker": sorted_segments[0]["speaker"],
            "start": sorted_segments[0]["start"],
            "end": sorted_segments[0]["end"],
            "text": sorted_segments[0]["text"],
            "duration": sorted_segments[0]["duration"]
        }
        
        for segment in sorted_segments[1:]:
            # Determine whether to merge based on multiple factors
            should_merge = False
            
            # Same speaker is primary condition
            if segment["speaker"] == current_group["speaker"]:
                # Check time proximity - relaxed for same speaker
                time_gap = segment["start"] - current_group["end"]
                
                # Merge if segments are close in time (within 2 seconds for same speaker)
                if time_gap < 2.0:
                    should_merge = True
                # Even with a bigger gap, sometimes we should merge if it's clearly the same speech
                elif time_gap < 4.0 and segment["text"].startswith(("and", "but", "or", "so", ",", ".", "...", "because")):
                    # Likely continuation of the same thought
                    should_merge = True
            
            # For different speakers, we're more strict about timing
            else:
                # Only consider merging different speakers in rare cases where:
                # 1. Current segment is very short (likely a misattribution)
                # 2. The gap is extremely small (virtually zero)
                time_gap = segment["start"] - current_group["end"]
                if segment["duration"] < 0.8 and time_gap < 0.1:
                    # Likely a misattribution of a short utterance
                    # Keep the dominant speaker
                    segment["speaker"] = current_group["speaker"]
                    should_merge = True
            
            if should_merge:
                # Merge by extending the current group
                current_group["end"] = segment["end"]
                current_group["duration"] = current_group["end"] - current_group["start"]
                
                # Insert appropriate connector between texts based on context
                if current_group["text"].endswith((".", "!", "?", "...", ":", ";")) or segment["text"].startswith(("I", "We", "They", "He", "She")):
                    current_group["text"] += " " + segment["text"]
                else:
                    # Add connector for smoother flow if needed
                    current_group["text"] += ", " + segment["text"]
            else:
                # Add current group to results and start a new one
                grouped.append(current_group)
                current_group = {
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "duration": segment["duration"]
                }
                
        # Don't forget the last group
        grouped.append(current_group)
        
        # Final post-processing: identify and fix potential speaker confusion
        if len(grouped) >= 3:
            for i in range(1, len(grouped) - 1):
                prev_speaker = grouped[i-1]["speaker"]
                curr_speaker = grouped[i]["speaker"]
                next_speaker = grouped[i+1]["speaker"]
                
                # If a speaker appears only briefly between segments from another speaker,
                # it might be a misidentification
                if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                    # Check if current segment is short
                    if grouped[i]["duration"] < 1.5:
                        # Fix the attribution
                        grouped[i]["speaker"] = prev_speaker
                        
                        # Consider merging all three segments
                        if grouped[i]["start"] - grouped[i-1]["end"] < 0.5 and grouped[i+1]["start"] - grouped[i]["end"] < 0.5:
                            # Merge all three - mark for later processing
                            grouped[i]["_merge_with_prev"] = True
                            grouped[i+1]["_merge_with_prev"] = True
        
        # Apply marked merges
        i = 0
        while i < len(grouped) - 1:
            if grouped[i+1].get("_merge_with_prev", False):
                # Merge with next
                grouped[i]["end"] = grouped[i+1]["end"]
                grouped[i]["duration"] = grouped[i]["end"] - grouped[i]["start"]
                grouped[i]["text"] += " " + grouped[i+1]["text"]
                
                # Remove the merged segment
                grouped.pop(i+1)
            else:
                i += 1
        
        # Clean up any temporary attributes
        for segment in grouped:
            if "_merge_with_prev" in segment:
                del segment["_merge_with_prev"]
            if "duration" in segment:
                del segment["duration"]
        
        return grouped
        
    def _find_speaker_for_segment(self, segment_start: float, segment_end: float, 
                                speaker_turns: List[Dict]) -> str:
        """
        Find the most likely speaker for a given segment with improved logic.
        Uses weighted overlap and speaker continuity for better assignment.
        """
        if not speaker_turns:
            return "UNKNOWN"
            
        max_overlap = 0
        best_speaker = "UNKNOWN"
        segment_duration = segment_end - segment_start
        
        # Speakers who spoke just before this segment (temporal continuity)
        recent_speakers = set()
        for turn in speaker_turns:
            # Consider speakers active in the past 2 seconds as recent
            if turn["end"] < segment_start and turn["end"] > segment_start - 2.0:
                recent_speakers.add(turn["speaker"])
        
        # First pass: find maximum overlap
        overlapping_speakers = {}
        for turn in speaker_turns:
            turn_start = turn["start"]
            turn_end = turn["end"]
            
            # Calculate overlap
            overlap_start = max(segment_start, turn_start)
            overlap_end = min(segment_end, turn_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                # Track all overlapping speakers with their overlap amount
                if turn["speaker"] not in overlapping_speakers:
                    overlapping_speakers[turn["speaker"]] = 0
                overlapping_speakers[turn["speaker"]] += overlap
                
                # Track maximum single overlap
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker"]
        
        # If we have meaningful overlap with a speaker
        if max_overlap > 0.2 * segment_duration:
            return best_speaker
            
        # If no strong overlap but multiple speakers with some overlap, use additional logic
        if overlapping_speakers:
            # Find speaker with maximum total overlap
            best_overall_speaker = max(overlapping_speakers.items(), key=lambda x: x[1])[0]
            
            # If best overall speaker is also a recent speaker, prefer them
            if best_overall_speaker in recent_speakers:
                return best_overall_speaker
                
            # Otherwise just return the speaker with most overall overlap
            return best_overall_speaker
            
        # If no overlap at all but we have recent speakers, use the most recent one
        if recent_speakers:
            most_recent_speaker = None
            most_recent_time = float('-inf')
            
            for turn in speaker_turns:
                if turn["speaker"] in recent_speakers and turn["end"] > most_recent_time and turn["end"] < segment_start:
                    most_recent_time = turn["end"]
                    most_recent_speaker = turn["speaker"]
                    
            if most_recent_speaker:
                return most_recent_speaker
                
        # Final fallback
        return "UNKNOWN"
        
    def _save_results(self, results: Dict, output_dir: Path):
        """Save processing results to JSON."""
        audio_results_path = output_dir / "audio_results.json"
        
        try:
            # Ensure we use UTF-8 encoding for all text to handle non-ASCII characters
            with open(audio_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved audio processing results to {audio_results_path}")
        except Exception as e:
            logger.error(f"Error saving audio results to JSON: {str(e)}")
            logger.warning("Attempting to save with ASCII encoding as fallback")
            try:
                # Fall back to ASCII with escaping if UTF-8 fails
                with open(audio_results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                    logger.info("Saved audio results with ASCII encoding")
            except Exception as e2:
                logger.error(f"Failed to save audio results even with fallback encoding: {str(e2)}")
        
        # Also create a text file with formatted scene-by-scene dialogue
        dialogue_path = output_dir / "dialogue.txt"
        
        try:
            with open(dialogue_path, 'w', encoding='utf-8') as f:
                f.write("VIDEO DIALOGUE TRANSCRIPT\n")
                f.write("=======================\n\n")
            
                for scene_idx, scene in results["scenes"].items():
                    scene_num = int(scene_idx) + 1
                    scene_start = self._format_time(scene.get("start_time", 0))
                    scene_end = self._format_time(scene.get("end_time", 0))
                
                f.write(f"SCENE {scene_num} [{scene_start} - {scene_end}]\n")
                f.write("-----------------------------------\n")
                
                    # Handle different scene data formats
                if "segments" in scene:
                        segments = scene["segments"]
                        for segment in segments:
                            speaker = segment.get("speaker", "Unknown")
                            text = segment.get("text", "").strip()
                            time_start = self._format_time(segment.get("start", 0))
                            
                            f.write(f"{speaker} [{time_start}]: {text}\n")
                elif "dialogue" in scene:
                        # Handle new format with dialogue array
                        for segment in scene.get("dialogue", []):
                            speaker = segment.get("speaker", "Unknown")
                            text = segment.get("text", "").strip()
                            time_start = self._format_time(segment.get("start", 0))
                    
                        f.write(f"{speaker} [{time_start}]: {text}\n")
                    
                f.write("\n")
                
            logger.info(f"Saved dialogue transcript to {dialogue_path}")
        except Exception as e:
            logger.error(f"Error saving dialogue transcript: {str(e)}")
            logger.warning("Transcript save failed, continuing without it")
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_timestamp(self, start_time: float, end_time: float) -> str:
        """
        Format time range as readable timestamp.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Formatted timestamp string (HH:MM:SS - HH:MM:SS)
        """
        return f"{self._format_time(start_time)} - {self._format_time(end_time)}"
        
    def _create_dummy_diarization(self) -> Dict:
        """Create dummy diarization results when actual diarization fails."""
        return {
            "speakers": [],
            "num_speakers": 0
        }

    def _split_on_speech_gaps(self, audio_path: str) -> List[str]:
        """
        Split audio into speech-containing chunks using Silero VAD, each ≤ 30s.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            List of file paths to .wav chunks containing speech.
        """
        try:
            import torch
            import torchaudio
            # Try to import Silero VAD from whisperx or silero
            try:
                from whisperx.vad import SileroVAD
                vad_model = SileroVAD(self.device)
                get_speech_timestamps = vad_model.get_speech_timestamps
            except ImportError:
                try:
                    from silero import silero_vad
                    vad_model, utils = silero_vad.get_silero_vad_model()
                    get_speech_timestamps = utils[0]
                except ImportError:
                    logger.error("Silero VAD is required for speech splitting.")
                    return []
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
            wav = wav.squeeze().numpy()
            # Run VAD
            if callable(get_speech_timestamps):
                speech_timestamps = get_speech_timestamps(torch.tensor(wav), vad_model, sampling_rate=sr)
            else:
                speech_timestamps = get_speech_timestamps(torch.tensor(wav), sampling_rate=sr)
            # Merge close segments and split into ≤30s chunks
            max_chunk = 30.0  # seconds
            chunks = []
            for seg in speech_timestamps:
                start = seg['start'] / sr
                end = seg['end'] / sr
                # Split long segments
                seg_len = end - start
                seg_start = start
                while seg_len > max_chunk:
                    seg_end = seg_start + max_chunk
                    chunks.append((seg_start, seg_end))
                    seg_start = seg_end
                    seg_len = end - seg_start
                chunks.append((seg_start, end))
            # Save chunks to temp files
            chunk_paths = []
            for i, (start, end) in enumerate(chunks):
                temp_fd, temp_path = tempfile.mkstemp(suffix=f'_chunk_{i}.wav')
                os.close(temp_fd)
                sf.write(temp_path, wav[int(start*sr):int(end*sr)], sr)
                chunk_paths.append(temp_path)
            logger.info(f"Split audio into {len(chunk_paths)} speech chunks (≤30s each)")
            return chunk_paths
        except Exception as e:
            logger.error(f"Error in _split_on_speech_gaps: {e}")
            return []

    def _split_on_speech_gaps_with_diarization(self, audio_path: str, auth_token: str = None) -> List[Tuple[float, float, str]]:
        """
        Split audio into speech-containing chunks using WebRTC VAD + Silero VAD, then refine segments using per-segment speaker diarization.

        Args:
            audio_path: Path to the input audio file.
            auth_token: HuggingFace authentication token for pyannote.audio.

        Returns:
            List of tuples (start_time, end_time, speaker_id) for each refined segment.
        """
        try:
            # Step 1: WebRTC VAD coarse segmentation
            def read_wave(path):
                with wave.open(path, 'rb') as wf:
                    if wf.getnchannels() != 1:
                        raise ValueError("Audio must be mono")
                    sample_rate = wf.getframerate()
                    audio = wf.readframes(wf.getnframes())
                return audio, sample_rate

            def frame_generator(frame_duration_ms, audio, sample_rate):
                n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
                offset = 0
                timestamp = 0.0
                duration = (float(n) / sample_rate) / 2.0
                while offset + n < len(audio):
                    yield audio[offset:offset + n], timestamp
                    timestamp += duration
                    offset += n

            def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
                num_padding_frames = int(padding_duration_ms / frame_duration_ms)
                ring_buffer = collections.deque(maxlen=num_padding_frames)
                triggered = False
                voiced_segments = []
                segment_start = 0.0
                for frame, timestamp in frames:
                    is_speech = vad.is_speech(frame, sample_rate)
                    if not triggered:
                        ring_buffer.append((frame, timestamp, is_speech))
                        num_voiced = sum(f[2] for f in ring_buffer)
                        if num_voiced > 0.9 * ring_buffer.maxlen:
                            triggered = True
                            segment_start = ring_buffer[0][1]
                    else:
                        ring_buffer.append((frame, timestamp, is_speech))
                        num_voiced = sum(f[2] for f in ring_buffer)
                        if num_voiced < 0.1 * ring_buffer.maxlen:
                            segment_end = ring_buffer[-1][1]
                            voiced_segments.append((segment_start, segment_end))
                            triggered = False
                            ring_buffer.clear()
                if triggered and ring_buffer:
                    segment_end = ring_buffer[-1][1]
                    voiced_segments.append((segment_start, segment_end))
                return voiced_segments

            audio, sample_rate = read_wave(audio_path)
            vad = webrtcvad.Vad(1)
            frames = frame_generator(30, audio, sample_rate)
            segments = vad_collector(sample_rate, 30, 300, vad, frames)

            # Step 2: Refine with Silero VAD
            import torchaudio
            from silero_vad import read_audio as silero_read_audio, get_speech_timestamps, load_silero_vad
            wav = silero_read_audio(audio_path, sampling_rate=16000)
            final_segments = []
            for start, end in segments:
                start_sample = int(start * 16000)
                end_sample = int(end * 16000)
                segment_wav = wav[start_sample:end_sample]
                model = load_silero_vad()
                silero_timestamps = get_speech_timestamps(
                    segment_wav,
                    model,
                    threshold=0.16,
                    sampling_rate=16000,
                    visualize_probs=False,
                    min_speech_duration_ms=200,
                    return_seconds=True,
                    min_silence_duration_ms=1000
                )
                for speech in silero_timestamps:
                    abs_start = start + speech['start']
                    abs_end = start + speech['end']
                    final_segments.append([abs_start, abs_end])

            # Step 3: Merge short/adjacent segments
            def merge_vad_segments(segments, gap_threshold=1.0, short_segment_threshold=2.0, max_segments_merged=3):
                if not segments:
                    return []
                merged = []
                current_start = None
                current_end = None
                current_count = 0
                def finalize_if_active():
                    if current_start is not None and current_end is not None:
                        merged.append([current_start, current_end])
                for seg in segments:
                    seg_start, seg_end = seg
                    seg_duration = seg_end - seg_start
                    if seg_duration >= short_segment_threshold:
                        finalize_if_active()
                        current_start = None
                        current_end = None
                        current_count = 0
                        merged.append([seg_start, seg_end])
                        continue
                    if current_start is None:
                        current_start = seg_start
                        current_end = seg_end
                        current_count = 1
                    else:
                        gap = seg_start - current_end
                        if gap < gap_threshold and current_count < max_segments_merged:
                            current_end = max(current_end, seg_end)
                            current_count += 1
                        else:
                            finalize_if_active()
                            current_start = seg_start
                            current_end = seg_end
                            current_count = 1
                finalize_if_active()
                return merged
            vad_segments = merge_vad_segments(final_segments)

            # Step 4: Per-segment diarization and speaker merge
            from pyannote.audio import Pipeline
            from pydub import AudioSegment
            import tempfile
            import time
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            audio = AudioSegment.from_file(audio_path)
            result_segments = []
            for vad_start, vad_end in vad_segments:
                segment_duration_ms = (vad_end - vad_start) * 1000
                segment_audio = audio[vad_start * 1000:vad_end * 1000]
                if segment_duration_ms < 500:
                    result_segments.append([vad_start, vad_end, None])
                    continue
                tmp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file_path = tmp_file.name
                        segment_audio.export(tmp_file_path, format='wav')
                    diarization = pipeline(tmp_file_path)
                    segment_diarization = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        adjusted_start = vad_start + turn.start
                        adjusted_end = vad_start + turn.end
                        segment_diarization.append((adjusted_start, adjusted_end, speaker))
                    unique_speakers = set(speaker for _, _, speaker in segment_diarization)
                    if len(segment_diarization) == 0:
                        result_segments.append([vad_start, vad_end, None])
                    elif len(unique_speakers) == 1:
                        result_segments.append([vad_start, vad_end, list(unique_speakers)[0]])
                    else:
                        split_points = set([vad_start, vad_end])
                        for diar_start, diar_end, _ in segment_diarization:
                            if vad_start < diar_start < vad_end:
                                split_points.add(diar_start)
                            if vad_start < diar_end < vad_end:
                                split_points.add(diar_end)
                        sorted_split_points = sorted(list(split_points))
                        temp_segments = []
                        for i in range(len(sorted_split_points) - 1):
                            start = sorted_split_points[i]
                            end = sorted_split_points[i + 1]
                            subsegment_diar = [
                                diar for diar in segment_diarization
                                if diar[0] < end and diar[1] > start
                            ]
                            if subsegment_diar:
                                speaker_overlap = {}
                                for diar_start, diar_end, speaker in subsegment_diar:
                                    overlap_start = max(start, diar_start)
                                    overlap_end = min(end, diar_end)
                                    overlap_duration = overlap_end - overlap_start
                                    if speaker in speaker_overlap:
                                        speaker_overlap[speaker] += overlap_duration
                                    else:
                                        speaker_overlap[speaker] = overlap_duration
                                dominant_speaker = max(speaker_overlap, key=speaker_overlap.get)
                                temp_segments.append([start, end, dominant_speaker])
                            else:
                                temp_segments.append([start, end, None])
                        if temp_segments:
                            merged_segments = [temp_segments[0]]
                            for segment in temp_segments[1:]:
                                if segment[2] == merged_segments[-1][2]:
                                    merged_segments[-1][1] = segment[1]
                                else:
                                    merged_segments.append(segment)
                            result_segments.extend(merged_segments)
                except Exception as e:
                    logger.error(f"Error processing segment {vad_start}-{vad_end}: {e}")
                    result_segments.append([vad_start, vad_end, None])
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        try:
                            time.sleep(0.1)
                            os.unlink(tmp_file_path)
                        except Exception as e:
                            logger.warning(f"Could not delete temporary file: {tmp_file_path}")
            result_segments = [segment for segment in result_segments if segment[2] is not None]
            return result_segments
        except Exception as e:
            logger.error(f"Error in _split_on_speech_gaps_with_diarization: {e}")
            return []

    def _asr_ensemble(self, audio_chunks: List[str]) -> Dict:
        """
        Run ASR ensemble (Whisper-Large, Faster-Whisper, Google STT) on each audio chunk and fuse results with majority-vote token strategy.

        Args:
            audio_chunks: List of paths to audio chunk files (wav).

        Returns:
            Dictionary with ensemble transcription results, including per-chunk and fused outputs.
        """
        import difflib
        results = {
            "chunks": [],
            "fused_text": "",
            "fused_tokens": []
        }
        # Helper to run each engine
        def run_whisper_large(chunk):
            try:
                import openai.whisper
                model = openai.whisper.load_model("large-v3", device=self.device)
                out = model.transcribe(chunk, word_timestamps=True)
                return [w["word"] for s in out.get("segments", []) for w in s.get("words", [])]
            except Exception as e:
                return None
        def run_faster_whisper(chunk):
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel("large-v3", device=self.device, compute_type="float16" if self.device=="cuda" else "int8")
                segments, _ = model.transcribe(chunk, word_timestamps=True)
                return [w.word for s in segments for w in getattr(s, "words", [])]
            except Exception as e:
                return None
        def run_google_stt(chunk):
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile(chunk) as source:
                    audio = r.record(source)
                text = r.recognize_google(audio)
                return text.split()  # crude tokenization
            except Exception as e:
                return None
        # Process each chunk
        all_tokens = []
        for chunk_path in audio_chunks:
            chunk_result = {"chunk": chunk_path, "whisper": None, "faster_whisper": None, "google": None}
            whisper_tokens = run_whisper_large(chunk_path)
            faster_tokens = run_faster_whisper(chunk_path)
            google_tokens = run_google_stt(chunk_path)
            chunk_result["whisper"] = whisper_tokens
            chunk_result["faster_whisper"] = faster_tokens
            chunk_result["google"] = google_tokens
            # Align all available token lists
            token_lists = [t for t in [whisper_tokens, faster_tokens, google_tokens] if t]
            if not token_lists:
                chunk_result["fused"] = []
                results["chunks"].append(chunk_result)
                continue
            # Use the longest as reference
            ref = max(token_lists, key=len)
            aligned = []
            for i, token in enumerate(ref):
                votes = [tokens[i] if i < len(tokens) else None for tokens in token_lists]
                # Majority vote
                vote_counts = {}
                for v in votes:
                    if v is not None:
                        vote_counts[v] = vote_counts.get(v, 0) + 1
                if vote_counts:
                    fused = max(vote_counts.items(), key=lambda x: x[1])[0]
                else:
                    fused = token
                aligned.append(fused)
            chunk_result["fused"] = aligned
            all_tokens.extend(aligned)
            results["chunks"].append(chunk_result)
        results["fused_tokens"] = all_tokens
        results["fused_text"] = " ".join(all_tokens)
        return results

    def _forced_align_with_gentle(self, audio_path: str, transcript_path: str) -> Dict:
        """
        Perform forced alignment using Gentle.

        Args:
            audio_path: Path to the audio file (wav)
            transcript_path: Path to the transcript or subtitle file (txt)

        Returns:
            Dictionary with alignment results from Gentle
        """
        # Assumes Gentle is running as a local server (default: http://localhost:8765)
        url = "http://localhost:8765/transcriptions?async=false"
        with open(audio_path, "rb") as audio_file, open(transcript_path, "r", encoding="utf-8") as transcript_file:
            files = {
                "audio": ("audio.wav", audio_file, "audio/wav"),
                "transcript": ("transcript.txt", transcript_file, "text/plain")
            }
            response = requests.post(url, files=files)
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"Gentle alignment failed: {response.status_code} {response.text}")

# Helper function
def process_video_audio(video_path: str, scenes: List[Dict], output_dir: str, config: Dict = None) -> Dict:
    """
    Convenience function to process audio from a video file.
    
    Args:
        video_path: Path to the video file
        scenes: List of scene dictionaries
        output_dir: Directory to save results
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with audio processing results
    """
    processor = AudioProcessor(config)
    return processor.process_video_audio(video_path, scenes, output_dir) 