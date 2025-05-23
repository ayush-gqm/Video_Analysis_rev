"""
Audio transcription and diarization module.

This module handles transcription of audio from videos and diarization of speakers.
"""

import os
import logging
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import subprocess
import numpy as np

# Check if running in CPU-only mode
CPU_ONLY_MODE = os.environ.get("SCREENWRITER_CPU_ONLY", "0") == "1"

# Force TensorFlow to be disabled when in CPU-only mode
if CPU_ONLY_MODE:
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["KERAS_BACKEND"] = "torch"
    logging.warning("Running in CPU-only mode. TensorFlow and WhisperX disabled.")

# Conditional imports to avoid TensorFlow dependencies in CPU-only mode
WHISPERX_AVAILABLE = False
TRANSFORMERS_ASR_AVAILABLE = False

if not CPU_ONLY_MODE:
    try:
        import torch
        import whisperx
        WHISPERX_AVAILABLE = True
    except ImportError:
        WHISPERX_AVAILABLE = False
        logging.warning("WhisperX not available. Using basic transcription fallback.")

# Try importing minimal transcription tools that don't rely on TensorFlow
try:
    import torch
    
    # Only try to import transformers if we're not in CPU-only mode, or with extra precautions
    if not CPU_ONLY_MODE:
        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
        TRANSFORMERS_ASR_AVAILABLE = True
    else:
        # In CPU-only mode, avoid transformers entirely
        logging.warning("In CPU-only mode, skipping transformers imports")
        TRANSFORMERS_ASR_AVAILABLE = False
except ImportError:
    TRANSFORMERS_ASR_AVAILABLE = False
    logging.warning("Transformers ASR not available. Using even more basic transcription.")

from screenwriter.models import TranscriptLine, Timestamp
from screenwriter.config import AUDIO_PROCESSING


class AudioTranscriber:
    """
    Audio transcription and diarization class.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the audio transcriber.
        
        Args:
            device: Device to use for inference ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.config = AUDIO_PROCESSING
        
        # Set compute type based on device
        self.compute_type = self.config["whisperx"].get("compute_type", "float16")
        if device == "cpu" and self.compute_type == "float16":
            self.compute_type = "float32"  # Use float32 on CPU
            
        self.logger.info(f"Audio transcriber initialized on {device} with compute type {self.compute_type}")
        
        # Check for TensorFlow-free operation
        if CPU_ONLY_MODE:
            self.logger.info("Running in CPU-only mode - TensorFlow dependencies disabled")
        
    def transcribe_and_diarize(self, video_path: Union[str, Path]) -> List[TranscriptLine]:
        """
        Transcribe and diarize audio from a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of TranscriptLine objects
        """
        video_path = Path(video_path)
        self.logger.info(f"Transcribing and diarizing audio from {video_path}")
        
        # Choose transcription method based on availability and mode
        if WHISPERX_AVAILABLE and not CPU_ONLY_MODE:
            self.logger.info("Using WhisperX for transcription")
            return self._transcribe_with_whisperx(video_path)
        elif TRANSFORMERS_ASR_AVAILABLE and not CPU_ONLY_MODE:
            self.logger.info("Using Transformers for transcription (WhisperX unavailable)")
            return self._transcribe_with_transformers(video_path)
        else:
            self.logger.info("Using basic transcription (CPU-only mode or dependencies unavailable)")
            return self._fallback_transcription(video_path)
            
    def _transcribe_with_whisperx(self, video_path: Path) -> List[TranscriptLine]:
        """
        Transcribe and diarize using WhisperX with proper error handling.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of TranscriptLine objects
        """
        try:
            # Load audio
            self.logger.info("Loading audio from video")
            device = self.device
            audio = whisperx.load_audio(str(video_path))
            
            # Load model
            model_name = self.config["whisperx"].get("model_name", "large-v3")
            self.logger.info(f"Loading WhisperX model: {model_name}")
            
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=self.compute_type,
                language=self.config["whisperx"].get("language", "auto"),
            )
            
            # Transcribe audio
            self.logger.info("Transcribing audio")
            result = model.transcribe(
                audio, 
                batch_size=self.config["whisperx"].get("batch_size", 16)
            )
            
            # Align timestamps
            self.logger.info("Aligning timestamps")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device
            )
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                device
            )
            
            # Diarize audio (assign speakers)
            self.logger.info("Diarizing audio (assigning speakers)")
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=None,
                    device=device
                )
                
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=self.config["diarization"].get("min_speakers", 1),
                    max_speakers=self.config["diarization"].get("max_speakers", 10)
                )
                
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                self.logger.error(f"Error during diarization: {str(e)}")
                self.logger.warning("Continuing without speaker assignment")
            
            # Convert result to TranscriptLine objects
            transcript_lines = []
            for segment in result.get("segments", []):
                speaker = segment.get("speaker", "UNKNOWN")
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                if not text:
                    continue
                    
                line = TranscriptLine(
                    start_time=start,
                    end_time=end,
                    text=text,
                    speaker=speaker
                )
                transcript_lines.append(line)
                
            self.logger.info(f"Transcription complete: {len(transcript_lines)} lines")
            return transcript_lines
            
        except Exception as e:
            self.logger.error(f"Error in WhisperX transcription: {str(e)}")
            self.logger.warning("Falling back to basic transcription")
            return self._fallback_transcription(video_path)
            
    def _transcribe_with_transformers(self, video_path: Path) -> List[TranscriptLine]:
        """
        Transcribe using only Transformers (no WhisperX) with careful TensorFlow avoidance.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of TranscriptLine objects
        """
        try:
            # Extract audio from video
            temp_audio_path = self._extract_audio(video_path)
            
            # Load model
            self.logger.info("Loading Transformers ASR model with TensorFlow disabled")
            model_id = "openai/whisper-small" # Use smaller model to reduce memory usage
            
            # Use TensorFlow-free approach
            os.environ["TRANSFORMERS_NO_TF"] = "1"
            
            # Use CPU-specific optimizations
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=1,
                return_timestamps=True,
                torch_dtype=torch.float32,
                device=self.device,
            )
            
            # Transcribe
            self.logger.info("Transcribing with Transformers ASR")
            result = pipe(str(temp_audio_path), return_timestamps="word")
            
            # Convert result to TranscriptLine objects
            transcript_lines = []
            
            if "chunks" in result:
                # If result has chunks property
                for chunk in result["chunks"]:
                    start = chunk.get("timestamp", [0, 0])[0]
                    end = chunk.get("timestamp", [0, 0])[1]
                    text = chunk.get("text", "").strip()
                    
                    if not text:
                        continue
                        
                    line = TranscriptLine(
                        start_time=start,
                        end_time=end,
                        text=text,
                        speaker="SPEAKER_01"  # Unknown speaker
                    )
                    transcript_lines.append(line)
            else:
                # For standard result format
                text = result.get("text", "")
                chunks = result.get("chunks", [])
                
                for i, chunk in enumerate(chunks):
                    start_time = chunk.get("timestamp", [0, 0])[0]
                    end_time = chunk.get("timestamp", [0, 0])[1]
                    chunk_text = chunk.get("text", "").strip()
                    
                    if not chunk_text:
                        continue
                        
                    line = TranscriptLine(
                        start_time=start_time,
                        end_time=end_time,
                        text=chunk_text,
                        speaker="SPEAKER_01"  # Default speaker without diarization
                    )
                    transcript_lines.append(line)
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            self.logger.info(f"Transformers transcription complete: {len(transcript_lines)} lines")
            return transcript_lines
            
        except Exception as e:
            self.logger.error(f"Error in Transformers transcription: {str(e)}")
            self.logger.warning("Falling back to basic transcription")
            return self._fallback_transcription(video_path)
            
    def _fallback_transcription(self, video_path: Path) -> List[TranscriptLine]:
        """
        Very basic fallback transcription using FFmpeg to extract subtitles if available.
        If no subtitles, returns a minimal placeholder transcript.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of TranscriptLine objects (may be empty or placeholder)
        """
        self.logger.warning("Using basic fallback transcription - results will be limited")
        transcript_lines = []
        
        try:
            # Try to extract subtitles with FFmpeg
            subtitle_path = self._extract_subtitles(video_path)
            
            if subtitle_path and os.path.exists(subtitle_path):
                # Parse SRT file
                self.logger.info(f"Using extracted subtitles from {subtitle_path}")
                with open(subtitle_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple SRT parsing
                blocks = content.strip().split('\n\n')
                for block in blocks:
                    lines = block.split('\n')
                    if len(lines) >= 3:
                        # Parse timecode like 00:00:20,000 --> 00:00:22,000
                        timecode = lines[1]
                        times = timecode.split(' --> ')
                        if len(times) == 2:
                            start_time = self._parse_srt_time(times[0])
                            end_time = self._parse_srt_time(times[1])
                            text = ' '.join(lines[2:])
                            
                            line = TranscriptLine(
                                start_time=start_time,
                                end_time=end_time,
                                text=text,
                                speaker="UNKNOWN"
                            )
                            transcript_lines.append(line)
                
                # Clean up
                os.remove(subtitle_path)
                
                self.logger.info(f"Parsed {len(transcript_lines)} lines from subtitles")
                if len(transcript_lines) > 0:
                    return transcript_lines
            
            # If no subtitles were found or parsed, create placeholder
            self.logger.info("No valid subtitles found, creating placeholder transcript")
            
            # Create simple placeholder transcript with basic timing
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Create some segments
            segment_count = max(1, int(duration / 30))  # One segment per 30 seconds
            for i in range(segment_count):
                start_time = i * 30
                end_time = min((i + 1) * 30, duration)
                
                line = TranscriptLine(
                    start_time=start_time,
                    end_time=end_time,
                    text=f"[Scene content {i+1}]",
                    speaker=f"SPEAKER_{(i % 3) + 1:02d}"  # Alternate between 3 speakers
                )
                transcript_lines.append(line)
        
        except Exception as e:
            self.logger.error(f"Error in fallback transcription: {str(e)}")
            # Create a single placeholder transcript
            line = TranscriptLine(
                start_time=0,
                end_time=60,
                text="[Transcription unavailable]",
                speaker="UNKNOWN"
            )
            transcript_lines.append(line)
        
        self.logger.info(f"Fallback transcription complete: {len(transcript_lines)} lines")
        return transcript_lines
        
    def _extract_audio(self, video_path: Path) -> str:
        """Extract audio from video to a temporary WAV file."""
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            self.logger.info(f"Extracting audio from {video_path} to {temp_audio_path}")
            subprocess.run([
                'ffmpeg', '-y', '-i', str(video_path), 
                '-ac', '1', '-ar', '16000', '-vn', temp_audio_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return temp_audio_path
        except Exception as e:
            self.logger.error(f"Error extracting audio: {str(e)}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            raise
            
    def _extract_subtitles(self, video_path: Path) -> Optional[str]:
        """Extract subtitles from video if available."""
        temp_srt = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
        temp_srt_path = temp_srt.name
        temp_srt.close()
        
        try:
            self.logger.info(f"Attempting to extract subtitles from {video_path}")
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(video_path), 
                '-map', '0:s:0', temp_srt_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(temp_srt_path) and os.path.getsize(temp_srt_path) > 0:
                self.logger.info(f"Subtitles extracted to {temp_srt_path}")
                return temp_srt_path
            else:
                self.logger.warning("No subtitles found in video")
                if os.path.exists(temp_srt_path):
                    os.remove(temp_srt_path)
                return None
        except Exception as e:
            self.logger.warning(f"Error extracting subtitles: {str(e)}")
            if os.path.exists(temp_srt_path):
                os.remove(temp_srt_path)
            return None
            
    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT format time string into seconds."""
        # Format: 00:00:20,000
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0 