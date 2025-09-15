"""
Audio Transcriber Agent
Handles real-time speech-to-text conversion using Whisper ASR with academic vocabulary enhancement.
"""

import whisper
import torch
import numpy as np
import soundfile as sf
import os
from typing import Dict, Any, List, Tuple
import time
import re
from pathlib import Path

from .base_agent import BaseAgent
import config

class AudioTranscriberAgent(BaseAgent):
    """
    Agent responsible for converting audio to text with timestamp tracking.
    Uses OpenAI Whisper for high-quality transcription with academic content optimization.
    """
    
    def __init__(self):
        super().__init__("AudioTranscriber")
        self.model = None
        self.academic_vocabulary = self._load_academic_vocabulary()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model."""
        try:
            self.logger.info(f"Loading Whisper model: {config.WHISPER_MODEL}")
            self.model = whisper.load_model(config.WHISPER_MODEL)
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _load_academic_vocabulary(self) -> List[str]:
        """Load academic and technical vocabulary for enhanced recognition."""
        # Common academic terms that might be mispronounced
        vocabulary = [
            # Mathematics
            "polynomial", "derivative", "integral", "logarithm", "exponential",
            "coefficient", "variable", "equation", "theorem", "hypothesis",
            
            # Science
            "molecule", "chromosome", "mitochondria", "photosynthesis", "catalyst",
            "equilibrium", "entropy", "electromagnetic", "quantum", "photon",
            
            # Computer Science
            "algorithm", "recursion", "iteration", "complexity", "binary",
            "database", "encryption", "compilation", "debugging", "inheritance",
            
            # General Academic
            "methodology", "paradigm", "empirical", "correlation", "statistical",
            "hypothesis", "analysis", "synthesis", "evaluation", "implementation"
        ]
        return vocabulary
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process audio file and return transcription with timestamps.
        
        Args:
            input_data: Path to audio file or audio array
            
        Returns:
            Dict containing transcription, segments with timestamps, and metadata
        """
        start_time = time.time()
        
        try:
            # Handle different input types
            if isinstance(input_data, str):
                audio_path = Path(input_data)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {input_data}")
                audio_data = self._load_audio_file(str(audio_path))
            else:
                audio_data = input_data
            
            # Validate audio data before transcription
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("Audio data is empty or None")
            
            # Check for minimum audio length (at least 0.1 seconds)
            min_length = int(0.1 * config.AUDIO_SAMPLE_RATE)  # 0.1 seconds
            if len(audio_data) < min_length:
                raise ValueError(f"Audio too short: {len(audio_data)} samples (minimum: {min_length})")
            
            # Check for valid audio range
            if np.max(np.abs(audio_data)) == 0:
                raise ValueError("Audio data is silent (all zeros)")
            
            self.logger.info(f"Audio validation passed: {len(audio_data)} samples, max amplitude: {np.max(np.abs(audio_data)):.4f}")
            
            # Transcribe with Whisper
            result = self._transcribe_audio(audio_data)
            
            # Post-process for academic content
            enhanced_result = self._enhance_academic_content(result)
            
            # Calculate metrics
            processing_time = self.log_performance(start_time, "transcription")
            confidence = self._calculate_confidence(enhanced_result)
            
            # Store quality score for status reporting
            self.last_quality_score = min(confidence + 0.1, 1.0)
            
            return {
                "transcription": enhanced_result["text"],
                "segments": enhanced_result["segments"],
                "language": enhanced_result.get("language", "unknown"),
                "confidence": confidence,
                "processing_time": processing_time,
                "word_count": len(enhanced_result["text"].split()),
                "duration": enhanced_result.get("duration", 0),
                "output_quality": self.last_quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return {
                "transcription": "",
                "segments": [],
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _load_audio_file(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file with robust error handling."""
        try:
            self.logger.info(f"Loading audio file: {file_path}")
            
            # Try to load the audio file with multiple methods for robustness
            try:
                # First try with soundfile (handles most formats well)
                audio, sample_rate = sf.read(file_path)
            except Exception as sf_error:
                self.logger.warning(f"soundfile failed, trying librosa: {sf_error}")
                try:
                    # Fallback to librosa which handles more formats and metadata issues
                    import librosa
                    audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
                except Exception as librosa_error:
                    self.logger.warning(f"librosa failed, trying pydub: {librosa_error}")
                    try:
                        # Final fallback to pydub which can handle corrupted metadata
                        from pydub import AudioSegment
                        import io
                        
                        # Load with pydub (handles metadata errors better)
                        audio_segment = AudioSegment.from_file(file_path)
                        
                        # Convert to numpy array
                        audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                        sample_rate = audio_segment.frame_rate
                        
                        # Handle stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                    except Exception as pydub_error:
                        self.logger.error(f"All audio loading methods failed: {pydub_error}")
                        raise Exception(f"Could not load audio file: {file_path}. Tried soundfile, librosa, and pydub.")
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Ensure we have valid audio data
            if len(audio) == 0:
                raise ValueError("Audio file resulted in empty array")
            
            # Ensure float32 dtype for Whisper compatibility
            audio = audio.astype(np.float32)
            
            # Check for silent audio
            if np.max(np.abs(audio)) == 0:
                self.logger.warning("Audio appears to be silent")
                # Don't raise error here, let the validation in process() handle it
            
            # Normalize to [-1, 1] range if not silent
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Remove any NaN or infinite values
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Resample to 16kHz if needed (Whisper requirement)
            if sample_rate != config.AUDIO_SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=config.AUDIO_SAMPLE_RATE)
                audio = audio.astype(np.float32)  # Ensure dtype after resampling
            
            self.logger.info(f"Audio loaded successfully: {len(audio)} samples at {config.AUDIO_SAMPLE_RATE}Hz")
            return audio
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {e}")
            # Log additional information for debugging
            self.logger.error(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                self.logger.error(f"File size: {os.path.getsize(file_path)} bytes")
            raise
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio using Whisper with academic optimizations."""
        try:
            # Ensure model is loaded
            if self.model is None:
                raise RuntimeError("Whisper model not initialized. Call _initialize_model() first.")
            
            # Final validation of audio data
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("Audio data is empty for transcription")
            
            # Ensure proper dtype for Whisper
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to prevent overflow
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Ensure audio is not all zeros (which can cause tensor issues)
            if np.max(np.abs(audio_data)) == 0:
                # Return empty result for silent audio
                self.logger.warning("Audio is silent, returning empty transcription")
                return {
                    "text": "",
                    "segments": [],
                    "language": "en",
                    "duration": len(audio_data) / config.AUDIO_SAMPLE_RATE
                }
            
            # Pad audio if too short (Whisper needs minimum length)
            min_samples = config.AUDIO_SAMPLE_RATE  # 1 second minimum
            if len(audio_data) < min_samples:
                self.logger.info(f"Padding audio from {len(audio_data)} to {min_samples} samples")
                audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), mode='constant')
            
            self.logger.info(f"Transcribing audio: {len(audio_data)} samples, amplitude range: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_data,
                word_timestamps=True,
                initial_prompt="This is an academic lecture with technical terminology.",
                temperature=0.0,  # Deterministic output
                fp16=False  # Use FP32 on CPU
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def _enhance_academic_content(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance transcription for academic content."""
        enhanced_result = result.copy()
        
        # Fix common academic term transcription errors
        text = result["text"]
        
        # DISABLED: Apply academic vocabulary corrections - too aggressive
        # for term in self.academic_vocabulary:
        #     # Simple fuzzy matching for common mispronunciations
        #     patterns = [
        #         (rf'\b{re.escape(term.lower()[:3])}[a-z]*\b', term),
        #         (rf'\b{re.escape(term[:4])}[a-z]*\b', term)
        #     ]
        #     
        #     for pattern, replacement in patterns:
        #         text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix common mathematical expressions
        math_fixes = {
            r'\bx squared\b': 'x²',
            r'\bx cubed\b': 'x³',
            r'\balpha\b': 'α',
            r'\bbeta\b': 'β',
            r'\bgamma\b': 'γ',
            r'\bdelta\b': 'δ',
            r'\btheta\b': 'θ',
            r'\bpi\b': 'π',
            r'\bsigma\b': 'σ'
        }
        
        for pattern, replacement in math_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        enhanced_result["text"] = text
        
        # Enhance segments with corrected text
        if "segments" in enhanced_result:
            for segment in enhanced_result["segments"]:
                segment_text = segment["text"]
                for pattern, replacement in math_fixes.items():
                    segment_text = re.sub(pattern, replacement, segment_text, flags=re.IGNORECASE)
                segment["text"] = segment_text
        
        return enhanced_result
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall confidence score from Whisper result."""
        if "segments" not in result:
            return 0.0
        
        total_confidence = 0.0
        total_duration = 0.0
        
        for segment in result["segments"]:
            if "avg_logprob" in segment and "end" in segment and "start" in segment:
                duration = segment["end"] - segment["start"]
                confidence = np.exp(segment["avg_logprob"])
                total_confidence += confidence * duration
                total_duration += duration
        
        return total_confidence / total_duration if total_duration > 0 else 0.0
    
    def process_streaming(self, audio_stream) -> List[Dict[str, Any]]:
        """Process streaming audio in chunks (future enhancement)."""
        # Placeholder for real-time streaming implementation
        chunks = []
        chunk_size = config.CHUNK_LENGTH_S * config.AUDIO_SAMPLE_RATE
        
        for i in range(0, len(audio_stream), chunk_size):
            chunk = audio_stream[i:i + chunk_size]
            if len(chunk) > 0:
                result = self.process(chunk)
                chunks.append(result)
        
        return chunks
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return [".wav", ".mp3", ".m4a", ".flac", ".ogg"]