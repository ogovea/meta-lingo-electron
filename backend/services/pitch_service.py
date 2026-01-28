"""
TorchCrepe Pitch Extraction Service
Uses CREPE model for F0 (fundamental frequency) extraction
"""

import os
import sys
import logging
import torch
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from config import MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TorchCrepe module path - 使用 config.py 中的 MODELS_DIR
TORCHCREPE_PATH = str(MODELS_DIR / "multimodal_analyzer" / "torchcrepe-master")
TORCHCREPE_ASSETS_PATH = os.path.join(TORCHCREPE_PATH, "torchcrepe", "assets")

# Add torchcrepe to Python path if not already there
if TORCHCREPE_PATH not in sys.path:
    sys.path.insert(0, TORCHCREPE_PATH)


@dataclass
class PitchResult:
    """Pitch extraction result"""
    enabled: bool
    hop_length_ms: float
    sample_rate: int
    fmin: float
    fmax: float
    f0: List[float]
    periodicity: List[float]
    times: List[float]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PitchService:
    """Service for pitch extraction using TorchCrepe"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = 16000  # TorchCrepe requires 16kHz
        self._initialized = False
        self._torchcrepe = None
    
    def initialize(self) -> bool:
        """Initialize the TorchCrepe model"""
        if self._initialized:
            return True
        
        try:
            logger.info(f"Loading TorchCrepe from: {TORCHCREPE_PATH}")
            
            # Check if torchcrepe path exists
            if not os.path.exists(TORCHCREPE_PATH):
                logger.error(f"TorchCrepe path does not exist: {TORCHCREPE_PATH}")
                return False
            
            # Check if model weights exist
            full_model_path = os.path.join(TORCHCREPE_ASSETS_PATH, "full.pth")
            if not os.path.exists(full_model_path):
                logger.error(f"TorchCrepe model not found: {full_model_path}")
                return False
            
            # Import torchcrepe
            import torchcrepe
            self._torchcrepe = torchcrepe
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Pre-load model by doing a dummy prediction
            # This ensures the model is loaded into memory
            logger.info("Pre-loading TorchCrepe full model...")
            dummy_audio = torch.zeros(1, 16000)  # 1 second of silence
            _ = torchcrepe.predict(
                dummy_audio,
                16000,
                hop_length=160,  # 10ms at 16kHz
                fmin=50,
                fmax=550,
                model='full',
                batch_size=512,
                device=self.device,
                return_periodicity=True
            )
            
            logger.info("TorchCrepe model loaded successfully")
            self._initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import torchcrepe: {e}")
            logger.error("Make sure resampy and other dependencies are installed")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TorchCrepe: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and resample audio to 16kHz"""
        import resampy
        
        # Use soundfile directly to avoid torchaudio/torchcodec issues
        data, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            data = resampy.resample(data, sr, self.sample_rate)
        
        # Convert to torch tensor with shape (1, samples)
        waveform = torch.from_numpy(data).unsqueeze(0)
        
        return waveform, self.sample_rate
    
    def extract_pitch(self, audio_path: str, hop_length_ms: float = 10.0,
                      fmin: float = 50.0, fmax: float = 550.0,
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Extract pitch (F0) from audio file
        
        Args:
            audio_path: Path to audio file
            hop_length_ms: Hop length in milliseconds (default 10ms)
            fmin: Minimum frequency in Hz (default 50Hz for speech)
            fmax: Maximum frequency in Hz (default 550Hz for speech)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing pitch extraction results
        """
        if not self._initialized:
            if not self.initialize():
                return {"success": False, "error": "Failed to initialize pitch model"}
        
        try:
            if progress_callback:
                progress_callback(10, "Loading audio for pitch extraction...")
            
            # Load audio
            waveform, sr = self._load_audio(audio_path)
            duration = waveform.shape[1] / sr
            
            if progress_callback:
                progress_callback(30, "Extracting pitch...")
            
            # Calculate hop length in samples
            hop_length = int(sr * hop_length_ms / 1000)
            
            # Extract pitch using TorchCrepe
            pitch, periodicity = self._torchcrepe.predict(
                waveform,
                sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                model='full',
                batch_size=512,
                device=self.device,
                return_periodicity=True,
                decoder=self._torchcrepe.decode.viterbi
            )
            
            if progress_callback:
                progress_callback(80, "Processing pitch data...")
            
            # Convert to numpy and then to list
            pitch_np = pitch.squeeze().cpu().numpy()
            periodicity_np = periodicity.squeeze().cpu().numpy()
            
            # Generate time array
            num_frames = len(pitch_np)
            times = [round(i * hop_length_ms / 1000, 4) for i in range(num_frames)]
            
            # Convert NaN to 0 for JSON serialization
            f0_list = []
            for p in pitch_np:
                if np.isnan(p):
                    f0_list.append(0.0)
                else:
                    f0_list.append(round(float(p), 2))
            
            periodicity_list = [round(float(p), 4) for p in periodicity_np]
            
            if progress_callback:
                progress_callback(100, "Pitch extraction complete")
            
            logger.info(f"Extracted pitch: {num_frames} frames, duration {duration:.2f}s")
            
            return {
                "success": True,
                "enabled": True,
                "hop_length_ms": hop_length_ms,
                "sample_rate": sr,
                "fmin": fmin,
                "fmax": fmax,
                "f0": f0_list,
                "periodicity": periodicity_list,
                "times": times,
                "duration": duration,
                "num_frames": num_frames
            }
            
        except Exception as e:
            logger.error(f"Pitch extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


# Singleton instance
_pitch_service: Optional[PitchService] = None


def get_pitch_service() -> PitchService:
    """Get or create the singleton PitchService instance"""
    global _pitch_service
    if _pitch_service is None:
        _pitch_service = PitchService()
    return _pitch_service
