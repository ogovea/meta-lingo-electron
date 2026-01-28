"""
Wav2Vec2 Forced Alignment Service
Uses Wav2Vec2 model for CTC-based forced alignment of audio with transcript
Reference: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
"""

import os
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

# Model path - 使用 config.py 中的 MODELS_DIR
WAV2VEC2_MODEL_PATH = str(MODELS_DIR / "multimodal_analyzer" / "wav2vec2-base-960h")


@dataclass
class AlignedWord:
    """Word-level alignment result"""
    word: str
    start: float
    end: float
    score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlignedChar:
    """Character-level alignment result"""
    char: str
    start: float
    end: float
    score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AlignmentService:
    """Service for forced alignment using Wav2Vec2"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.sample_rate = 16000
        self.labels = None
        self.dictionary = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the Wav2Vec2 model for forced alignment"""
        if self._initialized:
            return True
        
        try:
            logger.info(f"Loading Wav2Vec2 model from: {WAV2VEC2_MODEL_PATH}")
            
            # Check if model path exists
            if not os.path.exists(WAV2VEC2_MODEL_PATH):
                logger.error(f"Model path does not exist: {WAV2VEC2_MODEL_PATH}")
                return False
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load model using transformers from local path
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            logger.info("Loading Wav2Vec2 model with transformers...")
            self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_PATH)
            self.model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
            
            # Get vocabulary/labels from processor
            vocab = self.processor.tokenizer.get_vocab()
            # Sort by index to get labels in order
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            self.labels = [item[0] for item in sorted_vocab]
            self.dictionary = vocab
            self.sample_rate = 16000  # Wav2Vec2 uses 16kHz
            
            logger.info(f"Wav2Vec2 model loaded successfully from local path")
            logger.info(f"Vocabulary size: {len(self.labels)}")
            logger.info(f"Sample rate: {self.sample_rate}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2 model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and resample audio to model's sample rate"""
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
    
    def _get_emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """Get frame-wise emission probabilities from model"""
        with torch.inference_mode():
            # Flatten waveform for processor (expects 1D array)
            waveform_np = waveform.squeeze().numpy()
            
            # Process with Wav2Vec2Processor
            inputs = self.processor(
                waveform_np, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(self.device)
            
            # Get logits from model
            outputs = self.model(input_values)
            logits = outputs.logits
            
            # Apply log softmax to get log probabilities
            emissions = torch.log_softmax(logits, dim=-1)
            
        return emissions[0].cpu().detach()
    
    def _get_trellis(self, emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
        """Generate alignment probability trellis"""
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        
        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1:, 0] = float("inf")
        
        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis
    
    def _backtrack(self, trellis: torch.Tensor, emission: torch.Tensor, 
                   tokens: List[int], blank_id: int = 0) -> List[Dict]:
        """Backtrack through trellis to find alignment path"""
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        
        path = [{"token_index": j, "time_index": t, "score": emission[t, blank_id].exp().item()}]
        
        while j > 0:
            assert t > 0
            
            # Figure out if the current position was stay or change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]
            
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change
            
            t -= 1
            if changed > stayed:
                j -= 1
            
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append({"token_index": j, "time_index": t, "score": prob})
        
        # Fill up the rest
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append({"token_index": j, "time_index": t - 1, "score": prob})
            t -= 1
        
        return path[::-1]
    
    def _merge_repeats(self, path: List[Dict], transcript: str) -> List[Dict]:
        """Merge repeated labels into segments"""
        i1, i2 = 0, 0
        segments = []
        
        while i1 < len(path):
            while i2 < len(path) and path[i1]["token_index"] == path[i2]["token_index"]:
                i2 += 1
            
            score = sum(path[k]["score"] for k in range(i1, i2)) / (i2 - i1)
            segments.append({
                "label": transcript[path[i1]["token_index"]],
                "start": path[i1]["time_index"],
                "end": path[i2 - 1]["time_index"] + 1,
                "score": score
            })
            i1 = i2
        
        return segments
    
    def _merge_words(self, segments: List[Dict], separator: str = "|") -> List[Dict]:
        """Merge character segments into words"""
        words = []
        i1, i2 = 0, 0
        
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2]["label"] == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg["label"] for seg in segs])
                    total_length = sum(seg["end"] - seg["start"] for seg in segs)
                    if total_length > 0:
                        score = sum(seg["score"] * (seg["end"] - seg["start"]) for seg in segs) / total_length
                    else:
                        score = sum(seg["score"] for seg in segs) / len(segs)
                    words.append({
                        "word": word,
                        "start": segments[i1]["start"],
                        "end": segments[i2 - 1]["end"],
                        "score": score
                    })
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        
        return words
    
    def align_audio(self, audio_path: str, transcript: str, 
                    progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Perform forced alignment of audio with transcript
        
        Args:
            audio_path: Path to audio file
            transcript: Text transcript to align
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing alignment results
        """
        if not self._initialized:
            if not self.initialize():
                return {"success": False, "error": "Failed to initialize alignment model"}
        
        try:
            if progress_callback:
                progress_callback(10, "Loading audio...")
            
            # Load audio
            waveform, sr = self._load_audio(audio_path)
            duration = waveform.shape[1] / sr
            
            if progress_callback:
                progress_callback(20, "Computing emissions...")
            
            # Get emissions
            emission = self._get_emissions(waveform)
            
            # Prepare transcript - convert to uppercase and add word boundaries
            transcript_clean = transcript.upper().strip()
            # Replace multiple spaces with single space
            transcript_clean = " ".join(transcript_clean.split())
            
            # Remove characters not in vocabulary (punctuation, numbers, etc.)
            # Keep only letters, apostrophe, and spaces
            filtered_chars = []
            for c in transcript_clean:
                if c in self.dictionary:
                    filtered_chars.append(c)
                elif c == " ":
                    filtered_chars.append(c)
                # Skip punctuation and other unknown characters
            transcript_filtered = "".join(filtered_chars)
            # Clean up multiple spaces that may result from removing punctuation
            transcript_filtered = " ".join(transcript_filtered.split())
            
            # Add word boundary markers
            transcript_with_boundaries = "|" + "|".join(transcript_filtered.split()) + "|"
            
            logger.info(f"Transcript length: {len(transcript_clean)}, filtered: {len(transcript_filtered)}, with boundaries: {len(transcript_with_boundaries)}")
            
            if progress_callback:
                progress_callback(40, "Building trellis...")
            
            # Convert transcript to tokens - now all chars should be in vocabulary
            tokens = []
            for c in transcript_with_boundaries:
                if c in self.dictionary:
                    tokens.append(self.dictionary[c])
                elif c == " ":
                    tokens.append(self.dictionary["|"])
                else:
                    # This shouldn't happen after filtering, but log if it does
                    logger.warning(f"Unexpected character not in vocab: '{c}'")
            
            if len(tokens) == 0:
                return {"success": False, "error": "No valid tokens in transcript"}
            
            logger.info(f"Token count: {len(tokens)}, transcript_with_boundaries length: {len(transcript_with_boundaries)}")
            
            # Build trellis
            trellis = self._get_trellis(emission, tokens)
            
            if progress_callback:
                progress_callback(60, "Backtracking alignment...")
            
            # Backtrack to find path
            path = self._backtrack(trellis, emission, tokens)
            
            if progress_callback:
                progress_callback(80, "Merging segments...")
            
            # Merge into character segments
            char_segments = self._merge_repeats(path, transcript_with_boundaries)
            
            # Merge into word segments
            word_segments = self._merge_words(char_segments)
            
            # Convert frame indices to time
            # Wav2Vec2 base model: 20ms per frame (320 samples at 16kHz)
            # Calculate actual frame duration from audio samples and emission frames
            samples_per_frame = waveform.shape[1] / emission.shape[0]
            frame_duration = samples_per_frame / sr  # seconds per frame
            logger.info(f"Frame info: {emission.shape[0]} frames, {samples_per_frame:.1f} samples/frame, {frame_duration*1000:.1f}ms/frame")
            
            # Convert character alignments to time
            char_alignments = []
            for seg in char_segments:
                if seg["label"] != "|":
                    char_alignments.append(AlignedChar(
                        char=seg["label"],
                        start=round(seg["start"] * frame_duration, 3),
                        end=round(seg["end"] * frame_duration, 3),
                        score=round(seg["score"], 3)
                    ))
            
            # Convert word alignments to time
            word_alignments = []
            for seg in word_segments:
                word_alignments.append(AlignedWord(
                    word=seg["word"],
                    start=round(seg["start"] * frame_duration, 3),
                    end=round(seg["end"] * frame_duration, 3),
                    score=round(seg["score"], 3)
                ))
            
            if progress_callback:
                progress_callback(100, "Alignment complete")
            
            logger.info(f"Aligned {len(word_alignments)} words, {len(char_alignments)} characters")
            
            return {
                "success": True,
                "enabled": True,
                "duration": duration,
                "word_alignments": [w.to_dict() for w in word_alignments],
                "char_alignments": [c.to_dict() for c in char_alignments]
            }
            
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


# Singleton instance
_alignment_service: Optional[AlignmentService] = None


def get_alignment_service() -> AlignmentService:
    """Get or create the singleton AlignmentService instance"""
    global _alignment_service
    if _alignment_service is None:
        _alignment_service = AlignmentService()
    return _alignment_service
