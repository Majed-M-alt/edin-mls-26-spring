"""
Minimal GLM-ASR inference using CuTile/CuPy.
"""

import cupy as cp
import numpy as np
from weight_loader import load_model_from_hf
from tokenizers import Tokenizer
from pathlib import Path


# Special token IDs for GLM-ASR
TOKEN_IDS = {
    'sop': 59250,
    'eop': 59251,
    'endoftext': 59246,
    'begin_of_audio': 59261,
    'end_of_audio': 59262,
    'user': 59253,
    'assistant': 59254,
    'audio': 59260,  # audio_token_id - placeholder for audio features
}


def load_tokenizer():
    """Load the tokenizer from HuggingFace cache."""
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("zai-org/GLM-ASR-Nano-2512")
    tokenizer_path = Path(model_dir) / "tokenizer.json"
    return Tokenizer.from_file(str(tokenizer_path))


def create_asr_inputs(audio_array, feature_extractor, num_audio_tokens=None):
    """
    Create inputs for ASR inference.

    Args:
        audio_array: numpy array of audio samples (16kHz)
        feature_extractor: WhisperFeatureExtractor
        num_audio_tokens: number of audio placeholder tokens (computed from audio length if None)

    Returns:
        input_features: mel spectrogram features (batch, mel_bins, time)
        input_ids: token IDs for the prompt
    """
    import torch

    # Extract mel spectrogram
    features = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np"
    )
    input_features = features.input_features  # (batch, mel_bins, time)

    # Calculate number of audio tokens (after conv layers: time / 2 / 2 = time / 4)
    # Then pooled by factor of 4 in projector = time / 16
    mel_time = input_features.shape[2]
    if num_audio_tokens is None:
        num_audio_tokens = mel_time // 4  # conv stride 2 twice, then pool 4

    # Build prompt: <|user|> <audio_tokens> <|user|> Please transcribe <|assistant|>
    # Format: <|begin_of_audio|> <audio>*N <|end_of_audio|> <|user|> instruction <|assistant|>
    prompt_tokens = [
        TOKEN_IDS['begin_of_audio'],
    ] + [TOKEN_IDS['audio']] * num_audio_tokens + [
        TOKEN_IDS['end_of_audio'],
        TOKEN_IDS['user'],
    ]

    # Add instruction text
    tokenizer = load_tokenizer()
    instruction = "Please transcribe this audio into text"
    instruction_tokens = tokenizer.encode(instruction).ids

    prompt_tokens.extend(instruction_tokens)
    prompt_tokens.append(TOKEN_IDS['assistant'])

    input_ids = np.array([prompt_tokens], dtype=np.int64)

    return input_features, input_ids


def transcribe(audio_array, model=None, processor=None, model_name="zai-org/GLM-ASR-Nano-2512"):
    """
    Transcribe audio to text using GLM-ASR CuTile model.

    Args:
        audio_array: numpy array of audio samples (16kHz)
        model: pre-loaded CuTile model (optional)
        processor: pre-loaded HF processor (WhisperFeatureExtractor)
        model_name: HuggingFace model name if model/processor not provided

    Returns:
        Transcription string
    """
    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_model_from_hf(model_name)

    # Create inputs
    input_features, input_ids = create_asr_inputs(audio_array, processor)

    # Convert to CuPy
    input_features_cp = cp.asarray(input_features, dtype=cp.float32)
    input_ids_cp = cp.asarray(input_ids, dtype=cp.int64)

    # Generate transcription
    generated_ids = model.generate(
        input_features_cp,
        input_ids=input_ids_cp,
        input_features_mask=None,
        max_new_tokens=500,
        temperature=1.0,
        top_k=1
    )

    # Decode output
    generated_np = cp.asnumpy(generated_ids)
    tokenizer = load_tokenizer()
    transcription = tokenizer.decode(generated_np[0].tolist(), skip_special_tokens=False)

    # Extract just the transcription (remove prompt and special tokens)
    if "<|assistant|>" in transcription:
        transcription = transcription.split("<|assistant|>")[-1]
    # Remove special tokens
    for token in ['<|endoftext|>', '<|user|>', '<|begin_of_audio|>', '<|end_of_audio|>']:
        transcription = transcription.replace(token, '')
    transcription = transcription.strip()

    return transcription


if __name__ == "__main__":
    import numpy as np
    import io

    print("Loading model...")
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    print("Loading test audio...")
    # Get sampling rate
    sampling_rate = getattr(processor, 'sampling_rate', None) or 16000

    # Try to load audio using soundfile from dataset bytes (avoids torchcodec)
    try:
        import soundfile as sf
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # Access the raw audio bytes without decoding
        audio_bytes = ds.data.column('audio')[0].as_py()['bytes']
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample if needed
        if sr != sampling_rate:
            import scipy.signal
            num_samples = int(len(audio_array) * sampling_rate / sr)
            audio_array = scipy.signal.resample(audio_array, num_samples)

        expected_text = ds.data.column('text')[0].as_py()
        print(f"Loaded LibriSpeech audio (original sr={sr})")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print("Using synthetic test audio instead...")
        # Generate a short synthetic audio for testing
        duration = 3.0  # seconds
        t = np.linspace(0, duration, int(sampling_rate * duration))
        # Simple tone (won't produce meaningful transcription, but tests the pipeline)
        audio_array = 0.5 * np.sin(2 * np.pi * 440 * t)
        expected_text = "(synthetic audio - no expected transcription)"

    print(f"Audio duration: {len(audio_array) / sampling_rate:.2f}s")

    print("\nTranscribing...")
    result = transcribe(audio_array, model, processor)

    print(f"\nTranscription: {result}")
    print(f"Expected: {expected_text}")
