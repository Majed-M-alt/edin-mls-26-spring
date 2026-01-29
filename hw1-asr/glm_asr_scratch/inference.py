"""
GLM-ASR Inference Script
Educational implementation from scratch using PyTorch only

This script demonstrates how to:
1. Load the GLM-ASR model weights
2. Process audio input
3. Generate transcription

Equivalent to test.py but without the transformers library.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

# Import our implementations
from config import GlmAsrConfig, AudioEncoderConfig, TextDecoderConfig, AudioProcessorConfig
from audio_features import WhisperFeatureExtractor, load_audio_file
from model import GlmAsrForConditionalGeneration
from tokenizer import SimpleTokenizer
from weight_loader import load_weights_into_model, load_safetensors, get_safetensors_metadata


class GlmAsrProcessor:
    """
    Processor for GLM-ASR that handles:
    - Audio feature extraction
    - Tokenization
    - Input preparation for the model
    """

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: SimpleTokenizer,
        audio_token_id: int = 59260,
        default_prompt: str = "Please transcribe this audio into text"
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_token_id = audio_token_id
        self.default_prompt = default_prompt

    def apply_transcription_request(
        self,
        audio: Union[np.ndarray, str],
        prompt: Optional[str] = None
    ) -> dict:
        """
        Prepare inputs for transcription.

        Args:
            audio: Audio waveform (numpy array) or path to audio file
            prompt: Custom transcription prompt (default: "Please transcribe...")

        Returns:
            Dictionary with input_ids, input_features, and attention_mask
        """
        # Load audio if path is provided
        if isinstance(audio, str):
            audio = load_audio_file(audio, target_sr=self.feature_extractor.sampling_rate)

        # Extract mel spectrogram features
        features = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding="max_length"
        )

        # Create input prompt with audio token placeholder
        if prompt is None:
            prompt = self.default_prompt

        # Simple chat template format:
        # <|user|>\n<audio>Please transcribe this audio into text<|end|>\n<|assistant|>\n
        # For simplicity, we'll use a basic format with audio token

        # In the actual model, the chat template creates token IDs like:
        # [bos, user_tokens, audio_tokens..., prompt_tokens, assistant_marker]

        # Create a simple input with audio placeholder
        # The actual sequence would be constructed by the chat template
        # Here we create a simplified version for demonstration

        # Number of audio tokens to insert (approximately audio_len / 2 after conv subsampling)
        audio_seq_len = features['input_features'].shape[1] // 2

        # Create input_ids with audio placeholders
        # Format: [BOS] [AUDIO_TOKENS...] [PROMPT_TOKENS] [ASSISTANT_START]
        input_ids = torch.tensor([[self.audio_token_id] * audio_seq_len], dtype=torch.long)

        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'input_features': features['input_features'],
            'attention_mask': attention_mask
        }

    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode generated token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens)


def load_model_from_pretrained(
    model_path: str,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16
) -> GlmAsrForConditionalGeneration:
    """
    Load GLM-ASR model from pretrained weights.

    Args:
        model_path: Path to model directory containing:
            - config.json
            - model.safetensors
            - tokenizer.json
        device: Device to load model on ("auto", "cuda", "cpu")
        dtype: Model dtype (default: bfloat16)

    Returns:
        Loaded model
    """
    model_path = Path(model_path)

    print(f"Loading model from {model_path}")

    # Load config
    import json
    config_file = model_path / "config.json"
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    # Create config objects
    audio_config = AudioEncoderConfig(
        hidden_size=config_dict['audio_config']['hidden_size'],
        intermediate_size=config_dict['audio_config']['intermediate_size'],
        num_hidden_layers=config_dict['audio_config']['num_hidden_layers'],
        num_attention_heads=config_dict['audio_config']['num_attention_heads'],
        num_key_value_heads=config_dict['audio_config']['num_key_value_heads'],
        head_dim=config_dict['audio_config']['head_dim'],
        num_mel_bins=config_dict['audio_config']['num_mel_bins'],
        max_position_embeddings=config_dict['audio_config']['max_position_embeddings'],
        hidden_act=config_dict['audio_config']['hidden_act'],
        partial_rotary_factor=config_dict['audio_config']['partial_rotary_factor'],
        rope_theta=config_dict['audio_config']['rope_parameters']['rope_theta']
    )

    text_config = TextDecoderConfig(
        hidden_size=config_dict['text_config']['hidden_size'],
        intermediate_size=config_dict['text_config']['intermediate_size'],
        num_hidden_layers=config_dict['text_config']['num_hidden_layers'],
        num_attention_heads=config_dict['text_config']['num_attention_heads'],
        num_key_value_heads=config_dict['text_config']['num_key_value_heads'],
        head_dim=config_dict['text_config']['head_dim'],
        vocab_size=config_dict['text_config']['vocab_size'],
        max_position_embeddings=config_dict['text_config']['max_position_embeddings'],
        hidden_act=config_dict['text_config']['hidden_act'],
        rms_norm_eps=config_dict['text_config']['rms_norm_eps'],
        attention_bias=config_dict['text_config']['attention_bias'],
        mlp_bias=config_dict['text_config']['mlp_bias'],
        rope_theta=config_dict['text_config']['rope_parameters']['rope_theta'],
        eos_token_ids=config_dict['text_config']['eos_token_id']
    )

    config = GlmAsrConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=config_dict['audio_token_id'],
        projector_hidden_act=config_dict['projector_hidden_act']
    )

    print(f"Model config:")
    print(f"  Audio encoder: {audio_config.num_hidden_layers} layers, "
          f"{audio_config.hidden_size} hidden")
    print(f"  Text decoder: {text_config.num_hidden_layers} layers, "
          f"{text_config.hidden_size} hidden, {text_config.vocab_size} vocab")

    # Create model
    model = GlmAsrForConditionalGeneration(config)

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load weights
    weights_file = model_path / "model.safetensors"
    if weights_file.exists():
        load_weights_into_model(model, str(weights_file), verbose=True)
    else:
        print(f"Warning: {weights_file} not found, using random weights")

    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def load_processor_from_pretrained(model_path: str) -> GlmAsrProcessor:
    """Load processor from pretrained model directory."""
    model_path = Path(model_path)

    # Load feature extractor config
    import json
    processor_file = model_path / "processor_config.json"
    with open(processor_file, 'r') as f:
        processor_config = json.load(f)

    audio_config = AudioProcessorConfig(
        sampling_rate=processor_config['feature_extractor']['sampling_rate'],
        n_fft=processor_config['feature_extractor']['n_fft'],
        hop_length=processor_config['feature_extractor']['hop_length'],
        chunk_length=processor_config['feature_extractor']['chunk_length'],
        n_samples=processor_config['feature_extractor']['n_samples'],
        feature_size=processor_config['feature_extractor']['feature_size'],
        nb_max_frames=processor_config['feature_extractor']['nb_max_frames']
    )

    feature_extractor = WhisperFeatureExtractor(audio_config)

    # Load tokenizer
    tokenizer = SimpleTokenizer.from_pretrained(str(model_path))

    return GlmAsrProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        default_prompt=processor_config.get('default_transcription_prompt',
                                            'Please transcribe this audio into text')
    )


def transcribe(
    model: GlmAsrForConditionalGeneration,
    processor: GlmAsrProcessor,
    audio: Union[np.ndarray, str],
    max_new_tokens: int = 500,
    do_sample: bool = False
) -> str:
    """
    Transcribe audio to text.

    Args:
        model: Loaded GLM-ASR model
        processor: Loaded processor
        audio: Audio waveform or file path
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling

    Returns:
        Transcribed text
    """
    # Prepare inputs
    inputs = processor.apply_transcription_request(audio)

    # Move to model device and dtype
    inputs = {
        'input_ids': inputs['input_ids'].to(model.language_model.model.embed_tokens.weight.device),
        'input_features': inputs['input_features'].to(
            model.language_model.model.embed_tokens.weight.device,
            dtype=model.language_model.model.embed_tokens.weight.dtype
        ),
        'attention_mask': inputs['attention_mask'].to(model.language_model.model.embed_tokens.weight.device)
    }

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            input_features=inputs['input_features'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    # Decode
    # Skip the input tokens when decoding
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded[0]


# Example usage (equivalent to test.py)
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("GLM-ASR Inference (Educational Implementation)")
    print("Without transformers library")
    print("=" * 60)

    # Check if model path is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default path - user should download the model first
        model_path = "./zai-org/GLM-ASR-Nano-2512"

    print(f"\nModel path: {model_path}")

    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("\nTo use this script, you need to:")
        print("1. Download the model files from HuggingFace:")
        print("   - config.json")
        print("   - model.safetensors (4.52 GB)")
        print("   - processor_config.json")
        print("   - tokenizer.json")
        print("\n2. Run this script with the model path:")
        print(f"   python inference.py /path/to/model")

        print("\n" + "=" * 60)
        print("Running with test data instead...")
        print("=" * 60)

        # Create test model with small config
        from config import GlmAsrConfig, AudioEncoderConfig, TextDecoderConfig

        test_audio_config = AudioEncoderConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=64
        )

        test_text_config = TextDecoderConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000
        )

        test_config = GlmAsrConfig(
            audio_config=test_audio_config,
            text_config=test_text_config,
            audio_token_id=999  # Use valid token ID for test vocab (1000)
        )

        model = GlmAsrForConditionalGeneration(test_config)
        model.eval()

        # Create test audio (1 second of 440Hz sine wave)
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Create feature extractor
        audio_config = AudioProcessorConfig()
        feature_extractor = WhisperFeatureExtractor(audio_config)

        # Extract features
        features = feature_extractor(test_audio)
        print(f"\nTest audio shape: {test_audio.shape}")
        print(f"Extracted features shape: {features['input_features'].shape}")

        # Create dummy input
        audio_seq_len = features['input_features'].shape[1] // 2
        input_ids = torch.tensor([[test_config.audio_token_id] * audio_seq_len], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                input_features=features['input_features']
            )

        print(f"Output logits shape: {outputs['logits'].shape}")
        print("\nTest completed successfully!")
        print("The model architecture is working correctly.")

    else:
        print("\nLoading model...")
        model = load_model_from_pretrained(str(model_path))
        processor = load_processor_from_pretrained(str(model_path))

        # Test with a sample audio file
        # You would replace this with actual audio
        print("\nTo transcribe audio, use:")
        print("  result = transcribe(model, processor, 'path/to/audio.wav')")
        print("  print(result)")
