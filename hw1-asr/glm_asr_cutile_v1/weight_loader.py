"""
Weight Loader for GLM-ASR
Educational implementation using NVIDIA CuTile for tile-based GPU programming

Loads pre-trained weights from safetensors format into the model.
"""

import cupy as cp
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import json

# Try to import safetensors
try:
    from safetensors import safe_open
    from safetensors.numpy import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed. Weight loading will be limited.")


def load_safetensor_to_cupy(filepath: str) -> Dict[str, cp.ndarray]:
    """
    Load a safetensors file and convert to CuPy arrays.

    Args:
        filepath: Path to .safetensors file

    Returns:
        Dictionary mapping tensor names to CuPy arrays
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors is required for weight loading")

    import torch

    tensors = {}
    # Use PyTorch framework to handle bfloat16
    with safe_open(filepath, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert bfloat16 to float32
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            tensor_np = tensor.numpy()
            tensors[key] = cp.array(tensor_np)

    return tensors


def load_model_weights(
    model,
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Load weights into GLM-ASR model from safetensors files.

    Args:
        model: GlmAsrForConditionalGeneration or GlmAsrModel instance
        model_path: Path to directory containing model weights
        config_path: Optional path to config.json
    """
    model_path = Path(model_path)

    # Find safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    print(f"Found {len(safetensor_files)} safetensor files")

    # Load all weights
    all_weights = {}
    for sf_file in safetensor_files:
        print(f"Loading {sf_file.name}...")
        weights = load_safetensor_to_cupy(str(sf_file))
        all_weights.update(weights)

    print(f"Loaded {len(all_weights)} tensors")

    # Map weights to model
    _map_weights_to_model(model, all_weights)


def _map_weights_to_model(model, weights: Dict[str, cp.ndarray]) -> None:
    """
    Map loaded weights to model parameters.

    The weight names follow HuggingFace naming conventions.
    """
    # Get the inner model if wrapped
    if hasattr(model, 'model'):
        inner_model = model.model
    else:
        inner_model = model

    loaded_count = 0
    skipped_keys = []

    for name, tensor in weights.items():
        try:
            if _load_single_weight(inner_model, name, tensor):
                loaded_count += 1
            else:
                skipped_keys.append(name)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            skipped_keys.append(name)

    print(f"Loaded {loaded_count} weights")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} weights (may be expected)")


def _load_single_weight(model, name: str, tensor: cp.ndarray) -> bool:
    """
    Load a single weight tensor into the model.

    Returns True if weight was loaded, False otherwise.
    """
    parts = name.split('.')

    # Navigate to the correct module
    obj = model
    found = True

    # Handle common prefixes
    if parts[0] == 'model':
        parts = parts[1:]

    # Audio encoder weights
    if parts[0] == 'audio_encoder' or parts[0] == 'encoder':
        obj = model.audio_encoder
        parts = parts[1:]

        if parts[0] == 'conv1':
            obj = obj.conv1
            parts = parts[1:]
        elif parts[0] == 'conv2':
            obj = obj.conv2
            parts = parts[1:]
        elif parts[0] == 'layers':
            layer_idx = int(parts[1])
            if layer_idx < len(obj.layers):
                obj = obj.layers[layer_idx]
                parts = parts[2:]
            else:
                return False
        elif parts[0] == 'layer_norm':
            obj = obj.layer_norm
            parts = parts[1:]
        else:
            return False

    # Text decoder weights
    elif parts[0] == 'text_decoder' or parts[0] == 'decoder':
        obj = model.text_decoder
        parts = parts[1:]

        if parts[0] == 'embed_tokens':
            obj = obj.embed_tokens
            parts = parts[1:]
        elif parts[0] == 'layers':
            layer_idx = int(parts[1])
            if layer_idx < len(obj.layers):
                obj = obj.layers[layer_idx]
                parts = parts[2:]
            else:
                return False
        elif parts[0] == 'norm':
            obj = obj.norm
            parts = parts[1:]
        else:
            return False

    # Multi-modal projector
    elif parts[0] == 'multi_modal_projector':
        obj = model.multi_modal_projector
        parts = parts[1:]

    # LM head
    elif parts[0] == 'lm_head':
        if hasattr(model, 'lm_head_weight'):
            model.lm_head_weight = tensor
            return True
        return False

    else:
        return False

    # Navigate deeper based on remaining parts
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and hasattr(obj, '__getitem__'):
            obj = obj[int(part)]
        else:
            return False

    # Set the final weight
    attr_name = parts[-1] if parts else 'weight'

    # Handle common weight names
    weight_map = {
        'weight': 'weight',
        'bias': 'bias',
        'gamma': 'weight',  # Some models use gamma for norm weight
        'beta': 'bias'       # Some models use beta for norm bias
    }

    attr_name = weight_map.get(attr_name, attr_name)

    if hasattr(obj, attr_name):
        current = getattr(obj, attr_name)
        if current is not None and current.shape == tensor.shape:
            setattr(obj, attr_name, tensor)
            return True
        elif current is None:
            setattr(obj, attr_name, tensor)
            return True

    return False


def get_model_info(model_path: Union[str, Path]) -> Dict:
    """
    Get model configuration and weight information.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary with model info
    """
    model_path = Path(model_path)

    info = {
        'path': str(model_path),
        'safetensor_files': [],
        'config': None,
        'total_params': 0
    }

    # Find safetensor files
    for sf_file in model_path.glob("*.safetensors"):
        info['safetensor_files'].append(sf_file.name)

    # Load config if available
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            info['config'] = json.load(f)

    # Count parameters
    if HAS_SAFETENSORS and info['safetensor_files']:
        total_params = 0
        for sf_file in info['safetensor_files']:
            filepath = model_path / sf_file
            with safe_open(str(filepath), framework="numpy") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    total_params += tensor.size
        info['total_params'] = total_params
        info['total_params_millions'] = total_params / 1e6

    return info


# ============================================================================
# HuggingFace Transformers Weight Loading
# ============================================================================

def create_config_from_hf(hf_config):
    """Create GlmAsrConfig from HuggingFace config."""
    from model import GlmAsrConfig

    ac = hf_config.audio_config
    tc = hf_config.text_config

    return GlmAsrConfig(
        # Audio encoder
        audio_hidden_size=ac.hidden_size,
        audio_num_heads=ac.num_attention_heads,
        audio_num_layers=ac.num_hidden_layers,
        audio_intermediate_size=ac.intermediate_size,
        audio_max_position_embeddings=getattr(ac, 'max_position_embeddings', 1500),

        # Text decoder
        text_hidden_size=tc.hidden_size,
        text_num_heads=tc.num_attention_heads,
        text_num_kv_heads=tc.num_key_value_heads,
        text_num_layers=tc.num_hidden_layers,
        text_intermediate_size=tc.intermediate_size,
        text_vocab_size=tc.vocab_size,
        text_max_position_embeddings=tc.max_position_embeddings,
        text_rope_base=getattr(tc, 'rope_theta', 10000.0),

        # Projector - uses intermediate size 4096, with 4x frame pooling
        projector_hidden_size=4096,  # Intermediate size (linear_1 output)
        projector_pool_factor=4,  # Concatenate 4 consecutive audio frames

        # Generation - get from text_config if not in main config
        pad_token_id=getattr(tc, 'pad_token_id', 0) if getattr(tc, 'pad_token_id', None) is not None else 0,
        bos_token_id=getattr(tc, 'bos_token_id', 1) if getattr(tc, 'bos_token_id', None) is not None else 1,
        # eos_token_id can be a list - pass all of them for proper stopping
        eos_token_id=getattr(tc, 'eos_token_id', 2),
    )


def load_linear_weight(cutile_linear, hf_weight, hf_bias=None):
    """Load weight (and optional bias) into CuTile Linear layer."""
    cutile_linear.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)
    if hf_bias is not None and cutile_linear.has_bias:
        cutile_linear.bias_param = cp.asarray(hf_bias.cpu().numpy(), dtype=cp.float32)


def load_conv1d_weight_from_hf(cutile_conv, hf_weight, hf_bias=None):
    """Load weight into CuTile Conv1d layer from HF format."""
    weight_np = hf_weight.cpu().numpy()
    out_channels, in_channels, kernel_size = weight_np.shape
    cutile_conv.weight = cp.asarray(
        weight_np.reshape(out_channels, in_channels * kernel_size),
        dtype=cp.float32
    )
    # Also update weight_padded if using CuTile
    if cutile_conv.use_cutile and (
        cutile_conv.col_size_padded != cutile_conv.col_size or
        cutile_conv.out_channels_padded != out_channels
    ):
        cutile_conv.weight_padded = cp.zeros(
            (cutile_conv.out_channels_padded, cutile_conv.col_size_padded),
            dtype=cp.float32
        )
        cutile_conv.weight_padded[:out_channels, :cutile_conv.col_size] = cutile_conv.weight
    else:
        cutile_conv.weight_padded = cutile_conv.weight
    if hf_bias is not None and cutile_conv.has_bias:
        cutile_conv.bias = cp.asarray(hf_bias.cpu().numpy(), dtype=cp.float32)


def load_layernorm_weight_from_hf(cutile_ln, hf_weight, hf_bias):
    """Load LayerNorm weights."""
    cutile_ln.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)
    cutile_ln.bias = cp.asarray(hf_bias.cpu().numpy(), dtype=cp.float32)


def load_rmsnorm_weight_from_hf(cutile_rms, hf_weight):
    """Load RMSNorm weight."""
    cutile_rms.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)


def load_embedding_weight_from_hf(cutile_emb, hf_weight):
    """Load Embedding weight."""
    cutile_emb.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)


def load_weights_from_hf_model(model, hf_model) -> None:
    """
    Load weights from HuggingFace GLM-ASR model into pure CuTile model.

    Args:
        model: Pure CuTile GlmAsrModel
        hf_model: HuggingFace GlmAsrForConditionalGeneration model
    """
    hf_state = hf_model.state_dict()

    print("Loading audio encoder weights...")

    # Audio encoder conv layers
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv1,
        hf_state['audio_tower.conv1.weight'],
        hf_state['audio_tower.conv1.bias']
    )
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv2,
        hf_state['audio_tower.conv2.weight'],
        hf_state['audio_tower.conv2.bias']
    )

    # Audio encoder positional embeddings (if learnable)
    if 'audio_tower.embed_positions.weight' in hf_state:
        model.audio_encoder.embed_positions = cp.asarray(
            hf_state['audio_tower.embed_positions.weight'].cpu().numpy(),
            dtype=cp.float32
        )

    # Audio encoder transformer layers
    for i, layer in enumerate(model.audio_encoder.layers):
        prefix = f'audio_tower.layers.{i}'

        # Input layernorm
        load_layernorm_weight_from_hf(
            layer.self_attn_layer_norm,
            hf_state[f'{prefix}.input_layernorm.weight'],
            hf_state[f'{prefix}.input_layernorm.bias']
        )

        # Attention projections
        load_linear_weight(
            layer.q_proj,
            hf_state[f'{prefix}.self_attn.q_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.q_proj.bias')
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f'{prefix}.self_attn.k_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.k_proj.bias')
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f'{prefix}.self_attn.v_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.v_proj.bias')
        )
        load_linear_weight(
            layer.out_proj,
            hf_state[f'{prefix}.self_attn.o_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.o_proj.bias')
        )

        # Post attention layernorm
        load_layernorm_weight_from_hf(
            layer.final_layer_norm,
            hf_state[f'{prefix}.post_attention_layernorm.weight'],
            hf_state[f'{prefix}.post_attention_layernorm.bias']
        )

        # MLP
        load_linear_weight(
            layer.fc1,
            hf_state[f'{prefix}.mlp.fc1.weight'],
            hf_state[f'{prefix}.mlp.fc1.bias']
        )
        load_linear_weight(
            layer.fc2,
            hf_state[f'{prefix}.mlp.fc2.weight'],
            hf_state[f'{prefix}.mlp.fc2.bias']
        )

    # Audio encoder final layernorm
    load_layernorm_weight_from_hf(
        model.audio_encoder.layer_norm,
        hf_state['audio_tower.norm.weight'],
        hf_state['audio_tower.norm.bias']
    )

    print("Loading multi-modal projector weights...")

    # Multi-modal projector
    load_linear_weight(
        model.multi_modal_projector.linear_1,
        hf_state['multi_modal_projector.linear_1.weight'],
        hf_state['multi_modal_projector.linear_1.bias']
    )
    load_linear_weight(
        model.multi_modal_projector.linear_2,
        hf_state['multi_modal_projector.linear_2.weight'],
        hf_state['multi_modal_projector.linear_2.bias']
    )

    print("Loading text decoder weights...")

    # Text decoder embeddings
    load_embedding_weight_from_hf(
        model.text_decoder.embed_tokens,
        hf_state['language_model.model.embed_tokens.weight']
    )

    # Text decoder transformer layers
    for i, layer in enumerate(model.text_decoder.layers):
        prefix = f'language_model.model.layers.{i}'

        # Input layernorm (RMSNorm)
        load_rmsnorm_weight_from_hf(
            layer.input_layernorm,
            hf_state[f'{prefix}.input_layernorm.weight']
        )

        # Attention projections (no bias in Llama-style)
        load_linear_weight(
            layer.q_proj,
            hf_state[f'{prefix}.self_attn.q_proj.weight']
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f'{prefix}.self_attn.k_proj.weight']
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f'{prefix}.self_attn.v_proj.weight']
        )
        load_linear_weight(
            layer.o_proj,
            hf_state[f'{prefix}.self_attn.o_proj.weight']
        )

        # Post attention layernorm (RMSNorm)
        load_rmsnorm_weight_from_hf(
            layer.post_attention_layernorm,
            hf_state[f'{prefix}.post_attention_layernorm.weight']
        )

        # MLP (SwiGLU: gate_proj, up_proj, down_proj)
        load_linear_weight(
            layer.mlp.gate_proj,
            hf_state[f'{prefix}.mlp.gate_proj.weight']
        )
        load_linear_weight(
            layer.mlp.up_proj,
            hf_state[f'{prefix}.mlp.up_proj.weight']
        )
        load_linear_weight(
            layer.mlp.down_proj,
            hf_state[f'{prefix}.mlp.down_proj.weight']
        )

    # Text decoder final norm
    load_rmsnorm_weight_from_hf(
        model.text_decoder.norm,
        hf_state['language_model.model.norm.weight']
    )

    # LM head
    load_linear_weight(
        model.lm_head,
        hf_state['language_model.lm_head.weight']
    )

    print("Weight loading complete!")


def load_model_from_hf(model_name: str = "zai-org/GLM-ASR-Nano-2512"):
    """
    Load GLM-ASR model from HuggingFace and create CuTile version.

    Returns:
        tuple: (cutile_model, hf_processor)
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from model import GlmAsrModel
    import json

    print(f"Loading HuggingFace model: {model_name}")

    # Download model files
    model_dir = snapshot_download(model_name)
    print(f"Model directory: {model_dir}")

    # Load config from JSON
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # Create config object manually
    class SimpleConfig:
        pass

    hf_config = SimpleConfig()
    hf_config.audio_config = SimpleConfig()
    hf_config.text_config = SimpleConfig()

    # Audio config
    audio_cfg = config_dict.get('audio_config', {})
    hf_config.audio_config.hidden_size = audio_cfg.get('hidden_size', 1280)
    hf_config.audio_config.num_attention_heads = audio_cfg.get('num_attention_heads', 20)
    hf_config.audio_config.num_hidden_layers = audio_cfg.get('num_hidden_layers', 32)
    hf_config.audio_config.intermediate_size = audio_cfg.get('intermediate_size', 5120)
    hf_config.audio_config.num_mel_bins = audio_cfg.get('num_mel_bins', 128)
    hf_config.audio_config.max_position_embeddings = audio_cfg.get('max_position_embeddings', 1500)
    hf_config.audio_config.head_dim = audio_cfg.get('head_dim', 64)
    hf_config.audio_config.num_key_value_heads = audio_cfg.get('num_key_value_heads', 20)

    # Text config
    text_cfg = config_dict.get('text_config', {})
    hf_config.text_config.hidden_size = text_cfg.get('hidden_size', 2048)
    hf_config.text_config.num_attention_heads = text_cfg.get('num_attention_heads', 16)
    hf_config.text_config.num_hidden_layers = text_cfg.get('num_hidden_layers', 28)
    hf_config.text_config.intermediate_size = text_cfg.get('intermediate_size', 6144)
    hf_config.text_config.vocab_size = text_cfg.get('vocab_size', 59264)
    hf_config.text_config.num_key_value_heads = text_cfg.get('num_key_value_heads', 4)
    hf_config.text_config.rms_norm_eps = text_cfg.get('rms_norm_eps', 1e-5)
    hf_config.text_config.head_dim = text_cfg.get('head_dim', 128)
    hf_config.text_config.max_position_embeddings = text_cfg.get('max_position_embeddings', 8192)

    # Global config
    hf_config.hidden_size = config_dict.get('hidden_size', 2048)
    hf_config.audio_token_id = config_dict.get('audio_token_id', 59260)

    cutile_config = create_config_from_hf(hf_config)

    print(f"Creating CuTile model with config:")
    print(f"  Audio: hidden={cutile_config.audio_hidden_size}, heads={cutile_config.audio_num_heads}, layers={cutile_config.audio_num_layers}")
    print(f"  Text: hidden={cutile_config.text_hidden_size}, heads={cutile_config.text_num_heads}, kv_heads={cutile_config.text_num_kv_heads}, layers={cutile_config.text_num_layers}")

    # Create CuTile model
    cutile_model = GlmAsrModel(cutile_config)

    # Load weights directly from safetensors
    print("Loading weights from safetensors...")
    safetensor_path = Path(model_dir) / "model.safetensors"
    hf_state = load_safetensor_to_cupy(str(safetensor_path))

    # Load weights using state dict
    _load_weights_from_state_dict_v1(cutile_model, hf_state)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print("Weight loading complete!")
    return cutile_model, processor


def _load_weights_from_state_dict_v1(model, hf_state):
    """Load weights directly from state dict (CuPy arrays)."""
    import numpy as np

    def load_linear(layer, weight, bias=None):
        layer.weight = cp.asarray(weight) if not isinstance(weight, cp.ndarray) else weight
        if bias is not None:
            layer.bias = cp.asarray(bias) if not isinstance(bias, cp.ndarray) else bias

    def load_layernorm(layer, weight, bias=None):
        layer.weight = cp.asarray(weight) if not isinstance(weight, cp.ndarray) else weight
        if bias is not None:
            layer.bias = cp.asarray(bias) if not isinstance(bias, cp.ndarray) else bias

    def load_rmsnorm(layer, weight):
        layer.weight = cp.asarray(weight) if not isinstance(weight, cp.ndarray) else weight

    def load_embedding(layer, weight):
        layer.weight = cp.asarray(weight) if not isinstance(weight, cp.ndarray) else weight

    def load_conv1d(layer, weight, bias=None):
        # HF weight is (out_channels, in_channels, kernel_size)
        # v1 conv.py expects (out_channels, in_channels * kernel_size)
        weight_arr = cp.asarray(weight) if not isinstance(weight, cp.ndarray) else weight
        if weight_arr.ndim == 3:
            out_channels, in_channels, kernel_size = weight_arr.shape
            weight_arr = weight_arr.reshape(out_channels, in_channels * kernel_size)
        layer.weight = weight_arr
        if bias is not None:
            layer.bias = cp.asarray(bias) if not isinstance(bias, cp.ndarray) else bias

    # Audio encoder
    print("  Loading audio encoder...")

    # Conv layers
    load_conv1d(model.audio_encoder.conv1,
                hf_state['audio_tower.conv1.weight'],
                hf_state.get('audio_tower.conv1.bias'))
    load_conv1d(model.audio_encoder.conv2,
                hf_state['audio_tower.conv2.weight'],
                hf_state.get('audio_tower.conv2.bias'))

    # Note: This model uses RoPE, no positional_embedding needed

    # Audio encoder layers
    for i, layer in enumerate(model.audio_encoder.layers):
        prefix = f'audio_tower.layers.{i}'

        # Layer norms (HF: input_layernorm -> self_attn_layer_norm)
        load_layernorm(layer.self_attn_layer_norm,
                       hf_state[f'{prefix}.input_layernorm.weight'],
                       hf_state.get(f'{prefix}.input_layernorm.bias'))

        # Attention projections
        load_linear(layer.q_proj, hf_state[f'{prefix}.self_attn.q_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.q_proj.bias'))
        load_linear(layer.k_proj, hf_state[f'{prefix}.self_attn.k_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.k_proj.bias'))
        load_linear(layer.v_proj, hf_state[f'{prefix}.self_attn.v_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.v_proj.bias'))
        load_linear(layer.out_proj, hf_state[f'{prefix}.self_attn.o_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.o_proj.bias'))

        # Layer norm (HF: post_attention_layernorm -> final_layer_norm)
        load_layernorm(layer.final_layer_norm,
                       hf_state[f'{prefix}.post_attention_layernorm.weight'],
                       hf_state.get(f'{prefix}.post_attention_layernorm.bias'))

        # MLP
        load_linear(layer.fc1, hf_state[f'{prefix}.mlp.fc1.weight'],
                    hf_state.get(f'{prefix}.mlp.fc1.bias'))
        load_linear(layer.fc2, hf_state[f'{prefix}.mlp.fc2.weight'],
                    hf_state.get(f'{prefix}.mlp.fc2.bias'))

    # Final layer norm (HF: audio_tower.norm -> layer_norm)
    load_layernorm(model.audio_encoder.layer_norm,
                   hf_state['audio_tower.norm.weight'],
                   hf_state.get('audio_tower.norm.bias'))

    # Projector
    print("  Loading projector...")
    load_linear(model.multi_modal_projector.linear_1, hf_state['multi_modal_projector.linear_1.weight'],
                hf_state.get('multi_modal_projector.linear_1.bias'))
    load_linear(model.multi_modal_projector.linear_2, hf_state['multi_modal_projector.linear_2.weight'],
                hf_state.get('multi_modal_projector.linear_2.bias'))

    # Text decoder
    print("  Loading text decoder...")

    # Embedding
    load_embedding(model.text_decoder.embed_tokens,
                   hf_state['language_model.model.embed_tokens.weight'])

    # Decoder layers
    for i, layer in enumerate(model.text_decoder.layers):
        prefix = f'language_model.model.layers.{i}'

        load_rmsnorm(layer.input_layernorm,
                     hf_state[f'{prefix}.input_layernorm.weight'])

        # Attention projections are directly on layer, not under self_attn
        load_linear(layer.q_proj, hf_state[f'{prefix}.self_attn.q_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.q_proj.bias'))
        load_linear(layer.k_proj, hf_state[f'{prefix}.self_attn.k_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.k_proj.bias'))
        load_linear(layer.v_proj, hf_state[f'{prefix}.self_attn.v_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.v_proj.bias'))
        load_linear(layer.o_proj, hf_state[f'{prefix}.self_attn.o_proj.weight'],
                    hf_state.get(f'{prefix}.self_attn.o_proj.bias'))

        load_rmsnorm(layer.post_attention_layernorm,
                     hf_state[f'{prefix}.post_attention_layernorm.weight'])

        load_linear(layer.mlp.gate_proj, hf_state[f'{prefix}.mlp.gate_proj.weight'])
        load_linear(layer.mlp.up_proj, hf_state[f'{prefix}.mlp.up_proj.weight'])
        load_linear(layer.mlp.down_proj, hf_state[f'{prefix}.mlp.down_proj.weight'])

    # Final norm
    load_rmsnorm(model.text_decoder.norm,
                 hf_state['language_model.model.norm.weight'])

    # LM head
    load_linear(model.lm_head, hf_state['language_model.lm_head.weight'])
