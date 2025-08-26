# Best Practices for Loading Pretrained Models in TorchTitan with FSDP

This document outlines the best practices for loading pretrained models (especially large models like T5-v1_1-xxl) in TorchTitan with FSDP parallelization.

## Overview

TorchTitan provides optimized support for loading pretrained text encoders (T5, CLIP) with memory-efficient techniques and FSDP parallelization. The implementation in `FluxEmbedder` demonstrates the recommended patterns.

## Configuration Options

### Basic Configuration

```python
# In your job config
encoder:
  t5_encoder: "google/t5-v1_1-xxl"  # Can be HuggingFace model name or local path
  clip_encoder: "openai/clip-vit-large-patch14" 
  use_meta_device: true  # Enable for large models to avoid OOM
  low_cpu_mem_usage: true  # Reduce CPU memory usage during loading
```

### Memory Optimization for Large Models

For very large models like `t5-v1_1-xxl`, enable memory optimizations:

```python
encoder:
  t5_encoder: "/path/to/t5-v1_1-xxl"
  use_meta_device: true      # Memory-efficient loading
  low_cpu_mem_usage: true    # Reduce CPU memory usage
```

## Loading Patterns

### 1. Standard Loading (Small to Medium Models)

```python
embedder = FluxEmbedder(
    version="google/t5-v1_1-small",
    random_init=False,
    use_meta_device=False,  # Standard loading
    low_cpu_mem_usage=True,
)
```

**Best for:** Models up to a few billion parameters that fit comfortably in memory.

### 2. Memory-Optimized Loading (Large Models)

```python
embedder = FluxEmbedder(
    version="google/t5-v1_1-xxl", 
    random_init=False,
    use_meta_device=True,       # Enable memory optimization
    low_cpu_mem_usage=True,
    target_device=device,       # Target device
    target_dtype=dtype,         # Target dtype (e.g., torch.bfloat16)
)
```

**Best for:** Large models (multi-billion parameters) like T5-XXL, where loading can cause OOM.

### 3. Test Mode (Development/CI)

```python
embedder = FluxEmbedder(
    version="google/t5-v1_1-small",
    random_init=True,  # Skip downloading weights
    use_meta_device=False,
)
```

**Best for:** Testing, CI, or when you don't need actual pretrained weights.

## FSDP Parallelization

The `parallelize_encoders` function applies FSDP specifically for text encoders:

```python
# T5 encoder gets FSDP (high computation/communication ratio)
for block in t5_model.hf_module.encoder.block:
    fully_shard(block, **fsdp_config)
fully_shard(t5_model.hf_module, **fsdp_config)

# CLIP encoder does NOT get FSDP (low computation/communication ratio)
# This is intentional for optimal performance
```

### Key FSDP Characteristics:

1. **Hierarchical sharding**: Individual blocks first, then whole model
2. **T5-only**: CLIP encoder is not sharded due to low compute/communication ratio
3. **Frozen parameters**: Models are set to `.eval().requires_grad_(False)`
4. **FSDP2**: Uses the newer `fully_shard()` API instead of FSDP1

## Memory Usage Patterns

### Without Memory Optimization
```
1. Load full model in CPU memory
2. Move to GPU 
3. Apply FSDP sharding
```
**Risk:** OOM during step 1 for very large models

### With Memory Optimization  
```
1. Load with device_map and torch_dtype optimization
2. Model is already on target device with correct dtype
3. Apply FSDP sharding
```
**Benefit:** Reduced peak memory usage, faster loading

## Parameter Freezing

All pretrained encoders are automatically frozen:

```python
self.hf_module = self.hf_module.eval().requires_grad_(False)
```

This ensures:
- Parameters don't update during training
- Gradients aren't computed for encoder parameters  
- Memory usage is optimized
- Behavior is consistent across training

## Example Usage

### For T5-v1_1-XXL (Recommended)

```python
# Job config
encoder:
  t5_encoder: "/mnt/workspace/cv_multimodal/aigc/huggingface/t5-v1_1-xxl"
  use_meta_device: true
  low_cpu_mem_usage: true

# In training code  
t5_encoder = FluxEmbedder(
    version=job_config.encoder.t5_encoder,
    random_init=job_config.training.test_mode,
    use_meta_device=job_config.encoder.use_meta_device,
    low_cpu_mem_usage=job_config.encoder.low_cpu_mem_usage,
    target_device=device,
    target_dtype=dtype,
)

# Apply FSDP
t5_encoder, clip_encoder = parallelize_encoders(
    t5_model=t5_encoder,
    clip_model=clip_encoder,
    parallel_dims=parallel_dims,
    job_config=job_config,
)
```

## Performance Considerations

1. **Large models**: Always use `use_meta_device=True` for multi-billion parameter models
2. **Dtype**: Use `torch.bfloat16` or `torch.float16` for memory savings
3. **Device placement**: Let the system handle device placement with `device_map="auto"`
4. **FSDP ordering**: Apply to individual blocks before the whole model

## Troubleshooting

### OOM During Loading
- Enable `use_meta_device=True`
- Enable `low_cpu_mem_usage=True` 
- Use a smaller dtype like `torch.bfloat16`

### Slow Loading
- Check network connection for HuggingFace models
- Use local model paths when possible
- Enable `low_cpu_mem_usage=True`

### FSDP Issues
- Ensure proper device placement before FSDP
- Verify model is in eval mode and frozen
- Check that block-level sharding is applied before model-level

## Migration from Direct HuggingFace Usage

### Before (Direct HuggingFace)
```python
from transformers import T5EncoderModel
hf_module = T5EncoderModel.from_pretrained("/mnt/workspace/cv_multimodal/aigc/huggingface/t5-v1_1-xxl")
```

### After (TorchTitan Optimized)
```python
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder

embedder = FluxEmbedder(
    version="/mnt/workspace/cv_multimodal/aigc/huggingface/t5-v1_1-xxl",
    use_meta_device=True,
    low_cpu_mem_usage=True,
    target_device=device,
    target_dtype=torch.bfloat16,
)
```

The TorchTitan approach provides automatic memory optimization, parameter freezing, and FSDP compatibility.