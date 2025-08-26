# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

from torch import nn, Tensor
import torch
from transformers import CLIPTextModel, T5EncoderModel


class FluxEmbedder(nn.Module):
    def __init__(
        self, 
        version: str, 
        random_init=False, 
        use_meta_device=False,
        low_cpu_mem_usage=True,
        target_device: Optional[torch.device] = None,
        target_dtype: Optional[torch.dtype] = None,
        **hf_kwargs
    ):
        """
        Initialize FluxEmbedder with support for memory-optimized loading of large pretrained models.
        
        Args:
            version: HuggingFace model name or local path
            random_init: Whether to randomly initialize for testing
            use_meta_device: Whether to use meta device initialization for large models
            low_cpu_mem_usage: Whether to use low CPU memory usage during loading
            target_device: Target device to move model to after loading
            target_dtype: Target dtype to cast model to after loading
            **hf_kwargs: Additional HuggingFace model arguments
        """
        super().__init__()
        self.is_clip = "clip" in version.lower()
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        
        # Prepare loading kwargs
        loading_kwargs = {"low_cpu_mem_usage": low_cpu_mem_usage, **hf_kwargs}
        
        if use_meta_device and not random_init:
            # Use meta device initialization for large models to avoid OOM
            self._init_with_meta_device(version, loading_kwargs, target_device, target_dtype)
        else:
            # Standard initialization
            self._init_standard(version, random_init, loading_kwargs)

        # Always set to eval mode and freeze parameters for pretrained encoders
        self.hf_module = self.hf_module.eval().requires_grad_(False)
    
    def _init_with_meta_device(
        self, 
        version: str, 
        loading_kwargs: dict,
        target_device: Optional[torch.device],
        target_dtype: Optional[torch.dtype]
    ):
        """Initialize model using meta device pattern to save memory during loading."""
        try:
            if self.is_clip:
                # For CLIP, use torch_dtype to save memory during loading
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                    version,
                    torch_dtype=target_dtype or torch.float32,
                    device_map="auto" if target_device is None else {"": target_device},
                    **loading_kwargs
                )
            else:  # T5 model
                # For T5, use torch_dtype and device_map for efficient loading
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                    version,
                    torch_dtype=target_dtype or torch.float32,
                    device_map="auto" if target_device is None else {"": target_device},
                    **loading_kwargs
                )
        except Exception as e:
            # Fallback to standard loading if meta device loading fails
            print(f"Meta device loading failed, falling back to standard loading: {e}")
            self._init_standard(version, False, loading_kwargs)
            
            # Apply target device and dtype manually
            if target_device is not None:
                self.hf_module = self.hf_module.to(device=target_device)
            if target_dtype is not None:
                self.hf_module = self.hf_module.to(dtype=target_dtype)
    
    def _init_standard(self, version: str, random_init: bool, loading_kwargs: dict):
        """Standard initialization without meta device."""
        if self.is_clip:
            if random_init:
                # Initialize CLIP model with random weights for test purpose only
                self.hf_module = CLIPTextModel._from_config(
                    CLIPTextModel.config_class.from_pretrained(
                        os.path.join(version, "config.json"), **loading_kwargs
                    )
                )
            else:
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                    version, **loading_kwargs
                )
        else:
            if random_init:
                # Initialize T5 model with random weights for test purpose only
                self.hf_module = T5EncoderModel._from_config(
                    T5EncoderModel.config_class.from_pretrained(
                        os.path.join(version, "config.json"), **loading_kwargs
                    )
                )
            else:
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                    version, **loading_kwargs
                )

    def forward(self, batch_tokens: Tensor) -> Tensor:
        """
        batch_tokens: [bsz, embedding_length]

        For T5 Encoder, embeding_length is 768
        For CLIP, embedding_length is 256
        """
        outputs = self.hf_module(
            input_ids=batch_tokens.to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
