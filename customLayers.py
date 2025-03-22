# ===========================================================================
# Project:        PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs - IOL Lab @ ZIB
# Paper:          arxiv.org/abs/2312.15230
# File:           customLayers.py
# Description:    Custom PyTorch layer implementations for various LoRA adaptations including pruning-aware, scaling, and masking variants.
# ===========================================================================

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class CustomLayerBaseClass(nn.Module):
    """Base class for custom layers, only for LoRA."""
    def __init__(self, **kwargs):
        super().__init__()
        self.lora_layer = kwargs['lora_layer']
        self.dropout_p = kwargs['dropout_p'] or 0.

    @abc.abstractmethod
    def _initialize(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def set_effective_weights(self):
        """Sets the effective weights in-place to avoid memory issues."""
        pass


class PruneLoraLayer(CustomLayerBaseClass):
    """Regular LoRA during training, but prunes the B@A matrix before merging. Doesn't use dropout and hence has faster forward pass."""

    def _initialize(self):
        # Use standard LoRA initialization
        pass

    def forward(self, x):
        # Standard LoRA forward pass but without dropout
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default']
        lora_B = self.lora_layer.lora_B['default']
        scaling = self.lora_layer.scaling['default']

        lora_contribution = (lora_B.weight @ lora_A.weight).reshape(original_weight.shape)   
        return F.linear(x, original_weight + scaling * lora_contribution, self.lora_layer.bias)

    @torch.no_grad()
    def set_effective_weights(self):
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight
        scaling = self.lora_layer.scaling['default']
        
        # Compute B@A directly into the shape of the original weight
        lora_contribution = scaling * (lora_B @ lora_A).view_as(self.lora_layer.weight)
        
        # Get the mask from the original weights
        mask = self.lora_layer.weight != 0
        lora_contribution.mul_(mask)
        
        # Add to the original weight in-place
        self.lora_layer.weight.add_(lora_contribution)



class ScaleLoRALayer(CustomLayerBaseClass):
    """ScaleLoRA as in the paper."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize()

    def _initialize(self):
        # Get LoRA matrices
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight

        # Initialize B and A to 1./sqrt(r)
        with torch.no_grad():
            nn.init.ones_(lora_B)
            nn.init.ones_(lora_A)
            r = lora_A.shape[0]
            lora_A.data *= (1 / sqrt(r))
            lora_B.data *= (1 / sqrt(r))

    def forward(self, x):
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight

        # Compute LoRA contribution
        lora_contribution = (lora_B @ lora_A).reshape(original_weight.shape)

        # Apply custom dropout to lora_contribution
        lora_contribution = self.custom_dropout(lora_contribution)

        # Hadamard product
        effective_weight = original_weight * lora_contribution

        return F.linear(x, effective_weight, self.lora_layer.bias)
    
    @torch.no_grad()
    def set_effective_weights(self):
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight
        
        # Compute B@A directly into the shape of the original weight
        lora_contribution = (lora_B @ lora_A).view_as(self.lora_layer.weight)
        
        # Multiply the original weight in-place with the lora contribution
        self.lora_layer.weight.mul_(lora_contribution)

@torch.jit.script
def fused_lora_masked_matmul(B: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Compute matmul and reshape in one go
    result = (B @ A).view_as(W)
    # Zero out elements in-place where W is zero
    result.masked_fill_(W == 0, 0)
    return result

class MaskLoRALayer(CustomLayerBaseClass):
    """MaskLoRA as in the paper."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize()

    def _initialize(self):
        # We use the same initialization as the original LoRA
        pass

    def forward(self, x):
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default']
        lora_B = self.lora_layer.lora_B['default']
        scaling = self.lora_layer.scaling['default']
        

        masked_lora_contribution = fused_lora_masked_matmul(lora_B.weight, lora_A.weight, original_weight)
        return F.linear(x, original_weight + scaling * masked_lora_contribution, self.lora_layer.bias)


    @torch.no_grad()
    def set_effective_weights(self):
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight
        scaling = self.lora_layer.scaling['default']
        
        # Compute B@A directly into the shape of the original weight
        lora_contribution = (lora_B @ lora_A).view_as(self.lora_layer.weight)
        
        # Apply sparsity mask and scaling in-place
        lora_contribution.mul_((self.lora_layer.weight != 0).to(dtype=lora_A.dtype)).mul_(scaling)
        
        # Add to the original weight in-place
        self.lora_layer.weight.add_(lora_contribution)