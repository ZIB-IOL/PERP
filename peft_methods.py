# ===========================================================================
# Project:        PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs - IOL Lab @ ZIB
# Paper:          arxiv.org/abs/2312.15230
# File:           peft_methods.py
# Description:    Implementation of Parameter-Efficient Fine-Tuning (PEFT) methods including various LoRA adaptations and selective fine-tuning strategies.
# ===========================================================================

import abc
import torch
from customLayers import ScaleLoRALayer, MaskLoRALayer, PruneLoraLayer
from utilities import SelectiveMethods

from peft import LoraConfig, get_peft_model
import sys
import torch.nn.utils.prune as prune

from peft.tuners.lora import LoraLayer

class PeftMethodBaseClass:
    """PEFT method base class - Important: Do not activate layers that have been pruned without keeping the mask, otherwise sparsity is destroyed."""
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.runner = kwargs['runner']
        self.config = kwargs['config']
        self.is_reconstruct = kwargs['is_reconstruct']

    @abc.abstractmethod
    def select_peft_layers(self, **kwargs):
        pass

    def at_train_end(self):
        """This is called at the end of training, to clean up the model and remove the peft layers, if possible."""
        pass


class FullFT(PeftMethodBaseClass):
    def at_train_end(self):
        # Remove the masks
        for name, module in self.model.named_modules():
            if prune.is_pruned(module) and hasattr(module, 'weight'):
                prune.remove(module, name='weight')

    def select_peft_layers(self, **kwargs):
        SelectiveMethods.activate_model(self.model)

class SelectivePEFT(PeftMethodBaseClass):
    """Base class for LoRA, to be used for all methods with slightly differing reparametrizations. Uses LN parameters + biases + LoRA on pruned layers, as well as lm head."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # PEFT Configuration
        self.use_peft_dict = {
            'biases': self.config.peft_use_bias or False,
            'layer_norm_params': self.config.peft_use_ln or False,
            'lm_head': self.config.peft_use_lm_head or False,
            'lora': self.config.peft_use_lora or False,
        }
        assert sum(self.use_peft_dict.values()) > 0, "At least one of the peft_use_* must be True."

        # LoRA Configuration
        self.lora_r = self.config.lora_r
        self.lora_alpha = self.config.lora_alpha
        self.lora_dropout = self.config.lora_dropout
        self.lora_type = self.config.lora_type

        self.CustomModule = None # To be set by below
        self.non_pruned_lora_layers = []    # Layers that are not pruned and hence use classical LoRA    

        self.target_modules = []
        if self.use_peft_dict['lm_head']:
            if not self.is_reconstruct:
                sys.stdout.write("SelectivePEFT: Using LM head.\n")
                self.target_modules.append("lm_head")
                self.non_pruned_lora_layers.append("lm_head")
            else:
                sys.stdout.write("lm_head has been specified for reconstruction, but it is not pruned. Skipping.\n")
        if self.use_peft_dict['lora']:
            sys.stdout.write("SelectivePEFT: Using LoRA.\n")
            # LoRA Configuration    
            self.target_modules = self.target_modules + [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "out_proj",     # OPT
                    "o_proj",       # Mistral/Llama
                    "gate_proj",    # Mistral/Llama
                    "up_proj",      # Mistral/Llama
                    "down_proj",    # Mistral/Llama
                    "fc1",          # OPT
                    "fc2",          # OPT
                    ]
            assert self.lora_type in ['lora', 'lora_prune', 'scale_lora', 'mask_lora'], "LoRA type must be in ['lora', 'lora_prune', 'scale_lora', 'mask_lora']."
            lora_layer_map = {
                'scale_lora': ScaleLoRALayer,
                'mask_lora': MaskLoRALayer,
                'lora_prune': PruneLoraLayer
            }
            
            if self.lora_type in lora_layer_map:
                self.CustomModule = lora_layer_map[self.lora_type]
                if self.lora_dropout > 0:
                    sys.stdout.write(f"Warning: Dropout probability specified as {self.lora_dropout}, setting this to zero since it may not be compatible with the current implementation.\n")
                    self.lora_dropout = 0.
    
    def get_lora_config(self) -> LoraConfig:
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        return config
    
    def select_peft_layers(self, **kwargs):
        SelectiveMethods.deactivate_model(self.model)

        if len(self.target_modules) > 0:
            config = self.get_lora_config()
            get_peft_model(self.model, config)

            # Replace LoRA layers with our custom implementation
            self._replace_lora_layers(self.model)

            if 'layer' in kwargs and kwargs['layer'] is not None:
                # Remove lora for all layers except the one specified
                self._remove_lora_layers_except_layer(self.model, kwargs['layer'])

        if self.use_peft_dict['biases']:
            sys.stdout.write("SelectivePEFT: Using biases.\n")
            SelectiveMethods.activate_biases(self.model)
        if self.use_peft_dict['layer_norm_params']:
            sys.stdout.write("SelectivePEFT: Using LN.\n")
            SelectiveMethods.activate_layer_norm_params(self.model)

    def _replace_lora_layers(self, model):
        """Replace the LoRA layers with the custom implementation."""
        for name, module in model.named_children():
            if isinstance(module, LoraLayer):
                if name in self.non_pruned_lora_layers:
                    # We skip the layers that use classical LoRA and are not pruned
                    continue
                elif self.CustomModule is not None:
                    setattr(model, name, self.CustomModule(
                        lora_layer=module, 
                        dropout_p=self.lora_dropout,
                    ))
            else:
                self._replace_lora_layers(module)

    def _remove_lora_layers_except_layer(self, model: torch.nn.Module, layer: torch.nn.Module):
        """Remove the LoRA layers except the one specified."""
        for name, module in model.named_children():
            if id(module) == id(layer):
                continue
            if self.CustomModule is not None and isinstance(module, self.CustomModule):
                setattr(model, name, module.lora_layer.base_layer)
            elif name in self.non_pruned_lora_layers:
                setattr(model, name, module.base_layer)
            else:
                self._remove_lora_layers_except_layer(module, layer)

    def at_train_end(self):
        self._merge_lora_layers(self.model)
    
    def _merge_lora_layers(self, model):
        """Merge the LoRA layers into the base model."""
        for name, module in model.named_children():
            if self.CustomModule is not None and isinstance(module, self.CustomModule):
                module.set_effective_weights()

                # Remove the custom LoRA layer and set the original one back
                setattr(model, name, module.lora_layer.base_layer)
                del module
            elif name in self.non_pruned_lora_layers:
                # We used classical LoRA for the classification layer, since not pruned, and we merge it in the classical way
                module.merge()
                setattr(model, name, module.base_layer)
                del module
            else:
                self._merge_lora_layers(module)