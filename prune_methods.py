# ===========================================================================
# Project:        PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs - IOL Lab @ ZIB
# Paper:          arxiv.org/abs/2312.15230
# File:           prune_methods.py
# Description:    Implementation of various pruning methods including magnitude, random, Wanda, and SparseGPT pruning. Further controls the reconstruction.
# ===========================================================================

import sys
from tqdm import tqdm
from typing import NamedTuple, Optional 
import numpy as np
import torch 
import torch.nn as nn 

import torch.nn.utils.prune as prune
from utilities import Utils, PruneUtils, WandaWrapper, SparseGPTWrapper
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import peft_methods

class PruneMethod:

    def __init__(self, runner, args: NamedTuple, prune_method: str, model: torch.nn.Module, prune_n: int = 0, prune_m: int = 0):
        self.runner = runner
        self.args = args
        self.prune_method = prune_method
        self.model = model
        self.prune_n = prune_n
        self.prune_m = prune_m       

        assert self.prune_method in ["magnitude", "random", "wanda", "sparsegpt"], "Invalid pruning method."
        self.requires_calibration_wrapper = self.prune_method in ["wanda", "sparsegpt"]

        self.reconstruction_error_initial, self.reconstruction_error_final = None, None

    @torch.no_grad()
    def prune(self):
        device = self.args.device
        layers = Utils.get_layerblock_list(model=self.model)

        use_cache = self.model.config.use_cache 
        self.model.config.use_cache = False
        initial_losses = []
        final_losses = []
        if self.args.reconstruct or self.requires_calibration_wrapper:
            sys.stdout.write("Loading calibration data.\n")
            dataloader = Utils.get_c4_for_calibration(nsamples=self.args.reconstruct_n_samples, seed=self.args.seed, seqlen=self.model.seqlen,tokenized_dataset=self.runner.get_dataset('c4'), tokenizer=self.runner.tokenizer)

            with torch.no_grad():
                inps, outs, attention_mask, position_ids = PruneUtils.prepare_calibration_input(self.args, self.model, dataloader, device)

        wrappers, hooks = {}, []
        for i in range(len(layers)):
            layer = layers[i]
            subset = Utils.get_layers_of_modules(layer)
            if self.args.reconstruct or self.requires_calibration_wrapper:
                if f"self.model.layers.{i}" in self.model.hf_device_map:
                    device = self.model.hf_device_map[f"self.model.layers.{i}"]
                    inps, outs = inps.to(device), outs.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    if position_ids is not None:
                        position_ids = position_ids.to(device)

                wrappers, hooks = self.get_wrappers_and_hooks(layer_subset=subset)

                Utils.get_outputs(args=self.args, layer=layer, inps=inps, outs=outs, attention_mask=attention_mask, position_ids=position_ids)  # Changes the outs dynamically
                for h in hooks:
                    h.remove()

            # Prune the block
            for name in subset:
                W = subset[name].weight.data

                # Determine Saliency criterion
                if self.prune_method == 'magnitude':
                    # Use the magnitude as saliency
                    W_metric = torch.abs(W)      
                elif self.prune_method == 'random':
                    # Random saliency
                    W_metric = torch.randn_like(W)
                elif self.prune_method == 'wanda':
                    # Wanda pruning criterion
                    W_metric = torch.abs(W) * torch.sqrt(wrappers[name].scaler_row.reshape((1,-1)))
                elif self.prune_method == 'sparsegpt':
                    # SparseGPT pruning criterion, the weights are changed automatically and we infer the mask from the weights, hence the saliency is None
                    wrappers[name].fasterprune(sparsity=self.args.sparsity_ratio, prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01, blocksize=self.args.reconstruct_n_samples)
                    wrappers[name].free()                
                    W_metric = None
                    W_mask = (subset[name].weight == 0)

                # Prune
                if W_metric is not None:    # True for all methods except sparsegpt
                    if self.prune_n != 0:
                        W_mask = PruneUtils.get_n_m_pruning_mask(W_saliency=W_metric, prune_n=self.prune_n, prune_m=self.prune_m)
                    else:
                        thresh = torch.sort(W_metric.flatten().to(device=device))[0][int(W.numel()*self.args.sparsity_ratio)].cpu()                
                        W_mask = (W_metric <= thresh)

                # Prune the weights
                prune.custom_from_mask(subset[name], name='weight', mask=~W_mask)

                if not self.args.keep_masks:
                    # We can remove the masks now already, since we either reconstruct non-pruned parameters or we handle this in the peft method
                    if prune.is_pruned(subset[name]):
                        prune.remove(subset[name], name='weight')

            if self.args.reconstruct:
                loss_initial = PruneUtils.eval_reconstruction_error(args=self.args, layer=layer, inps=inps, outs=outs, attention_mask=attention_mask, position_ids=position_ids)
                with torch.enable_grad():   # Selectively enable grad only for the reconstruction
                    final_loss = self.reconstruct_weights(layer=layer, inps=inps, outs=outs,
                                                           attention_mask=attention_mask, position_ids=position_ids, layer_idx=i, n_layers=len(layers))

                initial_losses.append(loss_initial)
                final_losses.append(final_loss)
            else:
                initial_losses.append(np.nan)
                final_losses.append(np.nan)
            

            # If we do not retrain further, we can now definitely remove the masks
            if not self.args.keep_masks or not self.args.training_mode == 'retrain':
                for name in subset:
                    if prune.is_pruned(subset[name]):
                        prune.remove(subset[name], name='weight')

            if self.args.reconstruct or self.requires_calibration_wrapper:
                # We update the inputs of the next layer (i.e. the outputs of the previous one) by performing another forward through the pruned (!) layer
                Utils.get_outputs(args=self.args, layer=layer, inps=inps, outs=outs, attention_mask=attention_mask, position_ids=position_ids)  # Changes the outs dynamically
                inps, outs = outs, inps
            torch.cuda.empty_cache()

        self.model.config.use_cache = use_cache 
        torch.cuda.empty_cache()

        # Print out the initial and final losses in each layer, as well as the total initial loss and total final loss
        if self.args.reconstruct:
            for i in range(len(layers)):
                sys.stdout.write(f"Layer {i}: Initial Loss: {initial_losses[i]}, Final Loss: {final_losses[i]}\n")
            sys.stdout.write(f"Total Initial Loss: {sum(initial_losses)}, Total Final Loss: {sum(final_losses)}\n")

            self.reconstruction_error_initial, self.reconstruction_error_final = sum(initial_losses), sum(final_losses)
        else:
            self.reconstruction_error_initial, self.reconstruction_error_final = None, None


    def reconstruct_weights(self, layer: torch.nn.Module, inps: torch.Tensor, outs: torch.Tensor, attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor], layer_idx: int, n_layers: int, ignore_reconstruction_method: bool = False) -> float:
        """Reconstructs the weights of the layer using the input-output pairs."""
        tensor_dataloader = Utils.get_tensor_dataloader(self.args, inps, outs)

        peft_strategy = self.runner.config.peft_strategy
        # Enable grad for all parameters that correspond to the peft strategy at stake
        assert hasattr(peft_methods, peft_strategy), f"PEFT strategy {peft_strategy} not implemented."
        peft_strategy = getattr(peft_methods, peft_strategy)(model=self.model, runner=self, config=self.runner.config, total_iterations=self.args.n_iterations, is_reconstruct=True)
        peft_strategy.select_peft_layers(layer=layer)

        original_dtype = next(iter(layer.parameters())).dtype
        for param in layer.parameters():
            if param.requires_grad:
                # Important: Set trainable parameters to float32, otherwise this won't work with fp16=True -> https://github.com/huggingface/peft/issues/341#issuecomment-1519460307
                param.data = param.data.float()

        n_iterations = self.args.n_iterations
        n_warmup_iterations = int(0.1 * n_iterations)
        lr = self.args.initial_lr
        train_config = [
            {"params": [p for n, p in layer.named_parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": 0.,
            },
        ]
        sys.stdout.write(f"Layer {layer_idx+1}/{n_layers}: Reconstructing with {self.runner.config.peft_strategy}, {Utils.get_percentage_of_trainable_parameters(layer):.4f}% trainable params\n")

        optimizer = AdamW(train_config)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_iterations, num_training_steps=n_iterations)
        criterion = nn.MSELoss(reduction="mean").cuda()
     
        gradScaler = torch.cuda.amp.GradScaler()
        layer.train()

        for step in tqdm(range(1, n_iterations + 1, 1)):
            # Reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len(tensor_dataloader) == 0:
                train_iterator = iter(tensor_dataloader)
            inputs, outps = next(train_iterator)
            with torch.cuda.amp.autocast():
                if position_ids is not None:
                    outputs = layer(inputs,
                                    attention_mask=attention_mask.expand(self.args.batch_size, -1, -1, -1) if attention_mask is not None else None,
                                    position_ids=position_ids,
                                    )[0]
                else:
                    outputs = layer(inputs,
                                    attention_mask=attention_mask.expand(self.args.batch_size, -1, -1, -1) if attention_mask is not None else None,
                                    )[0]
                loss = criterion(outputs, outps)

            gradScaler.scale(loss).backward()
            # Unscale the weights manually, normally this would be done by gradScaler.step(), but we might modify_grads()
            gradScaler.unscale_(optimizer)
            gradScaler.step(optimizer)
            gradScaler.update()
            scheduler.step()
            optimizer.zero_grad()
            layer.zero_grad()

        torch.cuda.empty_cache()

        final_loss = PruneUtils.eval_reconstruction_error(args=self.args, layer=layer, inps=inps, outs=outs, attention_mask=attention_mask, position_ids=position_ids)

        # Reset the original data type
        for param in layer.parameters():
            if param.requires_grad:
                param.data = param.data.to(original_dtype)

        peft_strategy.at_train_end()

        return final_loss
    
    def get_wrappers_and_hooks(self, layer_subset: dict) -> tuple[dict, list]:
        """Gets the wrappers and hooks needed for the pruning method."""
        wrappers, hooks = {}, []
        if self.prune_method in ["wanda", "sparsegpt"]:
            WrapperClass = WandaWrapper if self.prune_method == "wanda" else SparseGPTWrapper
            for name in layer_subset:
                wrappers[name] = WrapperClass(layer_subset[name])
            def define_hook_fn(name):
                def hook_fn(_, inp, out):
                    wrappers[name].add_batch(inp[0].data, out.data)
                return hook_fn

            for name in wrappers:
                hooks.append(layer_subset[name].register_forward_hook(define_hook_fn(name)))

        return wrappers, hooks

    def get_reconstruction_errors(self) -> tuple[float]:
        """Returns the initial and final reconstruction errors."""
        return self.reconstruction_error_initial, self.reconstruction_error_final