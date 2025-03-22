# ===========================================================================
# Project:        PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs - IOL Lab @ ZIB
# Paper:          arxiv.org/abs/2312.15230
# File:           utilities.py
# Description:    Utility functions.
# ===========================================================================

import math
import random
import sys
from typing import List, NamedTuple, Optional, Tuple
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm import tqdm
import torch
from datasets import load_dataset
from torch import nn
import transformers
import time



class PruneUtils:
    """Utilities for pruning"""

    @staticmethod
    def get_n_m_pruning_mask(W_saliency: torch.Tensor, prune_n: int, prune_m: int) -> torch.Tensor:
        """Normal n:m magnitude pruning. Prunes n out of every m weights, using the saliency in W. W should be the metric tensor, i.e. the absolute value tensor in the case of magnitude pruning."""
        W_mask = torch.zeros_like(W_saliency, dtype=torch.bool)
        for ii in range(W_saliency.shape[1]):
            if ii % prune_m == 0:
                tmp = W_saliency[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        return W_mask
    
    @staticmethod
    def prepare_calibration_input(args, model, dataloader, device):
        use_cache = model.config.use_cache
        model.config.use_cache = False
        #layers = model.model.decoder.layers
        layers = Utils.get_layerblock_list(model=model)

        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros((args.reconstruct_n_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']

                # Check if position_ids are passed, this is e.g. not the case for OPT models
                if 'position_ids' in kwargs:
                    cache['position_ids'] = kwargs['position_ids']
                raise ValueError
        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(device))
            except ValueError:
                pass 
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids

    @staticmethod
    @torch.no_grad()
    def eval_reconstruction_error(args: NamedTuple, layer: torch.nn.Module, inps: torch.Tensor, outs: torch.Tensor, attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor]) -> float:
        tensor_dataloader = Utils.get_tensor_dataloader(args, inps, outs)
        criterion = nn.MSELoss(reduction="mean").cuda()
        layer.eval()

        ret_loss = 0.
        for inputs, outs in tensor_dataloader:
            with torch.cuda.amp.autocast():
                if position_ids is not None:
                    outputs = layer(inputs,
                                    attention_mask=attention_mask.expand(args.batch_size, -1, -1, -1) if attention_mask is not None else None,
                                    position_ids=position_ids,
                                    )[0]
                else:
                    outputs = layer(inputs,
                                    attention_mask=attention_mask.expand(args.batch_size, -1, -1, -1) if attention_mask is not None else None,
                                    )[0]
                loss = criterion(outputs, outs)

            ret_loss += loss.item() * len(inputs)
        return ret_loss / len(inps)

class Utils:

    @staticmethod
    @torch.no_grad()
    def get_outputs(args: NamedTuple, layer: torch.nn.Module, inps: torch.Tensor, outs: torch.Tensor, attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor]):
        # Compute the outputs. Note: This should not use autocast, otherwise SparseGPT will run into problems
        for j in range(0, args.reconstruct_n_samples, args.batch_size):
            if position_ids is not None:
                outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size],
                                                            attention_mask=attention_mask.expand(args.batch_size, -1, -1, -1) if attention_mask is not None else None,
                                                            position_ids=position_ids
                                                            )[0]
            else:
                outs[j:j+args.batch_size] = layer(inps[j:j+args.batch_size],
                                                            attention_mask=attention_mask.expand(args.batch_size, -1, -1, -1) if attention_mask is not None else None,
                                                            )[0]

    @staticmethod
    def get_tensor_dataloader(args, inps, outs) -> torch.utils.data.DataLoader:
        """Return a TensorDataLoader for the given input and output tensors."""
        tensor_dataset = torch.utils.data.TensorDataset(inps, outs)
        tensor_dataloader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0,      
            pin_memory=False,  
        )
        return tensor_dataloader

    @staticmethod
    def change_model_state(model, train: bool):
        """Change the model state to train or eval mode."""
        sys.stdout.write(f"Changing model state to {'train' if train else 'eval'} mode.\n")
        if train:
            model.train()
            model.enable_input_require_grads() # Needed for PEFT: https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        else:
            model.eval()

    @staticmethod
    def compute_ppl_wikitext(model, testenc, bs=1, device=None):
        # Get input IDs
        testenc = testenc.input_ids

        # Calculate number of samples
        nsamples = testenc.numel() // model.seqlen

        # List to store negative log likelihoods
        nlls = []

        # Loop through each batch
        for i in tqdm(range(0, nsamples, bs)):
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
            inputs = inputs.reshape(j - i, model.seqlen)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * model.seqlen * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        sys.stdout.write(f"Computing perplexity for {nsamples} samples.\n")
        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

        sys.stdout.write(f"Perplexity: {ppl.item()} - Emptying CUDA Cache.\n")
        # Empty CUDA cache to save memory
        torch.cuda.empty_cache()
        sys.stdout.write(f"Emptying CUDA Cache done.\n")

        return ppl.item()
    
    
    @staticmethod
    def get_c4_for_calibration(nsamples: int, seed: int, seqlen: int, tokenized_dataset, tokenizer) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get calibration samples from pre-tokenized C4 dataset.
        
        Args:
            nsamples: Number of samples to generate
            seed: Random seed
            seqlen: Sequence length for each sample
            tokenized_dataset: Pre-tokenized C4 dataset -> we just use this to infer the dataset, we tokenize again to not have the truncation/padding issues
            tokenizer: HuggingFace tokenizer with defined pad_token_id
        """
        traindata = tokenized_dataset["train"]
        random.seed(seed)
        trainloader = []

        for _ in range(nsamples):
            max_iter = 10000
            it = 0
            while True:
                # Sample a random sequence
                i = random.randint(0, len(traindata) - 1)
                # Tokenize the sequence
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                # We have found a sequence that is long enough and must not be padded
                if trainenc.input_ids.shape[1] > seqlen:
                    break
                it += 1
                if it > max_iter:
                    raise ValueError("Could not find long enough sequence. If the model is Mistral, we might need to reduce the sequence length (its >30k).")
            # Sample a random start point for the subsequence
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen  # This is the end index
            inp = trainenc.input_ids[:, i:j]
            # Create target with all but last token masked
            tar = inp.clone()
            tar[:, :-1] = -100  # Sets everything except the last token to -100
            trainloader.append((inp, tar))

        return trainloader
    
    @staticmethod
    def get_percentage_of_trainable_parameters(model):
        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 100 * float(n_params_trainable) / n_params_total

    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                Utils.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d
    
    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = Utils.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict


    
    @staticmethod
    def eval_zero_shot(model_name, model, task_list):
        from lm_eval import evaluator, models

        model_args = f"pretrained={model_name},cache_dir=./llm_weights"
        limit = None 
        if "70b" in model_name or "65b" in model_name:
            limit = 2000

        model = models.huggingface.HFLM(pretrained=model)

        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,  # Now passing as dict instead of string
            tasks=task_list,
            num_fewshot=0,
            batch_size=None,
            device=None,
            limit=limit,
            check_integrity=False,
        )

        # Clean up the results, since some keys might end with ,0 or ,none
        cleaned_results = {'results': {}}
        for task in results['results'].keys():
            cleaned_results['results'][task] = {}
            for key, value in results['results'][task].items():
                clean_key = key.replace(",0", "").replace(",none", "")
                cleaned_results['results'][task][clean_key] = value
        results = cleaned_results

        return results

    @staticmethod
    def get_layers_of_modules(module, layers=[nn.Linear], name='') -> dict:
        """
        Recursively find the layers of a certain type in a module.

        Args:
            module (nn.Module): PyTorch module.
            layers (list): List of layer types to find.
            name (str): Name of the module.

        Returns:
            dict: Dictionary of layers of the given type(s) within the module.
        """
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(Utils.get_layers_of_modules(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
    
    @staticmethod
    def get_layerblock_list(model: torch.nn.Module):
        base_model = model
        while hasattr(base_model, "model"):
            base_model = base_model.model
        
        if hasattr(base_model, 'decoder'):
            # OPT CASE
            layer_list = base_model.decoder.layers
        elif hasattr(base_model, 'layers'):
            # LLAMA CASE
            layer_list = base_model.layers
        else:
            raise ValueError("Model does not have a 'decoder' or 'layers' attribute.")   
        return layer_list
    
    @staticmethod
    def check_sparsity(model):
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

        layers = Utils.get_layerblock_list(model=model)
        
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = Utils.get_layers_of_modules(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        model.config.use_cache = use_cache 
        return float(count)/total_params 


class SelectiveMethods:
    """Class of methods that can be used to selectively activate and deactivate certain layers."""

    @staticmethod
    def activate_model(model: torch.nn.Module):
        """Activate the entire model"""
        for param in model.parameters():
            param.requires_grad_(True)

    @staticmethod
    def deactivate_model(model: torch.nn.Module):
        """Deactivate the entire model"""
        for param in model.parameters():
            param.requires_grad_(False)


    @staticmethod
    def activate_specific_modules(moduleList: List[torch.nn.Module]):
        """Activate specific modules"""
        for module in moduleList:
            for param in module.parameters():
                param.requires_grad_(True)

    @staticmethod
    def change_biases_activation(model: torch.nn.Module, activate: bool):
        """Change the bias of a model to be trainable or not"""
        for module in model.modules():
            if hasattr(module, 'bias') and not isinstance(module.bias, type(None)):
                module.bias.requires_grad_(activate)

    @staticmethod
    def deactivate_biases(model: torch.nn.Module):
        """Deactivate the bias of a model"""
        SelectiveMethods.change_biases_activation(model, False)

    @staticmethod
    def activate_biases(model: torch.nn.Module):
        """Activate the bias of a model"""
        SelectiveMethods.change_biases_activation(model, True)

    @staticmethod
    def change_layer_norm_activation(model: torch.nn.Module, activate: bool):
        """Change the layer params of a model to be trainable or not"""
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm) or \
               isinstance(module, transformers.models.mistral.modeling_mistral.MistralRMSNorm) or \
               isinstance(module, transformers.models.llama.modeling_llama.LlamaRMSNorm) or \
               isinstance(module, transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm):
                module.requires_grad_(requires_grad=activate)

    @staticmethod
    def deactivate_layer_norm_params(model: torch.nn.Module):
        """Deactivate the layer norm params of a model"""
        SelectiveMethods.change_layer_norm_activation(model, False)

    @staticmethod
    def activate_layer_norm_params(model: torch.nn.Module):
        """Activate the layer norm params of a model"""
        SelectiveMethods.change_layer_norm_activation(model, True)


class WandaWrapper:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


class SparseGPTWrapper:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()