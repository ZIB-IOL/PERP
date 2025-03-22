# ===========================================================================
# Project:        PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs - IOL Lab @ ZIB
# Paper:          arxiv.org/abs/2312.15230
# File:           runner.py
# Description:    Core runner class that handles model loading, training, evaluation, and experiment orchestration.
# ===========================================================================

from collections import namedtuple
import os
import sys
import time
import wandb
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import peft_methods
from utilities import Utils

from transformers import TrainingArguments, Trainer
import datasets
from prune_methods import PruneMethod


class Runner:
    def __init__(self, config, tmp_dir, debug):
        self.config = config
        self.tmp_dir = tmp_dir
        self.debug = debug
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_opt = 'facebook/opt' in self.config.model
        self.is_llama = 'meta-llama/Llama-2' in self.config.model or 'meta-llama/Llama-3.' in self.config.model
        self.is_mistral = 'mistralai' in self.config.model
        assert self.is_opt or self.is_llama or self.is_mistral, f"Model family not supported for model {self.config.model}."

        # Define the cache base directory
        cache_base = os.path.join(os.getcwd(), 'cache')
        
        self.directoryDict = {
            'output': os.path.join(self.tmp_dir, 'output'),  # Directory for model checkpoints
            'pretrained_models': os.path.join(cache_base, 'pretrained_models'),
            'datasets': os.path.join(cache_base, 'datasets'),
            'tokenized_datasets': os.path.join(cache_base, 'tokenized_datasets'),
        }
            
        # Create all necessary directories
        for dir_path in self.directoryDict.values():
            os.makedirs(dir_path, exist_ok=True)

        # Variables to be defined
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_strategy = None

        # Other metrics
        self.time_for_pruning, self.time_for_retraining = None, None
        self.reconstruction_error_initial, self.reconstruction_error_final = None, None

    def get_llm(self, model_name):
        device_map = "auto"
        torch_dtype = torch.float16 # In the original setup, this was specified as torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=self.directoryDict['pretrained_models'],
            low_cpu_mem_usage=True,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )

        if model.config.max_position_embeddings > 4096:
            model.seqlen = 4096
            sys.stdout.write(f"Avoiding OOM by setting model.seqlen to 4096 for {model_name}.\n")
        else:
            model.seqlen = model.config.max_position_embeddings
        
        if self.is_opt:
            # Mistral uses flash attention 2 and is loaded with attn_implementation as above
            # Llama natively supports BetterTransformer since transformers>=4.36 (requiring torch>=2.1.1)
            model = model.to_bettertransformer()    # Flash attention as per https://huggingface.co/docs/transformers/perf_train_gpu_one
        return model

    def change_model_state(self, train: bool):
        sys.stdout.write(f"Changing model state to {'train' if train else 'eval'} mode.\n")
        if train:
            self.model.train()
            self.model.enable_input_require_grads() # Needed for PEFT: https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        else:
            self.model.eval()

    def eval_on_wikitext(self):
        train_state = self.model.training
        self.change_model_state(train=False)

        # Load test dataset
        testLoader = self.get_dataset('wikitext2', enable_tokenized_cache=False)

        sys.stdout.write(f"Evaluating wikitext2.\n")
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = Utils.compute_ppl_wikitext(self.model, testLoader, 1, self.device)
        self.change_model_state(train=train_state)
        return ppl_test


    def get_dataset(self, dataset_name: str, enable_tokenized_cache: bool = True) -> tuple:
        """Returns the tokenized datasets for the given dataset name.
        If enable_tokenized_cache is True, the tokenized datasets are cached locally.
        """
        sys.stdout.write(f"Loading {dataset_name}.\n")
        assert dataset_name in ['wikitext2', 'c4'], f"Dataset {dataset_name} not supported."

        if self.is_opt:
            prefix = 'opt-'
        elif self.is_llama:
            prefix = 'llama-'
        elif self.is_mistral:
            prefix = 'mistral-'

        tokenized_data_location = os.path.join(self.directoryDict['tokenized_datasets'], f"{prefix}{dataset_name}")

        if enable_tokenized_cache and os.path.exists(tokenized_data_location):
            # The tokenized dataset is cached locally already
            sys.stdout.write(f"Loading tokenized {dataset_name} dataset from {tokenized_data_location}.\n")
            tokenized_datasets = datasets.load_from_disk(tokenized_data_location)
        else:
            # The dataset is not cached locally, hence load the raw dataset and tokenize it
            sys.stdout.write(f"Tokenizing {dataset_name}.\n")
            if dataset_name == 'c4':
                raw_datasets = load_dataset('allenai/c4', 'en',
                                          data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                    'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                          cache_dir=self.directoryDict['datasets'],
                                          verification_mode=datasets.VerificationMode.NO_CHECKS,  # Otherwise leads to a bug with c4 validation set
                                          )

                tokenize_function = lambda examples: self.tokenizer(examples["text"], padding=True, truncation=True, max_length=self.model.seqlen)
                tokenized_datasets = raw_datasets.map( 
                    tokenize_function,
                    batched=True,
                    desc="Running tokenizer on dataset")
                
            elif dataset_name == 'wikitext2':
                # Load test dataset
                testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=self.directoryDict['datasets'])
                tokenized_datasets = self.tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

            # Save the tokenized datasets if caching is enabled
            if enable_tokenized_cache:
                os.makedirs(tokenized_data_location, exist_ok=True)
                tokenized_datasets.save_to_disk(tokenized_data_location)

        if dataset_name == 'c4':
            # Take only 100 random samples for validation
            tokenized_datasets['validation'] = tokenized_datasets['validation'].shuffle().select(range(100))

        return tokenized_datasets
            
    def make_model_param_efficient(self):
        sys.stdout.write(f"Percentage of parameters with grad without PEFT: {Utils.get_percentage_of_trainable_parameters(self.model)}\n")

        # Enable grad for all parameters that correspond to the peft strategy at stake
        assert hasattr(peft_methods, self.config.peft_strategy), f"PEFT strategy {self.config.peft_strategy} not implemented."
        self.peft_strategy = getattr(peft_methods, self.config.peft_strategy)(model=self.model, runner=self, config=self.config, total_iterations=self.config.n_iterations, is_reconstruct=False)
        self.peft_strategy.select_peft_layers()

        for param in self.model.parameters():
            if param.requires_grad:
                # Important: Set trainable parameters to float32, otherwise this won't work with fp16=True -> https://github.com/huggingface/peft/issues/341#issuecomment-1519460307
                param.data = param.data.float()
        
        sys.stdout.write(f"Percentage of parameters with grad with PEFT: {Utils.get_percentage_of_trainable_parameters(self.model)}\n")

    def log_metrics(self, state, additional_metrics=None):
        metrics = {
            'metrics/sparsity': Utils.check_sparsity(self.model),
            'metrics/ppl_test': self.eval_on_wikitext(),
            'metrics/reconstruction_error_initial': self.reconstruction_error_initial,
            'metrics/reconstruction_error_final': self.reconstruction_error_final,
        }

        if additional_metrics is not None:
            metrics.update(additional_metrics)

        for time_metric in ['time_for_pruning', 'time_for_retraining']:
            if getattr(self, time_metric) is not None:
                metrics[f'metrics/{time_metric}'] = getattr(self, time_metric)
                
                # Set to None to avoid logging the same metric twice
                setattr(self, time_metric, None)

        # Log the metrics to the wandb summary with the given state as prefix
        for key, val in metrics.items():
            wandb.run.summary[f'{state}/{key}'] = val

        step = self.trainer.state.global_step if self.trainer is not None else 0
        commit = self.trainer is not None
        sys.stdout.write(f"Logging metrics to wandb with step {step} and commit {commit}.\n")
        wandb.log(metrics, step=step, commit=commit)
        sys.stdout.write(f"Finished logging metrics to wandb with step {step} and commit {commit}.\n")

    def train_on_c4(self):
        self.change_model_state(train=True)

        # Load the tokenized dataset
        tokenized_datasets = self.get_dataset('c4')


        # Huggingface trainer approach
        training_args = TrainingArguments(
            # Training hyperparameters
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            max_steps=self.config.n_iterations,
            learning_rate=self.config.initial_lr,
            lr_scheduler_type=self.config.lr_scheduler_type or 'linear',  # Linear learning rate decay
            warmup_ratio=0.1,  # Warmup ratio for linear learning rate scheduler, keep fixed at 10%
            weight_decay=self.config.weight_decay or 0.,  # Strength of weight decay
            
            # Evaluation
            evaluation_strategy='no',
            eval_steps=100,            

            # Additional optimization parameters
            gradient_accumulation_steps=self.config.gradient_accumulation_steps or 1,  # Number of updates steps to accumulate before performing a backward/update pass.
            fp16=True,  # Use mixed precision
            gradient_checkpointing=self.config.gradient_checkpointing,  # If true, enables gradient checkpointing to save memory
            optim=self.config.optim or 'adamw_torch',

            # Logging
            report_to="wandb",  # Enable logging to W&B
            logging_steps=100,  # Log every X updates steps
            logging_first_step=True,    # Log also the first step
            include_tokens_per_second=self.config.include_tokens_per_second or False,   # Log the tokens per second, however this increases the overall runtime

            # Model Checkpointing
            output_dir=self.directoryDict['output'],
            overwrite_output_dir=True,
            save_strategy="no", # Do not save the model checkpoints
            
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        self.trainer.train()


    def prune_model(self, sparsity):
        """Prune the model using the specified method."""
        # Check whether we retrain and no peft method is selected, then we have to keep the masks (only working for unstructured magnitude pruning as of now)
        keep_masks = self.config.peft_strategy == 'FullFT' and (self.config.training_mode == 'retrain' or self.config.training_mode == 'reconstruct')
        sys.stdout.write(f"Keeping masks: {keep_masks}.\n")

        # Define the necessary args
        args = {
            'training_mode': self.config.training_mode,
            'reconstruct': self.config.training_mode == 'reconstruct',
            'reconstruct_n_samples': self.config.reconstruct_n_samples,
            'n_iterations': self.config.n_iterations,
            'batch_size': self.config.batch_size,
            'initial_lr': self.config.initial_lr,
            'seed': self.config.seed,
            'sparsity_ratio': sparsity,
            'cache_dir': self.directoryDict['datasets'],
            'tokenizer': self.tokenizer,
            'device': self.device,
            'keep_masks': keep_masks,
        }

        # Handle n:m sparsity
        prune_n, prune_m = 0, 0
        sparsity_type = self.config.sparsity_type or 'unstructured'
        assert sparsity_type in ['unstructured', '2:4', '4:8'], f"Sparsity type {sparsity_type} not supported."
        if sparsity_type != 'unstructured':
            assert self.config.goal_sparsity == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity. Note: currently, this is implemented as pruning n out of m, instead of m-n out of m."
            prune_n, prune_m = map(int, sparsity_type.split(":"))

        # Define a named tuple to pass the args (since then they can be accessed as attributes)
        NamedTupleClass = namedtuple('ArgTuple', args)
        args = NamedTupleClass(**args)

        pruneMethod = PruneMethod(runner=self, args=args, prune_method=self.config.prune_method, model=self.model, prune_n=prune_n, prune_m=prune_m)
        pruneMethod.prune()

        if self.config.training_mode == 'reconstruct':
            previous_train_state = self.model.training
            self.change_model_state(train=True)
            self.reconstruction_error_initial, self.reconstruction_error_final = pruneMethod.get_reconstruction_errors()
            self.change_model_state(train=previous_train_state)
        
    def get_zeroshot_metrics(self, debug=False):
        sys.stdout.write(f"Evaluating zero-shot performance.\n")
        train_state = self.model.training
        self.change_model_state(train=False)
        task_list = ["boolq", "rte", "hellaswag","winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        if debug:
            task_list = task_list[:1]
        results = Utils.eval_zero_shot(self.config.model, self.model, task_list)['results']
        zero_shot_acc = {f"metrics/zero_shot_acc_{task}": results[task]['acc'] for task in task_list}
        zero_shot_stderr = {f"metrics/zero_shot_accstderr_{task}": results[task]['acc_stderr'] for task in task_list}
        avg_zero_shot_acc = np.mean([results[task]['acc'] for task in task_list])
        zero_shot_metrics = {
            'metrics/avg_zero_shot_acc': avg_zero_shot_acc,
            **zero_shot_acc,
            **zero_shot_stderr,
        }
        self.change_model_state(train=train_state)
        return zero_shot_metrics

    def run(self):
        # Setting seeds for reproducibility
        np.random.seed(self.config.seed)
        torch.random.manual_seed(self.config.seed)
        sys.stdout.write(f"Running on node {self.config.computer} with seed {self.config.seed}.\n")

        sys.stdout.write(f"Loading LLM model {self.config['model']}.\n")
        # Load model and tokenizer
        self.model = self.get_llm(self.config['model'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], use_fast=False)

        if self.is_llama or self.is_mistral:
            self.tokenizer.pad_token = self.tokenizer.eos_token # For shorter sequences, the eos_token is used as padding token
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Reconfigure the device in the case of multiple GPUs (set to the device of lm_head)
        if torch.cuda.device_count() > 1:
            self.device = self.model.hf_device_map["lm_head"]
            sys.stdout.write(f"Using {torch.cuda.device_count()} GPUs - setting self.device = {self.device}.\n")
            

        # Prune the model
        if self.config.goal_sparsity > 0:
            t_start = time.time()
            self.prune_model(self.config['goal_sparsity'])  
            self.time_for_pruning = time.time() - t_start

        self.log_metrics(state='pruned')
        additional_metrics = None
        if self.config.training_mode == 'retrain':
            # Make the model parameter efficient
            self.make_model_param_efficient()

            # Fine-tune the model on C4
            t_start = time.time()
            self.train_on_c4()
            self.time_for_retraining = time.time() - t_start

            # Potentially merge the adapters, but first, collect additional metrics that are only available before merging
            additional_metrics = {'metrics/fraction_of_trainable_parameters': Utils.get_percentage_of_trainable_parameters(self.model)}
            self.peft_strategy.at_train_end()

        if self.config.training_mode == 'retrain' or self.config.training_mode == 'reconstruct':
            self.log_metrics(state='retrained', additional_metrics=additional_metrics)
        
        # Evaluate zero-shot performance
        if self.config.eval_zero_shot:
            zero_shot_metrics = self.get_zeroshot_metrics(debug=self.debug)

            # Log the metrics to the wandb summary
            for key, val in zero_shot_metrics.items():
                wandb.run.summary[key] = val
            wandb.log(zero_shot_metrics, commit=True)