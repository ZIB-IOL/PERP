# ===========================================================================
# Project:        PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs - IOL Lab @ ZIB
# Paper:          arxiv.org/abs/2312.15230
# File:           main.py
# Description:    Main entry point for running experiments, handling configuration, initialization, and orchestration of the pruning and retraining pipeline.
# ===========================================================================

import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import wandb

from runner import Runner
from utilities import Utils

debug = "--debug" in sys.argv

defaults = dict(
    # System
    seed=0,                                         # Random seed   

    # Models
    model='facebook/opt-125m',                      # Model to use, available models: facebook/opt-*, meta-llama/Llama-2-*, mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1, meta-llama/Llama-3.*

    # Pruning config
    goal_sparsity=0.5,                              # Goal sparsity for pruning, must be 0.5 for 2:4, 4:8
    prune_method='magnitude',                       # ['magnitude', 'random', 'wanda', 'sparsegpt']
    sparsity_type='unstructured',                   # [None, 'unstructured', '2:4', '4:8'], defaults to 'unstructured'
    
    # PEFT Config
    training_mode='retrain',                        # ['retrain', 'reconstruct', None]
    peft_strategy='SelectivePEFT',                  # ['SelectivePEFT', 'FullFT']
    peft_use_bias=True,                             # If peft_strategy is SelectivePEFT, this decides whether to train biases
    peft_use_ln=True,                               # If peft_strategy is SelectivePEFT, this decides whether to train LN params
    peft_use_lm_head=True,                          # If peft_strategy is SelectivePEFT, this decides whether to train the LM head
    peft_use_lora=True,                             # If peft_strategy is SelectivePEFT, this decides whether to use LoRA adapters
    lora_r=16,                                      # LoRA rank
    lora_alpha=32,                                  # LoRA scaling factor
    lora_dropout=0.,                                # LoRA dropout probability
    lora_type='mask_lora',                          # ['lora', 'lora_prune', 'scale_lora', 'mask_lora']

    # Reconstruction config
    reconstruct_n_samples=128,                      # Number of C4-samples to use for reconstruction/wanda/sparsegpt

    # Optimization config
    batch_size=1,                                   # Batch size
    n_iterations=20,                                # Number of iterations
    initial_lr=5e-04,                               # Initial learning rate
    weight_decay=None,                              # Defaults to 0
    gradient_accumulation_steps=2,                  # Defaults to 1
    gradient_checkpointing=True,                    # Whether to use gradient checkpointing
    optim=None,                                     # Defaults to AdamW
    lr_scheduler_type='linear',                     # Defaults to 'linear'

    # Evaluation
    eval_zero_shot=False,                           # Whether to evaluate the model on zero-shot tasks

    # Other
    include_tokens_per_second=False,                # Defaults to False since it increases the overall runtime
    )

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)


@contextmanager
def tempdir():
    """Create a temporary directory that will be cleaned up after use."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
