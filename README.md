# PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs

*Paper: [arxiv.org/abs/2312.15230](https://arxiv.org/abs/2312.15230)*

*Authors: [Max Zimmer](https://maxzimmer.org/), Megi Andoni, [Christoph Spiegel](http://www.christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the official implementation of PERP, a framework for pruning and retraining/reconstructing Large Language Models (LLMs). The code is built on PyTorch and uses [Weights & Biases](https://wandb.ai) for experiment tracking.

### Key Features

- Support for multiple LLM architectures (OPT, LLaMA-2, Mistral, Mixtral)
- Various pruning methods (magnitude, random, WANDA, SparseGPT)
- Sparsity-preserving reconstruction/retraining approaches such as MaskLoRA and ScaleLoRA
- Semi-Structured and unstructured sparsity patterns
- Parameter-Efficient Fine-Tuning (PEFT) integration
- Comprehensive experiment tracking with W&B

### Running Experiments

The main entry point is [`main.py`](main.py). If passed the `--debug` flag, the code will run in debug mode, executing the default configuration as specified in the `defaults` dictionary. Otherwise, the code expectes to be started by a WandB sweep agent. Hence, to run an experiment, either configure the parameters in `main.py` or use Weights & Biases sweeps. Key parameters include:

### Essential hyperparameters
- `training_mode`: Whether to retrain or reconstruct the model, must be one of `retrain`, `reconstruct`, or `None` (in which case the model is not retrained/reconstructed). Retraining is a full retraining of the model using the overall loss. Reconstruction operates layerwise by minimizing the per-layer $L_2$-deviation loss.
- `peft_strategy`: Whether to use SelectivePEFT or FullFT. FullFT retrains all parameters, consequently requires more memory. SelectivePEFT allows for more parameter-efficient fine-tuning, which can be further specified by the `peft_use_*` parameters.
- `lora_type`: When `SelectivePEFT` is used and `peft_use_lora` is `True`, this parameter specifies the LoRA-variant to use. Options are `lora`, `lora_prune`, `scale_lora`, `mask_lora`, corresponding to the variants proposed in the paper.

### Citation

If you find this work useful for your research, please consider citing:

```
@article{zimmer2023perp,
  title={Perp: Rethinking the prune-retrain paradigm in the era of llms},
  author={Zimmer, Max and Andoni, Megi and Spiegel, Christoph and Pokutta, Sebastian},
  journal={arXiv preprint arXiv:2312.15230},
  year={2023}
}
``` 