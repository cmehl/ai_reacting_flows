"""NN_manager with two opt-in regularisation knobs, read from the ``learning``
block of networks_params.yaml (both default to off, so an unchanged yaml
trains exactly like NN_manager):

    learning:
      weight_decay: 1.0e-5   # L2 penalty passed to torch.optim.Adam
      input_noise: 0.05      # std of Gaussian noise added to the (standardised)
                             # network input in training mode only
      seed: 1                # seeds python/numpy/torch/CUDA (initialisation and
                             # batch order; GPU kernels stay slightly non-deterministic)

Implemented as a subclass so NN_manager itself is untouched. The noise is a
forward pre-hook registered for the duration of train_model only and removed
before the model is saved, so the saved .pth / .h5 contain no hook.
"""

import os
import random

import numpy as np
import oyaml as yaml
import torch
import torch.optim as optim

import ai_reacting_flows.ann_model_generation.NN_manager as nnm


class NNManagerReg(nnm.NN_manager):

    def __init__(self, run_folder: str | None = None):
        super().__init__(run_folder)
        with open(os.path.join(self.run_folder, "networks_params.yaml"), "r") as file:
            learning = yaml.safe_load(file)["learning"]
        self.weight_decay = float(learning.get("weight_decay", 0.0))
        self.input_noise = float(learning.get("input_noise", 0.0))
        seed = learning.get("seed")
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        self._log(f"REG weight_decay={self.weight_decay}, input_noise={self.input_noise}, seed={seed}")

    def train_model(self, i_cluster, model, *args, **kwargs):
        handle = None
        if self.input_noise > 0:
            sigma = self.input_noise

            def _add_noise(module, inputs):
                if module.training:
                    x = inputs[0]
                    return (x + sigma * torch.randn_like(x),) + tuple(inputs[1:])

            handle = model.register_forward_pre_hook(_add_noise)
        try:
            return super().train_model(i_cluster, model, *args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

    def train_all_clusters(self):
        if self.weight_decay <= 0:
            return super().train_all_clusters()
        original_adam = optim.Adam
        wd = self.weight_decay
        optim.Adam = lambda params, **kw: original_adam(params, weight_decay=wd, **kw)
        try:
            return super().train_all_clusters()
        finally:
            optim.Adam = original_adam
