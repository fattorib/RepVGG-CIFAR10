import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EWMAModel:
    def __init__(self, base, alpha=0.9999):
        # Create model based on exponentially weighted moving average (EWMA) of paramaters over training

        self.model = base
        self.alpha = alpha

    def update(self, new_model_state):
        ewma_params_dict = self.model.state_dict()

        with torch.no_grad():
            for key in ewma_params_dict:
                if (
                    "running_var" in key
                    or "running_mean" in key
                    or "num_batches_tracked" in key
                ):
                    # No ewma on these BN params
                    pass
                else:
                    ewma_params_dict[key] = (
                        self.alpha * ewma_params_dict
                        + (1 - self.alpha) * new_model_state[key]
                    )

        self.model.load_state_dict(ewma_params_dict)

    def forward(self, x):

        return self.model(x)

    def __call__(self, x):
        return self.model(x)
