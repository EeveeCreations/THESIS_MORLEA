import numpy as np
import torch.optim as optim

import importlib
PPO = importlib.import_module("pymoo.PPO_CON")


PPO_SEARCH_SPACE = {
    "gamma": (0.90, 0.999),
    "lambda": (0.90, 0.99),
    "clip": (0.1, 0.4),

    # log-uniform range
    "learning_rate": (1e-6, 1e-3),

    "epochs": (3, 20),

    "entropy_coeff": (0.0, 0.1),

    "actor_loss_coeff": (0.1, 2.0),
}

def sample_ppo_config():

    config = {

        "gamma":
            np.random.uniform(*PPO_SEARCH_SPACE["gamma"]),

        "lambda":
            np.random.uniform(*PPO_SEARCH_SPACE["lambda"]),

        "clip":
            np.random.uniform(*PPO_SEARCH_SPACE["clip"]),

        # log-uniform sampling
        "learning_rate":
            10 ** np.random.uniform(
                np.log10(PPO_SEARCH_SPACE["learning_rate"][0]),
                np.log10(PPO_SEARCH_SPACE["learning_rate"][1])
            ),

        "epochs":
            np.random.randint(
                PPO_SEARCH_SPACE["epochs"][0],
                PPO_SEARCH_SPACE["epochs"][1] + 1
            ),

        "entropy_coeff":
            np.random.uniform(*PPO_SEARCH_SPACE["entropy_coeff"]),

        "actor_loss_coeff":
            np.random.uniform(*PPO_SEARCH_SPACE["actor_loss_coeff"]),
    }

    return config


config = sample_ppo_config()

agent = PPO(
    state_dim,
    action_dim,
    config
)