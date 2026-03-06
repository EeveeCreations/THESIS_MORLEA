import gymnasium
from gymnasium import spaces
import numpy as np
from pymoo.optimize import minimize

from OLD.MOEA_RL import algorithm

MUTATION = [0.01,0.1,0.3]
CROSSOVER = [0.6,0.8,0.9]

ACTION_SPACE = [ (m,c) for m in MUTATION for c in CROSSOVER]

N_ACTIONS = len(ACTION_SPACE)


class EAEnv(gymnasium.Env):
    def __init__(self, algorithem, problem):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 1.0]),
                                       high=np.array([1.0, 50.0]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.state = np.zeros(3)
        self.problem = problem
        self.algorithem = algorithem
        self.max_steps = 50
        self.step_count = 0

    def  step(self, action):
        current_state = self.state
        next_state = self.observation_space.sample(action)
        mutation, crossover = ACTION_SPACE[action]
        result = minimize(self.problem,self.algorithem, ('n_gen',1), seed=1)
        hyper_vol = compute_hypervolume(result.F)
        # diversity = 0  # Diversiity will be measure ocne we  have  tevrythign else goign too
        progress = self.step_count / self.max_steps

        self.state = np.array([progress, improvement], dtype=np.float32)

        # SImply look if the  hv
        improvement = hyper_vol - self.prev_hyper_vol #Reward
        self.prev_hyper_vol = hyper_vol
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self.state, improvement, terminated, truncated #  infor will ebadded  later, {}

    def reset(self):
        self.step_count = 0
        self.state = np.zeros(3)
        return self.state # migjt be abl to a dd some info if needed later on
