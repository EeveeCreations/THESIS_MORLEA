import gymnasium
from gymnasium import spaces
import numpy as np


from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM


MUTATION = [0.01,0.1,0.3]
CROSSOVER = [0.6,0.8,0.9]

ACTION_SPACE = [ (m,c) for m in MUTATION for c in CROSSOVER]

N_ACTIONS = len(ACTION_SPACE)


class EAEnv(gymnasium.Env):
    def __init__(self, algorithm, problem):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 1.0]),
                                       high=np.array([1.0, 50.0]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.state = np.zeros(3)
        self.problem = problem
        self.prev_hyper_vol = 0
        self.algorithm = algorithm
        self.max_steps = 50
        self.step_count = 0

    def step(self, action):
        current_state = self.state
        mutation, crossover = ACTION_SPACE[action]
        #Update    teh enxt step of the algorithem
        self.algorithm.mating.mutation.prob = mutation
        self.algorithm.mating.crossover.prob = crossover
        self.algorithm.next()
        NEXT_GEN = self.algorithm.pop
        NEXT_F = NEXT_GEN.get("F")

        hyper_vol = HV(ref_point=np.array([1.1, 1.1]))(NEXT_F)
        # diversity = 0  # Diversiity will be measure ocne we  have  tevrythign else goign too
        progress = self.step_count / self.max_steps

        # SImply look if the  hv has improved
        print(hyper_vol)
        improvement = (hyper_vol - self.prev_hyper_vol)  *1000#Reward
        self.prev_hyper_vol = hyper_vol
        self.step_count += 1

        self.state = np.array([progress, improvement, 0.0], dtype=np.float32)

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self.state, improvement, terminated, truncated #  infor will ebadded  later, {}

    def reset(self):
        self.step_count = 0
        self.state = np.zeros(3)
        return self.state, {} # migjt be abl to a dd some info if needed later on
