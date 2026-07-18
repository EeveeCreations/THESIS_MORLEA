import gymnasium
from fontTools.misc.cython import returns
from gymnasium import spaces
import numpy as np
from MOEA_RL import REWARD_SCALE, REF_POINT
from pymoo.indicators.hv import HV


class EAEnv(gymnasium.Env):
    def __init__(self, algorithm, problem):
        super().__init__()

        # CONTINUOUS ACTION SPACE  FIRST CROSS THENMUTAT
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        self.truncation_condition =  0.001

        self.algorithm = algorithm
        self.problem = problem

        self.last_improvement = 0
        self.state = np.zeros(3, dtype=np.float32)
        self.prev_hv = 0.0
        self.max_steps = 50 # MAX STEPS   PER EPISODE  MAybe find it to be vribale  epsilon close
        self.step_count = 0

    def decide_max(self, new_hv):

        current_sc = self.step_count

        hv_gain_ratio = (new_hv - self.prev_hv) / max(self.prev_hv, 1e-12)

        if hv_gain_ratio < self.truncation_condition:

            self.max_steps  = current_sc ## This way in trucrates early


    def calculate_hv(self):
        F = self.algorithm.pop.get("F")
        return HV(ref_point=REF_POINT)(F)


    def step(self, action):
        mutation = float(np.clip(action[0], 0.0, 1.0))
        crossover = float(np.clip(action[1], 0.0, 1.0))

        # apply EA params
        self.algorithm.mating.mutation.prob = mutation
        self.algorithm.mating.crossover.prob = crossover

        self.algorithm.next()
        F = self.algorithm.pop.get("F")
        hv =  HV(ref_point=REF_POINT)(F)

        improvement = (hv - self.prev_hv) * REWARD_SCALE
        self.last_improvement = improvement
        self.prev_hv = hv

        progress = self.step_count / self.max_steps

        self.state = np.array([progress, improvement, hv], dtype=np.float32)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        return self.state, improvement, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.algorithm.setup(self.problem)
        self.algorithm.next()

        F = self.algorithm.pop.get("F")
        self.prev_hv = HV(ref_point=REF_POINT)(F)

        self.step_count = 0
        self.state = np.zeros(3, dtype=np.float32)

        return self.state, {}
