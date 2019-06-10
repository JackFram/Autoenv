from envs.base import AutoEnv
from envs.multi_agent_env import MultiAgentAutoEnv
from envs.utils import build_space


def make_env(env_id: str, env_params: dict):
    try:
        if env_id == "NGSIMEnv":
            return AutoEnv(env_params)
        elif env_id == "MultiAgentAutoEnv":
            return MultiAgentAutoEnv(env_params)
        else:
            raise ValueError("No such env name!")

    except RuntimeError as e:
        print(e)


class Env:
    def __init__(self, env_id, env_params):
        self.env = make_env(env_id, env_params)
        self._observation_space = build_space(*(self.env.observation_space_spec()))
        self._action_space = build_space(*(self.env.action_space_spec()))

    def reset(self, dones=None, **kwargs):
        return self.env.reset(dones, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def obs_names(self):
        return self.env.obs_names()

    def vec_env_executor(self, *args, **kwargs):
        return self

    @property
    def num_envs(self):
        return self.env.num_envs()

    @property
    def vectorized(self):
        return self.env.vectorized()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


