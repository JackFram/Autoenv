from envs.base import AutoEnv
from envs.multi_agent_env import MultiAgentAutoEnv


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


