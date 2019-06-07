import os
from envs.base import AutoEnv


def build_ngsim_env(
        args,
        exp_dir='/tmp',
        alpha=0.001,
        vectorize=False,
        render_params=None,
        videoMaking=False):
    basedir = os.path.expanduser('~/Autoenv/data')
    filepaths = [os.path.join(basedir, args.ngsim_filename)]
    # if render_params is None:
    #     render_params = dict(
    #         viz_dir=os.path.join(exp_dir, 'imitate/viz'),
    #         zoom=5.
    #     )
    env_params = dict(
        trajectory_filepaths=filepaths,
        H=args.env_H,
        primesteps=args.env_primesteps,
        action_repeat=args.env_action_repeat,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        # render_params=render_params,
        n_envs=args.n_envs,
        n_veh=args.n_envs,
        remove_ngsim_veh=args.remove_ngsim_veh,
        reward=args.env_reward
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must
    # also be true
    env_id = 'NGSIMEnv'

    env = AutoEnv(params=env_params)

    # get low and high values for normalizing _real_ actions
    low, high = env.action_space["low"], env.action_space["high"]
    return env, low, high