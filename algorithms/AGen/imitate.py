import numpy as np
import os
import algorithms.utils as utils
from envs.utils import load_data
from algorithms.RL_Algorithm.GAIL.gail import GAIL


def run(args):
    print("loading from:", args.params_filepath)
    print("saving to:", args.exp_name)
    exp_dir = utils.set_up_experiment(exp_name=args.exp_name, phase='imitate')
    saver_dir = os.path.join(exp_dir, 'imitate', 'log')
    saver_filepath = os.path.join(saver_dir, 'checkpoint')
    np.savez(os.path.join(saver_dir, 'args'),  args=args)

    # build components
    env, act_low, act_high = utils.build_ngsim_env(args, exp_dir, vectorize=args.vectorize)
    data = load_data(
        args.expert_filepath,
        act_low=act_low,
        act_high=act_high,
        min_length=args.env_H + args.env_primesteps,
        clip_std_multiple=args.normalize_clip_std_multiple,
        ngsim_filename=args.ngsim_filename
    )

    critic = utils.build_critic(args, data, env)
    policy = utils.build_policy(args, env)
    baseline = utils.build_baseline(args, env)
    reward_handler = utils.build_reward_handler(args)

    # build algo
    sampler_args = dict(n_envs=args.n_envs) if args.vectorize else None

    if args.policy_recurrent:
        optimizer = ConjugateGradientOptimizer(
            max_backtracks=50,
            hvp_approach=FiniteDifferenceHvp
        )
    else:
        optimizer = None

    algo = GAIL(
        critic=critic,
        recognition=None,
        reward_handler=reward_handler,
        env=env,
        policy=policy,
        baseline=baseline,
        validator=None,
        batch_size=args.batch_size,
        max_path_length=args.max_path_length,
        n_itr=args.n_itr,
        discount=args.discount,
        step_size=args.trpo_step_size,
        saver=None,
        saver_filepath=saver_filepath,
        force_batch_sampler=False if args.vectorize else True,
        sampler_args=sampler_args,
        snapshot_env=False,
        plot=False,
        optimizer=optimizer,
        optimizer_args=dict(
            max_backtracks=50,
            debug_nan=True
        )
    )

