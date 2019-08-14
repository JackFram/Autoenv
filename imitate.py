import numpy as np
import os
import algorithms.utils as utils
from envs.utils import load_data
from algorithms.RL_Algorithm.GAIL.gail import GAIL

import envs.hyperparams as hyperparams


def run(args):
    print("loading from:", args.params_filepath)
    print("saving to:", args.exp_name)
    exp_dir = utils.set_up_experiment(exp_name=args.exp_name, phase='imitate')
    saver_dir = os.path.join(exp_dir, 'imitate', 'log')
    saver_filepath = './data/experiments/NGSIM-gail/imitate/model'
    print("saver file path is {}".format(saver_filepath))
    np.savez(os.path.join(saver_dir, 'args'),  args=args)

    # build components
    env, trajinfos, act_low, act_high = utils.build_ngsim_env(args)
    # # TODO: need to extract expert data first
    # data, veh_2_index = load_data(
    #     args.expert_filepath,
    #     act_low=act_low,
    #     act_high=act_high,
    #     min_length=args.env_H + args.env_primesteps,
    #     clip_std_multiple=args.normalize_clip_std_multiple,
    #     ngsim_filename=args.ngsim_filename
    # )
    # print("Finish loading the data!")
    #
    # critic = utils.build_critic(args, data, env)
    # print("Finish building our critic!")
    policy = utils.build_policy(args, env)
    print("Finish building our policy!")
    baseline = utils.build_baseline(args, env)
    print("Finish building our baseline!")
    reward_handler = utils.build_reward_handler(args)
    print("Finish building our reward handler!")

    # build algo
    sampler_args = dict(n_envs=args.n_envs) if args.vectorize else dict(n_envs=None)

    algo = GAIL(
        critic=None,
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
        max_kl=args.max_kl,
        damping=args.damping,
        l2_reg=args.l2_reg,
        policy_filepath=args.policy_param,
        critic_filepath=args.critic_param,
        env_filepath=args.env_param,
        cuda_enable=False,
        args=args
    )
    print("Finish building GAIL!")
    print("Start training:\n")
    algo.train()

# setup
args = hyperparams.parse_args()
run(args)

