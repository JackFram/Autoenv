import os
import tensorflow as tf
from envs.make import make_env, Env
from envs.utils import add_kwargs_to_reset

from rllab.envs.normalized_env import normalize as normalize_env

from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.envs.spec_wrapper_env import SpecWrapperEnv
from hgail.policies.categorical_latent_sampler import CategoricalLatentSampler
from hgail.samplers.hierarchy_sampler import HierarchySampler
from hgail.algos.hgail_impl import Level
from hgail.envs.vectorized_normalized_env import vectorized_normalized_env
from hgail.policies.gaussian_latent_var_gru_policy import GaussianLatentVarGRUPolicy
from hgail.policies.gaussian_latent_var_mlp_policy import GaussianLatentVarMLPPolicy
from hgail.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import hgail.misc.utils


def build_ngsim_env(
        args,
        exp_dir='/tmp',
        n_veh=1,
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
        n_veh=n_veh,
        remove_ngsim_veh=args.remove_ngsim_veh,
        reward=args.env_reward
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must
    # also be true
    if args.env_multiagent:
        env_id = "MultiAgentAutoEnv"
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env
    else:
        env_id = "NGSIMEnv"
        normalize_wrapper = normalize_env

    print(env_params)
    env = Env(env_id=env_id, env_params=env_params)
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha))
    # get low and high values for normalizing _real_ actions
    add_kwargs_to_reset(env)
    return env, low, high


def build_reward_handler(args, writer=None):
    reward_handler = hgail.misc.utils.RewardHandler(
        use_env_rewards=args.reward_handler_use_env_rewards,
        max_epochs=args.reward_handler_max_epochs,  # epoch at which final scales are used
        critic_final_scale=args.reward_handler_critic_final_scale,
        recognition_initial_scale=0.,
        recognition_final_scale=args.reward_handler_recognition_final_scale,
        summary_writer=writer,
        normalize_rewards=True,
        critic_clip_low=-100,
        critic_clip_high=100,
    )
    return reward_handler


def build_hierarchy(args, env, writer=None):
    levels = []

    latent_sampler = UniformlyRandomLatentSampler(
        name='base_latent_sampler',
        dim=args.latent_dim,
        scheduler=ConstantIntervalScheduler(k=args.env_H)
    )
    for level_idx in [1, 0]:
        # wrap env in different spec depending on level
        if level_idx == 0:
            level_env = env
        else:
            level_env = SpecWrapperEnv(
                env,
                action_space=Discrete(args.latent_dim),
                observation_space=env.observation_space
            )

        with tf.variable_scope('level_{}'.format(level_idx)):
            # recognition_model = build_recognition_model(args, level_env, writer)
            recognition_model = None
            if level_idx == 0:
                policy = build_policy(args, env, latent_sampler=latent_sampler)
            else:
                scheduler = ConstantIntervalScheduler(k=args.scheduler_k)
                policy = latent_sampler = CategoricalLatentSampler(
                    scheduler=scheduler,
                    name='latent_sampler',
                    policy_name='latent_sampler_policy',
                    dim=args.latent_dim,
                    env_spec=level_env.spec,
                    latent_sampler=latent_sampler,
                    max_n_envs=args.n_envs
                )
            baseline = build_baseline(args, level_env)
            if args.vectorize:
                force_batch_sampler = False
                if level_idx == 0:
                    sampler_args = dict(n_envs=args.n_envs)
                else:
                    sampler_args = None
            else:
                force_batch_sampler = True
                sampler_args = None

            sampler_cls = None if level_idx == 0 else HierarchySampler
            algo = TRPO(
                env=level_env,
                policy=policy,
                baseline=baseline,
                batch_size=args.batch_size,
                max_path_length=args.max_path_length,
                n_itr=args.n_itr,
                discount=args.discount,
                step_size=args.trpo_step_size,
                sampler_cls=sampler_cls,
                force_batch_sampler=force_batch_sampler,
                sampler_args=sampler_args,
                optimizer_args=dict(
                    max_backtracks=50,
                    debug_nan=True
                )
            )
            reward_handler = build_reward_handler(args, writer)
            level = Level(
                depth=level_idx,
                algo=algo,
                reward_handler=reward_handler,
                recognition_model=recognition_model,
                start_itr=0,
                end_itr=0 if level_idx == 0 else np.inf
            )
            levels.append(level)

    # by convention the order of the levels should be increasing
    # but they must be built in the reverse order
    # so reverse the list before returning it
    return list(reversed(levels))

'''
    build policy functions
'''


def build_policy(args, env, latent_sampler=None):
    if args.use_infogail:
        if latent_sampler is None:
            latent_sampler = UniformlyRandomLatentSampler(
                scheduler=ConstantIntervalScheduler(k=args.scheduler_k),
                name='latent_sampler',
                dim=args.latent_dim
            )
        if args.policy_recurrent:
            policy = GaussianLatentVarGRUPolicy(
                name="policy",
                latent_sampler=latent_sampler,
                env_spec=env.spec,
                hidden_dim=args.recurrent_hidden_dim,
            )
        else:
            print("GaussianLatentVarMLPPolicy")
            policy = GaussianLatentVarMLPPolicy(
                name="policy",
                latent_sampler=latent_sampler,
                env_spec=env.spec,
                hidden_sizes=args.policy_mean_hidden_layer_dims,
                std_hidden_sizes=args.policy_std_hidden_layer_dims
            )
    else:
        if args.policy_recurrent:
            print("GaussianGRUPolicy")
            policy = GaussianGRUPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_dim=args.recurrent_hidden_dim,
                output_nonlinearity=None,
                learn_std=True
            )
        else:
            print("GaussianMLPPolicy")
            policy = GaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_sizes=args.policy_mean_hidden_layer_dims,
                std_hidden_sizes=args.policy_std_hidden_layer_dims,
                adaptive_std=True,
                output_nonlinearity=None,
                learn_std=True
            )
    return policy


def build_baseline(args, env):
    return GaussianMLPBaseline(env_spec=env.spec)

