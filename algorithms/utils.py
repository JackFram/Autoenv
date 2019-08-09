from algorithms.dataset.utils import KeyValueReplayMemory
from algorithms.dataset.CriticDataset import CriticDataset
from algorithms.AGen.critic.model import ObservationActionMLP
from algorithms.AGen.critic.base import Critic
from algorithms.policy.GaussianGRUPolicy import GaussianGRUPolicy
from algorithms.policy.GaussianMLPBaseline import GaussianMLPBaseline
from algorithms.RL_Algorithm.utils import RewardHandler
from envs.make import Env
from hgail.envs.vectorized_normalized_env import vectorized_normalized_env, VectorizedNormalizedEnv
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import NormalizedEnv
from envs.utils import add_kwargs_to_reset
import os
import numpy as np


def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def build_ngsim_env(
        args,
        n_veh=1,
        alpha=0.001):
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
        raise NotImplementedError("No single agent env")

    # print(env_params)
    env = Env(env_id=env_id, env_params=env_params)
    low, high = env.action_space.low, env.action_space.high
    trajinfos = env.trajinfos
    env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha))
    # get low and high values for normalizing _real_ actions
    add_kwargs_to_reset(env)
    return env, trajinfos, low, high


def build_critic(args, data, env, writer=None):
    if args.use_critic_replay_memory:
        critic_replay_memory = KeyValueReplayMemory(maxsize=3 * args.batch_size)
    else:
        critic_replay_memory = None

    critic_dataset = CriticDataset(
        data,
        replay_memory=critic_replay_memory,
        batch_size=args.critic_batch_size,
        flat_recurrent=args.policy_recurrent
    )

    critic_network = ObservationActionMLP(
        hidden_layer_dims=args.critic_hidden_layer_dims,
        dropout_keep_prob=args.critic_dropout_keep_prob,
        obs_size=env.observation_space.flat_dim,
        act_size=env.action_space.flat_dim
    )

    critic = Critic(
        obs_dim=env.observation_space.flat_dim,
        act_dim=env.action_space.flat_dim,
        dataset=critic_dataset,
        network=critic_network,
        n_train_epochs=args.n_critic_train_epochs,
        summary_writer=writer,
        grad_norm_rescale=args.critic_grad_rescale,
        verbose=2,
        debug_nan=True
    )

    return critic


def build_policy(args, env, mode: int=0):
    if args.policy_recurrent:
        policy = GaussianGRUPolicy(
            env_spec=env.spec,
            hidden_dim=args.recurrent_hidden_dim,
            output_nonlinearity=None,
            mode=mode
        )
    else:
        raise NotImplementedError("Hasn't implement none recurrent policy yet.")

    return policy


def build_baseline(args, env):
    return GaussianMLPBaseline(env_spec=env.spec)


def build_reward_handler(args, writer=None):
    reward_handler = RewardHandler(
        use_env_rewards=args.reward_handler_use_env_rewards,
        max_epochs=args.reward_handler_max_epochs,
        critic_final_scale=args.reward_handler_critic_final_scale,
        recognition_initial_scale=0.,
        recognition_final_scale=args.reward_handler_recognition_final_scale,
        summary_writer=writer,
        normalize_rewards=True,
        critic_clip_low=-100,
        critic_clip_high=100
    )
    return reward_handler


def set_up_experiment(
        exp_name,
        phase,
        exp_home='./data/experiments/',
        snapshot_gap=5):
    maybe_mkdir(exp_home)
    exp_dir = os.path.join(exp_home, exp_name)
    maybe_mkdir(exp_dir)
    phase_dir = os.path.join(exp_dir, phase)
    maybe_mkdir(phase_dir)
    return exp_dir


def save_params(output_dir, params, epoch, max_to_keep=None):
    # make sure output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save
    output_filepath = os.path.join(output_dir, 'itr_{}'.format(epoch))
    print("params are saved to: {}".format(output_filepath))
    np.savez(output_filepath, params=params)

    # delete files if in excess of max_to_keep
    if max_to_keep is not None:
        files = [os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
                and 'itr_' in f]
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
        if len(sorted_files) > max_to_keep:
            for filepath in sorted_files[max_to_keep:]:
                os.remove(filepath)


def load_params(filepath):
    return np.load(filepath)['params'].item()


'''
rllab utils
'''


def extract_wrapped_env(env, typ):
    while not isinstance(env, typ):
        # descend to wrapped env
        if hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        # not the desired type, and has no wrapped env, return None
        else:
            return None
    # reaches this point, then the env is of the desired type, return it
    return env


def extract_normalizing_env(env):
    normalizing_env = extract_wrapped_env(env, NormalizedEnv)
    if normalizing_env is None:
        normalizing_env = extract_wrapped_env(env, VectorizedNormalizedEnv)
    return normalizing_env




