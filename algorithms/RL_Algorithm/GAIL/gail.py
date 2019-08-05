import os
import time
import random
import julia

import algorithms.RL_Algorithm.utils
from algorithms.utils import save_params, extract_normalizing_env, load_params
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from algorithms.RL_Algorithm.optimizers.trpo import trpo_step
from algorithms.RL_Algorithm.optimizers.utils import *
from algorithms import utils
from envs.utils import load_data
from preprocessing.clean_holo import clean_data, csv2txt, create_lane
from src.trajdata import convert_raw_ngsim_to_trajdatas
from preprocessing.extract_feature import extract_ngsim_features


class GAIL(object):
    def __init__(self,
                 env,
                 policy,
                 baseline,
                 critic=None,
                 recognition=None,
                 step_size=0.01,
                 reward_handler=algorithms.RL_Algorithm.utils.RewardHandler(),
                 saver=None,
                 saver_filepath=None,
                 validator=None,
                 snapshot_env=True,
                 scope=None,
                 n_itr=500,
                 start_itr=0,
                 batch_size=5000,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 plot=False,
                 pause_for_plot=False,
                 center_adv=True,
                 positive_adv=False,
                 store_paths=False,
                 whole_paths=True,
                 fixed_horizon=False,
                 sampler_cls=None,
                 sampler_args=None,
                 force_batch_sampler=False,
                 max_kl=None,
                 damping=None,
                 l2_reg=None,
                 policy_filepath=None,
                 critic_filepath=None,
                 env_filepath=None,
                 args=None
                 ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = BatchSampler
            self.sampler_cls = sampler_cls
        self.sampler = sampler_cls(self, **sampler_args)
        self.sampler_args = sampler_args

        self.step_size = step_size

        self.critic = critic
        self.recognition = recognition
        self.reward_handler = reward_handler
        self.saver = saver
        self.saver_filepath = saver_filepath
        self.validator = validator
        self.snapshot_env = snapshot_env

        self.max_kl = max_kl
        self.damping = damping
        self.l2_reg = l2_reg

        self.critic_filepath = critic_filepath
        self.policy_filepath = policy_filepath
        self.env_filepath = env_filepath

        self.file_set = set()

        self.j = julia.Julia()
        self.j.using("NGSIM")

        self.args = args

    def start_worker(self):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        """
        Augment path rewards with critic and recognition model rewards

        Args:
            itr: iteration counter
            paths: list of dictionaries
                each containing info for a single trajectory
                each with keys 'observations', 'actions', 'agent_infos', 'env_infos', 'rewards'
        """
        # compute critic and recognition rewards and combine them with the path rewards
        critic_rewards = self.critic.critique(itr, paths) if self.critic else None
        recognition_rewards = self.recognition.recognize(itr, paths) if self.recognition else None
        paths = self.reward_handler.merge(paths, critic_rewards, recognition_rewards)
        return self.sampler.process_samples(itr, paths)

    def _save(self, itr):
        """
        Save a tf checkpoint of the session.
        """
        # using keep_checkpoint_every_n_hours as proxy for iterations between saves
        if (itr + 1) % 50 == 0:
            # collect params (or stuff to keep in general)
            params = dict()
            if self.critic:
                torch.save(self.critic.network.state_dict(),
                           os.path.join(self.saver_filepath, "critic_{}.pkl".format(itr)))
            torch.save(self.policy.state_dict(),
                       os.path.join(self.saver_filepath, "policy_{}.pkl".format(itr)))
            # if the environment is wrapped in a normalizing env, save those stats
            normalized_env = extract_normalizing_env(self.env)
            if normalized_env is not None:
                params['normalzing'] = dict(
                    obs_mean=normalized_env._obs_mean,
                    obs_var=normalized_env._obs_var
                )

            # save params
            save_dir = os.path.split(self.saver_filepath)[0]
            save_params(save_dir, params, itr + 1, max_to_keep=50)

    def load(self):
        '''
        Load parameters from a filepath. Symmetric to _save. This is not ideal,
        but it's easier than keeping track of everything separately.
        '''
        params = load_params(self.env_filepath)
        if self.policy is not None:
            self.load_policy(self.policy_filepath)
        if self.critic is not None:
            self.load_critic(self.critic_filepath)
        # self.policy.set_param_values(params['policy'])
        normalized_env = extract_normalizing_env(self.env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

    def load_critic(self, critic_param_path):
        self.critic.network.load_state_dict(torch.load(critic_param_path))

    def load_policy(self, policy_param_path):
        self.policy.load_state_dict(torch.load(policy_param_path))

    def _validate(self, itr, samples_data):
        """
        Run validation functions.
        """
        if self.validator:
            objs = dict(
                policy=self.policy,
                critic=self.critic,
                samples_data=samples_data,
                env=self.env)
            self.validator.validate(itr, objs)

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def get_itr_snapshot(self, itr, samples_data):
        """
        Snapshot critic and recognition model as well
        """
        self._save(itr)
        self._validate(itr, samples_data)
        snapshot = dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
        )
        if self.snapshot_env:
            snapshot['env'] = self.env
        if samples_data is not None:
            snapshot['samples_data'] = dict()
            if 'actions' in samples_data.keys():
                snapshot['samples_data']['actions'] = samples_data['actions'][:10]
            if 'mean' in samples_data.keys():
                snapshot['samples_data']['mean'] = samples_data['mean'][:10]

        return snapshot

    def optimize_policy(self, itr, samples_data):
        """
        Update the critic and recognition model in addition to the policy

        Args:
            itr: iteration counter
            samples_data: dictionary resulting from process_samples
                keys: 'rewards', 'observations', 'agent_infos', 'env_infos', 'returns',
                      'actions', 'advantages', 'paths'
                the values in the infos dicts can be accessed for example as:
                    samples_data['agent_infos']['prob']

                and the returned value will be an array of shape (batch_size, prob_dim)
        """
        obes = samples_data['observations']
        actions = samples_data['actions']
        advantages = samples_data['advantages']

        print("obs shape: {}, action shape: {}, return shape: {}, advantages shape: {}".format(
            samples_data['observations'].shape,
            samples_data['actions'].shape,
            samples_data['returns'].shape,
            samples_data['advantages'].shape
        ))

        trpo_step(
            self.policy.double(),
            obes,
            actions,
            advantages,
            self.max_kl,
            self.damping,
        )

        # train critic
        if self.critic is not None:
            self.critic.train(itr, samples_data)
        if self.recognition is not None:
            self.recognition.train(itr, samples_data)
        return dict()

    def init_env(self, itr):
        if len(self.file_set) == 0:
            data_base_dir = "./preprocessing/data"
            file_list = os.listdir(data_base_dir)
            dir_name = random.choice(file_list)
            while not os.path.isdir(os.path.join(data_base_dir, dir_name, "processed")):
                dir_name = random.choice(file_list)
            print("Sample from directory: {}".format(dir_name))
            paths = []
            for file_name in os.listdir(os.path.join(data_base_dir, dir_name, "processed")):
                if "section" in file_name:
                    orig_traj_file = os.path.join(dir_name, "processed", file_name)
                    paths.append(orig_traj_file)
            lane_file = os.path.join(dir_name, "processed", '{}_lane'.format(dir_name[:19]))
            create_lane(lane_file)
            base_dir = os.path.expanduser('~/Autoenv/data/')
            self.j.write_roadways_to_dxf(base_dir)
            self.j.write_roadways_from_dxf(base_dir)
            print("Finish generating roadway")
            self.file_set.update(paths)
        trajectory_file = random.choice(list(self.file_set))
        processed_data_path = 'holo_{}_perfect_cleaned.csv'.format(trajectory_file[5:19])
        self.file_set.remove(trajectory_file)
        df_len = clean_data(trajectory_file)
        if df_len == 0:
            print("Invalid file, skipping")
            return False
        csv2txt(processed_data_path)
        convert_raw_ngsim_to_trajdatas()
        extract_ngsim_features(output_filename="ngsim_holo_new.h5", n_expert_files=1)
        print("Finish converting and feature extraction")
        args = self.args
        env, trajinfos, act_low, act_high = utils.build_ngsim_env(args)
        data, veh_2_index = load_data(
            args.expert_filepath,
            act_low=act_low,
            act_high=act_high,
            min_length=args.env_H + args.env_primesteps,
            clip_std_multiple=args.normalize_clip_std_multiple,
            ngsim_filename=args.ngsim_filename
        )
        if data is None:
            return False
        torch.save(self.critic.network.state_dict(), './data/experiments/NGSIM-gail/imitate/model/critic_cache.pkl')
        critic = utils.build_critic(args, data, env)
        self.env = env
        self.critic = critic
        self.load_critic('./data/experiments/NGSIM-gail/imitate/model/critic_cache.pkl')
        # self.shutdown_worker()
        self.sampler = self.sampler_cls(self, **self.sampler_args)
        self.start_worker()
        return True

    def train(self):
        self.start_worker()
        start_time = time.time()
        print("loading critic and policy params from file")
        self.load()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            print("Initializing AutoEnv...")
            if not self.init_env(itr):
                print("Invalid data, skipping this iteration!")
                continue
            print("Obtaining samples...")
            paths = self.obtain_samples(itr)
            print("Processing samples...")
            samples_data = self.process_samples(itr, paths)
            print("Logging diagnostics...")
            # self.log_diagnostics(paths)
            print("Optimizing policy...")
            self.optimize_policy(itr, samples_data)
            print("Saving snapshot...")
            params = self.get_itr_snapshot(itr, samples_data)
            if self.store_paths:
                params["paths"] = samples_data["paths"]
            # logger.save_itr_params(itr, params)
            print("Saved")
            print('Time', time.time() - start_time)
            print('ItrTime', time.time() - itr_start_time)
        self.shutdown_worker()
