import numpy as np
import os
import time

import torch

import algorithms.RL_Algorithm.utils
from algorithms.utils import save_params, extract_normalizing_env, load_params
from algorithms.RL_Algorithm.optimizers import ext


class GAIL(object):
    def __init__(self,
                 env,
                 policy,
                 baseline,
                 critic=None,
                 recognition=None,
                 optimizer=None,
                 optimizer_args=None,
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
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size

        self.critic = critic
        self.recognition = recognition
        self.reward_handler = reward_handler
        self.saver = saver
        self.saver_filepath = saver_filepath
        self.validator = validator
        self.snapshot_env = snapshot_env

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
        if self.saver and (itr + 1) % self.saver._keep_checkpoint_every_n_hours == 0:

            # collect params (or stuff to keep in general)
            params = dict()
            if self.critic:
                params['critic'] = self.critic.network.get_param_values()
            if self.recognition:
                params['recognition'] = self.recognition.network.get_param_values()
            params['policy'] = self.policy.get_param_values()
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

    def load(self, filepath):
        '''
        Load parameters from a filepath. Symmetric to _save. This is not ideal,
        but it's easier than keeping track of everything separately.
        '''
        params = load_params(filepath)
        if self.critic and 'critic' in params.keys():
            self.critic.network.set_param_values(params['critic'])
        if self.recognition and 'recognition' in params.keys():
            self.recognition.network.set_param_values(params['recognition'])
        self.policy.set_param_values(params['policy'])
        normalized_env = extract_normalizing_env(self.env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

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

    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = torch.sum(kl * valid_var) / torch.sum(valid_var)
            surr_loss = - torch.sum(lr * advantage_var * valid_var) / torch.sum(valid_var)
        else:
            mean_kl = torch.mean(kl)
            surr_loss = - torch.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

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
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        print("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        print("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        print("Optimizing")
        self.optimizer.optimize(all_input_values)
        print("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        print("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        print('LossBefore', loss_before)
        print('LossAfter', loss_after)
        print('MeanKLBefore', mean_kl_before)
        print('MeanKL', mean_kl)
        print('dLoss', loss_before - loss_after)

        # train critic
        if self.critic is not None:
            self.critic.train(itr, samples_data)
        if self.recognition is not None:
            self.recognition.train(itr, samples_data)
        return dict()

    def train(self):
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            print("Obtaining samples...")
            paths = self.obtain_samples(itr)
            print("Processing samples...")
            samples_data = self.process_samples(itr, paths)
            print("Logging diagnostics...")
            self.log_diagnostics(paths)
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
