import torch.optim as optim
from torch.autograd import Variable
import torch

import numpy as np
import algorithms.AGen.critic.utils


class Critic(object):
    """
    Critic base class
    """

    def __init__(
            self,
            network,
            dataset,
            obs_dim,
            act_dim,
            optimizer=None,
            lr=0.0001,
            n_train_epochs=5,
            grad_norm_rescale=10000.,
            grad_norm_clip=10000.,
            summary_writer=None,
            debug_nan=False,
            verbose=0):
        self.network = network
        self.dataset = dataset
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if optimizer is None:
            self.optimizer = optim.RMSprop(network.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        self.lr = lr
        self.n_train_epochs = n_train_epochs
        self.grad_norm_rescale = grad_norm_rescale
        self.grad_norm_clip = grad_norm_clip
        self.summary_writer = summary_writer
        self.debug_nan = debug_nan
        self.verbose = verbose

    def critique(self, itr, paths):
        """
        Compute and return rewards based on the (obs, action) pairs in paths
            where rewards are a list of numpy arrays of equal length as the
            corresponding path rewards

        Args:
            itr: iteration count
            paths: list of dictionaries {'observations': obs(list), 'actions': act(list)}
        """
        # convert to batch and use network to critique
        obs = np.concatenate([d['observations'] for d in paths], axis=0)
        acts = np.concatenate([d['actions'] for d in paths], axis=0)

        # normalize
        if self.dataset.observation_normalizer:
            obs = self.dataset.observation_normalizer(obs)
        if self.dataset.action_normalizer:
            acts = self.dataset.action_normalizer(acts)

        # compute rewards
        rewards = self.network.forward(obs, acts)
        rewards = rewards.cpu().detach().numpy()
        if np.any(np.isnan(rewards)) and self.debug_nan:
            import ipdb
            ipdb.set_trace()

        # output as a list of numpy arrays, each of len equal to the rewards of
        # the corresponding trajectory
        path_lengths = [len(d['rewards']) for d in paths]
        path_rewards = algorithms.AGen.critic.utils.batch_to_path_rewards(rewards, path_lengths)

        self._log_critique(itr, paths, rewards)
        return path_rewards

    def _log_critique(self, itr, paths, critic_rewards):
        """
        Log information about the critique and paths
        Args:
            itr: algorithm batch iteration
            paths: list of dictionaries containing trajectory information
            critic_rewards: critic rewards
        """
        # only write summaries if have a summary writer
        print("reward shape: ", critic_rewards.shape)
        print("Wait to complete")

    def train(self, itr, samples_data):
        """
        Train the critic using real and sampled data

        Args:
            itr: iteration count
            samples_data: dictionary containing generated data
        """
        for train_itr in range(self.n_train_epochs):
            for batch in self.dataset.batches(samples_data, store=train_itr == 0):
                self._train_batch(batch)

    def _train_batch(self, batch):
        """
        Runs a single training batch

        Args:
            batch: dictionary with values needed for training network class member
        """
        self.rx = batch['rx']
        self.ra = batch['ra']
        self.gx = batch['gx']
        self.ga = batch['ga']
        self.eps = np.random.uniform(0, 1, len(batch['rx'])).reshape(-1, 1)
        rx, ra, gx, ga, eps = self.rx, self.ra, self.gx, self.ga, self.eps

        gp_loss = 0
        # gradient penalty

        # loss and train op
        self.optimizer.zero_grad()
        self.real_loss = real_loss = -torch.mean(self.network(rx, ra))
        self.gen_loss = gen_loss = torch.mean(self.network(gx, ga))
        self.loss = loss = real_loss + gen_loss + gp_loss
        loss.backward()
        self.optimizer.step()

    def _build_summaries(
            self,
            loss,
            real_loss,
            gen_loss,
            gradients,
            clipped_gradients,
            gradient_penalty=None,
            batch_size=None):
        summaries = None
        return summaries

    def _build_input_summaries(self, rx, ra, gx, ga):
        summaries = None
        return summaries
