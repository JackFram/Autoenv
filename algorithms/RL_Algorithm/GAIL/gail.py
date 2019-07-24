import numpy as np
import os
import algorithms.RL_Algorithm.utils


class GAIL(object):
    def __init__(self,
                 critic=None,
                 recognition=None,
                 reward_handler=algorithms.RL_Algorithm.utils.RewardHandler(),
                 policy_update_algo = None,
                 saver=None,
                 saver_filepath=None,
                 validator=None,
                 snapshot_env=True,
                 **kwargs):

        self.critic = critic
        self.recognition = recognition
        self.reward_handler = reward_handler
        self.saver = saver
        self.saver_filepath = saver_filepath
        self.validator = validator
        self.snapshot_env = snapshot_env

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
