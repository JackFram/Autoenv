import numpy as np


class RewardHandler(object):
    def __init__(
            self,
            use_env_rewards=True,
            critic_clip_low=-np.inf,
            critic_clip_high=np.inf,
            critic_initial_scale=1.,
            critic_final_scale=1.,
            recognition_initial_scale=1,
            recognition_final_scale=1.,
            augmentation_scale=1.,
            normalize_rewards=False,
            alpha=.01,
            max_epochs=10000,
            summary_writer=None):
        self.use_env_rewards = use_env_rewards
        self.critic_clip_low = critic_clip_low
        self.critic_clip_high = critic_clip_high

        self.critic_initial_scale = critic_initial_scale
        self.critic_final_scale = critic_final_scale
        self.critic_scale = critic_initial_scale

        self.recognition_initial_scale = recognition_initial_scale
        self.recognition_final_scale = recognition_final_scale
        self.recognition_scale = recognition_initial_scale

        self.augmentation_scale = augmentation_scale

        self.normalize_rewards = normalize_rewards
        self.alpha = alpha
        self.critic_reward_mean = 0.
        self.critic_reward_var = 1.
        self.recog_reward_mean = 0.
        self.recog_reward_var = 1.

        self.step = 0
        self.max_epochs = max_epochs
        self.summary_writer = summary_writer

    def _update_reward_estimate(self, rewards, reward_type):
        # unpack
        a = self.alpha
        mean = self.critic_reward_mean if reward_type == 'critic' else self.recog_reward_mean
        var = self.critic_reward_var if reward_type == 'critic' else self.recog_reward_var

        # update the reward mean using the mean of the rewards
        new_mean = (1 - a) * mean + a * np.mean(rewards)
        # update the variance with the mean of the individual variances
        new_var = (1 - a) * var + a * np.mean((rewards - mean) ** 2)

        # update class members
        if reward_type == 'critic':
            self.critic_reward_mean = new_mean
            self.critic_reward_var = new_var
        else:
            self.recog_reward_mean = new_mean
            self.recog_reward_var = new_var

    def _normalize_rewards(self, rewards, reward_type):
        self._update_reward_estimate(rewards, reward_type)
        var = self.critic_reward_var if reward_type == 'critic' else self.recog_reward_var
        return rewards / (np.sqrt(var) + 1e-8)

    def _update_scales(self):

        self.step += 1
        frac = np.minimum(self.step / self.max_epochs, 1)
        self.critic_scale = self.critic_initial_scale \
                            + frac * (self.critic_final_scale - self.critic_initial_scale)
        self.recognition_scale = self.recognition_initial_scale \
                                 + frac * (self.recognition_final_scale - self.recognition_initial_scale)

    def merge(
            self,
            paths,
            critic_rewards=None,
            recognition_rewards=None):
        """
        Add critic and recognition rewards to path rewards based on settings

        Args:
            paths: list of dictionaries as described in process_samples
            critic_rewards: list of numpy arrays of equal shape as corresponding path['rewards']
            recognition_rewards: same as critic rewards
        """
        # update relative reward scales
        self._update_scales()

        # combine the different rewards
        for (i, path) in enumerate(paths):

            shape = np.shape(path['rewards'])

            # env rewards
            if self.use_env_rewards:
                path['rewards'] = np.float32(path['rewards'])
            else:
                path['rewards'] = np.zeros(shape, dtype=np.float32)

            # critic rewards
            if critic_rewards is not None:
                critic_rewards[i] = np.clip(critic_rewards[i], self.critic_clip_low, self.critic_clip_high)
                if self.normalize_rewards:
                    critic_rewards[i] = self._normalize_rewards(
                        critic_rewards[i], reward_type='critic')
                path['rewards'] += self.critic_scale * np.reshape(critic_rewards[i], shape)

            # recognition rewards
            if recognition_rewards is not None:
                if self.normalize_rewards:
                    recognition_rewards[i] = self._normalize_rewards(
                        recognition_rewards[i], reward_type='recognition')
                path['rewards'] += self.recognition_scale * np.reshape(recognition_rewards[i], shape)

        return paths

