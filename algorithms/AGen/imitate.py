from algorithms.AGen.critic.model import ObservationActionMLP
from algorithms.AGen.critic.base import Critic
import os

# setup
env_id = "CartPole-v0"
exp_name = "CartPole-v0"
exp_dir = utils.set_up_experiment(exp_name=exp_name, phase='imitate')
saver_dir = os.path.join(exp_dir, 'imitate', 'log')
saver_filepath = os.path.join(saver_dir, 'checkpoint')

# constants
use_infogail = False
use_critic_replay_memory = True
latent_dim = 2
real_data_maxsize = None
batch_size = 8000
n_critic_train_epochs = 50
n_recognition_train_epochs = 30
scheduler_k = 20
trpo_step_size = .01
critic_learning_rate = .0001
critic_dropout_keep_prob = .6
recognition_learning_rate = .0001
initial_filepath = None # tf.train.latest_checkpoint(saver_dir)

# build the critic
critic_network = ObservationActionMLP(
    hidden_layer_dims=[64, 64],
    dropout_keep_prob=critic_dropout_keep_prob
)

critic = Critic(
        dataset=critic_dataset,
        network=critic_network,
        obs_dim=66,
        act_dim=2,
        n_train_epochs=n_critic_train_epochs,
        grad_norm_rescale=50.,
        verbose=2,
)