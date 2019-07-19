from algorithms.AGen.critic.model import ObservationActionMLP
from algorithms.AGen.critic.base import Critic
from algorithms.dataset.CriticDataset import CriticDataset
from algorithms.dataset.utils import KeyValueReplayMemory, load_dataset
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
initial_filepath = None  # tf.train.latest_checkpoint(saver_dir)
obs_size = 66
act_size = 2

# build the critic
critic_network = ObservationActionMLP(
    hidden_layer_dims=[64, 64],
    dropout_keep_prob=critic_dropout_keep_prob,
    obs_size=obs_size,
    act_size=act_size
)

expert_data_filepath = "Some file path here"

data = load_dataset(expert_data_filepath, maxsize=real_data_maxsize)

if use_critic_replay_memory:
    critic_replay_memory = KeyValueReplayMemory(maxsize=3 * batch_size)
else:
    critic_replay_memory = None

critic_dataset = CriticDataset(
    data,
    replay_memory=critic_replay_memory,
    batch_size=1000
)

critic = Critic(
        dataset=critic_dataset,
        network=critic_network,
        obs_dim=obs_size,
        act_dim=act_size,
        n_train_epochs=n_critic_train_epochs,
        grad_norm_rescale=50.,
        verbose=2,
)