def batch_to_path_rewards(rewards, path_lengths):
    '''
    Args:
        - rewards: numpy array of shape (batch size, reward_dim)
        - path_lengths: list of lengths to be selected in groups from the row of rewards
    '''
    assert len(rewards) == sum(path_lengths)

    path_rewards = []
    s = 0
    for path_length in path_lengths:
        e = s + path_length
        path_rewards.append(rewards[s:e])
        s = e
    return path_rewards

