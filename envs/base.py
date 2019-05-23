class Env(object):
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Returns a action space
        :rtype: action space type
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: observation space type
        """
        raise NotImplementedError

    # Helpers that derive from Spaces
    @property
    def action_dim(self):
        raise NotImplementedError

    def render(self):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def horizon(self):
        """
        Horizon of the environment, if it has one
        """
        raise NotImplementedError

    def terminate(self):
        """
        Clean up operation,
        """
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
