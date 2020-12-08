from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def act(self):
        """the agent receives the current state 
        and compute the action accordingly"""
        raise NotImplementedError

    @abstractmethod
    def observe(self, state, reward):
        """the agent adapts itself based on
        the reward from his last action"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError