from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def act(self):
        """the agent receives the current state 
        and compute the action accordingly"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state, reward):
        """the agent adapts itself based on
        the reward from his last action"""

    trial = 0 # == episode

    def reset(self):
        self.trial += 1