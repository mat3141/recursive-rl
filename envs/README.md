The structure of an RMDP environment is
```
class RMDPEnv():
    def __init__(self):
        pass
    
    def num_actions(self):
        """Returns the number of actions."""
        NotImplemented

    def num_exits(self):
        """Returns the number of exits."""
        NotImplemented
    
    def observation_dim(self):
        """Size of the state vector."""
        NotImplemented
        
    def reset(self):
        """Returns the initial state."""
        NotImplemented
        
    def step(self, action):
        """Returns (S, R, done, info) where
         - S := [next state, [box exits]]
         - R := reward
         - done := [box done, global done]
         - info := dict of extra information not used by the learner"""
        NotImplemented
```