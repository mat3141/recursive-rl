# Wrapper environment that treats an RMDP as an MDP, i.e.
# box enter and box exit treated as normal MDP transitions
class RMDPasMDP():
    def __init__(self, Env, **kwargs):
        self.env = Env(**kwargs)
        self.num_actions = self.env.num_actions
        self.num_exits = self.env.num_exits
        self.observation_dim = self.env.observation_dim
        self.metadata = {}
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        sPrimes, reward, dones, info = self.env.step(action)        
        return sPrimes[0], reward, dones[1], info

# Wrapper environment that treats an RMDP as an MDP 
# (enter and exit treated as normal MDP transitions), but maintains
# the interface of an RMDP environment
class RMDPFlatten():
    def __init__(self, Env, **kwargs):
        self.env = Env(**kwargs)
        self.num_actions = self.env.num_actions
        self.num_exits = self.env.num_exits
        self.observation_dim = self.env.observation_dim
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        sPrimes, reward, dones, info = self.env.step(action)        
        return [sPrimes[0], []], reward, [False, dones[1]], info