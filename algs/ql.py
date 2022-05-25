import copy
import numpy as np
from tqdm import tqdm

# Tabular Q-learning for MDPs
class Qlearning():
    def __init__(self, env, alpha=0.1, epsilon=0.1, discount=1.0, episode_limit=30):
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.num_actions = self.env.num_actions()
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.episode_limit = episode_limit
    
    def initQ(self, *states):
        for k in states:
            if k is None:
                continue
            if k not in self.Q:
                self.Q[k] = [0 for x in range(self.num_actions)]
    
    def getEpsilonGreedyAction(self, S):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            if S not in self.Q:
                self.initQ(S)
            return np.argmax(self.Q[S])
    
    def test(self, test_num=10):
        Rsum = 0
        
        for n in range(test_num):
            S = self.test_env.reset()
            episode_step = 0
            done = False
            
            while not done:
                self.initQ(S)
                A = np.argmax(self.Q[S])
                sPrime, R, done, info = self.test_env.step(A)
                Rsum += R
                episode_step += 1
                S = sPrime
                if episode_step >= self.episode_limit:
                    break
                
        return Rsum/test_num
    
    def train(self, train_steps=10000, test_freq=1000, test_num=10):
        log = [(0,self.test(test_num))]
        S = self.env.reset()
        self.initQ(S)
        episode_step = 0
        for t in tqdm(range(train_steps)):
            A = self.getEpsilonGreedyAction(S)
            sPrime, R, done, info = self.env.step(A)
            # Initialize Q-table
            self.initQ(sPrime)
            # Update
            if done: # Terminal
                self.Q[S][A] = (1-self.alpha) * self.Q[S][A] + self.alpha * (R)
            else: # Normal
                self.Q[S][A] = (1-self.alpha) * self.Q[S][A] + self.alpha * (R + self.discount*max(self.Q[sPrime]))
            
            episode_step += 1
            
            if done or episode_step >= self.episode_limit:
                S = self.env.reset()
                episode_step = 0
            else:
                S = sPrime
            
            if test_freq > 0 and t % test_freq == 0:
                log.append((t,self.test(test_num)))
        
        return log