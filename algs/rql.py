import copy
import numpy as np
from tqdm import tqdm

# Tabular Q-learning with discretized values
class RQlearning():
    def __init__(self, env, discretize_step=0.1, discretize_max=100, alpha=0.1, epsilon=0.1, episode_limit=30):
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.num_actions = self.env.num_actions()
        self.num_exits = self.env.num_exits()
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_limit = episode_limit
        self.discretize_step = discretize_step
        self.discretize_max = discretize_max
    
    def discretize(self, v):
        dis_v = []
        for x in v:
            x = self.discretize_step*np.round(x/self.discretize_step)
            x = max(-self.discretize_max, min(self.discretize_max, x))
            dis_v.append(x)
        return dis_v
    
    def initQ(self, *bS):
        for k in bS:
            if k is None:
                continue
            if k not in self.Q:
                self.Q[k] = [0 for x in range(self.num_actions)]
    
    def getEpsilonGreedyAction(self, bS):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            if bS not in self.Q:
                self.initQ(bS)
            return np.argmax(self.Q[bS])
    
    def test(self, test_num=10):
        Rsum = 0
        
        v = [0]*self.num_exits
        v_min = 0
        stack = []
        
        for n in range(test_num):
            S = self.test_env.reset()
            episode_step = 0
            done = False
            
            while not done:
                self.initQ((*S,*v))
                A = np.argmax(self.Q[(*S,*v)])
                sPrimes, R, dones, info = self.test_env.step(A)
                Rsum += R
                episode_step += 1
                S, bsExit = sPrimes
                if dones[0]: # Exited box
                    if not dones[1]:
                        v, v_min = stack.pop()
                elif len(bsExit) != 0: # Entered box
                    stack.append((v, v_min))
                    bsExV = [(*bsEx,*v) for bsEx in bsExit]
                    self.initQ(*bsExV)
                    v = [max(self.Q[x]) for x in bsExV]
                    v = self.discretize(v)
                    v_min = min(v)
                    v = [x - v_min for x in v]
                done = dones[1]
                if self.episode_limit is not None and episode_step >= self.episode_limit:
                    v = [0]*self.num_exits
                    v_min = 0
                    stack = []
                    break
                
        return Rsum/test_num 
    
    def train(self, train_steps=10000, test_freq=1000, test_num=10):
        log = [(0,self.test(test_num))]
        bS = self.env.reset()
        v = [0]*self.num_exits
        v_min = 0
        stack = []
        episode_step = 0
        for t in tqdm(range(train_steps)):
            A = self.getEpsilonGreedyAction((*bS, *v))
            nextStates, R, dones, info = self.env.step(A)
            # Unpack
            bsPrime, bsExit = nextStates
            boxDone, globalDone = dones
            # Initialize Q-table
            self.initQ((*bS, *v))
            # Update
            if boxDone: # Exit box
                v_exit = 0
                if not globalDone:
                    v_prime, v_min_prime = stack.pop()
                    v_exit = max(self.Q[(*bsPrime,*v_prime)]) - v_min
                self.Q[(*bS,*v)][A] = (1-self.alpha) * self.Q[(*bS,*v)][A] + self.alpha * (R + v_exit)
                if not globalDone:
                    v, v_min = v_prime, v_min_prime
            elif len(bsExit) > 0: # Enter box
                bsExV = [(*bsEx,*v) for bsEx in bsExit]
                self.initQ(*bsExV)
                v_prime = [max(self.Q[x]) for x in bsExV]
                v_prime = self.discretize(v_prime)
                v_min_prime = min(v_prime)
                v_prime = [x - v_min_prime for x in v_prime]
                self.initQ((*bsPrime,*v_prime))
                self.Q[(*bS, *v)][A] = (1-self.alpha) * self.Q[(*bS,*v)][A] + self.alpha * (R + max(self.Q[(*bsPrime,*v_prime)]) + v_min_prime)
                stack.append((v, v_min))
                v, v_min = v_prime, v_min_prime
            else: # Normal
                self.initQ((*bsPrime,*v))
                self.Q[(*bS,*v)][A] = (1-self.alpha) * self.Q[(*bS,*v)][A] + self.alpha * (R + max(self.Q[(*bsPrime,*v)]))

            episode_step += 1
            
            if globalDone or (self.episode_limit is not None and episode_step >= self.episode_limit):
                bS = self.env.reset()
                v = [0]*self.num_exits
                v_min = 0
                stack = []
                episode_step = 0
            else:
                bS = bsPrime
                
            if test_freq > 0 and t % test_freq == 0:
                log.append((t,self.test(test_num)))
        return log