import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm

# Simple feedforward network
class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=200, hidden_layers=2):
        super().__init__()
        
        seq = []
        # Input layer
        seq.append(nn.Linear(in_dim, hidden_dim))
        seq.append(nn.Tanh())
        # Hidden layers
        for _ in range(hidden_layers-1):
            seq.append(nn.Linear(hidden_dim, hidden_dim))
            seq.append(nn.Tanh())
        # Output layer
        seq.append(nn.Linear(hidden_dim, out_dim))
        
        self.model = nn.Sequential(*seq)
    
    def forward(self, x):
        return self.model(x)     
    
# Helper function for updating target network
def copy_weights(net_from, net_to):
    with torch.no_grad():
        net_to.load_state_dict(net_from.state_dict())


# Experience replay buffer
class Buffer():
    def __init__(self, buffer_size, sizes, is_rmdp=False, device=torch.device('cpu')):
        self.buffer = [torch.zeros((buffer_size, x), device=device) for x in sizes]
        self.buffer_size = buffer_size
        self.buffer_ind = 0
        self.buffer_full = False
        self.is_rmdp = is_rmdp
        self.device = device
    
    def bufferLen(self):
        return self.buffer_size if self.buffer_full else self.buffer_ind
    
    def isFull(self):
        return self.buffer_full

    def add(self, *xs):
        with torch.no_grad():
            for t, x in enumerate(xs):
                self.buffer[t][self.buffer_ind, ...] = torch.tensor(x)
        self.buffer_ind += 1
        if self.buffer_ind >= self.buffer_size:
            self.buffer_ind = 0
            self.buffer_full = True

    def sample(self, num):
        buffer_len = self.bufferLen()
        inds = torch.randint(buffer_len, (num,))
        rv = [self.buffer[t][inds].squeeze() for t in range(len(self.buffer))]
        
        return rv

# Class packaging test and train methods for Deep Recursive Q-learning
class DeepRQL():
    def __init__(self):
        pass

    def test(self, \
             env, \
             net, \
             device, \
             episode_limit = 100, \
             test_num = 10):
        Rsum = 0
        
        num_actions = env.num_actions()
        state_size = env.observation_dim()
        num_exits = env.num_exits()
        
        for _ in range(test_num):
            S = env.reset()
            v = np.zeros(num_exits)
            v_min = 0
            stack = []
            exits_stack = []
            done = False
            episode_step = 0

            while not done:
                Sv_np = np.concatenate((S.squeeze(), v))

                with torch.no_grad():
                    Sv = torch.tensor(Sv_np, device=device).float()
                    _, ind = net(Sv).max(dim=0)
                    A = ind.detach().cpu().numpy()

                sPrimes, R, dones, info = env.step(A)
                done = dones[1]
                episode_step += 1
                
                Rsum += R

                exits = sPrimes[1]

                enteredBox = len(exits) != 0
                exitedBox = dones[0]

                if enteredBox:
                    exits_stack.append(exits)
                    with torch.no_grad():
                        # Get v_prime, v_min_prime
                        exitsV = [[*e, *v] for e in exits]
                        q_exits = net(torch.tensor(exitsV, device=device).float())
                        v_prime = q_exits.max(dim=-1)[0]

                        v_min_prime = v_prime.min(dim=-1)[0]
                        v_prime = v_prime - v_min_prime

                        v_min_prime = v_min_prime.cpu().numpy()
                        v_prime = v_prime.cpu().numpy()

                        # Push to stack
                        stack.append((v, v_min))
                        v, v_min = v_prime, v_min_prime
                elif exitedBox:
                    # Get k
                    k = 0
                    if not done:
                        exits_k = exits_stack.pop()
                        for e in exits_k:
                            if np.all(e == sPrimes[0]):
                                break
                            else:
                                k += 1
                        if k == len(exits_k):
                            raise RuntimeError("Could not match current state with exit. " + str(sPrimes[0]) + ' :: ' + str(exits_k))

                    # Pop stack
                    if not done:
                        v, v_min = stack.pop()

                S = sPrimes[0]

                if episode_limit > 0 and episode_step >= episode_limit:
                    break

        return Rsum/test_num

    def train(self, \
              env, \
              hidden_dim = 128, \
              hidden_layers = 2, \
              epsilon = 0.1, \
              final_epsilon = 0.01, \
              epsilon_frac = 0.1, \
              num_train_steps = int(2e5), \
              warm_up = 1000, \
              episode_limit = 100, \
              buffer_size = int(5e4), \
              batch_size = 64, \
              target_update_freq = 400, \
              lr = 0.001, \
              log_freq = 100, \
              test_num = 10, \
              device=torch.device('cpu')):

        num_actions = env.num_actions()
        state_size = env.observation_dim()
        num_exits = env.num_exits()
        
        test_env = copy.deepcopy(env)
        
        epsilon_step = (final_epsilon - epsilon)/(epsilon_frac*num_train_steps)

        # Create network
        net = FFN(state_size+num_exits, num_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        net.to(device)

        # Create target network
        target_net = FFN(state_size+num_exits, num_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        target_net.to(device)
        copy_weights(net, target_net)

        optim = torch.optim.Adam(net.parameters(), lr=lr)

        # Parameters
        buffer = Buffer(buffer_size, [state_size+num_exits, 1, 1, state_size+num_exits, 1, 1, 1], device=device)
        target_update_step = 0
        train_step = 0
        log_step = 0
        logs = [(train_step, self.test(test_env, net, device, test_num=test_num), epsilon)]
        log_buffer_full = None
        
        with tqdm(total=num_train_steps) as pbar:
            while train_step <= num_train_steps:
                S = env.reset()
                v = np.zeros(num_exits)
                v_min = 0
                stack = []
                exits_stack = []
                done = False
                episode_step = 0

                while not done:
                    Sv_np = np.concatenate((S.squeeze(), v))

                    if train_step > warm_up: # Pick epsilon greedy action
                        if np.random.rand() <= epsilon:
                            A = np.random.randint(num_actions)
                        else:
                            with torch.no_grad():
                                Sv = torch.tensor(Sv_np, device=device).float()
                                _, ind = net(Sv).max(dim=0)
                                A = ind.detach().cpu().numpy()
                    else: # Pick random action
                        A = np.random.randint(num_actions)

                    sPrimes, R, dones, info = env.step(A)
                    done = dones[1]
                    train_step += 1
                    log_step += 1
                    pbar.update(1)
                    episode_step += 1

                    if train_step < epsilon_frac*num_train_steps:
                        epsilon += epsilon_step
                    
                    exits = sPrimes[1]

                    enteredBox = len(exits) != 0
                    exitedBox = dones[0]
                    
                    if enteredBox and exitedBox:
                        raise RuntimeError("Cannot exit box with exits listed.")

                    # Push update to buffer
                    if enteredBox:
                        exits_stack.append(exits)
                        with torch.no_grad():
                            # Get v_prime, v_min_prime
                            exitsV = [[*e, *v] for e in exits]
                            q_exits = net(torch.tensor(exitsV, device=device).float())
                            v_prime = q_exits.max(dim=-1)[0]
                            v_min_prime = v_prime.min(dim=-1)[0]
                            v_prime = v_prime - v_min_prime

                            v_min_prime = v_min_prime.cpu().numpy()
                            v_prime = v_prime.cpu().numpy()

                            # Push update
                            sPrimeV_np = np.array((*sPrimes[0], *v_prime))
                            buffer.add(Sv_np, A, R, sPrimeV_np, exitedBox, enteredBox, v_min_prime)

                            # Push to stack
                            stack.append((v, v_min))
                            v, v_min = v_prime, v_min_prime
                    elif exitedBox:
                        # Get k
                        k = 0
                        if not done:
                            exits_k = exits_stack.pop()
                            for e in exits_k:
                                if np.all(e == sPrimes[0]):
                                    break
                                else:
                                    k += 1
                            if k == len(exits_k):
                                raise RuntimeError("Could not match current state with exit.")

                        # Push update
                        sPrimeV_np = np.zeros(state_size+num_exits) # Dummy state
                        buffer.add(Sv_np, A, R, sPrimeV_np, exitedBox, enteredBox, v[k])

                        # Pop stack
                        if not done:
                            v, v_min = stack.pop()
                    else:
                        # Push update
                        sPrimeV = torch.tensor((*sPrimes[0], *v))
                        buffer.add(Sv_np, A, R, sPrimeV, exitedBox, enteredBox, 0)

                    # Update network if buffer is ready
                    if train_step > warm_up:
                        # Save training step where buffer was first filled
                        if log_buffer_full == None:
                            log_buffer_full = train_step

                        # Sample from buffer
                        states, actions, rewards, states_prime, dones, enters, cs = \
                            buffer.sample(batch_size)
                            
                        # Compute targets
                        with torch.no_grad():
                            qmax_sprimes, _ = target_net(states_prime).max(dim=-1)
                            qmax_sprimes = qmax_sprimes.squeeze()

                            targs = rewards + qmax_sprimes*(1-dones) + torch.logical_or(dones, enters)*cs

                        # Update network
                        qs = net(states)[torch.arange(batch_size), actions.long()]
                        loss = (qs - targs).square().mean()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()            

                        # Update target network
                        target_update_step += 1
                        if target_update_step >= target_update_freq:
                            copy_weights(net, target_net)
                            target_update_step = 0

                    S = sPrimes[0]

                    if episode_limit > 0 and episode_step >= episode_limit:
                        break
            
                    # Log
                    if log_step > log_freq:
                        log_step = 0
                        logs.append((train_step, self.test(test_env, net, device, test_num=test_num), epsilon))
        
        return net, logs