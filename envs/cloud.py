import numpy as np

class CloudEnv():
    def __init__(self, encoding='human'):
        self.encoding = encoding
        self.stack = []
    
    def encode(self, state):
        if state is None:
            return None
        if self.encoding == 'human':
            return state
        elif self.encoding == 'vector':
            b = np.zeros(3)
            S = np.zeros(5)
            if state[0] == 'T':
                b[0] = 1
            elif state[0] == 'S':
                b[1] = 1
            else:
                b[2] = 1
            S[state[1]] = 1
            return np.hstack((b,S))
        else:
            raise NotImplementedError(self.encoding)
    
    def num_actions(self):
        return 2
    
    def observation_dim(self):
        return 8
    
    def num_exits(self):
        return 1
    
    def reset(self):
        self.stack = []
        self.state = ('T', 0)
        return self.encode(self.state)
    
    def step(self, action):
        R = -1
        box_done = False
        global_done = False
        box_exits = []
        info = {}
        
        current_box = self.state[0]
        current_node = self.state[1]
        
        if current_box == 'T':
            if current_node == 0:
                if action == 0: # Decompose
                    self.state = ('S', 0)
                    box_exits = [('T', 1)]
                    self.stack.append(('T', 1))
                    R = -0.5
                elif action == 1: # Monolithic
                    box_done = True
                    global_done = True
                    R = -8
            elif current_node == 1:
                self.state = ('S', 0)
                box_exits = [('T', 2)]
                self.stack.append(('T', 2))
                R = 0
            elif current_node == 2:
                self.state = ('S', 0)
                box_exits = [('T', 3)]
                self.stack.append(('T', 3))
                R = 0
            elif current_node == 3:
                box_done = True
                global_done = True
                R = -0.5
        elif current_box == 'S':
            if current_node == 0:
                if action == 0: # Reliable
                    R = -1.5#-1.6
                    if np.random.rand() < 0.3:
                        self.state = ('H', 0)
                        box_exits = [('S', 2), ('S', 3)]
                        self.stack.append(box_exits)
                    else:
                        box_done = True
                        self.state = self.stack.pop()
                elif action == 1: # Fast
                    if np.random.rand() < 0.4:
                        self.state = ('S', 0)
                        box_exits = [('S', 1)]
                        self.stack.append(('S', 1))
                    else:
                        box_done = True
                        self.state = self.stack.pop()
                    R = -1
            elif current_node == 1:
                box_done = True
                self.state = self.stack.pop()
                R = 0
            elif current_node == 2:
                box_done = True
                self.state = self.stack.pop()
                R = 0.2
            elif current_node == 3:
                box_done = True
                self.state = self.stack.pop()
                R = 0.2
        elif current_box == 'H':
            if current_node == 0:
                if action == 0: # Don't upgrade
                    if np.random.rand() < 0.7:
                        box_done = True
                        self.state = self.stack.pop()[1]
                        R = 0
                    else:
                        self.state = ('H', 0)
                        box_exits = [('H', 1), ('H', 2)]
                        self.stack.append(box_exits)
                    R = -0.01
                else: # Upgrade
                    box_done = True
                    self.state = self.stack.pop()[0]
                    R = -0.2
            elif current_node == 1: # Box 1, exit upgrade
                box_done = True
                self.state = self.stack.pop()[0]
                R = 0#-0.2
            elif current_node == 2: # Box 1, exit don't upgrade
                self.state = ('H', 0)
                box_exits = [('H', 3), ('H', 4)]
                self.stack.append(box_exits)
                R = 0
            elif current_node == 3: # Box 2, exit upgrade
                box_done = True
                self.state = self.stack.pop()[0]
                R = 0
            elif current_node == 4:# Box 2, exit don't upgrade
                box_done = True
                self.state = self.stack.pop()[1]
                R = 0
        
        return (self.encode(self.state), [self.encode(be) for be in box_exits]), R, (box_done, global_done), info