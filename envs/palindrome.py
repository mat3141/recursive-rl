import numpy as np

class PalindromeEnv():
    def __init__(self, encoding='human', pop_chance=0.01):
        self.encoding = encoding
        self.n = 3
        self.goal = (1,1)
        self.pop_chance = pop_chance
    
    def num_actions(self):
        return 5
    
    def num_exits(self):
        return 2*self.n**2
    
    def observation_dim(self):
        return 7
    
    def encode(self, state):
        if self.encoding == 'human':
            return tuple(state)
        elif self.encoding == 'vector':
            return np.array(state)
        else:
            raise RuntimeError("Encoding " + self.encoding + " not supported")
    
    def reset(self):
        self.stack = []
        goal_x = 1
        goal_y = 1
        self.goal = (goal_x,goal_y)
        x = np.random.randint(self.n)
        y = np.random.randint(self.n)
        self.state = [x,y,False,4,False,goal_x,goal_y]
        return self.encode(self.state)
    
    def move(self, state, action):
        # North
        if action == 0:
            state[0] = max(0, state[0]-1)
        # East
        if action == 1:
            state[1] = min(self.n-1, state[1]+1)
        # South
        if action == 2:
            state[0] = min(self.n-1, state[0]+1)
        # West
        if action == 3:
            state[1] = max(0, state[1]-1)
        # Pop
        if action == 4:
            pass
        if action > 4:
            raise RuntimeError("Invalid move")
        
        return state
    
    def getExits(self, action):
        exits = []
        for x in range(self.n):
            for y in range(self.n):
                for b in [False, True]:
                    exits.append((x,y,True,action,b,self.goal[0],self.goal[1]))
        return exits
    
    def step(self, action):
        if np.random.rand() < self.pop_chance: # Action corrupted to pop to ensure properness
            action = 4
        
        info = {}
        reward = -1
        boxDone = False
        globalDone = False
        exits = []
        
        # State:
        # x, y, popped, action, trap, goal_x, goal_y
        # 0, 1, 2,      3,      4,    5,      6
        
        # Pop == true
        if self.state[2] == True:
            boxDone = True
            if len(self.stack) == 0 and self.state[3] == 4: # Done
                globalDone = True
                if action != 4:
                    self.state[4] = True
                    reward = -5
            elif len(self.stack) == 0:
                if action != self.state[3]:
                    self.state[4] = True
                    reward = -5
                self.state = self.move(self.state.copy(), action)
                self.state[3] = 4
            else: # Continue
                if action != self.state[3]:
                    self.state[4] = True
                    reward = -5
                self.state = self.move(self.state.copy(), action)
                self.state[3] = self.stack.pop()
        else:
            # Pop 
            if action == 4:
                self.state[2] = True
                if len(self.stack) > 0:
                    self.stack.pop()
            else:
                self.stack.append(action)
                self.state = self.move(self.state.copy(), action)
                exits = self.getExits(self.state[3])
                self.state[3] = action
        
        if globalDone and self.state[0] == self.goal[0] and self.state[1] == self.goal[1] and not self.state[4]:
            reward = 50
            
        return [self.encode(self.state), [self.encode(x) for x in exits]], reward, [boxDone, globalDone], info