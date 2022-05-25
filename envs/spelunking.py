import numpy as np

class SpelunkingEnv():
    def __init__(self, encoding='human', ascend_chance=0.01, trap_chance=0.5):
        self.encoding = encoding
        
        # o == normal, air
        # x == trap
        # w == wall (there are no walls in final environment)
        # r == rope/equipment
        
        self.levels =  {
            0:
            [
                ['o', 'x', 'o'],
                ['o', 'x', 'o'],
                ['o', 'x', 'o']
            ],
            
            1:
            [
                ['o', 'o', 'x'],
                ['x', 'o', 'o'],
                ['r', 'x', 'o']
            ]
        }
        self.n = 3 # Level dimensions are n x n

        # State :
        # [box_type, [x, y, has_rope, hole_x, hole_y]]
        self.init_state = [0, [0, 2, False, 0, 2]]
        self.ascend_chance = ascend_chance
        self.trap_chance = trap_chance
    
    def encode(self, state):
        if state is None:
            return None
        if self.encoding == 'human':
            return state
        elif self.encoding == 'vector':
            return np.array(state)
        else:
            raise NotImplementedError(self.encoding)
    
    def num_actions(self):
        return 5
        
    def observation_dim(self):
        return 6
        
    def num_exits(self):
        return 1
        
    def reset(self):
        self.S = [self.init_state[0], self.init_state[1].copy()]
        self.stack = []
        return self.encode((self.S[0], *self.S[1]))
    
    def check_fall(self, state):
        new_level = 0 if state[0] == 1 else 1
        has_rope = self.levels[new_level][state[1][0]][state[1][1]] == 'r' or state[1][2]
        new_state = [new_level, [
            state[1][0], state[1][1], has_rope, state[1][0], state[1][1]
        ]]
        
        return new_state, self.check(new_state)
    
    def move(self, state, action):
        new_state_coords = None
        if action == 0: # North
            new_state_coords = (max(state[1][0]-1, 0), state[1][1])
        if action == 1: # East
            new_state_coords = (state[1][0], min(state[1][1]+1, self.n-1))
        if action == 2: # South
            new_state_coords = (min(state[1][0]+1, self.n-1), state[1][1])
        if action == 3: # West
            new_state_coords = (state[1][0], max(state[1][1]-1, 0))
        
        grid_type = self.levels[state[0]][new_state_coords[0]][new_state_coords[1]]
        has_rope = grid_type == 'r' or state[1][2]
        if grid_type == 'w':
            return [state[0], state[1].copy()]
        new_state = [state[0], [*new_state_coords, has_rope, state[1][3], state[1][4]]]
        
        return new_state
    
    def check(self, state):
        """
        Returns state status.
        Status:
            0 == good
            1 == trap
            2 == invalid
            3 == rope
        """
        status = 0
        grid_type = self.levels[state[0]][state[1][0]][state[1][1]]
        if grid_type == 'o':
            status = 0
        elif grid_type == 'x':
            status = 1
        elif grid_type == 'w':
            status = 2
        elif grid_type == 'r':
            status = 3
        return status
    
    def step(self, action):
        reward = -1
        box_done = False
        global_done = False
        box_exits = []
        info = {}

        # Update state
        fall_state, fall_status = self.check_fall(self.S)
        
        if self.check(self.S) == 1 and np.random.rand() < self.trap_chance and self.S[1][2] == False: # Trap fall
            fall_loc = [np.random.randint(3), np.random.randint(3)] # Get random location
            exit = [self.S[0], self.S[1].copy()]
            exit[1][2] = True
            exit[1][0] = fall_loc[0]
            exit[1][1] = fall_loc[1]
            self.stack.append(exit)
            box_exits.append((exit[0], *exit[1]))
            fall_state[1][0] = fall_loc[0]
            fall_state[1][1] = fall_loc[1]
            fall_state[1][3] = fall_loc[0]
            fall_state[1][4] = fall_loc[1]
            self.S = fall_state
            reward = -2
        else: # Continue
            if action == 4 and self.S[1][2] == False: # Fall
                exit = [self.S[0], self.S[1].copy()]
                exit[1][2] = True
                self.stack.append(exit)
                box_exits.append((exit[0], *exit[1]))
                self.S = fall_state
            elif action == 4 and self.S[1][2] == True \
                 and self.S[1][0] == self.S[1][3] and self.S[1][1] == self.S[1][4]: # Ascend
                box_done = True
                if len(self.stack) == 0:
                    global_done = True
                else:
                    self.S = self.stack.pop()
            elif action < 4:
                self.S = self.move(self.S, action)
            else:
                reward = -5 # Invalid
                
        # Discounting
        if np.random.rand() < self.ascend_chance and len(box_exits) == 0:
            box_done = True
            if len(self.stack) == 0:
                global_done = True
            else:
                self.S = self.stack.pop()
        
        return [self.encode((self.S[0], *self.S[1])), [self.encode(box_e) for box_e in box_exits]], \
               reward, [box_done, global_done], info