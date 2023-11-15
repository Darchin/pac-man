import numpy as np
import math


class game:
    class position:
        def __init__(self, position):
            self.position = position
            self.X = self.position % 18
            self.Y = math.floor(self.position/18)
        def getPos(self):
            return self.position
        def getX(self):
            return self.X
        def getY(self):
            return self.Y
        def updateCoords(self):
            self.X = self.position % 18
            self.Y = math.floor(self.position/18)
        def setPos(self, newPos):
            self.position = newPos
            self.updateCoords()
    class agent:
        def __init__(self, id, pos):
            self.id = id
            self.position = super().position(pos)
        def getPos(self):
            return self.position
    class state:
        def __init__(self, food, pacman, ghosts, score):
            self.food = food
            self.pacman = pacman
            self.ghosts = ghosts
            self.score = score
        def getPacmanPos(self):
            return self.pacman.getPos()
        def getGhosts(self):
            return self.state[1]
        def getFood(self):
            return self.state[2]
        def getScore(self):
            return self.state[3]    
        def getFoodIndices(self):
            return (self.getFood() == True).nonzero()[0]

        def setPacman(self, pos):
            self.pacman = pos 
        def setGhost(self, ghost_no, pos):
            self.ghosts[ghost_no] = pos
        def markFood(self, food_idx):
            self.food[food_idx] = False
        def updateScore(self, new_score):
            self.score = new_score
        def checkWin(self):
            if self.getFoodIndices.size == 0: return True
        def checkLose(self):
            for g in self.getGhosts:
                if g == self.getPacman: return False
        def copy(self):
            return state()
    def __init__(self, row_count, col_count, num_of_ghosts=2, obstacle_frequency=1/3):

        # Set grid_size, save row/col count
        self.GRID_SIZE = row_count * col_count
        self.HORIZONTAL_MAX = col_count
        self.VERTICAL_MAX = row_count


        # Food mask
        self.food = np.full(self.GRID_SIZE, True)

        randomized_nodes = np.random.randint(0, self.GRID_SIZE, obstacle_frequency*math.floor(self.GRID_SIZE) + num_of_ghosts)

        for i in randomized_nodes:
            self.food[i] = False

        # Obstacle mask
        obstacles = np.full(self.GRID_SIZE, False)
        for i in randomized_nodes[:obstacle_frequency]:
            obstacles[i] = True

        # Randomizing agent positions
        self.pacman = self.agent(0, randomized_nodes[-1])
        self.ghosts = []
        for i in range(2, num_of_ghosts+2):
            self.ghosts.append(self.agent(i-1, randomized_nodes[-i]))

        self.game_state = self.state(self.food, self.pacman, self.ghosts, self.score)
    def evalFunc(state):
        eval_score = 0

        ghost_distances = [manhattan_distance(state.getPacman(), ghost) for ghost in state.getGhosts()]
        for gd in ghost_distances:
            if gd == 1:
                return -99999
            else:
                eval_score += (1-(2/gd)^2)

        food_indices = state.getFoodIndices()
        food_distances = [manhattan_distance(state.getPacman(), f) for f in food_indices]

        eval_score += 5 * state.getGhosts().size * (1/np.min(food_distances))
        return eval_score
    def move(self, pos, direction):
        N = self.GRID_SIZE - 1
        # position from 0 to GRID_SIZE-1
        pos_INT = pos.getPosition()
        # 0: Up, 1: Right, 2: Down, 3: Left
        match direction:
            case 0:
                new_pos = pos_INT + self.HORIZONTAL_MAX
                if new_pos > N or self.obstacles[new_pos]: return pos
                else: return new_pos
            case 2:
                new_pos = pos - self.HORIZONTAL_MAX
                if new_pos < 0 or self.obstacles[new_pos]: return pos
                else: return new_pos
            case 1:
                if pos.getX() == self.HORIZONTAL_MAX - 1 or self.obstacles[pos+1]: return pos
                else: return pos + 1
            case 3:
                if pos.getX() == 0 or self.obstacles[pos-1]: return pos
                else: return pos - 1
    def transition(self, state, agent, action):
        new_state = self.state()
        if agent == 0:
            new_pos = self.move(state.getPacmanPos(), action)
            state.markFood(new_pos)
            

        
    
def manhattan_distance(pos1, pos2):
    x1 = pos1 % 18
    x2 = pos2 % 18
    y1 = math.floor(pos1/18)
    y2 = math.floor(pos2/18)
    return abs(x2-x1) + abs(y2-y1)

class Minimax:
    def maximize(state, depth):
        if state.checkWin() 
    def minimize(state, depth):
def minimax(state, depth, agent):
    if state.checkWin or state.:



def main():
    food = np.full(162, True)
    randomized_nodes = np.random.randint(0, 162, 53)

    for i in randomized_nodes:
        food[i] = False
    food_indices = (food == True).nonzero()[0]

    obstacles = np.full(162, False)
    for i in randomized_nodes[:50]:
        obstacles[i] = True

    pacman = randomized_nodes[50]
    ghost1 = randomized_nodes[51]
    ghost2 = randomized_nodes[52]

main()
