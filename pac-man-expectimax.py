# Basic utilities
import numpy as np
from copy import deepcopy

# Used to calculate heuristic
from scipy.sparse.csgraph import dijkstra

# Needed to pretty print game board
import os
import re

############################################################################################################################
# Utility function that colors indices in a string NumPy array - Used to display the game board in console                 #
# Taken from: https://stackoverflow.com/questions/72630094/how-to-change-the-font-color-of-the-specified-elements-in-numpy #
############################################################################################################################
class EscapeString(str):
    """A string that excludes SGR escape sequences from its length."""
    def __len__(self):
        return len(re.sub(r"\033\[[\d:;]*m", "", self))
    def __add__(self, other):
        return EscapeString(super().__add__(other))
    def __radd__(self, other):
        return EscapeString(str.__add__(other, self))

def color_array(arr, color, *lst, **kwargs):
    """Takes the same keyword arguments as np.array2string()."""
    # adjust some kwarg name differences from _make_options_dict()
    names = {"max_line_width": "linewidth", "suppress_small": "suppress"}
    options_kwargs = {names.get(k, k): v for k, v in kwargs.items() if v is not None}
    # this comes from arrayprint.array2string()
    overrides = np.core.arrayprint._make_options_dict(**options_kwargs)
    options = np.get_printoptions()
    options.update(overrides)
    # from arrayprint._array2string()
    format_function = np.core.arrayprint._get_format_function(arr, **options)

    # convert input index lists to tuples
    target_indices = set(map(tuple, lst))

    def color_formatter(i):
        # convert flat index to coordinates
        idx = np.unravel_index(i, arr.shape)
        s = format_function(arr[idx])
        if idx in target_indices:
            return EscapeString(f"\033[{30+color}m{s}\033[0m")
        return s

    # array of indices into the flat array
    indices = np.arange(arr.size).reshape(arr.shape)
    kwargs["formatter"] = {"int": color_formatter}
    return np.array2string(indices, **kwargs)
############################################################################################################################

#####################################################################################################################
#                                                                                                                   #
#   Function to check whether randomly generated obstacle map blocks any node from being accessed by the agents.    #
#       > Returns True if all non-obstacle nodes are reachable by all agents; returns False otherwise <             #
#                                                                                                                   #
#####################################################################################################################
def checkAccessible(obstacles_map, state):
    food_indices = (state[2] == True).nonzero()[0] # Generate list of nodes which have food
    adj_matrix = np.zeros(shape=(162,162),dtype=int) # Adjacency matrix for all nodes
    for i in range(162):
        if obstacles_map[i]: continue
        x = i % 18
        y = int(i/18)
        if x + 1 <= 17:
            if  not obstacles_map[i+1]:
                adj_matrix[i][i+1] = 1
        if x - 1 >= 0:
            if  not obstacles_map[i-1]:
                adj_matrix[i][i-1] = 1
        if y + 1 <= 8:
            if  not obstacles_map[i+18]:
                adj_matrix[i][i+18] = 1
        if y - 1 >= 0:
            if  not obstacles_map[i-18]:
                adj_matrix[i][i-18] = 1

    dist_matrix = dijkstra(csgraph=adj_matrix, directed=False, indices=state[0]) # Find shortest path from Pac-man to every other non-obstacle node.
    if (dist_matrix[food_indices] == np.inf).nonzero()[0].size != 0: return (False,) #[1] Check whether Pac-man can reach every food node.
    return (True, adj_matrix)                                                     #[2] If the output is True, there is no need to check the other agents,
    #                                                                             #[3] as they also lie on a food node.
    


#####################################################################
#                                                                   #
#                    Initialization of game                         #
#        Defined outside functions to make variables global         #
#                                                                   #
#####################################################################

# Game 'settings'
ghost_count = 2
row_count = 9
col_count = 18
N = col_count*row_count
score = 10
obstacle_count = 50
actions = ['up', 'down', 'left', 'right']

# Loop to re-generate game map if checkAccesible() returns False -- ie if map is not winnable.
while True:

    # Initialization of randomly generated variables
    obstacles = []
    food = []
    initial_state = ()
    pacman = 0
    adj_matrix = []
    ghosts = []
    # Randomized positions for non-food nodes
    randomized_nodes = np.random.randint(0, N, obstacle_count + ghost_count + 1)

    # Food boolean mask
    food = np.full(N, True)
    for i in randomized_nodes[:obstacle_count+1]:
        food[i] = False
    # for i in range(2, ghost_count+2):
    #     food[i] = True

    # Obstacle boolean mask
    obstacles = np.full(N, False)
    for i in randomized_nodes[:obstacle_count]:
        obstacles[i] = True

    # Randomized agent positions
    pacman = randomized_nodes[obstacle_count] 
    for i in range(1, ghost_count+1):
        ghosts.append(randomized_nodes[-i])

    # Store game state as array of 4
    initial_state = [pacman, ghosts, food, score]

    # Start game once suitable map has been generated
    result = checkAccessible(obstacles, initial_state)
    if result[0]:
        adj_matrix = result[1]
        break

###############################################################################################################
#                                                                                                             #
#                            Returns new agent position after taking an action                                #
#         > Returns input position unchanged if 'new position' is out of bounds or blocked by obstacle <      #
#                                                                                                             #
###############################################################################################################
def move(agentPosition, direction):
    match direction:
        case 'down':
            newAgentPosition = agentPosition + 18
            if newAgentPosition > 161 or obstacles[newAgentPosition]: 
                return agentPosition
            else: return newAgentPosition
        case 'up':
            newAgentPosition = agentPosition - 18
            if newAgentPosition < 0 or obstacles[newAgentPosition]: 
                return agentPosition
            else: return newAgentPosition
        case 'right':
            if (agentPosition % 18) == 17 or obstacles[agentPosition + 1]: 
                return agentPosition
            else: return agentPosition + 1
        case 'left':
            if (agentPosition % 18) == 0 or obstacles[agentPosition - 1]: 
                return agentPosition
            else: return agentPosition - 1

#####################################################################################################################
#                                                                                                                   #
#                  Calculate and return new game state after an agent takes an action                               #
#         > New state will be identical to old state (except for score) if move() returns position unchanged <      #
#                                                                                                                   #
#####################################################################################################################
def transition(state, agent_id, direction):
    # Initialize new state to old state
    new_state = deepcopy(state)

    # If it is Pac-man's turn
    if agent_id == 0:
        # Update Pac-man's position
        new_state[0] = move(state[0], direction)

        # Deduct 1 point for each tile moved (even if movement did not change position)
        new_state[3] -= 1

        # Eat food and gain points if there is food at new position
        if new_state[2][new_state[0]] == True:
            new_state[3] += 10
            new_state[2][new_state[0]] = False

    # If it is the ghosts turn
    else:
        # Update selected ghosts position
        new_state[1][agent_id-1] = move(state[1][agent_id-1], direction)

    return new_state

###################################################################################
#                                                                                 #
#      Evaluation function to assign a utility score to non-terminal nodes        #
#                                                                                 #
###################################################################################
def evaluationFunction(state):
    # Initialiaze utility to current state's score
    eval_score = state[3]

    # Calculate shortest path from Pac-man to every other node
    dist_matrix = dijkstra(csgraph=adj_matrix, directed=False, indices=state[0])

    # Increase score if ghosts are far.
    ghost_distances = [dist_matrix[ghost] for ghost in state[1]]
    for gd in ghost_distances:
        if gd <= 1:
            return -99999
        else:
            # As the closest distance a ghost can get is 2 (cases below 2 are handled above),
            # the (2/gd) expression can become 1 at most. The reason for raising to the 2nd power
            # is to make it so that "far" distances are all nearly equally as good and so that
            # Pac-man does not waste time trying to maximize distance from ghosts when he is not
            # in danger (meaning they are far away enough).
            eval_score += (1-(2/gd)**2)

    # Calculate minimum distance to a food node.
    food_indices = (state[2] == True).nonzero()[0]
    food_distances = np.array(dist_matrix[food_indices])
    minFoodDist = min(food_distances, default=999999) # Default value is so that the game does not crash when Pac-man is about to win.

    # Weighted sum of features. Minimum distance to food is (number of ghosts) times more valuable than the distance to each ghost.
    eval_score += 5 * len(state[1]) * (1/np.min(minFoodDist))
    return eval_score

#############################################################################
#       Fuction that returns True if all food nodes have been eaten         #
#############################################################################
def checkWin(state):
    if len((state[2] == True).nonzero()[0]) == 0: return True

#####################################################################
#       Fuction that returns True any ghost reaches Pac-man         #
#####################################################################
def checkLose(state):
    for ghost in state[1]:
        if ghost == state[0]: return True


#####################################################
#       Depth-limited Expectimax algorithm          #
#####################################################
def expectimax(state, agent, depth):
    # Check if 'end of tree' is reached at this state and evaluate said state if True
    if checkLose(state) or checkWin(state) or depth == 6:
        return (evaluationFunction(state),)

    # If it's Pac-man's turn, maximize eval score.
    if agent == 0:
        highest_eval = -9999999
        optimal_move = ''
        res = 0

        #[1] Since moving into obstacles or map boundaries simply does not move the agent,
        #[2] omit actions that would yield a duplicate end position.
        unique_actions = set()
        viable_actions = []
        for a in actions:
            prev_actions = set.copy(unique_actions)
            unique_actions.add(move(state[0],a))
            if prev_actions != unique_actions:
                viable_actions.append(a)

        # Perform search over actions with unique outcomes
        for a in viable_actions:
            res = expectimax(transition(state, 0, a), 1, depth+1)[0]
            if res > highest_eval:
                highest_eval = res
                optimal_move = a
        return (highest_eval, optimal_move)
    
    # If ghosts turn, calculate expected utility based on action probabilities (in our case all actions are equally as likely).
    elif agent >= 1:
        unique_actions = set()
        viable_actions = []
        expected_value = 0

        #[1] Since moving into obstacles or map boundaries simply does not move the agent,
        #[2] omit actions that would yield a duplicate end position.
        for a in actions:
            prev_actions = set.copy(unique_actions)
            unique_actions.add(move(state[1][agent-1],a))
            if prev_actions != unique_actions:
                viable_actions.append(a)
        
        # Weighted sum of each sub-trees expected utility (since all actions are equally likely, simply sum and divide by number of actions)
        for a in viable_actions:
            if agent == ghost_count: next_agent = 0
            else: next_agent = agent + 1
            expected_value += expectimax(transition(state, agent, a), next_agent, depth+1)[0]

        return (expected_value/len(viable_actions),)
                
#################################
#       Main game loop          #
#################################     
def main():
    game_state = initial_state

    while True:
        # Code to display the game board in console and print each agents position in addition to current game score
        os.system("cls")

        # Constructing the game board
        game_board = np.full(fill_value="-",shape=(9,18),dtype=str)
        food_indices = (game_state[2] == True).nonzero()[0]
        for i in food_indices:
            game_board[int(i/18)][i%18] = 'o'
        game_board[int(game_state[0]/18)][game_state[0] % 18] = 'P'
        i = 1
        for g in game_state[1]:
            game_board[int(g/18)][g % 18] = i
            i += 1
        game_board[np.reshape(obstacles, (9,18))] = 'X'

        # Display agent positions and game score
        print('Current Pac-man position: {}, Current score: {}, Ghost1 position: {}, Ghost2 position: {}'.format(game_state[0],game_state[3], game_state[1][0],game_state[1][1]))

        #[1] There are two ways to print the board:
        #[2] (1) using 'np.array2string' will omit the quotation marks from every cell, however Pac-man won't be colored.
        #[3] (2) using 'color_array' will not omit the quotation marks, however it will color Pac-man for easier visibility.
        # print(np.array2string(game_board, separator='  ', formatter={'str_kind': lambda x: x}))
        print(color_array(game_board, 1, [int(game_state[0]/18), game_state[0] % 18]))

        # Checking if Pac-man has lost or won every turn (ply)
        if checkLose(game_state):
            print("Pac-man lost.")
            break
        elif checkWin(game_state):
            print("Pac-man won!")
            break
        
        
        # Advacing the game based on agent behaviour (best move for Pac-man and random moves in the case of the ghosts).
        best_move = expectimax(game_state, 0, 0)[1]
        
        # Pac-man
        game_state = transition(game_state, 0, best_move)

        # Ghosts
        for i in range(1, ghost_count+1):
            game_state = transition(game_state, i, np.random.choice(actions))
        

main()