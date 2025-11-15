# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choi ce point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        # If winning or losing, make that dominate
        if successor_game_state.is_win():
            return float("inf")
        if successor_game_state.is_lose():
            return -float("inf")

        score = successor_game_state.get_score()

        # Distance to the closest food
        try:
            food_list = new_food.as_list()
        except AttributeError:
            food_list = new_food.asList()

        if food_list:
            food_distances = [manhattan_distance(new_pos, food_pos) for food_pos in food_list]
            min_food_dist = min(food_distances)
            # Closer food is better (use reciprocal)
            score += 2.0 / (min_food_dist + 1.0)
            # Fewer food dots is better
            score -= 4.0 * len(food_list)

        # Ghost distances
        ghost_positions = [g.get_position() for g in new_ghost_states]
        if ghost_positions:
            ghost_distances = [manhattan_distance(new_pos, g_pos) for g_pos in ghost_positions]

            for dist, scared_time in zip(ghost_distances, new_scared_times):
                if scared_time > 0:
                    # Scared ghosts: getting closer is slightly good (chase for points but not too aggressively)
                    if dist > 0:
                        score += 1.5 / (dist + 1.0)
                else:
                    # Normal ghosts: avoid being too close (if too close, penalize)
                    if dist <= 1:
                        score -= 250.0 # Large penalization when close
                    else:
                        score -= 1.0 / (dist + 1.0)

        # Slight penalty for stopping
        if action == Directions.STOP:
            score -= 5.0

        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agents = game_state.get_num_agents()

        def minimax_value(state, agent_index, depth):
            # Terminal state or depth limit
            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluation_function(state)

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            # Compute next agent and depth
            next_agent = (agent_index + 1) % num_agents
            # Increase depth only when all agents have already moved
            next_depth = depth + 1 if next_agent == 0 else depth

            if agent_index == 0:
                # Pacman (MAX)
                value = -float("inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, minimax_value(successor, next_agent, next_depth))
                return value
            else:
                # Ghosts (MIN)
                value = float("inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, minimax_value(successor, next_agent, next_depth))
                return value

        # Root: choose the action that maximizes the minimax value
        best_value = -float("inf")
        best_action = Directions.STOP
        legal_actions = game_state.get_legal_actions(0)

        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = minimax_value(successor, 1 % num_agents, 0)
            # Tracking the best option
            if value > best_value:
                best_value = value
                best_action = action # Produce/Make the better move

        return best_action
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        num_agents = game_state.get_num_agents()

        def alphabeta_value(state, agent_index, depth, alpha, beta):
            # Terminal state or depth limit
            if depth == self.depth or state.is_win() or state.is_lose():
                return self.evaluation_function(state)

            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == 0 else depth

            if agent_index == 0:
                # Pacman (MAX)
                value = -float("inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alphabeta_value(successor, next_agent, next_depth, alpha, beta))
                    if value > beta:  # Beta-pruning (no equality pruning)
                        return value # Prune the remaining 
                    alpha = max(alpha, value) # Update best option by taking the MAX
                return value
            else:
                # Ghosts (MIN)
                value = float("inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alphabeta_value(successor, next_agent, next_depth, alpha, beta))
                    if value < alpha:  # Alpha-pruning (no equality pruning)
                        return value # Prune the remaining
                    beta = min(beta, value) # Update best option by taking the MIN
                return value

        alpha = -float("inf") # Best MAX score
        beta = float("inf") # Best MIN score
        best_value = -float("inf")
        best_action = Directions.STOP
        legal_actions = game_state.get_legal_actions(0)

        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = alphabeta_value(successor, 1 % num_agents, 0, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value) # Update alpha

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
