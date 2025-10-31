# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    #Creating stack (DFS uses stack)
    stack = util.Stack()
    #Getting problem's initial state
    frontier = problem.get_start_state()
    #Push into stack initial state, initial cost and the empty list of the different actions
    stack.push([frontier, 0, []])
    #List to store the nodes we expand to
    expandedNodes = []

    while stack:
        #Pop the top items out of the stack to see the attributes of the next move
        [n, cost, action] = stack.pop()

        #Checking if node n is the goal state
        if problem.is_goal_state(n):
            return action
        
        #Check if n is not in expandedNodes
        if n not in expandedNodes:
            expandedNodes.append(n)
            #Getting node successors
            successors = problem.get_successors(n)

            #Iterating through all successors with the corresponding attribute
            for n_successor, action_sucessor, cost_sucessor in successors:
                if n_successor not in expandedNodes:
                    #Updating the cost
                    new_cost = cost + cost_sucessor
                    new_action = action + [action_sucessor]
                    #Push unvisited successor onto stack
                    stack.push([n_successor, new_cost, new_action]) 
    
    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))

    return []


def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #Instead of stack like in DFS, we use queue for BFS
    Queue = util.Queue()
    frontier = problem.get_start_state()
    Queue.push([frontier, 0, []]) 

    expandedNodes = []
    #Rest of the code the same as DFS
    while Queue:
        [n, cost, action] = Queue.pop()

        if problem.is_goal_state(n):
            return action

        if n not in expandedNodes:
            expandedNodes.append(n)
            successors = problem.get_successors(n)

            for n_successor, action_sucessor, cost_sucessor in successors:
                if n_successor not in expandedNodes:
                    new_cost = cost + cost_sucessor
                    new_action = action + [action_sucessor]
                    Queue.push([n_successor, new_cost, new_action])

    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    from util import PriorityQueue
    start = problem.get_start_state()
    frontier = PriorityQueue()
    frontier.push((start, []), 0)
    # Mejor coste conocido para cada estado
    best_g = {start: 0}

    while not (frontier.is_empty() if hasattr(frontier, "is_empty") else frontier.is_empty()):
        state, actions = frontier.pop()
        g = problem.get_cost_of_actions(actions)

        if problem.is_goal_state(state):
            return actions

        # Si aparece un camino más caro que uno ya conocido, saltamos
        if best_g.get(state, float("inf")) < g:
            continue

        for succ, action, step_cost in problem.get_successors(state):
            new_actions = actions + [action]
            new_g = g + step_cost
            if new_g < best_g.get(succ, float("inf")):
                best_g[succ] = new_g
                frontier.push((succ, new_actions), new_g)

    return []

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    start = problem.get_start_state()
    frontier = PriorityQueue()
    frontier.push((start, []), heuristic(start, problem))
    best_g = {start: 0}

    while not (frontier.is_empty() if hasattr(frontier, "is_empty") else frontier.is_empty()):
        state, actions = frontier.pop()
        g = problem.get_cost_of_actions(actions)

        if problem.is_goal_state(state):
            return actions

        # Si ya tenemos un g mejor para este estado, saltamos
        if best_g.get(state, float("inf")) < g:
            continue

        for succ, action, step_cost in problem.get_successors(state):
            new_actions = actions + [action]
            new_g = g + step_cost
            if new_g < best_g.get(succ, float("inf")):
                best_g[succ] = new_g
                f = new_g + heuristic(succ, problem)
                frontier.push((succ, new_actions), f)

    return []

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search