
#_______________________________________________
#               Submitted by:
#   Name: Prashanna Raj pandit
#   SID:800817018
#_______________________________________________





# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print (successorGameState)
        # print(newFood)
        # print(newPos)
        # print(newGhostStates)
        
        # return successorGameState.getScore()

        # Compute the Manhattan distance between Pacman's new position and each ghost's position.
        ghostDistance = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        # Compute the Manhattan distance between Pacman's new position and each food's position.
        foodsDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]
        # If there is food remaining (i.e., foodsDistance is not empty):
        if len(foodsDistance) != 0:
        # Evaluate the score by considering the following:
        # - The base score from the successorGameState.
        # - Subtract the minimum food distance (since the closer Pacman is to food, the better).
        # - Add the minimum ghost distance (since Pacman should ideally avoid ghosts).
            return(successorGameState.getScore()-min(foodsDistance)+min(ghostDistance))
        else:       # If there is no food left, just return the successorGameState's score.
            return(successorGameState.getScore())
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.ReturnAction(gameState, 0, 0)[0] # Return the action with the best score.
        # ReturnAction returns a tuple (action, score). Here, we're returning just the action (first element). i.e. [0]
    
    def ReturnAction(self, gameState, curDepth, agent):
        """
        Recursive function to calculate the minimax action and score for a given agent.

        Parameters:
        - gameState: the current state of the game.
        - curDepth: the current depth in the game tree (counts total moves, Pacman + ghosts).
        - agent: the current agent (Pacman = 0, ghosts >= 1).

        Returns a tuple (bestAction, bestScore).
        """
        numAgents = gameState.getNumAgents()  # Get the number of agents (Pacman + all ghosts).
        # Base case: if the game is over (win or lose) or if we have reached the maximum search depth.
        if gameState.isWin() or gameState.isLose() or curDepth == self.depth * numAgents:
            return (None, self.evaluationFunction(gameState)) # Return the evaluation score for the current state.
        
       # Increment agent index, i.e., switch from Pacman to the next ghost or vice versa.
        if curDepth != 0:
            agent = 0 if agent == numAgents - 1 else agent + 1   # Move to the next agent. If it's the last ghost, go back to Pacman.

        scores = []
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            res_act, res_score = self.ReturnAction(successor, curDepth + 1, agent)  # Recursively calculate the action and score for the successor state.
            scores.append(res_score)
        
        bestScore, actInd = self.getBestScore(agent, scores) # Get the best score and the index of the corresponding action.
        return (gameState.getLegalActions()[actInd], bestScore) # Return the best action and its score

    def getBestScore(self, agent, scores):
        """
        Helper function to select the best score for the current agent.

        - If the agent is Pacman (agent 0), we want the maximum score.
        - If the agent is a ghost (agent > 0), we want the minimum score (because ghosts minimize Pacman's score).
        
        Returns the best score and the index of the corresponding action.
        """
        if agent == 0: # Pacman is the maximizing player.
            return max(scores), scores.index(max(scores)) # Return the maximum score and the index of the best action.
        return min(scores), scores.index(min(scores)) # Ghosts minimize the score, so return the minimum.
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initialize alpha and beta values for alpha-beta pruning. #implementing algorithm
        alpha = -(float("inf"))
        beta = float("inf")

        def Maximizer(gameState, depth, alpha, beta):
            # Initialize v with negative infinity, since we are looking for the maximum value.
            v = -(float("inf"))
            takeAction = None    #This will hold the best action to take for Pacman.

            # Get Pacman's possible legal actions (directions: 'North', 'South', etc.)
            actionList = gameState.getLegalActions(0) # 0 is Pacman's index.
            # If no actions are available, or if it's a terminal state (win/lose), or max depth is reached,
            # we return the evaluated value of this state using the evaluation function.
            if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            

            for action in actionList:
                # Generate the successor game state after Pacman takes this action.
                # Then, we call min_value for the ghosts' turns (agentID 1).
                successorValue = Minimizer(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)[0]
                # If the value from the successor state is greater than the current v, we update v and the action.
                if (v < successorValue):
                    v, takeAction = successorValue, action

                if (v > beta):  # Alpha-beta pruning condition: If v is greater than beta, prune this branch.
                    return (v, takeAction)

                alpha = max(alpha, v) # Update alpha: alpha is the best value that Pacman can guarantee.

            return (v, takeAction)

        
        def Minimizer(gameState, agentID, depth, alpha, beta):   # This function computes the minimum value for the ghosts (Min agents).
            v = float("inf")   # Initialize v with positive infinity, since we are looking for the minimum value (ghosts minimize Pacman's score).
            takeAction = None

            actionList = gameState.getLegalActions(agentID) # Get the actions of the ghost
            if len(actionList) == 0:
              return (self.evaluationFunction(gameState), None)

            for action in actionList:
                if (agentID == gameState.getNumAgents() - 1):   # If it's the last ghost, call max_value to let Pacman take the next turn.
                    successorValue = Maximizer(gameState.generateSuccessor(agentID, action), depth + 1, alpha, beta)[0]
                else:  # Otherwise, call min_value recursively for the next ghost.
                    successorValue = Minimizer(gameState.generateSuccessor(agentID, action), agentID + 1, depth, alpha, beta)[0]
                if (successorValue < v): # If the successor value is less than v, we update v and the ghost's action.
                    v, takeAction = successorValue, action

                if (v < alpha):
                    return (v, takeAction)

                beta = min(beta, v)

            return (v, takeAction)
        
        return Maximizer(gameState, 0, alpha, beta)[1]  # Call the Maximizer function for Pacman's first move (at depth 0) and return the chosen action.
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState):

            """
            agent: The current agent (0 for Pacman, > 0 for ghosts)
            depth: The current depth in the game tree
            gameState: The current state of the game
            
            This function recursively calculates the expectimax value of each game state.
            """
            # Base case: return the utility (evaluation function result) if:
            # The game state is a win or a loss.
            # The maximum search depth has been reached.
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(gameState)
            if agent == 0:   # If the agent is Pacman (maximizing player, agent 0):
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # If the agent is a ghost (chance node):
                nextAgent = agent + 1  # Determine the next agent (or back to Pacman if all ghosts have moved)
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0  # back to pacman
                if nextAgent == 0:
                    depth += 1     # Increase depth after all agents (Pacman + ghosts) have moved
                # Expectimax for ghosts (average the values of all possible moves, since they act randomly)
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) 
                           for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

        # Performing maximizing task for the root node i.e. pacman
        maximum = float("-inf")
        action = Directions.WEST  # Default action to move west in case nothing better is found
        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"): # If the utility of this action is better than the maximum found so far, update the action and maximum value
                maximum = utility
                action = agentState

        return action     # Return the best action found for Pacman
        # util.raiseNotDefined()



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    MinimumFoodDistance = -1
    # GhostDistance tracks the total distance to all ghosts
    # proximity_to_ghosts tracks the number of ghosts within a distance of 1 (danger zone)
    GhostDistance = 1  # Start with 1 to avoid division by zero later
    proximityToGhosts = 0  # Counts how many ghosts are too close to Pacman


    for food in newFoodList: # Calculate distance to the closest food pellet using Manhattan Distance
        distance = util.manhattanDistance(newPos, food)
        if MinimumFoodDistance >= distance or MinimumFoodDistance == -1:  # If this is the closest food (or first food checked), update the minimum food distance
            MinimumFoodDistance = distance

    # Loop over all ghost positions to calculate distances
    for ghost_state in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost_state)
        GhostDistance += distance  # Accumulate the distance to all ghosts
        
        # If a ghost is too close (distance <= 1), increment the danger counter
        if distance <= 1:
            proximityToGhosts += 1


    # Extract the list of remaining power capsules and count them
    newCapsule = currentGameState.getCapsules()
    numberOfCapsules = len(newCapsule)

    # Return the overall score by combining several factors:
    # - The current score of the game state
    # - Pacman prefers states where the closest food is nearby (Minimum Food Distance is small)
    # - Pacman prefers to stay far from ghosts (large GhostDistance)
    # - Proximity to ghosts is dangerous, so we penalize states where ghosts are nearby
    # - The fewer remaining capsules, the better (since Pacman should consume them)
    return currentGameState.getScore() + (1 / float(MinimumFoodDistance)) - (1 / float(GhostDistance)) - proximityToGhosts - numberOfCapsules

# Abbreviation
better = betterEvaluationFunction

