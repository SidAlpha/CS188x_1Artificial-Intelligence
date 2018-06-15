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

import numpy as np

from math import sqrt
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        def dist(a,b):
          return sqrt ((b[0] - a[0])**2 + (b[1] - a[1])**2)

        if   len([dist(capDist,successorGameState.getPacmanPosition()) for capDist in currentGameState.getCapsules()]) > 0:
          CapDist = (min([dist(capDist,newPos) for capDist in currentGameState.getCapsules()]) +1)
        else: 
          CapDist = 0

        foodPos = successorGameState.getFood().asList() 
        foodDist = []
        for food in foodPos:
         foodDist.append(dist(food,newPos))


        if len(foodDist) > 1:
          closeFood = min(foodDist)
        else: 
          closeFood = 1

        #print(closeFood)

        if successorGameState.getNumFood() > 1:
          numFood = successorGameState.getNumFood()
        else:
          numFood = 10

        GhostDist = (min([dist(capDist,successorGameState.getPacmanPosition()) for capDist in successorGameState.getGhostPositions()]) +1)

        x = successorGameState.getScore() - CapDist + 2.03*GhostDist -  numFood -2.53*closeFood
        return x

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

        curDepth = 0
        curAgentindex = 0
        v = self.minimax(gameState,curDepth,curAgentindex)
        return v[1]
        
    def  minimax(self,gameState,curDepth,curAgentindex):


        if curAgentindex >= gameState.getNumAgents():
            curAgentindex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if curAgentindex == 0:
            return self.max_value(gameState, curDepth,curAgentindex)
        else:
            return self.min_value(gameState, curDepth,curAgentindex)

    def max_value(self,gameState,curDepth,curAgentindex):

        v = (-1*np.Infinity,'wait')
        
        if not gameState.getLegalActions(curAgentindex):
          return self.evaluationFunction(gameState)


        for action in gameState.getLegalActions(curAgentindex):

          if action == "Stop":
            continue

          retval = self.minimax(gameState.generateSuccessor(curAgentindex, action),curDepth,curAgentindex+1)

          if type(retval) is tuple:
                retval = retval[0] 


          vNew = max(v[0], retval)

          if vNew is not v[0]:
            v = (vNew,action)

        return v

    def min_value(self,gameState,curDepth,curAgentindex):

        v = (np.Infinity,'wait')
        
        if not gameState.getLegalActions(curAgentindex):
          return self.evaluationFunction(gameState)


        for action in gameState.getLegalActions(curAgentindex):

          if action == "Stop":
            continue

          retval = self.minimax(gameState.generateSuccessor(curAgentindex, action),curDepth,curAgentindex+1)


          if type(retval) is tuple:
                retval = retval[0] 

          vNew = min(v[0], retval)
          
          if vNew is not v[0]:
            v = (vNew,action)

        return v

           

        
        
        




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        curDepth = 0
        curAgentindex = 0
        alpha = -1*np.Infinity
        beta = np.Infinity
        v = self.alpha_beta(gameState,curDepth,curAgentindex,alpha,beta)
        return v[1]
        
    def  alpha_beta(self,gameState,curDepth,curAgentindex,alpha,beta):


        if curAgentindex >= gameState.getNumAgents():
            curAgentindex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if curAgentindex == 0:
            return self.max_value(gameState, curDepth,curAgentindex,alpha,beta)
        else:
            return self.min_value(gameState, curDepth,curAgentindex,alpha,beta)

    def max_value(self,gameState,curDepth,curAgentindex,alpha,beta):

        v = (-1*np.Infinity,'wait')
        
        if not gameState.getLegalActions(curAgentindex):
          return self.evaluationFunction(gameState)


        for action in gameState.getLegalActions(curAgentindex):

          if action == "Stop":
            continue

          retval = self.alpha_beta(gameState.generateSuccessor(curAgentindex, action),curDepth,curAgentindex+1,alpha,beta)

          if type(retval) is tuple:
                retval = retval[0] 

          if retval > beta:
            return (retval,action)


          alpha = max(retval,alpha)

          
          vNew = max(v[0], retval)

          if vNew is not v[0]:
            v = (vNew,action)

        return v

    def min_value(self,gameState,curDepth,curAgentindex,alpha,beta):

        v = (np.Infinity,'wait')
        
        if not gameState.getLegalActions(curAgentindex):
          return self.evaluationFunction(gameState)


        for action in gameState.getLegalActions(curAgentindex):

          if action == "Stop":
            continue

          retval = self.alpha_beta(gameState.generateSuccessor(curAgentindex, action),curDepth,curAgentindex+1,alpha,beta)


          

          if type(retval) is tuple:
                retval = retval[0] 

          if retval < alpha:
            return (retval,action)
          beta = min(retval,beta)

         

          vNew = min(v[0], retval)
          
          if vNew is not v[0]:
            v = (vNew,action)

        return v

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
        curDepth = 0
        curAgentindex = 0
        v = self.expectimax(gameState,curDepth,curAgentindex)
        return v[1]
        
    def  expectimax(self,gameState,curDepth,curAgentindex):


        if curAgentindex >= gameState.getNumAgents():
            curAgentindex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if curAgentindex == 0:
            return self.max_value(gameState, curDepth,curAgentindex)
        else:
            return self.exp_value(gameState, curDepth,curAgentindex)

    def max_value(self,gameState,curDepth,curAgentindex):

        v = (-1*np.Infinity,'wait')
        
        if not gameState.getLegalActions(curAgentindex):
          return self.evaluationFunction(gameState)


        for action in gameState.getLegalActions(curAgentindex):

          if action == "Stop":
            continue

          retval = self.expectimax(gameState.generateSuccessor(curAgentindex, action),curDepth,curAgentindex+1)

          if type(retval) is tuple:
                retval = retval[0] 


          vNew = max(v[0], retval)

          if vNew is not v[0]:
            v = (vNew,action)

        return v

    def exp_value(self,gameState,curDepth,curAgentindex):

        v = (0,'wait')
        vNew = 0
        
        if not gameState.getLegalActions(curAgentindex):
          return self.evaluationFunction(gameState)

        prob = 1.0/len(gameState.getLegalActions(curAgentindex))
        for action in gameState.getLegalActions(curAgentindex):

          if action == "Stop":
            continue

          retval = self.expectimax(gameState.generateSuccessor(curAgentindex, action),curDepth,curAgentindex+1)


          if type(retval) is tuple:
                retval = retval[0] 

          vNew += prob*retval
          
          if vNew is not v[0]:
            v = (vNew,action)

        return v


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
